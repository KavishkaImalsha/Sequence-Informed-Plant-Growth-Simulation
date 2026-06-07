"""
architectures_v2.py — Research-Grade Enhanced SI-PGS-R Architecture
=====================================================================
Major improvements over the baseline SI-PGS-R:

1. AttentionLSTMEncoder (S_θ):
   - Bidirectional LSTM over env sequence
   - Multi-Head Self-Attention aggregation over all time-steps
   - Captures WHICH environmental days matter most (not just the last)

2. UNet_Recurrent_CVAE_Generator (G_θ):
   - PriorFeatureExtractor: encodes y_{t-1} at 4 spatial scales
   - U-Net skip connections from prior features into decoder
     → preserves fine spatial structure → better SSIM_tw
   - Self-attention at bottleneck (gamma-learned)
   - Residual blocks at bottleneck for capacity
   - Larger latent space (256-dim)

3. MultiScale_PatchGAN_Discriminator (D_θ):
   - Two PatchGAN discriminators (original + 0.5× resolution)
   - PatchGAN output: grid of scores rather than single scalar
     → richer gradient signal → better FID
   - Spectral normalisation throughout
   - Returns intermediate features for feature-matching loss

Design resolution: 240×320 (actual GrowliFlowerL/R patch dimensions)

All tensor shapes documented inline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Residual block: Conv-IN-ReLU → Conv-IN → + input."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)   # [B, C, H, W]


class _SelfAttention2D(nn.Module):
    """Lightweight spatial self-attention with learnable γ scale.

    Starts as identity (γ=0), progressively learns global structure.
    Input/output: [B, C, H, W]
    """

    def __init__(self, in_channels: int):
        super().__init__()
        mid = max(in_channels // 8, 1)
        self.q = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.k = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        q = self.q(x).view(B, -1, N).permute(0, 2, 1)   # [B, N, mid]
        k = self.k(x).view(B, -1, N)                      # [B, mid, N]
        scale = math.sqrt(k.shape[1])
        attn = F.softmax(torch.bmm(q, k) / scale, dim=-1) # [B, N, N]
        v = self.v(x).view(B, C, N)                        # [B, C, N]
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


def _sn_conv(in_ch: int, out_ch: int, **kwargs) -> nn.Conv2d:
    """Spectral-normalised Conv2d."""
    return spectral_norm(nn.Conv2d(in_ch, out_ch, **kwargs))


# ===========================================================================
# 1.  PRIOR FEATURE EXTRACTOR  (shared between encoder & decoder paths)
# ===========================================================================

class PriorFeatureExtractor(nn.Module):
    """
    Encode y_{t-1} (prior frame) at 4 spatial scales.
    The multi-scale feature maps serve as U-Net skip connections inside the
    decoder, preserving fine spatial details and enforcing temporal coherence.

    All weights can optionally be frozen after the generator is trained
    (useful for ablation studies).

    Input : y_prior [B, 3, H, W]   H=240, W=320

    Returns: list of 4 tensors
        skip_0 : [B,  64, H/2,  W/2 ]  = [B,  64, 120, 160]
        skip_1 : [B, 128, H/4,  W/4 ]  = [B, 128,  60,  80]
        skip_2 : [B, 256, H/8,  W/8 ]  = [B, 256,  30,  40]
        skip_3 : [B, 512, H/16, W/16]  = [B, 512,  15,  20]
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def _block(ic, oc, first=False):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1, bias=False)]
            if not first:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.block0 = _block(in_channels, 64,  first=True)  # → [B, 64, H/2, W/2]
        self.block1 = _block(64,          128)               # → [B, 128, H/4, W/4]
        self.block2 = _block(128,         256)               # → [B, 256, H/8, W/8]
        self.block3 = _block(256,         512)               # → [B, 512, H/16, W/16]

    def forward(self, y_prior: torch.Tensor) -> List[torch.Tensor]:
        s0 = self.block0(y_prior)  # [B, 64, 120, 160]
        s1 = self.block1(s0)       # [B, 128, 60, 80]
        s2 = self.block2(s1)       # [B, 256, 30, 40]
        s3 = self.block3(s2)       # [B, 512, 15, 20]
        return [s0, s1, s2, s3]


# ===========================================================================
# 2.  ATTENTION LSTM ENCODER  S_θ
# ===========================================================================

class AttentionLSTMEncoder(nn.Module):
    """
    Encodes the cumulative environmental sequence c(τ) → conditional vector x.

    Architecture:
        Bidirectional LSTM  →  per-step hidden states  H  [B, T, 2·hidden]
                            ↓
        Multi-Head Self-Attention over H  → attended context A  [B, T, 2·hidden]
                            ↓
        Mean pool + FC-Tanh  →  x  [B, output_size]

    Self-attention lets the encoder focus on the most informative days in the
    environmental history rather than relying solely on the last LSTM state.

    Args:
        feature_size  (int): Env feature dimension (4).
        hidden_size   (int): Per-direction LSTM hidden size.
        num_layers    (int): LSTM depth.
        output_size   (int): Output embedding dimension (= conditional_size for generator).
        num_heads     (int): Multi-head attention heads.
        dropout       (float): LSTM inter-layer dropout.
    """

    def __init__(
        self,
        feature_size:  int = 4,
        hidden_size:   int = 256,
        num_layers:    int = 4,
        output_size:   int = 128,
        num_heads:     int = 4,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.params = dict(
            feature_size=feature_size, hidden_size=hidden_size,
            num_layers=num_layers, output_size=output_size,
            num_heads=num_heads, dropout=dropout,
        )
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        lstm_out_dim = hidden_size * 2   # bidirectional

        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Self-attention over LSTM outputs
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(lstm_out_dim)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, output_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 2, output_size),
            nn.Tanh(),
        )

    def forward(
        self,
        padded_sequences: torch.Tensor,   # [B, T, feature_size]
        sequence_lengths: torch.Tensor,   # [B]   (LongTensor)
    ) -> torch.Tensor:
        """
        Returns:
            x : [B, output_size]  — environmental conditional embedding
        """
        B, T, _ = padded_sequences.shape
        seq_lens = sequence_lengths.cpu().clamp(min=1)

        # 1. Bidirectional LSTM
        packed = pack_padded_sequence(
            padded_sequences, seq_lens, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, lengths = pad_packed_sequence(packed_out, batch_first=True)
        # out: [B, T', 2*hidden]  (T' ≤ T, actual sequence length)

        # Pad/trim to T for consistent shape (needed for attention mask)
        if out.shape[1] < T:
            pad = torch.zeros(
                B, T - out.shape[1], out.shape[2],
                device=out.device, dtype=out.dtype
            )
            out = torch.cat([out, pad], dim=1)   # [B, T, 2*hidden]

        # 2. Build padding mask for attention (True = ignore)
        mask = torch.arange(T, device=out.device).unsqueeze(0) >= seq_lens.unsqueeze(1).to(out.device)
        # mask: [B, T]  — True for padding positions

        # 3. Multi-head self-attention over LSTM outputs
        attn_out, _ = self.attn(
            out, out, out,
            key_padding_mask=mask.bool(),
        )   # [B, T, 2*hidden]
        attn_out = self.norm(attn_out + out)   # residual + layernorm

        # 4. Mean-pool over valid time steps
        valid = (~mask).float().unsqueeze(-1)   # [B, T, 1]
        context = (attn_out * valid).sum(1) / valid.sum(1).clamp(min=1)   # [B, 2*hidden]

        # 5. Project to output_size
        x = self.fc(context)   # [B, output_size]
        return x


# ===========================================================================
# 3.  U-NET RECURRENT CVAE GENERATOR  G_θ
# ===========================================================================

class UNet_Recurrent_CVAE_Generator(nn.Module):
    """
    SI-PGS-R Generator with U-Net skip connections from y_prior.

    The U-Net design fundamentally differs from the baseline:
      • Baseline decoder: z + x + y_prior → CNN → ŷ_t  (no skip)
      • This decoder : z + x + skip_0..3  → ConvT×4 + fuse → ŷ_t

    Skip connections carry fine spatial details from y_{t-1} into every
    decoder resolution, directly improving temporal SSIM and MSE_tw.

    Training forward:  encode(y, y_prior, x, Δt) → z → decode → ŷ_t
    Inference forward: sample z ~ N(0,I)          →    decode → ŷ_t

    Args:
        img_shape        (list): [C, H, W]  default [3, 240, 320]
        latent_size      (int) : Latent variable dimension z.
        conditional_size (int) : LSTM output dimension (= output_size of encoder).
        n_res_blocks     (int) : Residual blocks at bottleneck.
    """

    def __init__(
        self,
        img_shape        = (3, 240, 320),
        latent_size:  int = 256,
        conditional_size: int = 128,
        n_res_blocks: int = 2,
    ):
        super().__init__()
        self.params = dict(
            img_shape=list(img_shape),
            latent_size=latent_size,
            conditional_size=conditional_size,
            n_res_blocks=n_res_blocks,
        )
        C, H, W = img_shape
        self.img_shape        = list(img_shape)
        self.latent_size      = latent_size
        self.cond_size        = conditional_size + 1   # +1 for time_diff scalar
        self.n_res_blocks     = n_res_blocks

        enc_in = 2 * C + self.cond_size   # y + y_prior + cond broadcast

        # ── Prior Feature Extractor (shared) ─────────────────────────────
        self.prior_feat = PriorFeatureExtractor(C)
        # Produces skips: [64,H/2,W/2]  [128,H/4,W/4]  [256,H/8,W/8]  [512,H/16,W/16]

        # ── CVAE Encoder F_φ ─────────────────────────────────────────────
        def enc_block(ic, oc, first=False):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1, bias=False)]
            if not first:
                layers += [nn.InstanceNorm2d(oc, affine=True)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)

        self.enc_block0 = enc_block(enc_in, 64,  first=True)  # [B, 64, H/2, W/2]
        self.enc_block1 = enc_block(64,     128)               # [B, 128, H/4, W/4]
        self.enc_block2 = enc_block(128,    256)               # [B, 256, H/8, W/8]
        self.enc_block3 = enc_block(256,    512)               # [B, 512, H/16, W/16]

        # Compute bottleneck spatial size from a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, enc_in, H, W)
            bot   = self.enc_block3(
                        self.enc_block2(
                            self.enc_block1(
                                self.enc_block0(dummy))))
            self.bot_shape = tuple(bot.shape[1:])   # (512, H/16, W/16)
            self.bot_flat  = int(bot.view(1, -1).shape[1])

        self.mu_fc     = nn.Linear(self.bot_flat, latent_size)
        self.logvar_fc = nn.Linear(self.bot_flat, latent_size)

        # ── Bottleneck: z → feature volume ─────────────────────────────
        z_in = latent_size + self.cond_size
        self.z_fc = nn.Sequential(
            nn.Linear(z_in, self.bot_flat),
            nn.ReLU(inplace=True),
        )

        # Residual blocks + self-attention at bottleneck
        self.res_blocks = nn.Sequential(
            *[_ResBlock(self.bot_shape[0]) for _ in range(n_res_blocks)]
        )
        self.attn = _SelfAttention2D(self.bot_shape[0])

        # ── Decoder U-Net blocks ──────────────────────────────────────────
        # Each block: ConvT → cat(skip from PFE) → conv to fuse → IN + ReLU
        # skip sizes:  [B,512,H/16] → [B,256,H/8] → [B,128,H/4] → [B,64,H/2]

        def up_block(in_ch, skip_ch, out_ch):
            return nn.ModuleDict({
                "upsample": nn.Sequential(
                    nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(in_ch, affine=True),
                    nn.ReLU(inplace=True),
                ),
                "fuse": nn.Sequential(
                    nn.Conv2d(in_ch + skip_ch, out_ch, 3, 1, 1, bias=False),
                    nn.InstanceNorm2d(out_ch, affine=True),
                    nn.ReLU(inplace=True),
                ),
            })

        self.up3 = up_block(512, 256, 256)   # bot → [B,256,H/8]  + skip_2(256)
        self.up2 = up_block(256, 128, 128)   # → [B,128,H/4]  + skip_1(128)
        self.up1 = up_block(128,  64,  64)   # → [B, 64,H/2]  + skip_0( 64)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(64, C, 4, 2, 1),
            nn.Sigmoid(),
        )   # → [B, C, H, W]

    # ── Encode ─────────────────────────────────────────────────────────────
    def encode(
        self,
        x:         torch.Tensor,   # [B, cond_size-1]
        y:         torch.Tensor,   # [B, C, H, W]
        y_prior:   torch.Tensor,   # [B, C, H, W]
        time_diff: torch.Tensor,   # [B, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """F_φ → μ [B, latent], logvar [B, latent]"""
        C, H, W = self.img_shape
        B = x.shape[0]
        cond = torch.cat([x.view(B, -1), time_diff], dim=1)       # [B, cond_size]
        cond_sp = cond.view(B, self.cond_size, 1, 1).expand(B, -1, H, W)
        enc_in  = torch.cat([y, y_prior, cond_sp], dim=1)         # [B, enc_in, H, W]

        f = self.enc_block0(enc_in)   # [B, 64,  H/2, W/2]
        f = self.enc_block1(f)        # [B, 128, H/4, W/4]
        f = self.enc_block2(f)        # [B, 256, H/8, W/8]
        f = self.enc_block3(f)        # [B, 512, H/16, W/16]
        f = f.view(B, -1)             # [B, bot_flat]

        return self.mu_fc(f), self.logvar_fc(f)   # [B, latent], [B, latent]

    # ── Reparameterisation ─────────────────────────────────────────────────
    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + ε·σ,  ε ~ N(0,I).  [B, latent]"""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ── Decode ─────────────────────────────────────────────────────────────
    def decode(
        self,
        z:         torch.Tensor,              # [B, latent]
        x:         torch.Tensor,              # [B, cond_size-1]
        y_prior:   torch.Tensor,              # [B, C, H, W]
        time_diff: torch.Tensor,              # [B, 1]
        skips:     Optional[List] = None,     # pre-computed prior features
    ) -> torch.Tensor:
        """G_θ with U-Net skip connections.  Returns ŷ_t [B, C, H, W]."""
        B = z.shape[0]
        C, H, W = self.img_shape

        # Compute prior skip features if not provided
        if skips is None:
            skips = self.prior_feat(y_prior)   # [s0, s1, s2, s3]
        s0, s1, s2, s3 = skips

        cond = torch.cat([x.view(B, -1), time_diff], dim=1)   # [B, cond_size]
        z_in = torch.cat([z, cond], dim=1)                     # [B, latent + cond_size]

        # Map to bottleneck volume
        feat = self.z_fc(z_in)                                 # [B, bot_flat]
        feat = feat.view(B, *self.bot_shape)                   # [B, 512, H/16, W/16]

        # Residual blocks + attention
        feat = self.res_blocks(feat)   # [B, 512, H/16, W/16]
        feat = self.attn(feat)         # [B, 512, H/16, W/16]

        # U-Net decoder
        f = self.up3["upsample"](feat)                         # [B, 512, H/8, W/8]
        f = self.up3["fuse"](torch.cat([f, s2], dim=1))        # [B, 256, H/8, W/8]

        f = self.up2["upsample"](f)                            # [B, 256, H/4, W/4]
        f = self.up2["fuse"](torch.cat([f, s1], dim=1))        # [B, 128, H/4, W/4]

        f = self.up1["upsample"](f)                            # [B, 128, H/2, W/2]
        f = self.up1["fuse"](torch.cat([f, s0], dim=1))        # [B, 64, H/2, W/2]

        y_hat = self.up0(f)                                    # [B, C, H, W]
        return y_hat

    # ── Training forward ───────────────────────────────────────────────────
    def forward(
        self,
        x:         torch.Tensor,   # [B, cond_size-1]
        y:         torch.Tensor,   # [B, C, H, W]
        y_prior:   torch.Tensor,   # [B, C, H, W]
        time_diff: torch.Tensor,   # [B, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            y_hat  : [B, C, H, W]
            mu     : [B, latent_size]
            logvar : [B, latent_size]
        """
        # Compute prior features once (shared between encode path and decode)
        skips = self.prior_feat(y_prior)
        mu, logvar = self.encode(x, y, y_prior, time_diff)
        z   = self.reparam(mu, logvar)
        y_hat = self.decode(z, x, y_prior, time_diff, skips=skips)
        return y_hat, mu, logvar

    # ── Inference-only (no y_t required) ───────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        x:           torch.Tensor,   # [B, cond_size-1]
        y_prior:     torch.Tensor,   # [B, C, H, W]
        time_diff:   torch.Tensor,   # [B, 1]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample z ~ N(0,I·temp²) and decode → ŷ_t [B, C, H, W]."""
        B = x.shape[0]
        z = torch.randn(B, self.latent_size, device=x.device) * temperature
        return self.decode(z, x, y_prior, time_diff)


# ===========================================================================
# 4.  MULTI-SCALE PATCHGAN DISCRIMINATOR  D_θ
# ===========================================================================

class _PatchGAN(nn.Module):
    """
    N-Layer PatchGAN (N=4) with spectral normalisation.

    Outputs a spatial grid of real/fake scores rather than a single value.
    Intermediate feature maps are also returned for feature-matching loss.

    Input  : [B, 3, H, W]
    Output : {
        'pred'    : [B, 1, H_p, W_p]    — patch prediction grid
        'features': list of 4 tensors   — intermediate features
    }
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        def _blk(ic, oc, stride=2, norm=True):
            layers = [_sn_conv(ic, oc, kernel_size=4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.block0 = _blk(in_channels, 64,  stride=2, norm=False)  # → H/2
        self.block1 = _blk(64,          128, stride=2)               # → H/4
        self.block2 = _blk(128,         256, stride=2)               # → H/8
        self.block3 = _blk(256,         512, stride=1)               # → H/8 (stride 1, larger RF)
        self.pred   = _sn_conv(512, 1, kernel_size=4, stride=1, padding=1)  # patch output

    def forward(self, x: torch.Tensor) -> Dict:
        f0 = self.block0(x)    # [B, 64,  H/2, W/2]
        f1 = self.block1(f0)   # [B, 128, H/4, W/4]
        f2 = self.block2(f1)   # [B, 256, H/8, W/8]
        f3 = self.block3(f2)   # [B, 512, H/8, W/8]
        p  = self.pred(f3)     # [B, 1,   H_p, W_p]
        return {"pred": p, "features": [f0, f1, f2, f3]}


class MultiScale_PatchGAN_Discriminator(nn.Module):
    """
    Two-scale PatchGAN discriminator.

    D1 operates on full resolution;  D2 on 0.5× downsampled input.
    Multi-scale discrimination captures both global composition (D2)
    and local texture quality (D1), directly improving FID.

    Spectral normalisation enforces the Lipschitz constraint,
    stabilising training without needing gradient penalty.

    Input : y [B, 3, H, W]
    Output: dict with D1/D2 predictions and features
    """

    def __init__(self, img_shape=(3, 240, 320)):
        super().__init__()
        self.params = {"img_shape": list(img_shape)}
        C = img_shape[0]
        self.D1 = _PatchGAN(C)
        self.D2 = _PatchGAN(C)
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, y: torch.Tensor) -> Dict:
        """
        Args:
            y : [B, 3, H, W]

        Returns dict:
            'D1' : {'pred': [B,1,H_p,W_p], 'features': [...]}
            'D2' : {'pred': [B,1,H_p//2,W_p//2], 'features': [...]}
        """
        out_d1 = self.D1(y)
        out_d2 = self.D2(self.downsample(y))
        return {"D1": out_d1, "D2": out_d2}


# ===========================================================================
# 5.  Factory
# ===========================================================================

def build_sipgs_r_v2(
    img_shape        = (3, 240, 320),
    env_feature_size: int = 4,
    lstm_hidden:      int = 256,
    lstm_layers:      int = 4,
    lstm_output:      int = 128,
    lstm_heads:       int = 4,
    latent_size:      int = 256,
    n_res_blocks:     int = 2,
    device:           str = "cpu",
):
    """
    Instantiate the full SI-PGS-R v2 model trio.

    Returns:
        encoder      : AttentionLSTMEncoder
        generator    : UNet_Recurrent_CVAE_Generator
        discriminator: MultiScale_PatchGAN_Discriminator
    """
    encoder = AttentionLSTMEncoder(
        feature_size=env_feature_size,
        hidden_size=lstm_hidden,
        num_layers=lstm_layers,
        output_size=lstm_output,
        num_heads=lstm_heads,
    ).to(device)

    generator = UNet_Recurrent_CVAE_Generator(
        img_shape=img_shape,
        latent_size=latent_size,
        conditional_size=lstm_output,
        n_res_blocks=n_res_blocks,
    ).to(device)

    discriminator = MultiScale_PatchGAN_Discriminator(
        img_shape=img_shape
    ).to(device)

    return encoder, generator, discriminator


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG    = (3, 240, 320)
    B      = 2

    enc, gen, disc = build_sipgs_r_v2(img_shape=IMG, device=DEVICE)

    env_seq   = torch.randn(B, 14, 4).to(DEVICE)
    seq_len   = torch.tensor([14, 12]).to(DEVICE)
    y_prior   = torch.rand(B, *IMG).to(DEVICE)
    y         = torch.rand(B, *IMG).to(DEVICE)
    time_diff = torch.rand(B, 1).to(DEVICE)

    x              = enc(env_seq, seq_len)
    print(f"x              : {x.shape}")         # [2, 128]

    y_hat, mu, lv  = gen(x, y, y_prior, time_diff)
    print(f"y_hat          : {y_hat.shape}")      # [2, 3, 240, 320]
    print(f"mu             : {mu.shape}")          # [2, 256]

    d_out = disc(y)
    print(f"D1 pred        : {d_out['D1']['pred'].shape}")   # [2, 1, 28, 38] ≈
    print(f"D2 pred        : {d_out['D2']['pred'].shape}")

    ep = sum(p.numel() for p in enc.parameters())
    gp = sum(p.numel() for p in gen.parameters())
    dp = sum(p.numel() for p in disc.parameters())
    print(f"\nEncoder      : {ep:>12,} params")
    print(f"Generator    : {gp:>12,} params")
    print(f"Discriminator: {dp:>12,} params")
    print(f"Total        : {ep+gp+dp:>12,} params")
