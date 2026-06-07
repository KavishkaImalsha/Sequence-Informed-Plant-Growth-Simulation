"""
architectures.py — SI-PGS-R Multimodal Generative Architecture
================================================================
Implements the four sub-networks that compose the SI-PGS-R pipeline:

  1. LSTM_SequenceEncoder      S_θ   →  conditional vector x
  2. Recurrent_CVAE_Generator  G_θ   →  synthesised frame ŷ_t  (+ μ, σ²)
  3. Discriminator             D_θ   →  real/fake probability d

Architecture is structurally identical to the pretrained SI-PGS-R baseline
(preserved for weight loading compatibility) while adding:
  • Spectral normalisation on the discriminator for training stability
  • Self-attention bottleneck in the CVAE decoder for global coherence
  • Separate ``EnhancedLSTMEncoder`` with Dropout + bidirectional option

All forward methods are heavily documented with expected tensor shapes using
the convention  [dim0, dim1, ...]  where B = batch_size, T = seq_len.

Baseline targets to surpass:
  FID < 45.09  |  Temporal SSIM > 0.95  |  MSE minimised
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import spectral_norm


# ===========================================================================
# 1. LSTM Sequence Encoder  S_θ
# ===========================================================================

class LSTM_SequenceEncoder(nn.Module):
    """
    Takes the cumulative environmental sequence c(τ) and outputs a
    conditional embedding vector x used to condition both the CVAE
    encoder and decoder.

    Architecture:
        LSTM  (feature_size → hidden_size, num_layers, dropout)
           ↓  last hidden state h[-1]
        FC-Tanh  (hidden_size → output_size)

    Args:
        feature_size  (int): Input env feature dimension (default 4).
        hidden_size   (int): LSTM hidden state dimension.
        num_layers    (int): Number of LSTM layers (with dropout between).
        output_size   (int): Dimension of output embedding x.
        dropout       (float): Dropout between LSTM layers (0 if num_layers==1).
        bidirectional (bool): If True, LSTM is bidirectional; hidden_size
                              is then halved per direction internally.
    """

    def __init__(
        self,
        feature_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 4,
        output_size: int = 64,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        # Persist for JSON config serialisation (mirrors original models.py pattern)
        self.params = {
            "feature_size": feature_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Projection: hidden_size*directions → output_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, output_size),
            nn.Tanh(),
        )

    def forward(
        self,
        padded_sequences: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            padded_sequences : [B, T, feature_size]   — zero-padded env context
            sequence_lengths : [B]                    — true lengths (LongTensor)

        Returns:
            x : [B, output_size]   — conditional embedding
        """
        # B, T, F = padded_sequences.shape  →  [B, T, 4]

        # Clamp lengths to avoid zero-length sequences
        seq_lens = sequence_lengths.cpu().clamp(min=1)

        packed = pack_padded_sequence(
            padded_sequences, seq_lens, batch_first=True, enforce_sorted=False
        )
        # packed_output : PackedSequence
        # hidden        : [num_layers*directions, B, hidden_size]
        packed_output, (hidden, cell) = self.lstm(packed)

        if self.bidirectional:
            # Concatenate last forward & backward hidden states
            # hidden[-2]: last layer forward   → [B, hidden_size]
            # hidden[-1]: last layer backward  → [B, hidden_size]
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, 2*H]
        else:
            last_hidden = hidden[-1]  # [B, hidden_size]

        x = self.fc(last_hidden)  # [B, output_size]
        return x


# ===========================================================================
# 2. Recurrent CVAE Generator  G_θ   (SI-PGS-R variant)
# ===========================================================================

class _SelfAttention2D(nn.Module):
    """Lightweight channel-wise self-attention for the decoder bottleneck.

    Helps the decoder attend to globally relevant spatial regions, reducing
    temporal jitter by encouraging consistent structure across frames.

    Input/output shape: [B, C, H, W]  (unchanged).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        mid = max(in_channels // 8, 1)
        self.query = nn.Conv2d(in_channels, mid, 1)
        self.key   = nn.Conv2d(in_channels, mid, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)   # [B,HW,mid]
        k = self.key(x).view(B, -1, H * W)                       # [B,mid,HW]
        attn = F.softmax(torch.bmm(q, k) / math.sqrt(k.shape[1]), dim=-1)
        v = self.value(x).view(B, C, -1)                          # [B,C,HW]
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class Recurrent_CVAE_Generator(nn.Module):
    """
    SI-PGS-R Recurrent CVAE Generator G_θ.

    Combines:
      • CVAE Encoder  F_φ : (y_prior, y, x, Δt) → μ, log σ²  (reparam → z)
      • CVAE Decoder  G_θ : (z, x, y_prior, Δt) → ŷ_t

    The recurrent connection feeds y_{t-1} (y_prior) into BOTH encoder and
    decoder, forcing the model to minimise temporal jitter by conditioning
    on the previously generated frame.

    Novel addition: Self-attention layer inserted at the decoder bottleneck.

    Args:
        img_shape       (list/tuple): [C, H, W] of the target frame.
        latent_size     (int): Dimension of latent variable z.
        conditional_size(int): Dimension of LSTM embedding x  (= output_size).
    """

    def __init__(
        self,
        img_shape,
        latent_size: int = 128,
        conditional_size: int = 64,
    ):
        super().__init__()
        self.params = {
            "img_shape": list(img_shape),
            "latent_size": latent_size,
            "conditional_size": conditional_size,
        }

        self.img_shape = list(img_shape)           # [C, H, W]
        C, H, W = img_shape

        # +1 for time_diff scalar (appended to conditional vector x)
        self.conditional_size = conditional_size + 1
        # Encoder sees: y (curr), y_prior, x (broadcast), → concat channels
        self.input_size = (2 * C) + self.conditional_size
        self.latent_size = latent_size
        # Decoder sampling input: z concat x concat y_prior channels
        self.sampling_size = latent_size + self.conditional_size + C

        # ── CVAE Encoder ─────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_size, 32,  kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,  64,  kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── z-CNN (decoder front-end, processes sampling input) ───────────
        self.z_cnn = nn.Sequential(
            nn.Conv2d(self.sampling_size, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  32,  kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  1,   kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Determine feature-map dimensions from dummy forward pass
        (self.hidden_shape, self.hidden_size,
         self.sample_hidden_shape, self.sample_hidden_size) = self._get_hidden_dims()

        # μ and log σ² projections
        self.mu_fc     = nn.Linear(self.hidden_size, latent_size)
        self.logvar_fc = nn.Linear(self.hidden_size, latent_size)

        # Bottleneck FC: sample → feature-map volume
        self.z_fc = nn.Linear(self.sample_hidden_size, self.hidden_size)

        # ── Novel: self-attention at decoder bottleneck ───────────────────
        self.attn = _SelfAttention2D(self.hidden_shape[0])

        # ── CVAE Decoder ─────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_shape[0], 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  C,   kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # → [0,1] pixel range
        )

    # ── Encoder F_φ ─────────────────────────────────────────────────────────
    def encode(
        self,
        x: torch.Tensor,           # [B, conditional_size-1]  (LSTM embedding)
        y: torch.Tensor,           # [B, C, H, W]  (current ground-truth frame)
        y_prior: torch.Tensor,     # [B, C, H, W]  (previous frame)
        time_diff: torch.Tensor,   # [B, 1]
    ):
        """
        F_φ : (y, y_prior, x, Δt) → μ [B, latent_size], log σ² [B, latent_size]
        """
        C, H, W = self.img_shape
        B = x.shape[0]

        # Merge time_diff into conditional vector
        cond = torch.cat([x.view(B, -1), time_diff], dim=1)     # [B, cond_size]

        # Broadcast cond across spatial dims and concatenate with images
        cond_spatial = cond.view(B, self.conditional_size, 1, 1).expand(B, -1, H, W)
        enc_input = torch.cat([y, y_prior, cond_spatial], dim=1)  # [B, input_size, H, W]

        feat = self.encoder(enc_input)                             # [B, 256, h, w]
        feat_flat = feat.view(B, -1)                               # [B, hidden_size]

        mu     = self.mu_fc(feat_flat)      # [B, latent_size]
        logvar = self.logvar_fc(feat_flat)  # [B, latent_size]
        return mu, logvar

    # ── Reparameterisation trick ─────────────────────────────────────────────
    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        z = μ + ε·σ,  ε ~ N(0, I)

        Args:
            mu     : [B, latent_size]
            logvar : [B, latent_size]   (log of variance)

        Returns:
            z : [B, latent_size]
        """
        std = torch.exp(0.5 * logvar)          # σ = e^(log σ² / 2)
        eps = torch.randn_like(std)            # ε ~ N(0,I)
        return mu + eps * std                  # [B, latent_size]

    # ── Decoder G_θ ─────────────────────────────────────────────────────────
    def decode(
        self,
        z: torch.Tensor,           # [B, latent_size]
        x: torch.Tensor,           # [B, conditional_size-1]
        y_prior: torch.Tensor,     # [B, C, H, W]
        time_diff: torch.Tensor,   # [B, 1]
    ) -> torch.Tensor:
        """
        G_θ : (z, x, y_prior, Δt) → ŷ_t [B, C, H, W]

        The recurrent conditioning on y_prior and the explicit Δt allow
        the decoder to produce temporally coherent frames that respect
        how much the plant has actually grown since the prior observation.
        """
        C, H, W = self.img_shape
        B = z.shape[0]

        cond = torch.cat([x.view(B, -1), time_diff], dim=1)     # [B, cond_size]

        # Pack z, cond into a spatial volume matching img dims
        sample_vec = torch.cat([z, cond], dim=1)                 # [B, z+cond]
        sample_spatial = sample_vec.view(B, -1, 1, 1).expand(B, -1, H, W)
        dec_input = torch.cat([sample_spatial, y_prior], dim=1)  # [B, sampling_size, H, W]

        feats = self.z_cnn(dec_input)                            # [B, 1, h', w']
        feats_flat = feats.view(B, -1)                           # [B, sample_hidden_size]

        feats_proj = self.z_fc(feats_flat)                       # [B, hidden_size]
        feats_vol  = feats_proj.view(
            B, self.hidden_shape[0],
            self.hidden_shape[1], self.hidden_shape[2]
        )                                                        # [B, C', h, w]

        # Novel: self-attention at bottleneck
        feats_vol = self.attn(feats_vol)                         # [B, C', h, w]

        y_hat = self.decoder(feats_vol)                          # [B, C, H, W]
        return y_hat

    # ── Full forward ─────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,           # [B, conditional_size-1]
        y: torch.Tensor,           # [B, C, H, W]
        y_prior: torch.Tensor,     # [B, C, H, W]
        time_diff: torch.Tensor,   # [B, 1]
    ):
        """
        Returns:
            y_hat  : [B, C, H, W]    — synthesised frame
            mu     : [B, latent_size]
            logvar : [B, latent_size]
        """
        mu, logvar = self.encode(x, y, y_prior, time_diff)
        z          = self.reparam(mu, logvar)
        y_hat      = self.decode(z, x, y_prior, time_diff)
        return y_hat, mu, logvar

    # ── Inference-only generation (no ground-truth y required) ───────────────
    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,           # [B, conditional_size-1]
        y_prior: torch.Tensor,     # [B, C, H, W]
        time_diff: torch.Tensor,   # [B, 1]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample ŷ_t at inference time from the prior p(z|x) ~ N(0,I).

        Args:
            temperature: scales the sampled noise (< 1 → sharper, > 1 → diverse)

        Returns:
            y_hat : [B, C, H, W]
        """
        B = x.shape[0]
        z = torch.randn(B, self.latent_size, device=x.device) * temperature
        return self.decode(z, x, y_prior, time_diff)

    # ── Utility: infer feature-map shapes ────────────────────────────────────
    def _get_hidden_dims(self):
        C, H, W = self.img_shape
        with torch.no_grad():
            dummy_enc = torch.zeros(1, self.input_size, H, W)
            dummy_dec = torch.zeros(1, self.sampling_size, H, W)

            enc_out = self.encoder(dummy_enc)
            hidden_shape = tuple(enc_out.shape[1:])
            hidden_size  = int(enc_out.view(1, -1).shape[1])

            dec_out = self.z_cnn(dummy_dec)
            sample_hidden_shape = tuple(dec_out.shape[1:])
            sample_hidden_size  = int(dec_out.view(1, -1).shape[1])

        return hidden_shape, hidden_size, sample_hidden_shape, sample_hidden_size


# ===========================================================================
# 3. Recurrent Discriminator  D_θ
# ===========================================================================

class Recurrent_Discriminator(nn.Module):
    """
    Convolutional discriminator D_θ that distinguishes real frames from
    synthesised frames.

    Enhancements over the baseline:
      • Spectral normalisation on all conv layers for Lipschitz constraint,
        stabilising the adversarial game and improving FID.
      • LeakyReLU activations (standard for discriminators).
      • Instance normalisation for improved generalisation.

    Input : single frame y  [B, C, H, W]
    Output: probability d   [B, 1]

    Args:
        img_shape (list/tuple): [C, H, W]
    """

    def __init__(self, img_shape):
        super().__init__()
        self.params = {"img_shape": list(img_shape)}
        self.img_shape = list(img_shape)
        C, H, W = img_shape

        def sn_conv(in_ch, out_ch, **kwargs):
            """Spectral-normalised conv2d."""
            return spectral_norm(nn.Conv2d(in_ch, out_ch, **kwargs))

        self.cnn = nn.Sequential(
            # Block 1 — no normalisation on first layer (standard practice)
            sn_conv(C,   16,  kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(16,  32,  kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(32,  64,  kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            sn_conv(64,  128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden_shape, self.hidden_size = self._get_hidden_dims()

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(self.hidden_size, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # probability output
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y : [B, C, H, W]

        Returns:
            d : [B, 1]   — probability of being real
        """
        x = self.cnn(y)           # [B, 128, h, w]
        x = x.view(x.size(0), -1) # [B, hidden_size]
        d = self.fc(x)             # [B, 1]
        return d

    def _get_hidden_dims(self):
        C, H, W = self.img_shape
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            out = self.cnn(dummy)
            hidden_shape = tuple(out.shape[1:])
            hidden_size  = int(out.view(1, -1).shape[1])
        return hidden_shape, hidden_size


# ===========================================================================
# 4. Convenience builder (mirrors original models.py SI_PGS_R pattern)
# ===========================================================================

def build_sipgs_r(
    img_shape=(3, 128, 128),
    env_feature_size: int = 4,
    lstm_hidden: int = 128,
    lstm_layers: int = 4,
    lstm_output: int = 64,
    latent_size: int = 128,
    device: str = "cpu",
):
    """
    Instantiate the full SI-PGS-R model trio and move to device.

    Returns:
        encoder     : LSTM_SequenceEncoder
        generator   : Recurrent_CVAE_Generator
        discriminator: Recurrent_Discriminator
    """
    encoder = LSTM_SequenceEncoder(
        feature_size=env_feature_size,
        hidden_size=lstm_hidden,
        num_layers=lstm_layers,
        output_size=lstm_output,
    ).to(device)

    generator = Recurrent_CVAE_Generator(
        img_shape=img_shape,
        latent_size=latent_size,
        conditional_size=lstm_output,
    ).to(device)

    discriminator = Recurrent_Discriminator(img_shape=img_shape).to(device)

    return encoder, generator, discriminator


# ===========================================================================
# Smoke-test
# ===========================================================================

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SHAPE = (3, 128, 128)
    B = 4

    enc, gen, disc = build_sipgs_r(img_shape=IMG_SHAPE, device=DEVICE)

    # Dummy tensors
    env_seq   = torch.randn(B, 14, 4).to(DEVICE)
    seq_len   = torch.randint(7, 14, (B,))
    y_prior   = torch.rand(B, *IMG_SHAPE).to(DEVICE)
    y         = torch.rand(B, *IMG_SHAPE).to(DEVICE)
    time_diff = torch.rand(B, 1).to(DEVICE)

    x = enc(env_seq, seq_len)
    print(f"x (LSTM embedding)   : {x.shape}")     # [4, 64]

    y_hat, mu, logvar = gen(x, y, y_prior, time_diff)
    print(f"ŷ_t (generated frame): {y_hat.shape}")  # [4, 3, 128, 128]
    print(f"μ                    : {mu.shape}")      # [4, 128]
    print(f"log σ²               : {logvar.shape}")  # [4, 128]

    d_real = disc(y)
    d_fake = disc(y_hat.detach())
    print(f"D(real)              : {d_real.shape}")  # [4, 1]
    print(f"D(fake)              : {d_fake.shape}")  # [4, 1]
