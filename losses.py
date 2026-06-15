"""
losses.py — Composite Loss Functions for SI-PGS-R v2
======================================================
Implements all loss terms of the composite training objective:

    L_G = L_CVAE  +  λ_adv · L_LSGAN  +  λ_fm · L_FM
            +  λ_perc · L_VGG  +  λ_temp · L_temp  +  λ_ssim · L_SSIM

Where:
    L_CVAE  = β·D_KL + (1-β)·L_recon         (β=0.90 default)
    L_LSGAN = E[(D(ŷ) - 1)²]                 (least-squares GAN, more stable than BCE)
    L_FM    = Σ_i ||D_i(y) - D_i(ŷ)||_1      (feature matching at each D scale)
    L_VGG   = Σ_l w_l · ||φ_l(ŷ) - φ_l(y)||²_F  (VGG-16 perceptual)
    L_temp  = e^{-Δt} · ||ŷ - y_prior||²_2   (temporal coherence, weighted by gap)
    L_SSIM  = 1 - SSIM(ŷ, y)                 (directly optimises SSIM_tw)

Key design choices:
  • LSGAN replaces BCE → more stable gradients, less mode collapse
  • VGG perceptual loss is the #1 driver of FID improvement
  • Feature matching (λ_fm=10) prevents discriminator from over-powering generator
  • Temporal loss (weighted by e^{-Δt}) penalises large jumps for nearby frames
  • SSIM loss directly minimises the SSIM_tw target metric

All shapes documented inline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Tuple, Optional


# ===========================================================================
# 1.  VGG Perceptual Loss
# ===========================================================================

class VGGPerceptualLoss(nn.Module):
    """
    Multi-layer perceptual loss using VGG-16 feature maps.

    Extracts features at relu1_2, relu2_2, relu3_3, relu4_3 and computes
    weighted MSE between real and generated feature maps.

    Deeper layers encode more semantic information; weights increase accordingly:
        relu1_2 : w = 1/32   (edge/colour level)
        relu2_2 : w = 1/16   (texture level)
        relu3_3 : w = 1/8    (pattern level)
        relu4_3 : w = 1/4    (semantic level)

    Args:
        resize (bool): Resize input to 224×224 before VGG (recommended for
                       inputs far from ImageNet training resolution).
        normalize (bool): Apply ImageNet normalisation to inputs.

    Input : y_hat, y  [B, 3, H, W]  in [0, 1]
    Output: scalar loss
    """

    # VGG-16 layer indices for relu1_2, relu2_2, relu3_3, relu4_3
    _FEATURE_LAYERS = [3, 8, 15, 22]
    _LAYER_WEIGHTS  = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4]

    def __init__(self, resize: bool = False, normalize: bool = True):
        super().__init__()
        self.resize    = resize
        self.normalize = normalize

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Slice the feature network at each target layer
        self.slices = nn.ModuleList()
        prev_idx = 0
        for idx in self._FEATURE_LAYERS:
            self.slices.append(nn.Sequential(*list(vgg.features[prev_idx:idx + 1])))
            prev_idx = idx + 1

        # Freeze VGG — only used for feature extraction
        for p in self.parameters():
            p.requires_grad_(False)

        # ImageNet normalisation constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] in [0,1] → VGG-ready tensor."""
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        if self.normalize:
            x = (x - self.mean.to(x)) / self.std.to(x)
        return x

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_hat : [B, 3, H, W]  generated frame  [0,1]
            y     : [B, 3, H, W]  ground-truth frame [0,1]

        Returns:
            loss : scalar
        """
        y_hat_p = self._preprocess(y_hat)
        y_p     = self._preprocess(y)

        loss = torch.tensor(0.0, device=y_hat.device)
        x_hat, x_real = y_hat_p, y_p
        for w, sl in zip(self._LAYER_WEIGHTS, self.slices):
            x_hat = sl(x_hat)
            x_real = sl(x_real.detach())   # detach real — no grad through VGG on real
            loss = loss + w * F.mse_loss(x_hat, x_real.detach())
        return loss


# ===========================================================================
# 2.  β-CVAE Loss
# ===========================================================================

def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Analytical KL: D_KL(N(μ,σ²) ‖ N(0,I)) = -0.5·Σ(1 + logσ² - μ² - σ²)

    Args:
        mu     : [B, latent_size]
        logvar : [B, latent_size]

    Returns: scalar (mean over batch and latent dims)
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pixel-wise MSE reconstruction loss.

    MSE is used instead of BCE because:
      1. It is safe inside torch.amp.autocast (BCE is not)
      2. Works correctly for continuous pixel values in [0,1]
      3. Standard choice in modern VAE/CVAE literature

    Args:
        y_hat : [B, C, H, W]  in [0,1]
        y     : [B, C, H, W]  in [0,1]

    Returns: scalar
    """
    return F.mse_loss(y_hat, y.float())


def cvae_loss(
    y_hat:  torch.Tensor,
    y:      torch.Tensor,
    mu:     torch.Tensor,
    logvar: torch.Tensor,
    beta:   float = 0.90,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    β-CVAE objective:
        L_CVAE = β · D_KL + (1-β) · L_recon

    β=0.90 biases toward prior regularisation → generates realistic structures.

    Returns:
        total_cvae : scalar
        kl         : scalar
        recon      : scalar
    """
    kl    = kl_divergence_loss(mu, logvar)
    recon = reconstruction_loss(y_hat, y)
    total = beta * kl + (1.0 - beta) * recon
    return total, kl, recon


# ===========================================================================
# 3.  Least-Squares GAN Losses  (LSGAN)
# ===========================================================================

def _extract_preds(d_output: Dict) -> List[torch.Tensor]:
    """Collect patch prediction grids from multi-scale D output."""
    return [d_output["D1"]["pred"], d_output["D2"]["pred"]]


def generator_lsgan_loss(d_fake_output: Dict) -> torch.Tensor:
    """
    LSGAN generator loss:  E[(D(ŷ) - 1)²]  (want D to output 1 for fake)

    More stable than BCE: provides non-zero gradients even when discriminator
    is saturated (common in early training).

    Args:
        d_fake_output : dict from MultiScale_PatchGAN_Discriminator(y_hat)

    Returns: scalar
    """
    loss = torch.tensor(0.0, device=d_fake_output["D1"]["pred"].device)
    for pred in _extract_preds(d_fake_output):
        loss = loss + F.mse_loss(pred, torch.ones_like(pred))
    return loss / 2.0   # average over scales


def discriminator_lsgan_loss(
    d_real_output: Dict,
    d_fake_output: Dict,
    label_smooth:  float = 0.9,   # one-sided label smoothing: real=0.9 not 1.0
) -> Tuple[torch.Tensor, float]:
    """
    LSGAN discriminator loss with one-sided label smoothing.

    Without label smoothing, D reaches near-zero loss in 1-2 epochs and
    the generator receives no useful adversarial gradient.
    Setting real_label=0.9 keeps D slightly uncertain, maintaining a
    useful gradient signal throughout training.

        L_D = 0.5 · [E[(D(y)-0.9)²]  +  E[D(ŷ)²]]

    Args:
        d_real_output : dict from D(y)
        d_fake_output : dict from D(ŷ.detach())
        label_smooth  : real target label (0.85-0.95 recommended)

    Returns: (loss tensor, scalar float for logging)
    """
    loss = torch.tensor(0.0, device=d_real_output["D1"]["pred"].device)
    preds_real = _extract_preds(d_real_output)
    preds_fake = _extract_preds(d_fake_output)
    for pr, pf in zip(preds_real, preds_fake):
        real_target = torch.full_like(pr, label_smooth)  # 0.9, not 1.0
        loss = loss + 0.5 * (
            F.mse_loss(pr, real_target) +
            F.mse_loss(pf, torch.zeros_like(pf))
        )
    disc_val = (loss / len(preds_real)).item()
    return loss / len(preds_real), disc_val


def compute_discriminator_loss(
    d_real_output: Dict,
    d_fake_output: Dict,
    label_smooth:  float = 0.9,
) -> Tuple[torch.Tensor, float]:
    """Alias kept for backward compatibility with train_v2.py."""
    return discriminator_lsgan_loss(d_real_output, d_fake_output, label_smooth)


# ===========================================================================
# 4.  Feature Matching Loss
# ===========================================================================

def feature_matching_loss(
    d_real_output: Dict,
    d_fake_output: Dict,
    lambda_fm:     float = 1.0,   # NOTE: weighting is now applied externally
) -> torch.Tensor:
    """
    Match intermediate discriminator features between real and fake frames.

    L_FM = (1/N) · Σ_{s∈{D1,D2}} Σ_{i} ||D_s_i(y) - D_s_i(ŷ)||_1

    The lambda_fm weight is applied EXTERNALLY in compute_total_generator_loss.
    This function returns the raw (unweighted) feature matching loss.

    Args:
        d_real_output : dict from D(y)    — real features
        d_fake_output : dict from D(ŷ)   — fake features
        lambda_fm     : kept for API compatibility, not used internally

    Returns: raw (unweighted) scalar loss
    """
    loss = torch.tensor(0.0, device=d_real_output["D1"]["pred"].device)
    n_scales = 0
    for scale in ["D1", "D2"]:
        real_feats = d_real_output[scale]["features"]
        fake_feats = d_fake_output[scale]["features"]
        for rf, ff in zip(real_feats, fake_feats):
            n_elements = rf.numel() / rf.shape[0]   # per-sample element count
            loss = loss + F.l1_loss(ff, rf.detach()) / n_elements
        n_scales += 1
    return lambda_fm * loss / n_scales


# ===========================================================================
# 5.  Temporal Coherence Loss
# ===========================================================================

def temporal_coherence_loss(
    y_hat:     torch.Tensor,   # [B, C, H, W]  generated frame ŷ_t
    y_prior:   torch.Tensor,   # [B, C, H, W]  prior frame y_{t-1}
    time_diff: torch.Tensor,   # [B, 1]        normalised Δt ∈ [0,1]
    lambda_temp: float = 0.5,
) -> torch.Tensor:
    """
    Penalise large frame-to-frame changes relative to the time gap.

        L_temp = mean(e^{-Δt} · ||ŷ_t - y_{t-1}||²)

    Physics: if Δt is small, ŷ_t should look very similar to y_{t-1}.
    If Δt is large, larger changes are acceptable (plant has grown more).

    The exponential weighting e^{-Δt} down-weights long-gap pairs,
    focusing the loss on adjacent frames where temporal consistency matters most.

    Args:
        y_hat, y_prior : [B, C, H, W]
        time_diff      : [B, 1]  normalised to [0,1]

    Returns: scalar
    """
    # Per-pixel squared difference
    sq_diff = (y_hat - y_prior).pow(2)   # [B, C, H, W]

    # Temporal weight: exp(-Δt), shape [B, 1, 1, 1]
    w = torch.exp(-time_diff).view(-1, 1, 1, 1)

    return lambda_temp * (w * sq_diff).mean()


# ===========================================================================
# 6.  SSIM Loss
# ===========================================================================

def ssim_loss_fn(
    y_hat: torch.Tensor,
    y:     torch.Tensor,
    window_size: int = 11,
    sigma:       float = 1.5,
) -> torch.Tensor:
    """
    SSIM loss: 1 - SSIM(ŷ, y)

    Directly optimises for the SSIM_tw evaluation metric.
    Uses a 2D Gaussian window for structural similarity computation.

    Args:
        y_hat       : [B, C, H, W]
        y           : [B, C, H, W]
        window_size : Gaussian window size (default 11)
        sigma       : Gaussian sigma (default 1.5)

    Returns: scalar  (0 = identical, 2 = maximally dissimilar)
    """
    C = y_hat.shape[1]

    # 1-D Gaussian kernel
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-((x - window_size // 2) ** 2) / (2 * sigma ** 2)))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()

    # 2-D separable kernel [1, 1, ws, ws]
    kernel_1d = gauss.unsqueeze(1)                        # [ws, 1]
    kernel_2d = kernel_1d.mm(kernel_1d.t()).float()       # [ws, ws]
    kernel    = kernel_2d.unsqueeze(0).unsqueeze(0)       # [1, 1, ws, ws]
    kernel    = kernel.expand(C, 1, window_size, window_size).to(y_hat.device)

    pad = window_size // 2

    def _ssim_pass(x1, x2):
        mu1  = F.conv2d(x1, kernel, padding=pad, groups=C)
        mu2  = F.conv2d(x2, kernel, padding=pad, groups=C)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

        s1  = F.conv2d(x1 * x1, kernel, padding=pad, groups=C) - mu1_sq
        s2  = F.conv2d(x2 * x2, kernel, padding=pad, groups=C) - mu2_sq
        s12 = F.conv2d(x1 * x2, kernel, padding=pad, groups=C) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
        return ssim_map.mean()

    return 1.0 - _ssim_pass(y_hat, y)


# ===========================================================================
# 7.  Combined Loss Calculators
# ===========================================================================

def compute_total_generator_loss(
    y_hat:         torch.Tensor,
    y:             torch.Tensor,
    mu:            torch.Tensor,
    logvar:        torch.Tensor,
    y_prior:       torch.Tensor,
    time_diff:     torch.Tensor,
    d_fake_output: Dict,
    d_real_output: Dict,
    perceptual_fn: VGGPerceptualLoss,
    # Loss weights
    beta:          float = 0.90,
    lambda_adv:    float = 0.05,
    lambda_fm:     float = 10.0,
    lambda_perc:   float = 1.0,
    lambda_temp:   float = 0.5,
    lambda_ssim:   float = 1.0,
    lambda_mse:    float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute total generator loss with all components.

    Returns:
        total_loss : scalar tensor (with grad)
        log_dict   : dict of scalar floats for logging
    """
    # Individual terms
    l_cvae, l_kl, l_recon = cvae_loss(y_hat, y, mu, logvar, beta=beta)
    l_adv  = generator_lsgan_loss(d_fake_output)
    l_fm   = feature_matching_loss(d_real_output, d_fake_output, lambda_fm=lambda_fm)
    l_perc = perceptual_fn(y_hat, y)
    l_temp = temporal_coherence_loss(y_hat, y_prior, time_diff, lambda_temp=lambda_temp)
    l_ssim = ssim_loss_fn(y_hat, y)
    l_mse  = F.mse_loss(y_hat, y)

    total = (
        l_cvae
        + lambda_adv  * l_adv
        + lambda_fm   * l_fm      # λ_fm applied here ONLY (not inside feature_matching_loss)
        + lambda_perc * l_perc
        + lambda_ssim * l_ssim
        + lambda_mse  * l_mse
        + l_temp
    )

    log_dict = {
        "loss_gen":   total.item(),
        "loss_cvae":  l_cvae.item(),
        "loss_kl":    l_kl.item(),
        "loss_recon": l_recon.item(),
        "loss_adv_g": l_adv.item(),
        "loss_fm":    l_fm.item(),
        "loss_perc":  l_perc.item(),
        "loss_temp":  l_temp.item(),
        "loss_ssim":  l_ssim.item(),
        "loss_mse":   l_mse.item(),
    }
    return total, log_dict


def compute_discriminator_loss(
    d_real_output: Dict,
    d_fake_output: Dict,
) -> Tuple[torch.Tensor, float]:
    """
    LSGAN discriminator loss.

    Returns:
        loss_disc : scalar tensor
        loss_val  : float for logging
    """
    loss, disc_val = discriminator_lsgan_loss(d_real_output, d_fake_output)
    return loss, disc_val


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == "__main__":
    B, C, H, W = 2, 3, 240, 320
    y_hat  = torch.sigmoid(torch.randn(B, C, H, W))
    y      = torch.sigmoid(torch.randn(B, C, H, W))
    mu     = torch.randn(B, 256)
    logvar = torch.randn(B, 256)
    y_prior   = torch.sigmoid(torch.randn(B, C, H, W))
    time_diff = torch.rand(B, 1)

    # Fake D output structure
    def _fake_d_out(B, device="cpu"):
        return {
            "D1": {
                "pred": torch.rand(B, 1, 28, 38),
                "features": [torch.rand(B, 64, 120, 160),
                             torch.rand(B, 128, 60, 80),
                             torch.rand(B, 256, 30, 40),
                             torch.rand(B, 512, 30, 40)],
            },
            "D2": {
                "pred": torch.rand(B, 1, 14, 19),
                "features": [torch.rand(B, 64, 60, 80),
                             torch.rand(B, 128, 30, 40),
                             torch.rand(B, 256, 15, 20),
                             torch.rand(B, 512, 15, 20)],
            },
        }

    d_real = _fake_d_out(B)
    d_fake = _fake_d_out(B)

    vgg_loss = VGGPerceptualLoss(resize=True)

    total, logs = compute_total_generator_loss(
        y_hat, y, mu, logvar, y_prior, time_diff,
        d_fake, d_real, vgg_loss
    )
    print(f"Total generator loss: {total.item():.4f}")
    for k, v in logs.items():
        print(f"  {k:15s}: {v:.4f}")

    d_loss, _ = compute_discriminator_loss(d_real, d_fake)
    print(f"Discriminator loss  : {d_loss.item():.4f}")
    print("losses.py smoke test PASSED")
