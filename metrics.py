"""
metrics.py — Research-Grade Evaluation Metrics for SI-PGS-R
============================================================
Implements the full metric suite from the SI-PGS paper:

    ┌─────────────────────────────────────────────────────────┐
    │  Generation Quality   │  Frame Coherence                │
    │  ─────────────────── │  ──────────────────────────────  │
    │  FID                 │  MSE_tw   SSIM_tw               │
    │  MSE (pixel-scale)   │  CS-MSE_tw  CS-SSIM_tw          │
    └─────────────────────────────────────────────────────────┘

Metric definitions:
    MSE        : mean((ŷ_t - y_t)² · 255²)    per-pixel MSE in 0-255 scale
    FID        : Fréchet Inception Distance
    MSE_tw     : mean((ŷ_t - ŷ_{t-1})²·255²)  temporal smoothness of generated seq
    SSIM_tw    : SSIM computed over consecutive generated frame pairs (window-averaged)
    CS-MSE_tw  : Same as MSE_tw but on REAL sequences (cross-sequence reference)
    CS-SSIM_tw : Same as SSIM_tw but on REAL sequences (cross-sequence reference)

These are sequence-level metrics — the `SequenceEvaluator` class
groups validation pairs by plant sequence and computes them correctly.

Usage:
    evaluator = SequenceEvaluator(device)
    evaluator.update(y_hat, y, y_prior, y_real_prior, seq_ids)
    results = evaluator.compute()
    evaluator.reset()
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from torchvision.models import inception_v3
from torchvision import transforms as TF
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


# ===========================================================================
# FID utilities
# ===========================================================================

class InceptionFeatureExtractor(nn.Module):
    """
    Extract Inception-v3 pool-3 feature vectors for FID computation.

    Resizes inputs to 299×299 (required by Inception).
    Returns [B, 2048] numpy array.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        inc = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
        # Build feature extractor up to AvgPool
        self.features = nn.Sequential(
            inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2),
            inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c,
            inc.Mixed_6d, inc.Mixed_6e,
            inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.features.eval().to(device)
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.resize = TF.Resize((299, 299), antialias=True)

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> np.ndarray:
        """
        Args:
            imgs : [B, 3, H, W]  in [0, 1]

        Returns:
            feats : [B, 2048]  numpy
        """
        x = self.resize(imgs.to(self.device))
        f = self.features(x)                          # [B, 2048, 1, 1]
        return f.squeeze(-1).squeeze(-1).cpu().numpy()  # [B, 2048]


def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """
    Fréchet Inception Distance:
        FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2·√(Σ_r·Σ_f))

    Args:
        real_feats : [N, 2048]
        fake_feats : [N, 2048]

    Returns:
        fid : float (lower is better)
    """
    from scipy import linalg
    mu1, sigma1 = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return max(fid, 0.0)   # numerical safety


# ===========================================================================
# Per-frame metrics (pixel scale = 0-255)
# ===========================================================================

def compute_mse_pixel(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    MSE in raw pixel scale (0-255 squared), matching the paper's table values.

    Args:
        y_hat : [B, C, H, W]  in [0, 1]
        y     : [B, C, H, W]  in [0, 1]

    Returns:
        mse : float   (same scale as table: e.g. 2441, 6858)
    """
    return (F.mse_loss(y_hat, y) * (255.0 ** 2)).item()


def compute_ssim_value(y_hat: torch.Tensor, y: torch.Tensor, device: str = "cpu") -> float:
    """
    SSIM between generated and real frames.

    Args:
        y_hat : [B, C, H, W]  in [0, 1]
        y     : [B, C, H, W]  in [0, 1]

    Returns:
        ssim : float  in [-1, 1]  (1.0 = identical)
    """
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return float(metric(y_hat.to(device), y.to(device)).item())


# ===========================================================================
# Temporal metrics (sequence-level)
# ===========================================================================

def compute_mse_tw(
    y_hat_sequence: torch.Tensor,   # [T, C, H, W]  generated sequence
) -> float:
    """
    Temporal MSE (MSE_tw):
        MSE_tw = mean_t( ||ŷ_t - ŷ_{t-1}||² · 255² )

    Measures temporal smoothness of the generated sequence.
    Lower = smoother (more temporally coherent) generation.

    Args:
        y_hat_sequence : [T, C, H, W]  concatenated generated frames (T ≥ 2)

    Returns:
        mse_tw : float
    """
    if y_hat_sequence.shape[0] < 2:
        return 0.0
    diff = y_hat_sequence[1:] - y_hat_sequence[:-1]     # [T-1, C, H, W]
    return float((diff.pow(2).mean() * (255.0 ** 2)).item())


def compute_ssim_tw(
    y_hat_sequence: torch.Tensor,   # [T, C, H, W]
    device:         str = "cpu",
) -> float:
    """
    Temporal SSIM (SSIM_tw):
        SSIM_tw = mean_t( SSIM(ŷ_t, ŷ_{t-1}) )

    Measures structural similarity between consecutive generated frames.
    Higher = more temporally coherent.

    Args:
        y_hat_sequence : [T, C, H, W]  (T ≥ 2)

    Returns:
        ssim_tw : float  (higher is better, target > 0.95)
    """
    if y_hat_sequence.shape[0] < 2:
        return 1.0

    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_vals = []
    for t in range(1, y_hat_sequence.shape[0]):
        a = y_hat_sequence[t - 1:t].to(device)
        b = y_hat_sequence[t:t + 1].to(device)
        ssim_vals.append(float(metric(a, b).item()))
        metric.reset()
    return float(np.mean(ssim_vals))


def compute_cs_mse_tw(
    y_sequence: torch.Tensor,   # [T, C, H, W]  REAL sequence
) -> float:
    """
    Cross-Sequence Temporal MSE (CS-MSE_tw):
    Same as MSE_tw but computed on the REAL frame sequence.
    Serves as a reference baseline for how much real plants change frame-to-frame.

    Args:
        y_sequence : [T, C, H, W]  real ground-truth frames

    Returns:
        cs_mse_tw : float
    """
    return compute_mse_tw(y_sequence)


def compute_cs_ssim_tw(
    y_sequence: torch.Tensor,
    device:     str = "cpu",
) -> float:
    """
    Cross-Sequence Temporal SSIM (CS-SSIM_tw):
    Same as SSIM_tw but on the REAL sequence.
    Expected to be ≥ 0.999 (real plant sequences are very smooth).

    Args:
        y_sequence : [T, C, H, W]

    Returns:
        cs_ssim_tw : float
    """
    return compute_ssim_tw(y_sequence, device=device)


# ===========================================================================
# Sequence Evaluator — groups pairs by sequence for temporal metrics
# ===========================================================================

class SequenceEvaluator:
    """
    Accumulates generated/real frame pairs over the validation set,
    groups them by plant sequence ID, and computes all 6 research metrics.

    Usage:
        evaluator = SequenceEvaluator(inception, device)

        for batch in val_loader:
            # Run model to get y_hat ...
            evaluator.update(
                y_hat      = y_hat,          # [B, C, H, W]
                y          = batch['y'],     # [B, C, H, W]
                y_prior    = batch['y_prior'],
                seq_ids    = batch['seq_id'],  # list[str] of sequence identifiers
                dap_values = batch['dap_curr'],
            )

        results = evaluator.compute()
        evaluator.reset()

    Args:
        inception_extractor : InceptionFeatureExtractor
        device : str
    """

    def __init__(
        self,
        inception_extractor: InceptionFeatureExtractor,
        device: str = "cpu",
    ):
        self.inception = inception_extractor
        self.device    = device
        self.reset()

    def reset(self):
        """Clear all accumulated data."""
        self._real_feats:  List[np.ndarray] = []
        self._fake_feats:  List[np.ndarray] = []
        self._mse_list:    List[float]       = []

        # Sequence-indexed storage for temporal metrics
        # key: seq_id → list of (dap, y_hat, y) sorted by dap
        self._seq_hat:  Dict[str, List] = defaultdict(list)
        self._seq_real: Dict[str, List] = defaultdict(list)

    def update(
        self,
        y_hat:      torch.Tensor,         # [B, C, H, W]
        y:          torch.Tensor,         # [B, C, H, W]
        seq_ids:    List[str],
        dap_values: Optional[torch.Tensor] = None,  # [B] — days after planting
    ):
        """Accumulate one batch."""
        y_hat = y_hat.detach().cpu()
        y     = y.detach().cpu()

        # Per-frame MSE (pixel scale)
        self._mse_list.append(compute_mse_pixel(y_hat, y))

        # Inception features for FID
        self._real_feats.append(self.inception(y))     # [B, 2048]
        self._fake_feats.append(self.inception(y_hat))

        # Store frames per sequence for temporal metrics
        for i, sid in enumerate(seq_ids):
            dap = float(dap_values[i]) if dap_values is not None else float(len(self._seq_hat[sid]))
            self._seq_hat[sid].append((dap, y_hat[i]))   # (dap, [C, H, W])
            self._seq_real[sid].append((dap, y[i]))

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics over accumulated data.

        Returns dict with keys:
            mse, fid, mse_tw, ssim_tw, cs_mse_tw, cs_ssim_tw, ssim
        """
        results: Dict[str, float] = {}

        # ── MSE (pixel scale) ─────────────────────────────────────────────
        results["mse"] = float(np.mean(self._mse_list))

        # ── FID ───────────────────────────────────────────────────────────
        rf = np.concatenate(self._real_feats, axis=0)  # [N, 2048]
        ff = np.concatenate(self._fake_feats, axis=0)
        results["fid"] = compute_fid(rf, ff)

        # ── SSIM (per frame) ──────────────────────────────────────────────
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        for sid in self._seq_hat:
            for (_, yh), (_, yr) in zip(
                sorted(self._seq_hat[sid]),
                sorted(self._seq_real[sid])
            ):
                ssim_metric.update(yh.unsqueeze(0), yr.unsqueeze(0))
        results["ssim"] = float(ssim_metric.compute().item())

        # ── Temporal metrics (per sequence, then averaged) ─────────────────
        mse_tw_vals    = []
        ssim_tw_vals   = []
        cs_mse_tw_vals = []
        cs_ssim_tw_vals= []

        for sid in self._seq_hat:
            # Sort by DAP within sequence
            hat_sorted  = [f for _, f in sorted(self._seq_hat[sid])]
            real_sorted = [f for _, f in sorted(self._seq_real[sid])]

            if len(hat_sorted) < 2:
                continue

            hat_seq  = torch.stack(hat_sorted)    # [T, C, H, W]
            real_seq = torch.stack(real_sorted)   # [T, C, H, W]

            mse_tw_vals.append(compute_mse_tw(hat_seq))
            ssim_tw_vals.append(compute_ssim_tw(hat_seq, device=self.device))
            cs_mse_tw_vals.append(compute_cs_mse_tw(real_seq))
            cs_ssim_tw_vals.append(compute_cs_ssim_tw(real_seq, device=self.device))

        results["mse_tw"]     = float(np.mean(mse_tw_vals))     if mse_tw_vals     else 0.0
        results["ssim_tw"]    = float(np.mean(ssim_tw_vals))    if ssim_tw_vals    else 0.0
        results["cs_mse_tw"]  = float(np.mean(cs_mse_tw_vals))  if cs_mse_tw_vals  else 0.0
        results["cs_ssim_tw"] = float(np.mean(cs_ssim_tw_vals)) if cs_ssim_tw_vals else 0.0

        return results

    def format_table(self, results: Dict[str, float]) -> str:
        """Format results as a paper-style table string."""
        return (
            "\n" + "=" * 60 + "\n"
            "  SI-PGS-R v2 — Evaluation Results\n"
            + "=" * 60 + "\n"
            f"  Generation Quality\n"
            f"    FID   ↓ : {results.get('fid', float('nan')):>10.2f}   (target < 42)\n"
            f"    MSE   ↓ : {results.get('mse', float('nan')):>10.1f}   (target < 3000)\n"
            f"\n  Frame Coherence\n"
            f"    MSE_tw   ↓ : {results.get('mse_tw',  float('nan')):>8.1f}   (target < 1500)\n"
            f"    SSIM_tw  ↑ : {results.get('ssim_tw', float('nan')):>8.4f}   (target > 0.940)\n"
            f"    CS-MSE_tw  : {results.get('cs_mse_tw',  float('nan')):>8.4f}   (real seq ref)\n"
            f"    CS-SSIM_tw : {results.get('cs_ssim_tw', float('nan')):>8.4f}   (real seq ref)\n"
            + "=" * 60
        )


# ===========================================================================
# Convenience: compute metrics from a single batch (for quick validation)
# ===========================================================================

def batch_metrics(
    y_hat: torch.Tensor,
    y:     torch.Tensor,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Quick per-batch metric computation (no sequence-level temporal metrics).
    Used for logging during training.

    Returns dict: {mse, ssim}
    """
    return {
        "mse":  compute_mse_pixel(y_hat.cpu(), y.cpu()),
        "ssim": compute_ssim_value(y_hat.cpu(), y.cpu(), device=device),
    }
