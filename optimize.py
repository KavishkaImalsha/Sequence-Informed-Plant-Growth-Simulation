"""
optimize.py — Environmental Optimizer (Novel Research Module)
=============================================================
Implements the EnvironmentalOptimizer class — the core research contribution
of this work.

Goal
----
Given:
  • An initial plant state image  y₀   [B, C, H, W]
  • A trained generative model   (encoder S_θ, generator G_θ)
  • A target number of future time-steps  T

Find the *optimal* future environmental sequence  c_opt(τ)  [T, ENV_FEATURES]
that **maximises a phenotypic trait** (plant biomass / green pixel area) in
the synthesised future frames.

Mechanism
---------
1. The trained generative model weights are **frozen**.
2. The environmental input sequence  c(τ)  is re-parameterised as a
   **learnable parameter** tensor (requires_grad=True).
3. A differentiable biomass reward function  R(ŷ_t)  is computed on each
   synthesised frame.
4. Gradient ascent (negative gradient descent) is performed for N iterations:

       c(τ) ← c(τ) + α · ∇_{c(τ)} Σ_t R(ŷ_t)

5. Physical plausibility constraints are enforced by projecting c(τ) back
   into the valid [0,1] normalised range at each step (projected gradient).

Biomass Reward Options
-----------------------
Three differentiable reward signals are available:

• 'green_ratio'   : Fraction of pixels with high green channel relative to
                    red and blue. Directly correlates with chlorophyll-rich
                    plant tissue.  R = mean(G > max(R,B)+threshold)

• 'ndvi_proxy'    : Approximate NDVI using (G-R)/(G+R+ε). Sensitive to
                    photosynthetically active tissue in RGB images.

• 'pixel_area'    : Total normalised pixel brightness mass — proxy for
                    overall plant coverage.  R = mean(ŷ_t)

Outputs
-------
• optimal_env_seq   : [T, ENV_FEATURES]   — optimal normalised env sequence
• reward_history    : [n_iters]           — reward per iteration (convergence plot)
• predicted_frames  : [T, C, H, W]       — forward-pass frames under optimal env

Usage (standalone)
------------------
    python optimize.py \\
        --checkpoint output/sipgs_r/ckpt_best \\
        --initial_frame datasets/GrowliFlowerL/images/Train/patch_2020_08_12_7246.jpg \\
        --horizon 10 \\
        --reward_fn green_ratio \\
        --n_iters 300 \\
        --lr 0.02
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from architectures import (
    LSTM_SequenceEncoder,
    Recurrent_CVAE_Generator,
    build_sipgs_r,
)
from data import ENV_FEATURE_DIM, EnvNormaliser
from utils import load_model_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EnvironmentalOptimizer")


# ===========================================================================
# Reward Functions (all differentiable, return scalar)
# ===========================================================================

def reward_green_ratio(
    y_hat: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """
    Fraction of pixels where the Green channel exceeds both R and B
    by at least `threshold`.

    This is a differentiable relaxation using sigmoid to avoid hard thresholds.

    Args:
        y_hat     : [B, 3, H, W]  in [0,1]   (channel order: RGB)
        threshold : float   margin for green dominance

    Returns:
        reward : scalar tensor   (higher = more green/biomass)
    """
    R = y_hat[:, 0:1, :, :]   # [B, 1, H, W]
    G = y_hat[:, 1:2, :, :]   # [B, 1, H, W]
    B = y_hat[:, 2:3, :, :]   # [B, 1, H, W]

    # Soft indicator: G > R + threshold  and  G > B + threshold
    green_over_red  = torch.sigmoid(100 * (G - R - threshold))  # [B,1,H,W]
    green_over_blue = torch.sigmoid(100 * (G - B - threshold))  # [B,1,H,W]

    # Combined: pixel is "green" if both conditions hold (product ~ AND)
    green_mask = green_over_red * green_over_blue                # [B,1,H,W]
    reward = green_mask.mean()                                   # scalar
    return reward


def reward_ndvi_proxy(y_hat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    NDVI-proxy using RGB channels:
        NDVI_rgb = (G - R) / (G + R + ε)

    Higher value → more green/photosynthetically active tissue.

    Args:
        y_hat : [B, 3, H, W]  in [0,1]

    Returns:
        reward : scalar tensor
    """
    R = y_hat[:, 0, :, :]
    G = y_hat[:, 1, :, :]
    ndvi = (G - R) / (G + R + eps)   # [B, H, W]  in [-1, 1]
    # Shift to [0,1] for a clean reward signal
    return ((ndvi + 1.0) / 2.0).mean()


def reward_pixel_area(y_hat: torch.Tensor) -> torch.Tensor:
    """
    Mean pixel brightness — simple proxy for overall plant coverage/biomass.

    Args:
        y_hat : [B, 3, H, W]

    Returns:
        reward : scalar
    """
    return y_hat.mean()


REWARD_FNS = {
    "green_ratio": reward_green_ratio,
    "ndvi_proxy":  reward_ndvi_proxy,
    "pixel_area":  reward_pixel_area,
}


# ===========================================================================
# Environmental Constraint Projector
# ===========================================================================

class EnvConstraints:
    """
    Defines plausible physical bounds for each environmental attribute and
    provides methods to project optimised values back into valid ranges.

    Bounds are in the *normalised* [0, 1] space since the LSTM operates on
    normalised inputs. The normaliser is passed at construction time for
    optional denormalisation in reporting.

    The projection at each iteration implements *projected gradient ascent*,
    which guarantees the optimised sequence stays within a physically
    meaningful region.

    Attribute order (must match ENV_COLUMNS in data.py):
        [temperature, relative_humidity, solar_radiation, precipitation]
    """

    # Normalised lower and upper bounds per feature (after min-max scaling)
    # Slightly tightened from [0,1] to prevent degenerate extremes
    LOW  = torch.tensor([0.02, 0.02, 0.02, 0.00])
    HIGH = torch.tensor([0.98, 0.98, 0.98, 0.98])

    def __init__(self, normaliser: Optional[EnvNormaliser] = None):
        self.normaliser = normaliser

    def project(self, env_seq: torch.Tensor) -> torch.Tensor:
        """
        Clamp env_seq to [LOW, HIGH] per feature.

        Args:
            env_seq : [T, ENV_FEATURES]  or  [B, T, ENV_FEATURES]

        Returns:
            projected : same shape, values clamped
        """
        lo = self.LOW.to(env_seq.device)   # [4]
        hi = self.HIGH.to(env_seq.device)  # [4]
        return torch.clamp(env_seq, min=lo, max=hi)


# ===========================================================================
# Environmental Optimizer
# ===========================================================================

class EnvironmentalOptimizer:
    """
    Gradient-ascent search for the optimal environmental sequence that
    maximises a target phenotypic reward over a prediction horizon.

    The trained generator weights are completely frozen. Only the
    environmental input sequence is optimised.

    Args:
        encoder      : Trained LSTM_SequenceEncoder (frozen)
        generator    : Trained Recurrent_CVAE_Generator (frozen)
        img_shape    : (C, H, W)
        device       : torch device string
        normaliser   : EnvNormaliser for constraint checking
        reward_fn    : One of 'green_ratio', 'ndvi_proxy', 'pixel_area'
        n_iters      : Gradient ascent iterations
        lr           : Step size for gradient ascent
        horizon      : Number of future frames to predict (T)
        temperature  : Generator sampling temperature (< 1 → less noisy)
    """

    def __init__(
        self,
        encoder:       LSTM_SequenceEncoder,
        generator:     Recurrent_CVAE_Generator,
        img_shape:     Tuple[int, int, int] = (3, 128, 128),
        device:        str = "cpu",
        normaliser:    Optional[EnvNormaliser] = None,
        reward_fn:     str = "green_ratio",
        n_iters:       int = 300,
        lr:            float = 0.02,
        horizon:       int = 10,
        temperature:   float = 0.8,
        context_window: int = 14,
    ):
        self.device         = device
        self.img_shape      = img_shape
        self.n_iters        = n_iters
        self.lr             = lr
        self.horizon        = horizon
        self.temperature    = temperature
        self.context_window = context_window

        # ── Freeze the generative model ────────────────────────────────────
        self.encoder   = encoder.to(device).eval()
        self.generator = generator.to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.generator.parameters():
            p.requires_grad_(False)

        # ── Reward function ────────────────────────────────────────────────
        if reward_fn not in REWARD_FNS:
            raise ValueError(
                f"Unknown reward_fn '{reward_fn}'. "
                f"Choose from: {list(REWARD_FNS.keys())}"
            )
        self.reward_fn = REWARD_FNS[reward_fn]
        self.reward_fn_name = reward_fn

        # ── Constraints ────────────────────────────────────────────────────
        self.constraints = EnvConstraints(normaliser)

        logger.info(
            f"EnvironmentalOptimizer ready: "
            f"reward={reward_fn}, horizon={horizon}, "
            f"n_iters={n_iters}, lr={lr}, device={device}"
        )

    # ────────────────────────────────────────────────────────────────────────
    def _simulate_trajectory(
        self,
        y0:        torch.Tensor,     # [1, C, H, W]   initial frame
        env_param: torch.Tensor,     # [horizon, ENV_FEATURES]  (optimisable)
        time_diffs: torch.Tensor,    # [horizon, 1]   normalised Δt per step
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward-simulate the generative model for `horizon` steps using the
        given (learnable) environmental sequence.

        At each step t:
          1. Build LSTM context from env_param up to step t
          2. Encode via S_θ → x_t
          3. Generate ŷ_t via G_θ(z, x_t, ŷ_{t-1}, Δt_t)

        Returns:
            frames  : [horizon, C, H, W]   — synthesised trajectory
            rewards : [horizon]            — per-step reward
        """
        frames  = []
        rewards = []
        y_prev  = y0  # [1, C, H, W]

        for t in range(self.horizon):
            # Build cumulative env context for step t
            # We use env_param rows [0..t] as the context ending at step t
            ctx_len = min(t + 1, self.context_window)
            ctx_start = max(0, t + 1 - self.context_window)

            ctx = env_param[ctx_start:t + 1, :]  # [ctx_len, 4]

            # Zero-pad to context_window at the front
            pad_len = self.context_window - ctx_len
            if pad_len > 0:
                pad = torch.zeros(
                    pad_len, ENV_FEATURE_DIM,
                    device=self.device, dtype=env_param.dtype
                )
                ctx = torch.cat([pad, ctx], dim=0)  # [context_window, 4]

            # Add batch dim → [1, context_window, 4]
            ctx_batch   = ctx.unsqueeze(0)
            seq_len_t   = torch.tensor([ctx_len], dtype=torch.long, device=self.device)
            td_t        = time_diffs[t].unsqueeze(0)            # [1, 1]

            # Encode environment → x_t  [1, lstm_output]
            x_t = self.encoder(ctx_batch, seq_len_t)

            # Generate next frame  ŷ_t  [1, C, H, W]
            y_hat = self.generator.generate(
                x_t, y_prev, td_t, temperature=self.temperature
            )

            reward_t = self.reward_fn(y_hat)   # scalar tensor
            frames.append(y_hat)               # [1, C, H, W]
            rewards.append(reward_t)

            # Recurrent: ŷ_t becomes y_{t-1} for next step
            y_prev = y_hat

        frames_tensor  = torch.cat(frames, dim=0)     # [horizon, C, H, W]
        rewards_tensor = torch.stack(rewards)         # [horizon]

        return frames_tensor, rewards_tensor

    # ────────────────────────────────────────────────────────────────────────
    def optimise(
        self,
        initial_frame:    torch.Tensor,      # [1, C, H, W]  or  [C, H, W]
        target_dap_gap:   float = 7.0,       # days between prediction steps
        max_dap:          float = 100.0,     # for Δt normalisation
        init_env_seq:     Optional[torch.Tensor] = None,  # [horizon, 4] warm-start
    ) -> Dict:
        """
        Run the gradient-ascent optimisation loop.

        Args:
            initial_frame  : Starting plant image (will be unsqueezed if 3D).
            target_dap_gap : Expected days between successive predicted frames.
                             Used to compute normalised time_diffs.
            max_dap        : Maximum DAP in the dataset (for normalisation).
            init_env_seq   : Optional warm-start for c(τ); if None, initialised
                             from N(0.5, 0.05) clamped to [0,1].

        Returns:
            A dict containing:
              'optimal_env_seq'   : [horizon, 4]    (normalised, numpy)
              'reward_history'    : [n_iters]        (numpy)
              'predicted_frames'  : [horizon, C, H, W]  (numpy, [0,1])
              'total_reward'      : float
              'reward_fn'         : str
        """
        if initial_frame.dim() == 3:
            initial_frame = initial_frame.unsqueeze(0)    # [1, C, H, W]
        initial_frame = initial_frame.to(self.device)

        # ── Initialise learnable env sequence ─────────────────────────────
        if init_env_seq is not None:
            env_data = init_env_seq.clone().float().to(self.device)
        else:
            # Start near the centre of the normalised range
            env_data = torch.clamp(
                torch.randn(self.horizon, ENV_FEATURE_DIM, device=self.device) * 0.05 + 0.5,
                0.0, 1.0
            )

        # Make it a leaf variable for gradient ascent
        env_param = env_data.detach().requires_grad_(True)

        # Fixed time differences (normalised Δt between each step)
        normalised_dt = target_dap_gap / max_dap
        time_diffs = torch.full(
            (self.horizon, 1), normalised_dt,
            dtype=torch.float32, device=self.device
        )  # [horizon, 1]

        # ── Gradient ascent optimizer ──────────────────────────────────────
        opt = torch.optim.Adam([env_param], lr=self.lr)

        reward_history = []
        best_reward    = -float("inf")
        best_env_seq   = env_param.detach().clone()
        best_frames    = None

        logger.info(
            f"Starting gradient ascent: "
            f"{self.n_iters} iters, horizon={self.horizon}, "
            f"reward_fn={self.reward_fn_name}"
        )

        for iteration in range(1, self.n_iters + 1):
            opt.zero_grad()

            # Forward simulate the trajectory under current env_param
            frames, rewards = self._simulate_trajectory(
                initial_frame, env_param, time_diffs
            )

            # Total reward = sum over horizon (maximise cumulative biomass)
            total_reward = rewards.sum()   # scalar

            # Gradient ASCENT: negate reward → descend on negative reward
            loss = -total_reward
            loss.backward()

            opt.step()

            # ── Project back to valid bounds ───────────────────────────────
            with torch.no_grad():
                env_param.data = self.constraints.project(env_param.data)

            reward_val = total_reward.item()
            reward_history.append(reward_val)

            # Track best
            if reward_val > best_reward:
                best_reward   = reward_val
                best_env_seq  = env_param.detach().clone()
                best_frames   = frames.detach()

            if iteration % max(1, self.n_iters // 20) == 0:
                logger.info(
                    f"  Iter {iteration:4d}/{self.n_iters}  "
                    f"reward={reward_val:.5f}  "
                    f"best={best_reward:.5f}"
                )

        logger.info(
            f"Optimisation complete. "
            f"Best total reward ({self.reward_fn_name})={best_reward:.5f}"
        )

        return {
            "optimal_env_seq":   best_env_seq.cpu().numpy(),            # [T, 4]
            "reward_history":    np.array(reward_history),               # [n_iters]
            "predicted_frames":  best_frames.cpu().numpy(),             # [T, C, H, W]
            "total_reward":      best_reward,
            "reward_fn":         self.reward_fn_name,
        }

    # ────────────────────────────────────────────────────────────────────────
    def report(
        self,
        result:      Dict,
        normaliser:  Optional[EnvNormaliser] = None,
        output_dir:  Optional[str] = None,
    ):
        """
        Print and optionally save the optimisation results.

        Args:
            result      : Dict returned by optimise().
            normaliser  : If provided, denormalise env values for reporting.
            output_dir  : If given, save frames as PNGs and results as JSON.
        """
        env_norm = result["optimal_env_seq"]  # [T, 4]

        print("\n" + "="*60)
        print(f"  Environmental Optimization Results")
        print(f"  Reward Function  : {result['reward_fn']}")
        print(f"  Total Reward     : {result['total_reward']:.5f}")
        print(f"  Horizon          : {len(env_norm)} steps")
        print("="*60)

        headers = ["temperature", "rel_humidity", "solar_rad", "precip"]

        if normaliser is not None:
            env_real = normaliser.denormalise(env_norm)  # [T, 4] original scale
        else:
            env_real = env_norm

        print(f"\nOptimal environmental sequence (per step):")
        print(f"  {'Step':>4}  " + "  ".join(f"{h:>14}" for h in headers))
        print("  " + "-"*70)
        for t, row in enumerate(env_real):
            fmt = "  ".join(f"{v:>14.4f}" for v in row)
            print(f"  {t+1:>4}  {fmt}")

        print(f"\nReward convergence:")
        rh = result["reward_history"]
        print(f"  Initial : {rh[0]:.5f}")
        print(f"  Final   : {rh[-1]:.5f}")
        print(f"  Max     : {rh.max():.5f}")

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Save frames as PNGs
            frames = result["predicted_frames"]  # [T, C, H, W]
            for t, frame in enumerate(frames):
                img_arr = (np.transpose(frame, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_arr).save(out_path / f"optimal_frame_{t+1:02d}.png")

            # Save result dict (without frames, as JSON)
            json_result = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result.items()
                if k != "predicted_frames"
            }
            if normaliser is not None:
                json_result["optimal_env_seq_original_scale"] = env_real.tolist()
                json_result["env_feature_names"] = headers
            with open(out_path / "optimization_result.json", "w") as f:
                json.dump(json_result, f, indent=2)

            logger.info(f"Results saved to {output_dir}/")


# ===========================================================================
# Loading Trained Models from Checkpoint
# ===========================================================================

def load_trained_models(
    checkpoint_dir: str,
    device:         str = "cpu",
) -> Tuple[LSTM_SequenceEncoder, Recurrent_CVAE_Generator]:
    """
    Load trained encoder and generator from a checkpoint directory.

    The checkpoint directory must contain:
      • feature_encoder.json + feature_encoder.pt
      • generator.json       + generator.pt

    Returns:
        encoder, generator   (eval mode, on device)
    """
    # Import here to avoid circular dependency
    from architectures import LSTM_SequenceEncoder, Recurrent_CVAE_Generator

    enc_params, enc_pt = load_model_config(
        os.path.join(checkpoint_dir, "feature_encoder.json")
    )
    gen_params, gen_pt = load_model_config(
        os.path.join(checkpoint_dir, "generator.json")
    )

    encoder = LSTM_SequenceEncoder(**enc_params).to(device)
    generator = Recurrent_CVAE_Generator(**gen_params).to(device)

    enc_path = os.path.join(checkpoint_dir, enc_pt.split("/")[-1])
    gen_path = os.path.join(checkpoint_dir, gen_pt.split("/")[-1])

    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    generator.load_state_dict(torch.load(gen_path, map_location=device))

    encoder.eval()
    generator.eval()

    logger.info(f"Loaded encoder from  {enc_path}")
    logger.info(f"Loaded generator from {gen_path}")

    return encoder, generator


# ===========================================================================
# Image Loader Utility
# ===========================================================================

def load_initial_frame(
    image_path: str,
    img_size:   Tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """
    Load and preprocess a single JPEG frame → [1, 3, H, W] tensor in [0,1].
    """
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0)   # [1, 3, H, W]


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Optimize environmental conditions for maximum plant growth"
    )
    p.add_argument("--checkpoint",
                   required=True,
                   help="Path to trained model checkpoint dir (ckpt_best/).")
    p.add_argument("--initial_frame",
                   required=True,
                   help="Path to initial plant state image (JPEG/PNG).")
    p.add_argument("--env_csv",
                   default="datasets/growliflower_environmental_data.csv",
                   help="Path to environmental CSV for normaliser fitting.")
    p.add_argument("--horizon",
                   type=int, default=10,
                   help="Number of future frames to predict.")
    p.add_argument("--reward_fn",
                   choices=list(REWARD_FNS.keys()),
                   default="green_ratio",
                   help="Phenotypic reward signal to maximise.")
    p.add_argument("--n_iters",
                   type=int, default=300,
                   help="Gradient ascent iterations.")
    p.add_argument("--lr",
                   type=float, default=0.02,
                   help="Gradient ascent step size.")
    p.add_argument("--target_dap_gap",
                   type=float, default=7.0,
                   help="Expected days between successive predicted frames.")
    p.add_argument("--max_dap",
                   type=float, default=100.0,
                   help="Maximum DAP in dataset for normalisation.")
    p.add_argument("--img_size",
                   type=int, default=128,
                   help="Image resolution (square).")
    p.add_argument("--temperature",
                   type=float, default=0.8,
                   help="Generator sampling temperature (< 1 = sharper).")
    p.add_argument("--context_window",
                   type=int, default=14,
                   help="Environmental context window (must match training).")
    p.add_argument("--output_dir",
                   default="output/optimization",
                   help="Directory to save optimal frames and results JSON.")
    return p.parse_args()


if __name__ == "__main__":
    import pandas as pd

    args = parse_args()

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Load models ────────────────────────────────────────────────────────
    encoder, generator = load_trained_models(args.checkpoint, device=device)

    # ── Fit normaliser ─────────────────────────────────────────────────────
    env_df = pd.read_csv(args.env_csv, parse_dates=["date"])
    from data import EnvNormaliser, ENV_COLUMNS
    normaliser = EnvNormaliser(env_df)

    # ── Load initial frame ─────────────────────────────────────────────────
    y0 = load_initial_frame(
        args.initial_frame, img_size=(args.img_size, args.img_size)
    ).to(device)
    logger.info(f"Initial frame loaded: {y0.shape}")

    # ── Build optimizer ────────────────────────────────────────────────────
    opt = EnvironmentalOptimizer(
        encoder         = encoder,
        generator       = generator,
        img_shape       = (3, args.img_size, args.img_size),
        device          = device,
        normaliser      = normaliser,
        reward_fn       = args.reward_fn,
        n_iters         = args.n_iters,
        lr              = args.lr,
        horizon         = args.horizon,
        temperature     = args.temperature,
        context_window  = args.context_window,
    )

    # ── Run optimisation ───────────────────────────────────────────────────
    result = opt.optimise(
        initial_frame  = y0,
        target_dap_gap = args.target_dap_gap,
        max_dap        = args.max_dap,
    )

    # ── Report & save ──────────────────────────────────────────────────────
    opt.report(result, normaliser=normaliser, output_dir=args.output_dir)

    # Plot convergence curve if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result["reward_history"], linewidth=1.5)
        ax.set_xlabel("Gradient Ascent Iteration")
        ax.set_ylabel(f"Total Reward ({args.reward_fn})")
        ax.set_title("Environmental Optimization Convergence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            Path(args.output_dir) / "convergence_curve.png",
            dpi=150, bbox_inches="tight"
        )
        logger.info(f"Convergence curve saved → {args.output_dir}/convergence_curve.png")
    except Exception:
        pass
