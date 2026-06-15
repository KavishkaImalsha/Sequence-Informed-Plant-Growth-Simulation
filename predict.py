"""
predict.py — SI-PGS-R v2 Inference
=====================================
Given a trained checkpoint, an initial plant image, and future environmental
conditions, this script:

  1. Generates a sequence of N predicted future plant frames
  2. Computes growth rate (green pixel ratio change per day)
  3. Saves all outputs to a results directory

Usage:
    python predict.py \
        --checkpoint  /path/to/ckpt_best \
        --initial_img datasets/GrowliFlowerL/images/Train/patch_2020_08_12_7246.jpg \
        --env_csv     datasets/growliflower_environmental_data.csv \
        --start_date  2020-08-12 \
        --horizon     14 \
        --output_dir  output/predictions/run1

Outputs:
    output/predictions/run1/
        frame_00_day012.jpg   ← generated plant images
        frame_01_day013.jpg
        ...
        growth_rate.csv       ← green ratio and growth rate per predicted day
        growth_chart.png      ← visualised growth curve
        prediction_grid.png   ← all frames in one image
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SI-PGS-R Predict")

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_H, IMG_W = 240, 320
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

to_tensor = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

def denorm(t: torch.Tensor) -> np.ndarray:
    """Denormalise tensor [C,H,W] → uint8 numpy [H,W,C]."""
    mean = torch.tensor(IMG_MEAN, device=t.device).view(3, 1, 1)
    std  = torch.tensor(IMG_STD,  device=t.device).view(3, 1, 1)
    img  = t * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


# =============================================================================
# Growth Rate Computation
# =============================================================================

def green_pixel_ratio(img_tensor: torch.Tensor) -> float:
    """
    Compute the fraction of pixels that are predominantly green.
    A pixel is 'green' if G > R * 1.1 AND G > B * 1.1.

    This is a proxy for plant biomass / canopy coverage.

    Args:
        img_tensor: [C, H, W] normalised tensor in ImageNet stats
    Returns:
        float in [0, 1]  — fraction of green pixels
    """
    # Denormalise to [0,1]
    mean = torch.tensor(IMG_MEAN, device=img_tensor.device).view(3, 1, 1)
    std  = torch.tensor(IMG_STD,  device=img_tensor.device).view(3, 1, 1)
    rgb  = (img_tensor * std + mean).clamp(0, 1)  # [3, H, W]

    R, G, B = rgb[0], rgb[1], rgb[2]
    green_mask = (G > R * 1.10) & (G > B * 1.10) & (G > 0.15)
    return float(green_mask.float().mean().item())


def compute_growth_rate(ratios: list, dap_values: list) -> list:
    """
    Growth rate = Δ(green_ratio) / Δ(DAP) per day.

    Args:
        ratios    : list of green pixel ratios per frame
        dap_values: list of days-after-planting for each frame
    Returns:
        list of growth rates (same length, first entry = 0)
    """
    rates = [0.0]
    for i in range(1, len(ratios)):
        delta_ratio = ratios[i] - ratios[i-1]
        delta_day   = max(1, dap_values[i] - dap_values[i-1])
        rates.append(delta_ratio / delta_day)
    return rates


# =============================================================================
# Environment Data Loading
# =============================================================================

def load_env_sequence(
    env_csv: str,
    start_date: str,
    context_window: int = 14,
    horizon: int = 14,
) -> tuple:
    """
    Load environmental conditioning data for:
      - context_window days BEFORE start_date (for LSTM context)
      - horizon days FROM start_date (for prediction targets)

    Returns:
        context_env : [context_window, 4]  float32 tensor
        future_daps : list of int  (DAP values for each predicted frame)
        future_dates: list of str
    """
    import pandas as pd

    df = pd.read_csv(env_csv, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Normalise env columns (z-score per column)
    ENV_COLS = ["temperature", "humidity", "co2", "light_intensity"]
    available = [c for c in ENV_COLS if c in df.columns]
    if len(available) < 4:
        logger.warning(f"Only {len(available)}/4 env columns found: {available}")
        # Pad missing with zeros
        for c in ENV_COLS:
            if c not in df.columns:
                df[c] = 0.0
    for c in ENV_COLS:
        mu, sigma = df[c].mean(), df[c].std()
        if sigma > 0:
            df[c] = (df[c] - mu) / sigma

    start_dt = pd.Timestamp(start_date)
    context_start = start_dt - pd.Timedelta(days=context_window)

    # Context window
    ctx_mask = (df["date"] >= context_start) & (df["date"] < start_dt)
    ctx_rows  = df[ctx_mask].tail(context_window)

    if len(ctx_rows) < context_window:
        logger.warning(
            f"Only {len(ctx_rows)} context days found, padding with zeros."
        )
        pad = context_window - len(ctx_rows)
        pad_df = pd.DataFrame(
            np.zeros((pad, 4)), columns=ENV_COLS
        )
        pad_df["date"] = context_start
        ctx_rows = pd.concat([pad_df, ctx_rows], ignore_index=True)

    context_env = torch.tensor(
        ctx_rows[ENV_COLS].values, dtype=torch.float32
    )  # [context_window, 4]

    # Compute DAP: assume plant starts at day 0 = first date in dataset
    plant_start = df["date"].min()
    start_dap   = int((start_dt - plant_start).days)

    future_daps  = [start_dap + i for i in range(horizon)]
    future_dates = [(start_dt + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(horizon)]

    return context_env, future_daps, future_dates


# =============================================================================
# Model Loading
# =============================================================================

def load_model(ckpt_dir: str, device: str):
    """
    Load trained encoder and generator from checkpoint directory.
    """
    from architectures_v2 import build_sipgs_r_v2

    ckpt = Path(ckpt_dir)

    # Load saved config if available
    gen_cfg_path = ckpt / "generator.json"
    if gen_cfg_path.exists():
        with open(gen_cfg_path) as f:
            cfg = json.load(f)
    else:
        # Fall back to defaults
        cfg = {
            "img_shape": [3, IMG_H, IMG_W],
            "env_feature_size": 4,
            "lstm_hidden": 256,
            "lstm_layers": 4,
            "lstm_output": 128,
            "lstm_heads":  4,
            "latent_size": 256,
            "n_res_blocks": 2,
        }

    encoder, generator, _ = build_sipgs_r_v2(
        img_shape=(3, IMG_H, IMG_W),
        device=device,
    )

    encoder.load_state_dict(
        torch.load(ckpt / "feature_encoder.pt", map_location=device)
    )
    generator.load_state_dict(
        torch.load(ckpt / "generator.pt", map_location=device)
    )

    encoder.eval()
    generator.eval()

    logger.info(f"✅ Loaded model from: {ckpt}")
    return encoder, generator


# =============================================================================
# Autoregressive Prediction Loop
# =============================================================================

@torch.no_grad()
def predict_sequence(
    encoder,
    generator,
    initial_frame: torch.Tensor,   # [1, C, H, W]
    context_env:   torch.Tensor,   # [context_window, 4]
    future_daps:   list,
    device:        str,
    context_window: int = 14,
) -> list:
    """
    Autoregressively generate future plant frames.

    At each step:
      1. Encoder processes the environmental context sequence
      2. Generator predicts the next frame from (env_code, prior_frame)
      3. The predicted frame becomes the prior for the next step

    Args:
        initial_frame  : [1, 3, H, W] normalised tensor — last known real frame
        context_env    : [context_window, 4] environmental history
        future_daps    : list of DAP values for each prediction step
        horizon        : number of frames to predict

    Returns:
        List of predicted frame tensors [1, 3, H, W] (one per future day)
    """
    horizon    = len(future_daps)
    prior      = initial_frame.to(device)             # [1, 3, H, W]
    env_seq    = context_env.unsqueeze(0).to(device)  # [1, T, 4]
    seq_len    = torch.tensor([context_env.shape[0]], device=device)

    # Encode environmental context (LSTM over context_window days)
    x = encoder(env_seq, seq_len)   # [1, 128]

    predictions = []

    for step in range(horizon):
        # Time difference: days between current and prior frame
        if step == 0:
            dt = torch.tensor([[1.0]], device=device)
        else:
            dt = torch.tensor(
                [[float(future_daps[step] - future_daps[step-1])]],
                device=device
            )

        # Generate next frame
        y_hat = generator.generate(x, prior, dt)   # [1, 3, H, W]
        predictions.append(y_hat.cpu())

        # Update prior for next step (autoregressive)
        prior = y_hat

    return predictions


# =============================================================================
# Visualisation
# =============================================================================

def save_prediction_grid(frames: list, output_path: str, dates: list):
    """Save all predicted frames as a single grid image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(frames)
        cols = min(7, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle("Predicted Plant Growth Sequence", fontsize=14, fontweight="bold")

        if rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        for i, (frame, ax) in enumerate(zip(frames, axes)):
            img_np = denorm(frame.squeeze(0))
            ax.imshow(img_np)
            ax.set_title(dates[i] if i < len(dates) else f"Day {i}", fontsize=8)
            ax.axis("off")

        # Hide unused subplots
        for ax in axes[len(frames):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved prediction grid → {output_path}")
    except ImportError:
        logger.warning("matplotlib not found — skipping grid image")


def save_growth_chart(daps, ratios, rates, output_path: str):
    """Plot green pixel ratio and daily growth rate over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle("Predicted Plant Growth Analysis", fontsize=13, fontweight="bold")

        ax1.plot(daps, ratios, "g-o", linewidth=2, markersize=5, label="Green Coverage")
        ax1.fill_between(daps, ratios, alpha=0.2, color="green")
        ax1.set_ylabel("Green Pixel Ratio", fontsize=10)
        ax1.set_ylim(0, max(0.5, max(ratios) * 1.2))
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(daps, rates, color=["green" if r >= 0 else "red" for r in rates],
                alpha=0.7, label="Daily Growth Rate")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("ΔGreen Ratio / Day", fontsize=10)
        ax2.set_xlabel("Days After Planting (DAP)", fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved growth chart → {output_path}")
    except ImportError:
        logger.warning("matplotlib not found — skipping growth chart")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SI-PGS-R v2 Inference")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to checkpoint dir (e.g. output/sipgs_r_v2/ckpt_best)")
    parser.add_argument("--initial_img", required=True,
                        help="Path to initial plant image (last known real frame)")
    parser.add_argument("--env_csv",     required=True,
                        help="Path to environmental data CSV")
    parser.add_argument("--start_date",  default=None,
                        help="Start date for prediction (YYYY-MM-DD). Default: today")
    parser.add_argument("--horizon",     type=int, default=14,
                        help="Number of future days to predict")
    parser.add_argument("--context_window", type=int, default=14,
                        help="Days of environmental history for LSTM")
    parser.add_argument("--output_dir", default="output/predictions",
                        help="Directory to save outputs")
    args = parser.parse_args()

    # Setup
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Device: {device}")

    start_date = args.start_date or datetime.now().strftime("%Y-%m-%d")

    # ── Load model ────────────────────────────────────────────────────────────
    encoder, generator = load_model(args.checkpoint, device)

    # ── Load initial frame ───────────────────────────────────────────────────
    pil_img = Image.open(args.initial_img).convert("RGB")
    initial_frame = to_tensor(pil_img).unsqueeze(0)   # [1, 3, 240, 320]
    logger.info(f"Initial frame: {args.initial_img}")

    # ── Load environmental data ───────────────────────────────────────────────
    context_env, future_daps, future_dates = load_env_sequence(
        args.env_csv, start_date,
        context_window=args.context_window,
        horizon=args.horizon,
    )
    logger.info(f"Predicting {args.horizon} frames from {start_date}")

    # ── Generate future frames ────────────────────────────────────────────────
    logger.info("Running autoregressive prediction...")
    predictions = predict_sequence(
        encoder, generator,
        initial_frame=initial_frame,
        context_env=context_env,
        future_daps=future_daps,
        device=device,
        context_window=args.context_window,
    )

    # ── Compute green ratio and growth rate ───────────────────────────────────
    # Include initial frame in growth analysis
    all_frames = [initial_frame] + predictions
    all_daps   = [future_daps[0] - 1] + future_daps
    all_dates  = [f"Initial ({start_date})"] + future_dates

    ratios = [green_pixel_ratio(f.squeeze(0)) for f in all_frames]
    rates  = compute_growth_rate(ratios, all_daps)

    # ── Save individual frames ────────────────────────────────────────────────
    for i, (frame, date, dap) in enumerate(zip(predictions, future_dates, future_daps)):
        img_np  = denorm(frame.squeeze(0))
        img_pil = Image.fromarray(img_np)
        fname   = out / f"frame_{i:02d}_day{dap:03d}_{date}.jpg"
        img_pil.save(fname, quality=95)

    logger.info(f"Saved {len(predictions)} predicted frames → {out}/")

    # ── Save growth CSV ───────────────────────────────────────────────────────
    import csv
    csv_path = out / "growth_rate.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "date", "dap", "green_ratio", "growth_rate_per_day"])
        for i, (date, dap, ratio, rate) in enumerate(
            zip(all_dates, all_daps, ratios, rates)
        ):
            writer.writerow([i, date, dap, f"{ratio:.4f}", f"{rate:.6f}"])

    logger.info(f"Saved growth analysis → {csv_path}")

    # ── Save visualisations ───────────────────────────────────────────────────
    save_prediction_grid(
        all_frames, str(out / "prediction_grid.png"), all_dates
    )
    save_growth_chart(
        all_daps, ratios, rates, str(out / "growth_chart.png")
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "="*55)
    logger.info("  GROWTH PREDICTION SUMMARY")
    logger.info("="*55)
    logger.info(f"  {'Date':<14} {'DAP':>5} {'Green%':>8} {'Growth/day':>12}")
    logger.info(f"  {'-'*43}")
    for date, dap, ratio, rate in zip(all_dates, all_daps, ratios, rates):
        logger.info(
            f"  {str(date):<14} {dap:>5} {ratio*100:>7.2f}% {rate*100:>+11.3f}%"
        )
    logger.info("="*55)
    net_growth = (ratios[-1] - ratios[0]) * 100
    avg_rate   = np.mean(rates[1:]) * 100
    logger.info(f"  Net green coverage change : {net_growth:+.2f}%")
    logger.info(f"  Average daily growth rate : {avg_rate:+.3f}%/day")
    logger.info(f"  Outputs saved to          : {out}/")
    logger.info("="*55)


if __name__ == "__main__":
    main()
