"""
data.py — Multimodal Data Processing & Alignment Pipeline
==========================================================
Implements the GrowliFlower + NASA Environmental dataset class for the
SI-PGS-R training pipeline.

Dataset: GrowliFlowerL (time-series RGB images) aligned with
         growliflower_environmental_data.csv (NASA-sourced environmental
         attributes: temperature, relative_humidity, solar_radiation,
         precipitation).

The core design challenge is *variable time intervals* between frames.
Rather than raw timestamps, we construct a **cumulative** environmental
context window c(τ) ending at each frame's capture date. The LSTM encoder
implicitly interpolates missing sensory readings by learning patterns over
these fixed-window sequences.

Key tensor shapes (documented inline):
  images      → [seq_len, 3, H, W]
  env_seq     → [total_seq_len, ENV_FEATURES]  (padded)
  seq_len     → scalar int
  time_diffs  → [1] (normalised day-gap to prior frame)
"""

import os
import re
import math
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

ENV_COLUMNS = ["temperature", "relative_humidity", "solar_radiation", "precipitation"]
# Environmental feature dimension fed to LSTM
ENV_FEATURE_DIM = len(ENV_COLUMNS)  # 4

# Default image resolution (matching GrowliFlowerL patch sizes)
DEFAULT_IMG_H = 128
DEFAULT_IMG_W = 128

# Window of historical environmental days fed as the cumulative LSTM context.
# Mirrors the 7-day rolling average used in the baseline prep_dataset.py.
CONTEXT_WINDOW_DAYS = 14

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics for min-max normalisation (computed over entire env CSV)
# ---------------------------------------------------------------------------

class EnvNormaliser:
    """Fits and applies per-feature min-max normalisation to env sequences.

    Args:
        df (pd.DataFrame): Full environmental DataFrame (date-indexed).
        columns (list): Column names to normalise.
    """

    def __init__(self, df: pd.DataFrame, columns: List[str] = ENV_COLUMNS):
        self.columns = columns
        self.mins = df[columns].min().values.astype(np.float32)   # [ENV_FEATURES]
        self.maxs = df[columns].max().values.astype(np.float32)   # [ENV_FEATURES]
        self.ranges = np.where(self.maxs - self.mins > 0,
                               self.maxs - self.mins, 1.0)        # avoid div-by-zero

    def normalise(self, arr: np.ndarray) -> np.ndarray:
        """Normalise a [T, ENV_FEATURES] array to [0, 1] per feature."""
        # arr shape: [T, ENV_FEATURES]
        return (arr - self.mins) / self.ranges

    def denormalise(self, arr: np.ndarray) -> np.ndarray:
        """Inverse transform from [0,1] back to original scale."""
        return arr * self.ranges + self.mins

    def normalise_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Normalise a [T, ENV_FEATURES] or [B, T, ENV_FEATURES] tensor."""
        mins = torch.tensor(self.mins, dtype=t.dtype, device=t.device)
        ranges = torch.tensor(self.ranges, dtype=t.dtype, device=t.device)
        return (t - mins) / ranges

    def denormalise_tensor(self, t: torch.Tensor) -> torch.Tensor:
        mins = torch.tensor(self.mins, dtype=t.dtype, device=t.device)
        ranges = torch.tensor(self.ranges, dtype=t.dtype, device=t.device)
        return t * ranges + mins


# ---------------------------------------------------------------------------
# Helper: parse image filename → date
# ---------------------------------------------------------------------------

def _parse_date_from_filename(filename: str) -> Optional[pd.Timestamp]:
    """Extract YYYY-MM-DD from patch_YYYY_MM_DD_XXXXX.jpg."""
    match = re.match(r"patch_(\d{4})_(\d{2})_(\d{2})_(\d+)\.jpg",
                     os.path.basename(filename))
    if match:
        y, m, d, _ = match.groups()
        return pd.Timestamp(f"{y}-{m}-{d}")
    return None


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class GrowliFlowerSequenceDataset(Dataset):
    """
    Multimodal time-series dataset aligning GrowliFlowerL/R images with
    NASA environmental data for the SI-PGS-R training pipeline.

    Each dataset *item* corresponds to a (prior_frame, current_frame) pair
    drawn from the same plant sequence:

        y_prior  : [3, H, W]        — previous RGB frame  (normalised [0,1])
        y        : [3, H, W]        — current  RGB frame  (normalised [0,1])
        env_seq  : [max_seq, 4]     — zero-padded cumulative env context c(τ)
        seq_len  : scalar int       — true length of env_seq before padding
        time_diff: [1]              — normalised Δt (days) between frames

    The environmental context window for frame t covers the CONTEXT_WINDOW_DAYS
    days *preceding* the capture date of that frame. Missing env days are
    forward-filled. Variable time intervals are encoded as an explicit
    `time_diff` scalar appended by the Recurrent_CVAE_Generator encoder.

    Args:
        manifest_path (str): Path to dataset_manifest.csv
        image_base_dir (str): Base directory for images
                              (images paths in manifest are relative to this).
        env_csv_path (str): Path to growliflower_environmental_data.csv
        img_size (tuple): (H, W) to resize images to.
        split (str): 'Train', 'Val', or 'Test' — filters manifest.
        context_window (int): Days of env history to pack into LSTM context.
        augment (bool): Apply random horizontal flip + colour jitter (Train only).
        normaliser (EnvNormaliser, optional): Pre-fitted normaliser; fitted
                                              from env_csv if not provided.
    """

    def __init__(
        self,
        manifest_path: str,
        image_base_dir: str,
        env_csv_path: str,
        img_size: Tuple[int, int] = (DEFAULT_IMG_H, DEFAULT_IMG_W),
        split: str = "Train",
        context_window: int = CONTEXT_WINDOW_DAYS,
        augment: bool = True,
        normaliser: Optional[EnvNormaliser] = None,
    ):
        super().__init__()
        self.image_base_dir = Path(image_base_dir)
        self.img_size = img_size
        self.split = split
        self.context_window = context_window

        # ── Image transforms ──────────────────────────────────────────────
        base_tf = [
            transforms.Resize(img_size),
            transforms.ToTensor(),  # → [3, H, W], values [0,1]
        ]
        if augment and split == "Train":
            aug_tf = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                       saturation=0.10, hue=0.05),
            ]
            self.transform = transforms.Compose(aug_tf + base_tf)
        else:
            self.transform = transforms.Compose(base_tf)

        # ── Load environmental CSV ─────────────────────────────────────────
        env_df = pd.read_csv(env_csv_path, parse_dates=["date"])
        env_df.set_index("date", inplace=True)
        # Forward-fill any missing dates in env data
        full_idx = pd.date_range(env_df.index.min(), env_df.index.max(), freq="D")
        env_df = env_df.reindex(full_idx).ffill().bfill()
        self.env_df = env_df  # DatetimeIndex → [ENV_FEATURES] rows

        # ── Fit normaliser ────────────────────────────────────────────────
        if normaliser is None:
            self.normaliser = EnvNormaliser(env_df)
        else:
            self.normaliser = normaliser

        # ── Load & filter manifest ────────────────────────────────────────
        manifest = pd.read_csv(manifest_path)
        # Deduplicate — manifest has many repetitions per (sequence, date)
        manifest = manifest.drop_duplicates(
            subset=["sequence_id", "elapsed_days", "image_path"]
        ).reset_index(drop=True)

        # Filter by split token in image_path  (Train/Val/Test prefix)
        manifest = manifest[
            manifest["image_path"].str.startswith(split)
        ].reset_index(drop=True)

        # Parse capture date from image filename
        manifest["capture_date"] = manifest["image_path"].apply(
            lambda p: _parse_date_from_filename(os.path.basename(p))
        )
        manifest = manifest.dropna(subset=["capture_date"]).reset_index(drop=True)

        # ── Build per-sequence lists of frames, sorted by DAP ─────────────
        # pair_list: list of (img_path_prior, img_path_curr,
        #                     date_curr, dap_curr, dap_prior)
        self.pair_list: List[Dict] = []

        max_dap = manifest["elapsed_days"].max()
        self.max_dap = float(max_dap) if max_dap > 0 else 1.0

        for seq_id, grp in manifest.groupby("sequence_id"):
            grp = grp.sort_values("elapsed_days").reset_index(drop=True)
            if len(grp) < 2:
                continue
            for i in range(1, len(grp)):
                prior_row = grp.iloc[i - 1]
                curr_row  = grp.iloc[i]
                self.pair_list.append({
                    "seq_id":         seq_id,
                    "img_prior":      str(self.image_base_dir / prior_row["image_path"]),
                    "img_curr":       str(self.image_base_dir / curr_row["image_path"]),
                    "date_curr":      curr_row["capture_date"],
                    "dap_curr":       curr_row["elapsed_days"],
                    "dap_prior":      prior_row["elapsed_days"],
                })

        logger.info(
            f"[{split}] GrowliFlowerSequenceDataset: "
            f"{len(self.pair_list)} (prior→curr) pairs "
            f"across {manifest['sequence_id'].nunique()} sequences."
        )

    # ────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.pair_list)

    # ────────────────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
          'y_prior'  : [3, H, W]           — previous frame (RGB, [0,1])
          'y'        : [3, H, W]           — current  frame (RGB, [0,1])
          'env_seq'  : [context_window, 4] — normalised environmental context
          'seq_len'  : scalar long         — true length of env_seq
          'time_diff': [1]                 — normalised Δt in days [0,1]
        """
        entry = self.pair_list[idx]

        # ── Load images ───────────────────────────────────────────────────
        y_prior = self._load_image(entry["img_prior"])  # [3, H, W]
        y       = self._load_image(entry["img_curr"])   # [3, H, W]

        # ── Build cumulative environmental context c(τ) ───────────────────
        # We take the CONTEXT_WINDOW_DAYS days leading up to the current frame.
        date_curr: pd.Timestamp = entry["date_curr"]
        env_seq, seq_len = self._build_env_context(date_curr)
        # env_seq  : [context_window, 4]   (zero-padded if < context_window)
        # seq_len  : true length (days with valid data)

        # ── Compute normalised time difference ───────────────────────────
        # Δt = days between prior and current frame, normalised by max_dap
        delta_days = float(entry["dap_curr"] - entry["dap_prior"])
        time_diff = torch.tensor(
            [delta_days / self.max_dap], dtype=torch.float32
        )  # shape: [1]

        return {
            "y_prior":   y_prior,               # [3, H, W]
            "y":         y,                     # [3, H, W]
            "env_seq":   env_seq,               # [context_window, 4]
            "seq_len":   torch.tensor(seq_len, dtype=torch.long),  # scalar
            "time_diff": time_diff,             # [1]
        }

    # ────────────────────────────────────────────────────────────────────────
    def _load_image(self, path: str) -> torch.Tensor:
        """Load a JPEG, apply transforms → [3, H, W]."""
        try:
            img = Image.open(path).convert("RGB")
        except FileNotFoundError:
            # Return a zero tensor if file missing (graceful degradation)
            logger.warning(f"Image not found: {path} — substituting zeros.")
            img = Image.fromarray(
                np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            )
        return self.transform(img)  # [3, H, W]

    # ────────────────────────────────────────────────────────────────────────
    def _build_env_context(
        self, date: pd.Timestamp
    ) -> Tuple[torch.Tensor, int]:
        """
        Construct a fixed-length environmental context window ending at `date`.

        Variable time intervals are handled here: we always return exactly
        `context_window` rows. If fewer real data-points exist (near the start
        of the season), the front is zero-padded so the LSTM sees zero-vectors
        for unobserved past time-steps — the sequence length scalar informs
        pack_padded_sequence of the true length.

        Returns:
            env_seq : [context_window, ENV_FEATURES]  (normalised, float32)
            seq_len : int — number of *real* time-steps in the context
        """
        start = date - pd.Timedelta(days=self.context_window - 1)
        end   = date

        # Slice the env dataframe
        mask = (self.env_df.index >= start) & (self.env_df.index <= end)
        window = self.env_df.loc[mask, ENV_COLUMNS]

        seq_len = min(len(window), self.context_window)

        # Build numpy array of shape [seq_len, 4], normalise
        if seq_len > 0:
            raw = window.values[-seq_len:].astype(np.float32)  # [seq_len, 4]
            raw = self.normaliser.normalise(raw)                # [seq_len, 4]
        else:
            raw = np.zeros((0, ENV_FEATURE_DIM), dtype=np.float32)

        # Zero-pad at the FRONT so that real data occupies the last seq_len rows
        padded = np.zeros((self.context_window, ENV_FEATURE_DIM), dtype=np.float32)
        if seq_len > 0:
            padded[-seq_len:] = raw  # real data at the end, zeros at front

        env_seq = torch.from_numpy(padded)  # [context_window, 4]
        return env_seq, seq_len


# ---------------------------------------------------------------------------
# Custom collate to batch variable-length data
# ---------------------------------------------------------------------------

def sipgs_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for GrowliFlowerSequenceDataset.

    Since env_seq is already padded to a fixed length (context_window),
    standard stacking works. seq_len values are stacked as a LongTensor.

    Batch tensor shapes:
        y_prior   : [B, 3, H, W]
        y         : [B, 3, H, W]
        env_seq   : [B, context_window, 4]
        seq_len   : [B]
        time_diff : [B, 1]
    """
    return {
        "y_prior":   torch.stack([b["y_prior"]   for b in batch]),  # [B,3,H,W]
        "y":         torch.stack([b["y"]         for b in batch]),  # [B,3,H,W]
        "env_seq":   torch.stack([b["env_seq"]   for b in batch]),  # [B,T,4]
        "seq_len":   torch.stack([b["seq_len"]   for b in batch]),  # [B]
        "time_diff": torch.stack([b["time_diff"] for b in batch]),  # [B,1]
    }


# ---------------------------------------------------------------------------
# Factory function — returns train / val dataloaders + normaliser
# ---------------------------------------------------------------------------

def build_dataloaders(
    manifest_path: str,
    image_base_dir: str,
    env_csv_path: str,
    img_size: Tuple[int, int] = (DEFAULT_IMG_H, DEFAULT_IMG_W),
    context_window: int = CONTEXT_WINDOW_DAYS,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, EnvNormaliser]:
    """
    Build Train, Val, and Test DataLoaders.

    The EnvNormaliser is fitted on the *entire* env CSV so normalisation
    statistics are consistent across all splits.

    Returns:
        train_loader, val_loader, test_loader, normaliser
    """
    # Fit normaliser once on entire env dataset
    env_df = pd.read_csv(env_csv_path, parse_dates=["date"])
    normaliser = EnvNormaliser(env_df)

    train_ds = GrowliFlowerSequenceDataset(
        manifest_path, image_base_dir, env_csv_path,
        img_size=img_size, split="Train",
        context_window=context_window, augment=True,
        normaliser=normaliser,
    )
    val_ds = GrowliFlowerSequenceDataset(
        manifest_path, image_base_dir, env_csv_path,
        img_size=img_size, split="Val",
        context_window=context_window, augment=False,
        normaliser=normaliser,
    )
    test_ds = GrowliFlowerSequenceDataset(
        manifest_path, image_base_dir, env_csv_path,
        img_size=img_size, split="Test",
        context_window=context_window, augment=False,
        normaliser=normaliser,
    )

    common_kwargs = dict(
        batch_size=batch_size,
        collate_fn=sipgs_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **common_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common_kwargs)

    logger.info(f"Train batches: {len(train_loader)} | "
                f"Val batches: {len(val_loader)} | "
                f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, normaliser


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    BASE = Path(__file__).parent
    MANIFEST   = str(BASE / "dataset_manifest.csv")
    IMAGE_DIR  = str(BASE / "datasets" / "GrowliFlowerL" / "images")
    ENV_CSV    = str(BASE / "datasets" / "growliflower_environmental_data.csv")

    train_loader, val_loader, test_loader, norm = build_dataloaders(
        MANIFEST, IMAGE_DIR, ENV_CSV,
        img_size=(128, 128),
        context_window=14,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(train_loader))
    print("Smoke-test shapes:")
    for k, v in batch.items():
        print(f"  {k:10s} → {tuple(v.shape)}")

    # Expected output (batch_size=4, context_window=14, img 128×128):
    # y_prior   → (4, 3, 128, 128)
    # y         → (4, 3, 128, 128)
    # env_seq   → (4, 14, 4)
    # seq_len   → (4,)
    # time_diff → (4, 1)
