"""
train.py — Complete SI-PGS-R Training Pipeline
================================================
Implements the composite training objective:

    L_Total = L_CVAE + λ_adv · L_adv + λ_mse · L_mse

where:

    L_CVAE = -β · D_KL(q_φ(z|x,y) ‖ p_θ(z|x))
             + (1-β) · E_{q_φ}[log p_θ(y|x,z)]

         β  = 0.90  (heavy KL regularisation → realistic plant structures)

    L_adv  = Standard min-max GAN binary cross-entropy
             Generator:      -log D(ŷ_t)
             Discriminator:  -log D(y) - log(1 - D(ŷ_t))

    L_mse  = MSE(y_hat, y)   (frame coherence, directly minimises target metric)

Training features:
  • Adam optimisers with independent LR schedules for generator & discriminator
  • Gradient clipping on LSTM parameters (max_norm=1.0 by default)
  • Early stopping with configurable patience (monitors val SSIM)
  • TensorBoard & optional W&B logging of FID, SSIM, MSE each epoch
  • Automatic checkpointing of best model by val SSIM
  • Mixed-precision training (torch.cuda.amp) when CUDA is available

Usage:
    python train.py --manifest dataset_manifest.csv \\
                    --image_dir datasets/GrowliFlowerL/images \\
                    --env_csv datasets/growliflower_environmental_data.csv \\
                    --output_dir output/sipgs_r_run1 \\
                    --epochs 200 --batch_size 16
"""

import os
import sys
import json
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import inception_v3
from torchvision import transforms as TF
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data import build_dataloaders, ENV_FEATURE_DIM
from architectures import (
    LSTM_SequenceEncoder,
    Recurrent_CVAE_Generator,
    Recurrent_Discriminator,
    build_sipgs_r,
)
from utils import save_model_config, initiate_dir

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SI-PGS-R Train")
# ---------------------------------------------------------------------------


# ===========================================================================
# Loss Functions
# ===========================================================================

def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Analytical KL divergence between N(μ, σ²) and N(0, I).

        D_KL = -0.5 · Σ (1 + log σ² - μ² - σ²)

    Args:
        mu     : [B, latent_size]
        logvar : [B, latent_size]   (log of variance)

    Returns:
        kl_loss : scalar  (mean over batch)
    """
    # [B, latent_size] → scalar (mean)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl


def reconstruction_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction term  E_{q_φ}[log p_θ(y|x,z)].
    Implemented as pixel-wise BCE (treating images as Bernoulli).

    Args:
        y_hat : [B, C, H, W]   (generated, in [0,1])
        y     : [B, C, H, W]   (ground-truth, in [0,1])

    Returns:
        recon_loss : scalar
    """
    return F.binary_cross_entropy(y_hat, y, reduction="mean")


def cvae_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.90,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    β-CVAE loss:
        L_CVAE = -β · D_KL + (1-β) · E[log p(y|x,z)]

    Note: negative KL is minimised (we want to MINIMISE this loss),
    so the sign convention is:
        L_CVAE = β · KL + (1-β) · reconstruction_loss

    Args:
        y_hat  : [B, C, H, W]
        y      : [B, C, H, W]
        mu     : [B, latent_size]
        logvar : [B, latent_size]
        beta   : float   KL weight (0.90 in baseline for realism)

    Returns:
        total_cvae : scalar
        kl         : scalar
        recon      : scalar
    """
    kl    = kl_divergence_loss(mu, logvar)       # scalar
    recon = reconstruction_loss(y_hat, y)         # scalar
    total = beta * kl + (1.0 - beta) * recon      # scalar
    return total, kl, recon


def generator_adv_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """
    Non-saturating generator loss:  -E[log D(ŷ)]

    Args:
        d_fake : [B, 1]  discriminator output on generated frames

    Returns:
        scalar
    """
    return F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))


def discriminator_adv_loss(
    d_real: torch.Tensor, d_fake: torch.Tensor
) -> torch.Tensor:
    """
    Standard discriminator loss:
        L_D = -E[log D(y)] - E[log(1 - D(ŷ))]

    Args:
        d_real : [B, 1]
        d_fake : [B, 1]  (detached from generator graph)

    Returns:
        scalar
    """
    real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
    fake_loss = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
    return (real_loss + fake_loss) * 0.5


# ===========================================================================
# FID Computation Utilities
# ===========================================================================

class InceptionFeatureExtractor(nn.Module):
    """Extract Inception-v3 pool-3 features for FID computation.

    Returns [B, 2048] feature vectors.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        inception = inception_v3(pretrained=True, transform_input=False)
        # Truncate at avgpool
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
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
            imgs : [B, 3, H, W]  in [0,1]

        Returns:
            feats : [B, 2048] numpy array
        """
        imgs = self.resize(imgs.to(self.device))
        feats = self.features(imgs)         # [B, 2048, 1, 1]
        return feats.squeeze(-1).squeeze(-1).cpu().numpy()  # [B, 2048]


def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """
    Fréchet Inception Distance.

        FID = ‖μ_r - μ_f‖² + Tr(Σ_r + Σ_f - 2·√(Σ_r Σ_f))

    Args:
        real_feats : [N, 2048]
        fake_feats : [N, 2048]

    Returns:
        fid : float
    """
    from scipy import linalg
    mu1, sigma1 = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


# ===========================================================================
# Training Configuration (dataclass-style dict for easy JSON serialisation)
# ===========================================================================

def default_config() -> Dict:
    return {
        # Data
        "manifest_path": "dataset_manifest.csv",
        "image_base_dir": "datasets/GrowliFlowerL/images",
        "env_csv_path": "datasets/growliflower_environmental_data.csv",
        "img_size": [128, 128],
        "context_window": 14,
        "batch_size": 16,
        "num_workers": 4,
        # Architecture
        "img_channels": 3,
        "env_feature_size": ENV_FEATURE_DIM,
        "lstm_hidden": 128,
        "lstm_layers": 4,
        "lstm_output": 64,
        "latent_size": 128,
        # Loss
        "beta": 0.90,
        "lambda_adv": 0.10,
        "lambda_mse": 1.00,
        # Optimisers
        "lr_gen": 2e-4,
        "lr_disc": 1e-4,
        "betas": [0.5, 0.999],
        "weight_decay": 1e-5,
        # Scheduler (CosineAnnealingLR)
        "lr_min": 1e-6,
        # Training
        "epochs": 200,
        "grad_clip_lstm": 1.0,
        "grad_clip_gen": 5.0,
        "early_stop_patience": 20,
        "early_stop_metric": "val_ssim",
        # Logging
        "output_dir": "output/sipgs_r",
        "use_wandb": False,
        "wandb_project": "SI-PGS-R",
        "log_every_n_batches": 50,
        "eval_every_n_epochs": 5,
        "save_every_n_epochs": 10,
    }


# ===========================================================================
# Training State & Helpers
# ===========================================================================

class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    For SSIM: higher is better (mode='max').
    For FID/MSE: lower is better (mode='min').
    """

    def __init__(self, patience: int = 20, mode: str = "max", delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = metric
            return False
        improved = (
            (metric > self.best + self.delta) if self.mode == "max"
            else (metric < self.best - self.delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ===========================================================================
# Main Trainer
# ===========================================================================

class SIPGSTrainer:
    """
    Encapsulates the complete SI-PGS-R training loop.

    Key design choices:
      • Generator updates on every step; discriminator every step
        (no k-step ratio — empirically stable with spectral norm D).
      • Gradient clipping applied only to the LSTM parameters
        (as specified) to prevent exploding gradients through long sequences.
      • Mixed-precision when CUDA is available.
    """

    def __init__(self, cfg: Dict, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Build models ───────────────────────────────────────────────────
        img_shape = (cfg["img_channels"], cfg["img_size"][0], cfg["img_size"][1])
        self.encoder, self.generator, self.discriminator = build_sipgs_r(
            img_shape=img_shape,
            env_feature_size=cfg["env_feature_size"],
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            lstm_output=cfg["lstm_output"],
            latent_size=cfg["latent_size"],
            device=device,
        )
        logger.info(
            f"Encoder params:       {sum(p.numel() for p in self.encoder.parameters()):,}"
        )
        logger.info(
            f"Generator params:     {sum(p.numel() for p in self.generator.parameters()):,}"
        )
        logger.info(
            f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}"
        )

        # ── Optimisers ─────────────────────────────────────────────────────
        # Generator group = LSTM encoder + CVAE generator
        gen_params = (
            list(self.encoder.parameters()) +
            list(self.generator.parameters())
        )
        self.opt_gen = torch.optim.Adam(
            gen_params,
            lr=cfg["lr_gen"],
            betas=cfg["betas"],
            weight_decay=cfg["weight_decay"],
        )
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg["lr_disc"],
            betas=cfg["betas"],
            weight_decay=cfg["weight_decay"],
        )

        # ── LR Schedulers (Cosine Annealing) ──────────────────────────────
        self.sched_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_gen, T_max=cfg["epochs"], eta_min=cfg["lr_min"]
        )
        self.sched_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_disc, T_max=cfg["epochs"], eta_min=cfg["lr_min"]
        )

        # ── Metrics ────────────────────────────────────────────────────────
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(device)
        self.inception = InceptionFeatureExtractor(device=device)

        # ── AMP ────────────────────────────────────────────────────────────
        self.use_amp = (device.startswith("cuda"))
        self.scaler  = GradScaler(enabled=self.use_amp)

        # ── Logging ────────────────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        if cfg.get("use_wandb") and WANDB_AVAILABLE:
            wandb.init(project=cfg.get("wandb_project", "SI-PGS-R"), config=cfg)
            self.use_wandb = True
        else:
            self.use_wandb = False

        # ── Early stopping ─────────────────────────────────────────────────
        self.early_stopper = EarlyStopping(
            patience=cfg["early_stop_patience"], mode="max"
        )

        # Tracking
        self.best_ssim = -1.0
        self.global_step = 0

    # ────────────────────────────────────────────────────────────────────────
    def _train_step(self, batch: Dict) -> Dict[str, float]:
        """
        One gradient step on a single batch.

        Returns dict of scalar losses for logging.
        """
        cfg = self.cfg
        dev = self.device

        # Unpack batch — all tensors moved to device
        y_prior   = batch["y_prior"].to(dev)    # [B, 3, H, W]
        y         = batch["y"].to(dev)           # [B, 3, H, W]
        env_seq   = batch["env_seq"].to(dev)     # [B, T, 4]
        seq_len   = batch["seq_len"].to(dev)     # [B]
        time_diff = batch["time_diff"].to(dev)   # [B, 1]

        # ── Generator forward ──────────────────────────────────────────────
        with autocast(enabled=self.use_amp):
            x = self.encoder(env_seq, seq_len)                    # [B, lstm_output]
            y_hat, mu, logvar = self.generator(x, y, y_prior, time_diff)

            # CVAE loss (reconstruction + KL)
            loss_cvae, loss_kl, loss_recon = cvae_loss(
                y_hat, y, mu, logvar, beta=cfg["beta"]
            )

            # Adversarial loss (fool the discriminator)
            d_fake_for_gen = self.discriminator(y_hat)            # [B, 1]
            loss_adv_g     = generator_adv_loss(d_fake_for_gen)

            # Frame coherence MSE
            loss_mse = F.mse_loss(y_hat, y)

            loss_gen = (
                loss_cvae
                + cfg["lambda_adv"] * loss_adv_g
                + cfg["lambda_mse"] * loss_mse
            )

        # ── Generator backward ─────────────────────────────────────────────
        self.opt_gen.zero_grad()
        self.scaler.scale(loss_gen).backward()

        # Gradient clipping on LSTM parameters only
        self.scaler.unscale_(self.opt_gen)
        nn.utils.clip_grad_norm_(
            self.encoder.parameters(), cfg["grad_clip_lstm"]
        )
        nn.utils.clip_grad_norm_(
            self.generator.parameters(), cfg["grad_clip_gen"]
        )

        self.scaler.step(self.opt_gen)
        self.scaler.update()

        # ── Discriminator forward ──────────────────────────────────────────
        with autocast(enabled=self.use_amp):
            d_real = self.discriminator(y)                         # [B, 1]
            d_fake = self.discriminator(y_hat.detach())            # [B, 1]
            loss_disc = discriminator_adv_loss(d_real, d_fake)

        self.opt_disc.zero_grad()
        self.scaler.scale(loss_disc).backward()
        self.scaler.step(self.opt_disc)
        self.scaler.update()

        self.global_step += 1

        return {
            "loss_gen":   loss_gen.item(),
            "loss_cvae":  loss_cvae.item(),
            "loss_kl":    loss_kl.item(),
            "loss_recon": loss_recon.item(),
            "loss_adv_g": loss_adv_g.item(),
            "loss_mse":   loss_mse.item(),
            "loss_disc":  loss_disc.item(),
            "d_real":     d_real.mean().item(),
            "d_fake":     d_fake.mean().item(),
        }

    # ────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, val_loader) -> Dict[str, float]:
        """
        Compute val MSE, SSIM, and FID over the validation set.

        Returns:
            metrics: {'val_mse', 'val_ssim', 'val_fid'}
        """
        self.encoder.eval()
        self.generator.eval()

        mse_list   = []
        real_feats = []
        fake_feats = []

        for batch in val_loader:
            y_prior   = batch["y_prior"].to(self.device)
            y         = batch["y"].to(self.device)
            env_seq   = batch["env_seq"].to(self.device)
            seq_len   = batch["seq_len"].to(self.device)
            time_diff = batch["time_diff"].to(self.device)

            x     = self.encoder(env_seq, seq_len)
            y_hat = self.generator.generate(x, y_prior, time_diff)   # [B, C, H, W]

            mse_list.append(F.mse_loss(y_hat, y).item())
            self.ssim_metric.update(y_hat, y)

            real_feats.append(self.inception(y))                      # [B, 2048]
            fake_feats.append(self.inception(y_hat))

        val_mse  = float(np.mean(mse_list))
        val_ssim = float(self.ssim_metric.compute().item())
        self.ssim_metric.reset()

        rf = np.concatenate(real_feats, axis=0)  # [N, 2048]
        ff = np.concatenate(fake_feats, axis=0)
        val_fid = compute_fid(rf, ff)

        self.encoder.train()
        self.generator.train()

        return {"val_mse": val_mse, "val_ssim": val_ssim, "val_fid": val_fid}

    # ────────────────────────────────────────────────────────────────────────
    def _log(self, metrics: Dict[str, float], step: int):
        """Write metrics to TensorBoard and optionally W&B."""
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        if self.use_wandb:
            wandb.log(metrics, step=step)

    # ────────────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, tag: str = "latest"):
        """Save model weights and configs."""
        ckpt_dir = self.output_dir / f"ckpt_{tag}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.encoder.state_dict(),       ckpt_dir / "feature_encoder.pt")
        torch.save(self.generator.state_dict(),     ckpt_dir / "generator.pt")
        torch.save(self.discriminator.state_dict(), ckpt_dir / "discriminator.pt")

        save_model_config(
            "LSTM_SequenceEncoder",
            f"ckpt_{tag}/feature_encoder.pt",
            self.encoder.params,
            str(ckpt_dir / "feature_encoder.json"),
        )
        save_model_config(
            "Recurrent_CVAE_Generator",
            f"ckpt_{tag}/generator.pt",
            self.generator.params,
            str(ckpt_dir / "generator.json"),
        )
        save_model_config(
            "Recurrent_Discriminator",
            f"ckpt_{tag}/discriminator.pt",
            self.discriminator.params,
            str(ckpt_dir / "discriminator.json"),
        )
        # Save optimiser states for resuming
        torch.save({
            "epoch":         epoch,
            "opt_gen":       self.opt_gen.state_dict(),
            "opt_disc":      self.opt_disc.state_dict(),
            "sched_gen":     self.sched_gen.state_dict(),
            "sched_disc":    self.sched_disc.state_dict(),
            "best_ssim":     self.best_ssim,
            "global_step":   self.global_step,
        }, ckpt_dir / "training_state.pt")

        logger.info(f"Checkpoint saved → {ckpt_dir}")

    # ────────────────────────────────────────────────────────────────────────
    def train(self, train_loader, val_loader):
        """Full training loop."""
        cfg = self.cfg
        logger.info(
            f"Starting training: {cfg['epochs']} epochs, "
            f"device={self.device}, amp={self.use_amp}"
        )

        for epoch in range(1, cfg["epochs"] + 1):
            # ── Train epoch ────────────────────────────────────────────────
            self.encoder.train()
            self.generator.train()
            self.discriminator.train()

            epoch_losses: Dict[str, list] = {
                k: [] for k in [
                    "loss_gen", "loss_cvae", "loss_kl", "loss_recon",
                    "loss_adv_g", "loss_mse", "loss_disc", "d_real", "d_fake"
                ]
            }

            for i, batch in enumerate(train_loader):
                step_losses = self._train_step(batch)
                for k, v in step_losses.items():
                    epoch_losses[k].append(v)

                if (i + 1) % cfg["log_every_n_batches"] == 0:
                    log_dict = {
                        f"batch/{k}": v for k, v in step_losses.items()
                    }
                    log_dict["lr/gen"]  = self.opt_gen.param_groups[0]["lr"]
                    log_dict["lr/disc"] = self.opt_disc.param_groups[0]["lr"]
                    self._log(log_dict, self.global_step)

                    logger.info(
                        f"Ep {epoch:03d} [{i+1}/{len(train_loader)}] "
                        f"G={step_losses['loss_gen']:.4f}  "
                        f"KL={step_losses['loss_kl']:.4f}  "
                        f"Recon={step_losses['loss_recon']:.4f}  "
                        f"MSE={step_losses['loss_mse']:.5f}  "
                        f"D={step_losses['loss_disc']:.4f}"
                    )

            # ── Epoch-level train metrics ──────────────────────────────────
            train_log = {
                f"train/{k}": float(np.mean(v))
                for k, v in epoch_losses.items()
            }
            self._log(train_log, epoch)

            # ── LR scheduling ──────────────────────────────────────────────
            self.sched_gen.step()
            self.sched_disc.step()

            # ── Validation ────────────────────────────────────────────────
            if epoch % cfg["eval_every_n_epochs"] == 0:
                val_metrics = self._evaluate(val_loader)
                val_log = {f"val/{k}": v for k, v in val_metrics.items()}
                self._log(val_log, epoch)

                logger.info(
                    f"[Val Ep {epoch:03d}] "
                    f"MSE={val_metrics['val_mse']:.5f}  "
                    f"SSIM={val_metrics['val_ssim']:.4f}  "
                    f"FID={val_metrics['val_fid']:.2f}"
                )

                # ── Best model checkpoint ──────────────────────────────────
                if val_metrics["val_ssim"] > self.best_ssim:
                    self.best_ssim = val_metrics["val_ssim"]
                    self._save_checkpoint(epoch, tag="best")
                    logger.info(
                        f"  ✓ New best SSIM={self.best_ssim:.4f} — saved best ckpt"
                    )

                # ── Early stopping ─────────────────────────────────────────
                if self.early_stopper.step(val_metrics["val_ssim"]):
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(patience={cfg['early_stop_patience']})"
                    )
                    break

            # ── Periodic checkpoint ────────────────────────────────────────
            if epoch % cfg["save_every_n_epochs"] == 0:
                self._save_checkpoint(epoch, tag="latest")

        # Final save
        self._save_checkpoint(epoch, tag="final")
        self.writer.close()
        logger.info("Training complete.")


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train SI-PGS-R")
    p.add_argument("--manifest",     default="dataset_manifest.csv")
    p.add_argument("--image_dir",    default="datasets/GrowliFlowerL/images")
    p.add_argument("--env_csv",      default="datasets/growliflower_environmental_data.csv")
    p.add_argument("--output_dir",   default="output/sipgs_r")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--img_size",     type=int,   default=128)
    p.add_argument("--lr_gen",       type=float, default=2e-4)
    p.add_argument("--lr_disc",      type=float, default=1e-4)
    p.add_argument("--beta",         type=float, default=0.90,
                   help="KL weight in β-CVAE (0.90 = baseline)")
    p.add_argument("--lambda_adv",   type=float, default=0.10)
    p.add_argument("--lambda_mse",   type=float, default=1.00)
    p.add_argument("--latent_size",  type=int,   default=128)
    p.add_argument("--context_window", type=int, default=14)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--use_wandb",    action="store_true")
    p.add_argument("--config",       type=str,   default=None,
                   help="Path to JSON config file (overrides all flags).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build config (JSON overrides CLI flags)
    cfg = default_config()
    cfg.update({
        "manifest_path":  args.manifest,
        "image_base_dir": args.image_dir,
        "env_csv_path":   args.env_csv,
        "output_dir":     args.output_dir,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "img_size":       [args.img_size, args.img_size],
        "lr_gen":         args.lr_gen,
        "lr_disc":        args.lr_disc,
        "beta":           args.beta,
        "lambda_adv":     args.lambda_adv,
        "lambda_mse":     args.lambda_mse,
        "latent_size":    args.latent_size,
        "context_window": args.context_window,
        "early_stop_patience": args.patience,
        "num_workers":    args.num_workers,
        "use_wandb":      args.use_wandb,
    })
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    # Save config for reproducibility
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["output_dir"]) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Device
    device = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader, normaliser = build_dataloaders(
        manifest_path  = cfg["manifest_path"],
        image_base_dir = cfg["image_base_dir"],
        env_csv_path   = cfg["env_csv_path"],
        img_size       = tuple(cfg["img_size"]),
        context_window = cfg["context_window"],
        batch_size     = cfg["batch_size"],
        num_workers    = cfg["num_workers"],
        pin_memory     = (device == "cuda"),
    )

    # Train
    trainer = SIPGSTrainer(cfg, device=device)
    trainer.train(train_loader, val_loader)

    # Final test evaluation
    logger.info("Running final test set evaluation...")
    test_metrics = trainer._evaluate(test_loader)
    logger.info(
        f"[TEST] MSE={test_metrics['val_mse']:.5f}  "
        f"SSIM={test_metrics['val_ssim']:.4f}  "
        f"FID={test_metrics['val_fid']:.2f}"
    )
    with open(Path(cfg["output_dir"]) / "test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
