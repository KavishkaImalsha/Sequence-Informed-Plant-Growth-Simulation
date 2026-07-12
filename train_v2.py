"""
train_v2.py — Complete SI-PGS-R v2 Training Pipeline
======================================================
Integrates all v2 components:
  • AttentionLSTMEncoder + UNet_Recurrent_CVAE_Generator + MultiScale_PatchGAN
  • LSGAN + VGG perceptual + feature matching + temporal + SSIM losses
  • Full 6-metric evaluation (FID, MSE, MSE_tw, SSIM_tw, CS-MSE_tw, CS-SSIM_tw)
  • Mixed precision (AMP), gradient clipping, cosine LR schedule
  • Two-time-scale update rule (TTUR): G_lr = 1e-4, D_lr = 4e-4

Resolution: 240×320  (actual GrowliFlowerL patch size)
β = 0.90              (matches best SI-PGS result)

TTUR note: Using G_lr < D_lr is a standard technique (Heusel et al. 2017)
that ensures the discriminator stays slightly ahead of the generator,
preventing mode collapse without extra gradient penalty terms.

Usage:
    python train_v2.py \
        --manifest dataset_manifest.csv \
        --image_dir datasets/GrowliFlowerL/images \
        --env_csv datasets/growliflower_environmental_data.csv \
        --output_dir output/sipgs_r_v2 \
        --epochs 200 --batch_size 8
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
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data import build_dataloaders, ENV_FEATURE_DIM
from architectures_v2 import (
    AttentionLSTMEncoder,
    UNet_Recurrent_CVAE_Generator,
    MultiScale_PatchGAN_Discriminator,
    build_sipgs_r_v2,
)
from losses import (
    VGGPerceptualLoss,
    compute_total_generator_loss,
    compute_discriminator_loss,
)
from metrics import (
    InceptionFeatureExtractor,
    SequenceEvaluator,
    batch_metrics,
)
from utils import save_model_config

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SI-PGS-R v2")
# ---------------------------------------------------------------------------


# ===========================================================================
# Default Configuration
# ===========================================================================

def default_config() -> Dict:
    return {
        # Data
        "manifest_path":  "dataset_manifest.csv",
        "image_base_dir": "datasets/GrowliFlowerL/images",
        "env_csv_path":   "datasets/growliflower_environmental_data.csv",
        "img_size":       [240, 320],       # actual GrowliFlowerL patch size
        "context_window": 14,
        "batch_size":     8,               # reduced vs v1 due to 240x320 resolution
        "num_workers":    2,
        # Architecture
        "img_channels":   3,
        "env_feature_size": ENV_FEATURE_DIM,
        "lstm_hidden":    256,
        "lstm_layers":    4,
        "lstm_output":    128,
        "lstm_heads":     4,
        "latent_size":    256,
        "n_res_blocks":   2,
        # Loss weights
        "beta":           0.20,            # FIX: was 0.90 — caused KL collapse (encoder ignores env conditioning)
        "kl_warmup_epochs": 10,            # FIX: was 30 — restart annealing from low beta after resume
        "lambda_adv":     0.15,            # FIX: was 0.10 — increase to revive near-zero discriminator gradient
        "lambda_fm":      1.0,             # FIX: was 2.0 — feature_matching_loss applies λ_fm internally;
                                           #      compute_total_generator_loss also multiplies by λ_fm,
                                           #      so effective weight was λ_fm² = 4×. Setting to 1.0 gives net weight=1.0
        "lambda_perc":    2.0,             # FIX: was 1.0 — FID is 10× above target; VGG loss is key driver
        "lambda_temp":    0.5,             # unchanged — temporal coherence (key for SSIM_tw)
        "lambda_ssim":    3.0,             # FIX: was 2.0 — train SSIM stuck at 0.388, push harder
        "lambda_mse":     1.0,
        "perc_freq":      1,               # FIX: was 5 — VGG was only active 20% of steps; now every step
        # Optimisers (TTUR: D_lr > G_lr)
        "lr_gen":         1e-4,
        "lr_disc":        2e-4,            # FIX: was 4e-4 — D saturates too fast; halved to keep gradient alive
        "betas_gen":      [0.5, 0.999],
        "betas_disc":     [0.5, 0.999],
        "weight_decay":   1e-5,
        # Scheduler
        "lr_min":         1e-6,
        # Training
        "epochs":         300,             # 300 needed to beat paper's SSIM_tw=0.951
        "grad_clip_lstm": 1.0,
        "grad_clip_gen":  5.0,
        # Early stopping (patience epochs, monitor SSIM_tw)
        "early_stop_patience": 30,          # stop if no SSIM_tw improvement for 30 ep
        # Logging
        "output_dir":     "output/sipgs_r_v2",
        "use_wandb":      False,
        "wandb_project":  "SI-PGS-R-v2",
        "log_every_n_batches": 50,
        "eval_every_n_epochs": 5,
        "save_every_n_epochs": 10,
    }


# ===========================================================================
# Early Stopping
# ===========================================================================

class EarlyStopping:
    """Stops training when val_ssim_tw stops improving."""

    def __init__(self, patience: int = 20, delta: float = 1e-4):
        self.patience = patience
        self.delta    = delta
        self.best:    Optional[float] = None
        self.counter: int = 0

    def step(self, val: float) -> bool:
        if self.best is None or val > self.best + self.delta:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ===========================================================================
# Trainer
# ===========================================================================

class SIPGSTrainer_V2:
    """
    Full SI-PGS-R v2 training loop.

    Key differences from v1:
      • TTUR (two-time-scale update rule): lr_disc > lr_gen
      • LSGAN loss (stable, no mode collapse)
      • VGG perceptual loss (improves FID)
      • Feature matching loss (stabilises generator)
      • Temporal coherence loss (improves SSIM_tw)
      • Multi-scale PatchGAN discriminator
      • Sequence-level evaluation metrics
    """

    def __init__(self, cfg: Dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device
        self.out    = Path(cfg["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)

        img_shape = (cfg["img_channels"], cfg["img_size"][0], cfg["img_size"][1])

        # ── Models ────────────────────────────────────────────────────────
        self.encoder, self.generator, self.discriminator = build_sipgs_r_v2(
            img_shape=img_shape,
            env_feature_size=cfg["env_feature_size"],
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            lstm_output=cfg["lstm_output"],
            lstm_heads=cfg["lstm_heads"],
            latent_size=cfg["latent_size"],
            n_res_blocks=cfg["n_res_blocks"],
            device=device,
        )

        # ── VGG Perceptual Loss ───────────────────────────────────────────
        self.vgg_loss = VGGPerceptualLoss(resize=True, normalize=True).to(device)

        # ── Inception for FID ─────────────────────────────────────────────
        self.inception = InceptionFeatureExtractor(device=device)
        self.seq_evaluator = SequenceEvaluator(self.inception, device=device)

        # ── Log model sizes ───────────────────────────────────────────────
        ep = sum(p.numel() for p in self.encoder.parameters())
        gp = sum(p.numel() for p in self.generator.parameters())
        dp = sum(p.numel() for p in self.discriminator.parameters())
        logger.info(f"Encoder      : {ep:>12,} params")
        logger.info(f"Generator    : {gp:>12,} params")
        logger.info(f"Discriminator: {dp:>12,} params")
        logger.info(f"Total        : {ep+gp+dp:>12,} params")

        # ── Optimisers (TTUR) ─────────────────────────────────────────────
        gen_params = (
            list(self.encoder.parameters()) +
            list(self.generator.parameters())
        )
        self.opt_gen  = torch.optim.Adam(
            gen_params,
            lr=cfg["lr_gen"], betas=cfg["betas_gen"],
            weight_decay=cfg["weight_decay"],
        )
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg["lr_disc"], betas=cfg["betas_disc"],
            weight_decay=cfg["weight_decay"],
        )

        # ── LR Schedulers ─────────────────────────────────────────────────
        self.sched_gen  = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_gen,  T_max=cfg["epochs"], eta_min=cfg["lr_min"]
        )
        self.sched_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_disc, T_max=cfg["epochs"], eta_min=cfg["lr_min"]
        )

        # ── Mixed Precision ───────────────────────────────────────────────
        self.use_amp = device.startswith("cuda")
        # GradScaler only meaningful on CUDA; kept disabled on MPS/CPU
        if self.use_amp:
            self.scaler = GradScaler("cuda", enabled=True)
        else:
            self.scaler = GradScaler("cpu", enabled=False)

        # ── TensorBoard / W&B ─────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(self.out / "tensorboard"))
        self.use_wandb = cfg.get("use_wandb", False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=cfg.get("wandb_project", "SI-PGS-R-v2"), config=cfg)

        # ── Early Stopping ────────────────────────────────────────────────
        self.stopper     = EarlyStopping(patience=cfg["early_stop_patience"])
        self.best_ssim_tw = -1.0
        self.global_step  = 0

    # ── KL Annealing helper ──────────────────────────────────────────────
    def _compute_beta(self, epoch: int) -> float:
        """
        Linear KL annealing: beta ramps from 0.01 to target over kl_warmup_epochs.

        Why: Setting beta=0.90 from epoch 1 forces the encoder to match N(0,I)
        immediately, before the decoder has learned anything. This causes the
        latent code to collapse to zero (KL ≈ 0.01) and the model ignores the
        environmental conditioning entirely.

        Annealing lets the model first learn pixel-level reconstruction
        (beta≈0), then progressively regularise the latent space.
        """
        warmup = max(1, self.cfg.get("kl_warmup_epochs", 30))
        target = self.cfg.get("beta", 0.90)
        frac   = min(1.0, epoch / warmup)
        return 0.01 + frac * (target - 0.01)

    # ── Single Training Step ───────────────────────────────────────────────
    def _train_step(self, batch: Dict, epoch: int = 1) -> Dict[str, float]:
        dev = self.device
        cfg = self.cfg

        y_prior   = batch["y_prior"].to(dev)   # [B, 3, H, W]
        y         = batch["y"].to(dev)          # [B, 3, H, W]
        env_seq   = batch["env_seq"].to(dev)    # [B, T, 4]
        seq_len   = batch["seq_len"].to(dev)    # [B]
        time_diff = batch["time_diff"].to(dev)  # [B, 1]

        # ── Generator forward ──────────────────────────────────────────────
        # VGG perceptual loss is expensive (~50ms/step on T4).
        # Compute every perc_freq steps for speed; skip on other steps.
        use_perc = (self.global_step % cfg.get("perc_freq", 5) == 0)
        lambda_perc_this_step = cfg["lambda_perc"] if use_perc else 0.0

        # KL-annealed beta for this step
        current_beta = self._compute_beta(getattr(self, '_current_epoch', 1))

        with autocast(device_type='cuda', enabled=self.use_amp):
            x = self.encoder(env_seq, seq_len)                      # [B, 128]
            y_hat, mu, logvar = self.generator(x, y, y_prior, time_diff)

            # Need D outputs for both real and fake (for feature matching)
            d_fake_for_gen = self.discriminator(y_hat)               # not detached
            d_real_for_fm  = self.discriminator(y.detach())          # for FM loss

            total_g, log_dict = compute_total_generator_loss(
                y_hat=y_hat, y=y, mu=mu, logvar=logvar,
                y_prior=y_prior, time_diff=time_diff,
                d_fake_output=d_fake_for_gen,
                d_real_output=d_real_for_fm,
                perceptual_fn=self.vgg_loss,
                beta=current_beta,            # annealed KL weight
                lambda_adv=cfg["lambda_adv"],
                lambda_fm=cfg["lambda_fm"],
                lambda_perc=lambda_perc_this_step,
                lambda_temp=cfg["lambda_temp"],
                lambda_ssim=cfg["lambda_ssim"],
                lambda_mse=cfg["lambda_mse"],
            )
            log_dict["beta"] = current_beta   # log annealed beta

        # ── Generator backward ─────────────────────────────────────────────
        self.opt_gen.zero_grad()
        self.scaler.scale(total_g).backward()
        self.scaler.unscale_(self.opt_gen)
        nn.utils.clip_grad_norm_(self.encoder.parameters(),   cfg["grad_clip_lstm"])
        nn.utils.clip_grad_norm_(self.generator.parameters(), cfg["grad_clip_gen"])
        self.scaler.step(self.opt_gen)
        self.scaler.update()

        # ── Discriminator forward / backward ──────────────────────────────
        with autocast(device_type='cuda', enabled=self.use_amp):
            d_real = self.discriminator(y)
            d_fake = self.discriminator(y_hat.detach())
            loss_disc, disc_val = compute_discriminator_loss(d_real, d_fake)

        self.opt_disc.zero_grad()
        self.scaler.scale(loss_disc).backward()
        self.scaler.step(self.opt_disc)
        self.scaler.update()

        log_dict["loss_disc"] = disc_val
        self.global_step += 1
        return log_dict

    # ── Evaluation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, val_loader) -> Dict[str, float]:
        """Compute all 6 research metrics over validation set."""
        self.encoder.eval()
        self.generator.eval()
        self.seq_evaluator.reset()

        for batch in val_loader:
            y_prior   = batch["y_prior"].to(self.device)
            y         = batch["y"].to(self.device)
            env_seq   = batch["env_seq"].to(self.device)
            seq_len   = batch["seq_len"].to(self.device)
            time_diff = batch["time_diff"].to(self.device)

            x     = self.encoder(env_seq, seq_len)
            y_hat = self.generator.generate(x, y_prior, time_diff)

            # Convert sequence ids to strings for the evaluator
            if "seq_id" in batch:
                seq_ids = [str(s) for s in batch["seq_id"]]
            else:
                # Fall back to batch index as proxy (no temporal grouping)
                seq_ids = [f"seq_{i}" for i in range(y_hat.shape[0])]

            dap = batch.get("dap_curr", None)

            self.seq_evaluator.update(
                y_hat=y_hat.cpu(),
                y=y.cpu(),
                seq_ids=seq_ids,
                dap_values=dap,
            )

        results = self.seq_evaluator.compute()
        self.encoder.train()
        self.generator.train()
        return results

    # ── Logging ───────────────────────────────────────────────────────────
    def _log(self, metrics: Dict[str, float], step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        if self.use_wandb:
            wandb.log(metrics, step=step)

    # ── Checkpoint ────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, tag: str = "latest"):
        ckpt = self.out / f"ckpt_{tag}"
        ckpt.mkdir(parents=True, exist_ok=True)

        torch.save(self.encoder.state_dict(),       ckpt / "feature_encoder.pt")
        torch.save(self.generator.state_dict(),     ckpt / "generator.pt")
        torch.save(self.discriminator.state_dict(), ckpt / "discriminator.pt")

        # Save JSON configs for model reconstruction
        def _save_json(name, obj, filename):
            save_model_config(
                name, f"ckpt_{tag}/{filename}.pt",
                obj.params, str(ckpt / f"{filename}.json")
            )

        _save_json("AttentionLSTMEncoder",            self.encoder,       "feature_encoder")
        _save_json("UNet_Recurrent_CVAE_Generator",   self.generator,     "generator")
        _save_json("MultiScale_PatchGAN_Discriminator",self.discriminator,"discriminator")

        torch.save({
            "epoch": epoch,
            "opt_gen":    self.opt_gen.state_dict(),
            "opt_disc":   self.opt_disc.state_dict(),
            "sched_gen":  self.sched_gen.state_dict(),
            "sched_disc": self.sched_disc.state_dict(),
            "best_ssim_tw": self.best_ssim_tw,
            "global_step":  self.global_step,
        }, ckpt / "training_state.pt")

        logger.info(f"Checkpoint saved → {ckpt}")

    # ── Load Checkpoint (for resuming after crash / Colab timeout) ────────
    def load_checkpoint(self, tag: str = "latest") -> int:
        """
        Load model weights, optimizer/scheduler states, and training metadata
        from a saved checkpoint.

        Returns:
            start_epoch (int): epoch to resume from (next epoch after the saved one)
        """
        ckpt = self.out / f"ckpt_{tag}"
        if not ckpt.exists():
            logger.info(f"No checkpoint found at {ckpt} — starting from scratch.")
            return 1

        # Verify checkpoint integrity before loading (detect mid-save corruption)
        required = ["feature_encoder.pt", "generator.pt", "discriminator.pt"]
        for fname in required:
            fpath = ckpt / fname
            if not fpath.exists():
                logger.warning(f"Checkpoint incomplete — missing {fname}. Starting fresh.")
                return 1
            try:
                torch.load(fpath, map_location="cpu", weights_only=False)  # integrity check
            except Exception as e:
                logger.warning(
                    f"Checkpoint '{tag}/{fname}' is CORRUPTED (mid-save kill detected).\n"
                    f"  Error: {e}\n"
                    f"  → Deleting corrupted checkpoint and starting from scratch."
                )
                import shutil
                shutil.rmtree(ckpt, ignore_errors=True)
                return 1

        # Load model weights (weights_only=False required for state_dicts in PyTorch 2.x)
        self.encoder.load_state_dict(
            torch.load(ckpt / "feature_encoder.pt", map_location=self.device,
                       weights_only=False))
        self.generator.load_state_dict(
            torch.load(ckpt / "generator.pt",       map_location=self.device,
                       weights_only=False))
        self.discriminator.load_state_dict(
            torch.load(ckpt / "discriminator.pt",   map_location=self.device,
                       weights_only=False))

        # Load training state
        state_path = ckpt / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device, weights_only=False)
            self.opt_gen.load_state_dict(state["opt_gen"])
            self.opt_disc.load_state_dict(state["opt_disc"])
            self.sched_gen.load_state_dict(state["sched_gen"])
            self.sched_disc.load_state_dict(state["sched_disc"])
            self.best_ssim_tw = state.get("best_ssim_tw", -1.0)
            self.global_step  = state.get("global_step",  0)
            resumed_epoch     = state.get("epoch", 0)
        else:
            resumed_epoch = 0

        start_epoch = resumed_epoch + 1
        logger.info(
            f"✅ Resumed from checkpoint '{tag}' "
            f"(epoch {resumed_epoch} → continuing from epoch {start_epoch}) "
            f"| best_ssim_tw={self.best_ssim_tw:.4f} | step={self.global_step}"
        )
        return start_epoch

    # ── Main Training Loop ────────────────────────────────────────────────
    def train(self, train_loader, val_loader, start_epoch: int = 1):
        cfg = self.cfg
        logger.info(
            f"Starting SI-PGS-R v2 training: {cfg['epochs']} epochs | "
            f"res={cfg['img_size']} | device={self.device} | amp={self.use_amp} "
            f"| resuming from epoch {start_epoch}"
        )

        for epoch in range(start_epoch, cfg["epochs"] + 1):
            self._current_epoch = epoch   # expose epoch to _train_step for KL annealing
            self.encoder.train()
            self.generator.train()
            self.discriminator.train()

            epoch_losses: Dict[str, List] = {}

            for i, batch in enumerate(train_loader):
                step_logs = self._train_step(batch, epoch=epoch)

                for k, v in step_logs.items():
                    epoch_losses.setdefault(k, []).append(v)

                if (i + 1) % cfg["log_every_n_batches"] == 0:
                    log_dict = {f"batch/{k}": v for k, v in step_logs.items()}
                    log_dict["lr/gen"]  = self.opt_gen.param_groups[0]["lr"]
                    log_dict["lr/disc"] = self.opt_disc.param_groups[0]["lr"]
                    self._log(log_dict, self.global_step)

                    logger.info(
                        f"Ep {epoch:03d} [{i+1:4d}/{len(train_loader)}] "
                        f"G={step_logs['loss_gen']:.4f} "
                        f"KL={step_logs['loss_kl']:.4f} "
                        f"Perc={step_logs['loss_perc']:.4f} "
                        f"MSE={step_logs['loss_mse']:.5f} "
                        f"SSIM_loss={step_logs['loss_ssim']:.4f} "
                        f"D={step_logs['loss_disc']:.4f}"
                    )

            # ── Epoch-level train metrics ─────────────────────────────────────────
            train_log = {f"train/{k}": float(np.mean(v)) for k, v in epoch_losses.items()}
            self._log(train_log, epoch)

            # ── Unified metrics/ namespace (same scale as val/) ───────────────
            # TensorBoard overlays tags with the same suffix on one graph
            # when you select them in the left panel.
            # MSE: convert from [0,1] scale → pixel scale (multiply by 255²)
            train_mse_01   = float(np.mean(epoch_losses.get("loss_mse",   [0])))
            train_ssim_los = float(np.mean(epoch_losses.get("loss_ssim",  [0])))
            train_g_loss   = float(np.mean(epoch_losses.get("loss_gen",   [0])))
            train_d_loss   = float(np.mean(epoch_losses.get("loss_disc",  [0])))
            train_kl       = float(np.mean(epoch_losses.get("loss_kl",    [0])))
            train_perc     = float(np.mean(epoch_losses.get("loss_perc",  [0])))

            unified_train = {
                # Pixel-scale MSE — matches val/mse scale
                "metrics/mse/train":      train_mse_01 * (255 ** 2),
                # SSIM: convert 1-SSIM loss → SSIM value, matches val/ssim
                "metrics/ssim/train":     1.0 - train_ssim_los,
                # Generator & Discriminator losses
                "metrics/loss_G/train":   train_g_loss,
                "metrics/loss_D/train":   train_d_loss,
                "metrics/kl/train":       train_kl,
                "metrics/perceptual/train": train_perc,
            }
            self._log(unified_train, epoch)

            logger.info(
                f"Ep {epoch:03d} TRAIN ─ "
                f"G={train_g_loss:.4f}  D={train_d_loss:.4f}  "
                f"MSE_px={train_mse_01*(255**2):.1f}  "
                f"SSIM={1-train_ssim_los:.4f}  "
                f"KL={train_kl:.4f}"
            )

            self.sched_gen.step()
            self.sched_disc.step()

            # ── Validation ─────────────────────────────────────────────────────
            if epoch % cfg["eval_every_n_epochs"] == 0:
                val_metrics = self._evaluate(val_loader)

                # Log under val/ (existing)
                val_log = {f"val/{k}": v for k, v in val_metrics.items()}
                self._log(val_log, epoch)

                # Also log under unified metrics/ namespace — same graph as train
                unified_val = {
                    "metrics/mse/val":        val_metrics.get("mse",     0),
                    "metrics/ssim/val":       val_metrics.get("ssim",    0),
                    "metrics/loss_G/val":     val_metrics.get("mse",     0),  # proxy
                    "metrics/mse_tw/val":     val_metrics.get("mse_tw",  0),
                    "metrics/ssim_tw/val":    val_metrics.get("ssim_tw", 0),
                    "metrics/fid/val":        val_metrics.get("fid",     0),
                }
                self._log(unified_val, epoch)

                # ── Overfitting diagnosis in terminal ────────────────────────
                val_mse    = val_metrics.get("mse",    0)
                val_ssim   = val_metrics.get("ssim",   0)
                tr_mse_px  = train_mse_01 * (255 ** 2)
                tr_ssim    = 1.0 - train_ssim_los

                mse_gap  = val_mse  - tr_mse_px
                ssim_gap = tr_ssim  - val_ssim

                if mse_gap > 1500 or ssim_gap > 0.10:
                    fit_status = "⚠️  OVERFITTING  (val >> train)"
                elif val_mse > 8000 and tr_mse_px > 7000:
                    fit_status = "⚠️  UNDERFITTING (both losses too high)"
                else:
                    fit_status = "✅  GOOD FIT"

                logger.info(
                    f"\n{'='*60}\n"
                    f"  Val Epoch {epoch:03d}          {fit_status}\n"
                    f"  {'Metric':<14} {'Train':>10} {'Val':>10} {'Gap':>10}\n"
                    f"  {'-'*46}\n"
                    f"  {'MSE (pixel)':<14} {tr_mse_px:>10.1f} {val_mse:>10.1f} "
                    f"{mse_gap:>+10.1f}\n"
                    f"  {'SSIM':<14} {tr_ssim:>10.4f} {val_ssim:>10.4f} "
                    f"{-ssim_gap:>+10.4f}\n"
                    f"  {'SSIM_tw':<14} {'---':>10} "
                    f"{val_metrics.get('ssim_tw',0):>10.4f} {'target>0.94':>10}\n"
                    f"  {'MSE_tw':<14} {'---':>10} "
                    f"{val_metrics.get('mse_tw',0):>10.1f} {'target<1500':>10}\n"
                    f"  {'FID':<14} {'---':>10} "
                    f"{val_metrics.get('fid',0):>10.2f} {'target<42':>10}\n"
                    f"{'='*60}"
                )

                ssim_tw = val_metrics.get("ssim_tw", -1.0)
                if ssim_tw > self.best_ssim_tw:
                    self.best_ssim_tw = ssim_tw
                    self._save_checkpoint(epoch, tag="best")
                    logger.info(f"  ✨ New best SSIM_tw={self.best_ssim_tw:.4f} — checkpoint saved")

                if self.stopper.step(ssim_tw):
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(patience={cfg['early_stop_patience']})"
                    )
                    break

            if epoch % cfg["save_every_n_epochs"] == 0:
                self._save_checkpoint(epoch, tag="latest")

            # ── Emergency save every epoch to minimise lost work ──────────
            # Colab free tier can disconnect unexpectedly — save lightweight
            # state every epoch so at most 1 epoch is ever re-done
            self._save_checkpoint(epoch, tag="latest")

        self._save_checkpoint(epoch, tag="final")
        self.writer.close()
        logger.info("Training complete.")


# ===========================================================================
# Picklable Dataset wrapper + Collate — required for Python 3.14 forkserver
# ===========================================================================

class SequenceAwareDataset(torch.utils.data.Dataset):
    """
    Wraps GrowliFlowerSequenceDataset to expose 'seq_id' and 'dap_curr'
    in each sample for sequence-level temporal metric computation.

    Implemented as a proper class (not monkey-patch) so it is fully
    picklable under Python 3.14's forkserver multiprocessing.
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]                    # dict from base dataset
        entry  = self.base.pair_list[idx]          # raw manifest entry
        sample["seq_id"]   = str(entry["seq_id"])
        sample["dap_curr"] = torch.tensor(
            float(entry["dap_curr"]), dtype=torch.float32
        )
        return sample


class SequenceAwareCollateFn:
    """
    Picklable collate function that extends sipgs_collate_fn with
    seq_id (list of str) and dap_curr (float tensor [B]).

    Must be a class (not a local function) so Python's forkserver
    multiprocessing can pickle it for worker processes.
    """

    def __call__(self, batch):
        from data import sipgs_collate_fn
        base = sipgs_collate_fn(batch)
        base["seq_id"]   = [b["seq_id"]   for b in batch]
        base["dap_curr"] = torch.stack([b["dap_curr"] for b in batch])
        return base


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train SI-PGS-R v2")
    p.add_argument("--manifest",       default="dataset_manifest.csv")
    p.add_argument("--image_dir",      default="datasets/GrowliFlowerL/images")
    p.add_argument("--env_csv",        default="datasets/growliflower_environmental_data.csv")
    p.add_argument("--output_dir",     default="output/sipgs_r_v2")
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--img_h",          type=int,   default=240)
    p.add_argument("--img_w",          type=int,   default=320)
    p.add_argument("--lr_gen",         type=float, default=1e-4)
    p.add_argument("--lr_disc",        type=float, default=4e-4)
    p.add_argument("--beta",           type=float, default=0.90)
    p.add_argument("--lambda_perc",    type=float, default=1.0)
    p.add_argument("--lambda_fm",      type=float, default=10.0)
    p.add_argument("--lambda_ssim",    type=float, default=1.0)
    p.add_argument("--lambda_temp",    type=float, default=0.5)
    p.add_argument("--latent_size",    type=int,   default=256)
    p.add_argument("--context_window", type=int,   default=14)
    p.add_argument("--patience",       type=int,   default=20)
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--resume",         action="store_true",
                   help="Resume training from ckpt_latest in output_dir")
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--config",         type=str,   default=None,
                   help="Path to JSON config (overrides all flags).")
    return p.parse_args()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from data import (
        GrowliFlowerSequenceDataset, EnvNormaliser,
        sipgs_collate_fn, build_dataloaders, ENV_COLUMNS
    )
    import pandas as pd

    args = parse_args()
    cfg  = default_config()
    cfg.update({
        "manifest_path":  args.manifest,
        "image_base_dir": args.image_dir,
        "env_csv_path":   args.env_csv,
        "output_dir":     args.output_dir,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "img_size":       [args.img_h, args.img_w],
        "lr_gen":         args.lr_gen,
        "lr_disc":        args.lr_disc,
        "beta":           args.beta,
        "lambda_perc":    args.lambda_perc,
        "lambda_fm":      args.lambda_fm,
        "lambda_ssim":    args.lambda_ssim,
        "lambda_temp":    args.lambda_temp,
        "latent_size":    args.latent_size,
        "context_window": args.context_window,
        "early_stop_patience": args.patience,
        "num_workers":    args.num_workers,
        "use_wandb":      args.use_wandb,
    })
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    # Save config
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["output_dir"]) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Device
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Device: {device}")

    # Build dataloaders with seq_id support
    env_df      = pd.read_csv(cfg["env_csv_path"], parse_dates=["date"])
    normaliser  = EnvNormaliser(env_df)
    img_size    = tuple(cfg["img_size"])

    # On MPS / CPU, multiprocessing workers are not beneficial and cause
    # pickling issues under Python 3.14's forkserver. Force num_workers=0.
    effective_workers = cfg["num_workers"] if device == "cuda" else 0
    if effective_workers == 0:
        logger.info("num_workers=0 (MPS/CPU device — single-process data loading)")

    train_base = GrowliFlowerSequenceDataset(
        cfg["manifest_path"], cfg["image_base_dir"], cfg["env_csv_path"],
        img_size=img_size, split="Train", context_window=cfg["context_window"],
        augment=True, normaliser=normaliser,
    )
    val_base = GrowliFlowerSequenceDataset(
        cfg["manifest_path"], cfg["image_base_dir"], cfg["env_csv_path"],
        img_size=img_size, split="Val", context_window=cfg["context_window"],
        augment=False, normaliser=normaliser,
    )
    test_base = GrowliFlowerSequenceDataset(
        cfg["manifest_path"], cfg["image_base_dir"], cfg["env_csv_path"],
        img_size=img_size, split="Test", context_window=cfg["context_window"],
        augment=False, normaliser=normaliser,
    )

    # Wrap with SequenceAwareDataset to expose seq_id + dap_curr (picklable)
    train_ds = SequenceAwareDataset(train_base)
    val_ds   = SequenceAwareDataset(val_base)
    test_ds  = SequenceAwareDataset(test_base)
    collate  = SequenceAwareCollateFn()   # picklable callable class

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate, num_workers=effective_workers,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=effective_workers,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=effective_workers,
        pin_memory=(device == "cuda"), drop_last=True,
    )

    logger.info(
        f"Data loaded: train={len(train_loader)} | "
        f"val={len(val_loader)} | test={len(test_loader)} batches"
    )

    # Train
    trainer     = SIPGSTrainer_V2(cfg, device=device)
    start_epoch = 1

    # ── Auto-resume if --resume flag is set ───────────────────────────────
    # Tries ckpt_latest first, falls back to ckpt_best, then epoch 1
    if args.resume:
        ckpt_dir = Path(cfg["output_dir"])
        if (ckpt_dir / "ckpt_latest" / "training_state.pt").exists():
            start_epoch = trainer.load_checkpoint(tag="latest")
        elif (ckpt_dir / "ckpt_best" / "training_state.pt").exists():
            start_epoch = trainer.load_checkpoint(tag="best")
            logger.info("ckpt_latest not found — resumed from ckpt_best")
        else:
            logger.info("No checkpoint found — starting from epoch 1")
    else:
        logger.info("Fresh training run (use --resume to continue from checkpoint)")

    trainer.train(train_loader, val_loader, start_epoch=start_epoch)

    # Final test evaluation
    logger.info("Running final test evaluation...")
    test_results = trainer._evaluate(test_loader)
    logger.info(trainer.seq_evaluator.format_table(test_results))

    with open(Path(cfg["output_dir"]) / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

