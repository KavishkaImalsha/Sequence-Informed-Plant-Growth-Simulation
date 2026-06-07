"""
colab_runner.py
================
Run this file directly in a Google Colab cell to set up and launch
the SI-PGS-R v2 training pipeline on a GPU instance.

Usage in Colab (after mounting Drive and cloning repo):
    exec(open('colab_runner.py').read())

OR paste the relevant sections as individual Colab cells.
"""

import os
import sys
import subprocess

# ============================================================
# CONFIG — Edit these before running
# ============================================================

GITHUB_REPO  = "https://github.com/KavishkaImalsha/Sequence-Informed-Plant-Growth-Simulation.git"
DRIVE_BASE   = "/content/drive/MyDrive/SI-PGS-R"   # where you uploaded data
REPO_DIR     = "/content/sipgs_r"
OUTPUT_DIR   = f"{DRIVE_BASE}/output/sipgs_r_v2"    # saved to Drive

# Training hyperparameters (Colab T4 optimised)
BATCH_SIZE   = 16    # T4 16GB: safe at 240x320. Reduce to 12 if OOM.
EPOCHS       = 200
IMG_H        = 240
IMG_W        = 320
LR_GEN       = 1e-4
LR_DISC      = 4e-4   # TTUR: disc learns faster
BETA         = 0.90
LAMBDA_PERC  = 1.0
LAMBDA_FM    = 10.0
LAMBDA_SSIM  = 1.0
LAMBDA_TEMP  = 0.5
CONTEXT_WIN  = 14

# ============================================================
# STEP 1 — Verify GPU
# ============================================================

def check_gpu():
    import torch
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected! Go to Runtime → Change runtime type → GPU")
        sys.exit(1)
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {name}  |  VRAM: {vram:.1f} GB")
    return True


# ============================================================
# STEP 2 — Clone / update code
# ============================================================

def setup_code():
    if not os.path.exists(REPO_DIR):
        print(f"Cloning {GITHUB_REPO}...")
        ret = subprocess.run(
            ["git", "clone", GITHUB_REPO, REPO_DIR], capture_output=True, text=True
        )
        if ret.returncode != 0:
            print(f"❌ Clone failed:\n{ret.stderr}")
            sys.exit(1)
    else:
        print("Pulling latest code...")
        subprocess.run(["git", "-C", REPO_DIR, "pull"], capture_output=True)

    os.chdir(REPO_DIR)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    print(f"✅ Code ready at {REPO_DIR}")


# ============================================================
# STEP 3 — Symlink datasets from Drive
# ============================================================

def setup_datasets():
    drive_datasets = f"{DRIVE_BASE}/datasets"
    local_datasets = os.path.join(REPO_DIR, "datasets")
    os.makedirs(local_datasets, exist_ok=True)

    links = [
        ("GrowliFlowerL",                       "GrowliFlowerL"),
        ("GrowliFlowerR",                       "GrowliFlowerR"),
        ("growliflower_environmental_data.csv", "growliflower_environmental_data.csv"),
    ]
    for src_name, dst_name in links:
        src = os.path.join(drive_datasets, src_name)
        dst = os.path.join(local_datasets, dst_name)
        if not os.path.exists(dst):
            if os.path.exists(src):
                os.symlink(src, dst)
                print(f"  Linked: datasets/{dst_name}")
            else:
                print(f"  ⚠️  NOT FOUND in Drive: {src}")
                print(f"      Upload your datasets to {drive_datasets}/")

    manifest_dst = os.path.join(REPO_DIR, "dataset_manifest.csv")
    manifest_src = f"{DRIVE_BASE}/dataset_manifest.csv"
    if not os.path.exists(manifest_dst) and os.path.exists(manifest_src):
        os.symlink(manifest_src, manifest_dst)
        print("  Linked: dataset_manifest.csv")

    print("✅ Dataset symlinks configured")


# ============================================================
# STEP 4 — Install Python dependencies
# ============================================================

def install_deps():
    print("Installing dependencies...")
    pkgs = ["torchmetrics", "tensorboard", "scipy"]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + pkgs,
        check=True
    )
    print("✅ Dependencies installed")


# ============================================================
# STEP 5 — Pre-cache VGG-16 weights
# ============================================================

def cache_vgg():
    print("Downloading VGG-16 weights (fast on Colab)...")
    from torchvision.models import vgg16, VGG16_Weights
    vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    print("✅ VGG-16 weights cached")


# ============================================================
# STEP 6 — Verify data pipeline
# ============================================================

def verify_data():
    from data import build_dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader, norm = build_dataloaders(
        manifest_path  = "dataset_manifest.csv",
        image_base_dir = "datasets/GrowliFlowerL/images",
        env_csv_path   = "datasets/growliflower_environmental_data.csv",
        img_size       = (IMG_H, IMG_W),
        context_window = CONTEXT_WIN,
        batch_size     = BATCH_SIZE,
        num_workers    = 2,
        pin_memory     = True,
    )
    batch = next(iter(train_loader))
    print("  Batch shapes:")
    for k, v in batch.items():
        print(f"    {k:12s} → {tuple(v.shape)}")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print("✅ Data pipeline verified")
    return train_loader, val_loader, test_loader, norm


# ============================================================
# STEP 7 — Launch Training
# ============================================================

def launch_training():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd = [
        sys.executable, "train_v2.py",
        "--manifest",      "dataset_manifest.csv",
        "--image_dir",     "datasets/GrowliFlowerL/images",
        "--env_csv",       "datasets/growliflower_environmental_data.csv",
        "--output_dir",    OUTPUT_DIR,
        "--epochs",        str(EPOCHS),
        "--batch_size",    str(BATCH_SIZE),
        "--img_h",         str(IMG_H),
        "--img_w",         str(IMG_W),
        "--lr_gen",        str(LR_GEN),
        "--lr_disc",       str(LR_DISC),
        "--beta",          str(BETA),
        "--lambda_perc",   str(LAMBDA_PERC),
        "--lambda_fm",     str(LAMBDA_FM),
        "--lambda_ssim",   str(LAMBDA_SSIM),
        "--lambda_temp",   str(LAMBDA_TEMP),
        "--context_window",str(CONTEXT_WIN),
        "--num_workers",   "2",
    ]
    print(f"\n🚀 Launching training → checkpoints saved to Drive:\n   {OUTPUT_DIR}\n")
    # Stream output live
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
    proc.wait()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  SI-PGS-R v2 — Colab Training Setup")
    print("=" * 55)

    check_gpu()
    setup_code()
    setup_datasets()
    install_deps()
    cache_vgg()
    verify_data()
    launch_training()
