"""
configs/config.py
-----------------
Central configuration — edit this file to change any hyperparameter or path.
All modules import from here so nothing is hardcoded elsewhere.
"""
from pathlib import Path
import random
import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"C:\one drive\OneDrive\SEM3\Capstone\UCI HAR Dataset\UCI HAR Dataset")
SAVE_DIR    = Path(r"C:\one drive\OneDrive\SEM3\Capstone")
CKPT_DIR    = Path(__file__).resolve().parents[1] / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark        = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cuda.matmul.allow_tf32 = True

# ── Dataset ────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Walking", "Walk Up", "Walk Down", "Sitting", "Standing", "Laying"]
N_CLASSES   = 6
FEAT_DIM    = 561
TS_CHANNELS = 3
TS_LEN      = 128

# ── DataLoader ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 256

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS_MLP    = 50
EPOCHS_FUSION = 50
EPOCHS_SEQ    = 50
EPOCHS_SSL    = 30
EPOCHS_KD     = 40
EPOCHS_FT     = 40

LR_MLP    = 1e-3
LR_FUSION = 5e-4
LR_SEQ    = 5e-4
LR_SSL    = 1e-3
LR_KD     = 1e-3
LR_FT     = 3e-4

PATIENCE = 8

# ── Knowledge Distillation ─────────────────────────────────────────────────────
KD_TEMP  = 4.0
KD_ALPHA = 0.7

# ── Robustness ─────────────────────────────────────────────────────────────────
ROB_NOISE_SIGMA  = 0.10
ROB_MISSING_COLS = 100
ROB_SEVERE_SIGMA = 0.30

# ── SHAP ───────────────────────────────────────────────────────────────────────
SHAP_BG_SAMPLES   = 500
SHAP_EVAL_SAMPLES = 200
SHAP_TOP_N        = 15

# ── Plotting ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 11})

COLORS6 = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6", "#795548"]
