"""
data/dataset.py
---------------
Loads the UCI HAR dataset from disk and returns PyTorch DataLoaders.

Two loader types are returned:
  - feature-only  (Xf, y)          → for MLP family
  - multi-modal   (Xf, Xt, y)      → for FusionNet, GatedFusionNet, BiLSTM, CNN-LSTM
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from configs.config import (
    DATA_ROOT, BATCH_SIZE, SEED,
    TS_CHANNELS, TS_LEN,
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_split(base, split: str):
    """Load features, labels, subjects, and raw inertial signals for one split."""
    sp   = base / split
    is_  = sp / "Inertial Signals"

    X    = np.loadtxt(sp / f"X_{split}.txt")
    y    = np.loadtxt(sp / f"y_{split}.txt").astype(int) - 1   # 0-indexed
    subj = np.loadtxt(sp / f"subject_{split}.txt").astype(int)

    # Body accelerometer (x, y, z) — shape (N, 128) each
    ts_x = np.loadtxt(is_ / f"body_acc_x_{split}.txt")
    ts_y = np.loadtxt(is_ / f"body_acc_y_{split}.txt")
    ts_z = np.loadtxt(is_ / f"body_acc_z_{split}.txt")

    return X, y, subj, ts_x, ts_y, ts_z


def _to_tensor_f32(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32)


def _to_tensor_i64(a: np.ndarray) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.long)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_uci_har(data_root=None, verbose: bool = True):
    """
    Load and preprocess the UCI HAR dataset.

    Returns
    -------
    dict with keys:
        Xf_tr, Xf_te   : (N, 561) scaled feature tensors
        Xt_tr, Xt_te   : (N, 3, 128) raw time-series tensors
        y_tr,  y_te    : (N,) label tensors
        scaler          : fitted StandardScaler (keep for inference)
        raw             : dict of unscaled numpy arrays for EDA
    """
    root = data_root or DATA_ROOT

    X_tr_raw, y_tr_raw, subj_tr, tsx_tr, tsy_tr, tsz_tr = _load_split(root, "train")
    X_te_raw, y_te_raw, subj_te, tsx_te, tsy_te, tsz_te = _load_split(root, "test")

    # Sanity checks
    overlap = set(subj_tr) & set(subj_te)
    assert len(overlap) == 0, f"Subject overlap detected: {overlap}"

    # Scale features — fit on train only, transform both
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw)
    X_te_sc = scaler.transform(X_te_raw)

    # Stack raw signals → (N, 3, 128)
    Xt_tr = np.stack([tsx_tr, tsy_tr, tsz_tr], axis=1)
    Xt_te = np.stack([tsx_te, tsy_te, tsz_te], axis=1)

    if verbose:
        print(f"UCI HAR loaded (subject-based 70/30 split)")
        print(f"  Train : {X_tr_sc.shape[0]:>5,} samples | {len(np.unique(subj_tr))} subjects")
        print(f"  Test  : {X_te_sc.shape[0]:>5,} samples | {len(np.unique(subj_te))} subjects")
        print(f"  Features : {X_tr_sc.shape[1]} | TS shape: {Xt_tr.shape[1:]}")
        print(f"  Subject overlap: {overlap} ✓")

    return {
        "Xf_tr" : _to_tensor_f32(X_tr_sc),
        "Xf_te" : _to_tensor_f32(X_te_sc),
        "Xt_tr" : _to_tensor_f32(Xt_tr),
        "Xt_te" : _to_tensor_f32(Xt_te),
        "y_tr"  : _to_tensor_i64(y_tr_raw),
        "y_te"  : _to_tensor_i64(y_te_raw),
        "scaler": scaler,
        "raw": {
            "X_train": X_tr_raw, "y_train": y_tr_raw, "subj_train": subj_tr,
            "X_test" : X_te_raw, "y_test" : y_te_raw, "subj_test" : subj_te,
            "ts_x_train": tsx_tr, "ts_y_train": tsy_tr, "ts_z_train": tsz_tr,
            "ts_x_test" : tsx_te, "ts_y_test" : tsy_te, "ts_z_test" : tsz_te,
        },
    }


def make_dataloaders(data: dict, batch_size: int = BATCH_SIZE):
    """
    Build four DataLoaders from the dict returned by load_uci_har().

    Returns
    -------
    dl_feat_tr, dl_feat_te  — feature-only (MLP family)
    dl_mm_tr,   dl_mm_te    — multi-modal  (FusionNet / LSTM variants)
    """
    Xf_tr, Xf_te = data["Xf_tr"], data["Xf_te"]
    Xt_tr, Xt_te = data["Xt_tr"], data["Xt_te"]
    y_tr,  y_te  = data["y_tr"],  data["y_te"]

    dl_feat_tr = DataLoader(
        TensorDataset(Xf_tr, y_tr),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    dl_feat_te = DataLoader(
        TensorDataset(Xf_te, y_te),
        batch_size=batch_size, shuffle=False,
    )
    dl_mm_tr = DataLoader(
        TensorDataset(Xf_tr, Xt_tr, y_tr),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    dl_mm_te = DataLoader(
        TensorDataset(Xf_te, Xt_te, y_te),
        batch_size=batch_size, shuffle=False,
    )

    return dl_feat_tr, dl_feat_te, dl_mm_tr, dl_mm_te
