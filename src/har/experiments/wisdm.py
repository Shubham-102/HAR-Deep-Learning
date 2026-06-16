"""
experiments/wisdm.py
---------------------
Real-World Generalisation Study — WISDM dataset.

Tests how models trained on UCI HAR (controlled lab, waist-mounted)
transfer to WISDM (free-living, pocket-mounted) — a realistic domain gap.

Three conditions:
  Zero-shot   — UCI HAR trained model, WISDM test, no fine-tuning
  10% FT      — fine-tuned on 10% of WISDM labelled data
  30% FT      — fine-tuned on 30%
  100% FT     — fine-tuned on all WISDM labelled data
"""

import urllib.request
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import stats as sp_stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from configs.config import SAVE_DIR, SEED, DEVICE


# ── Activity mapping: WISDM → UCI HAR class indices ──────────────────────────
ACT_MAP = {
    "Walking":    0,
    "Upstairs":   1,
    "Downstairs": 2,
    "Sitting":    3,
    "Standing":   4,
    "Jogging":    1,   # mapped to Walk Up (closest dynamic class)
}

WINDOW_SIZE = 50
STEP_SIZE   = 25


# ── Data loading ──────────────────────────────────────────────────────────────

def download_wisdm(save_dir=None):
    """Download and extract the WISDM AR v1.1 dataset."""
    save_dir = save_dir or SAVE_DIR
    url      = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
    tar_path = save_dir / "WISDM.tar.gz"
    print("Downloading WISDM dataset…")
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(save_dir)
    print("WISDM downloaded and extracted.")


def load_wisdm(save_dir=None):
    """Parse WISDM raw CSV into a cleaned DataFrame."""
    save_dir    = save_dir or SAVE_DIR
    wisdm_path  = save_dir / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
    cols        = ["user_id", "activity", "timestamp", "x_accel", "y_accel", "z_accel"]

    df = pd.read_csv(wisdm_path, header=None, names=cols,
                     lineterminator=";", on_bad_lines="skip")
    df["z_accel"] = df["z_accel"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    df = df[df["z_accel"] != ""].copy()
    for c in ["x_accel", "y_accel", "z_accel"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    print(f"WISDM: {len(df):,} rows | {df['user_id'].nunique()} users")
    print(df["activity"].value_counts().to_string())
    return df


def _extract_features(window: np.ndarray) -> np.ndarray:
    """Extract 36 statistical features from a (W, 3) window."""
    feats = []
    for ax in range(3):
        c = window[:, ax]
        feats += [
            c.mean(), c.std(), c.min(), c.max(), np.median(c),
            np.percentile(c, 25), np.percentile(c, 75),
            np.abs(np.diff(c)).sum(), np.abs(c).mean(),
            sp_stats.skew(c), sp_stats.kurtosis(c),
        ]
    s = window[:, :3]
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        feats.append(np.corrcoef(s[:, i], s[:, j])[0, 1])
    return np.array(feats)


def build_wisdm_windows(df: pd.DataFrame):
    """Segment WISDM signals into windows and extract 36 features."""
    df_m = df[df["activity"].isin(ACT_MAP)].copy()
    df_m["label"] = df_m["activity"].map(ACT_MAP)

    X_w, y_w = [], []
    sigs = df_m[["x_accel", "y_accel", "z_accel"]].values
    labs = df_m["label"].values
    usrs = df_m["user_id"].values

    for u in df_m["user_id"].unique():
        mask = usrs == u
        us, ul = sigs[mask], labs[mask]
        i = 0
        while i + WINDOW_SIZE <= len(us):
            X_w.append(_extract_features(us[i:i + WINDOW_SIZE]))
            y_w.append(sp_stats.mode(ul[i:i + WINDOW_SIZE], keepdims=True)[0][0])
            i += STEP_SIZE

    X_w = np.array(X_w)
    y_w = np.array(y_w)
    print(f"WISDM windows: {X_w.shape}")
    return X_w, y_w


# ── Evaluation ────────────────────────────────────────────────────────────────

class _WISDMNet(nn.Module):
    """Lightweight MLP for WISDM fine-tuning (36-feature input)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(36, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 5),
        )
    def forward(self, x):
        return self.net(x)


def run_wisdm_generalisation(
    feature_models: list,
    scaler,
    Xf_te,
    best_acc_uci: float,
    save_dir=None,
):
    """
    Full WISDM generalisation experiment.

    Parameters
    ----------
    feature_models : list of (name, model)  — UCI-trained feature-only models
    scaler         : fitted UCI HAR StandardScaler
    Xf_te          : UCI HAR test feature tensor (for reference acc lookup)
    best_acc_uci   : best UCI HAR accuracy to plot as reference line
    save_dir       : Path for WISDM data download

    Returns
    -------
    dict : {zero_shot_avg, ft_10, ft_30, ft_100}
    """
    download_wisdm(save_dir)
    df       = load_wisdm(save_dir)
    X_w, y_w = build_wisdm_windows(df)

    # ── Zero-shot evaluation ──────────────────────────────────────────────────
    # Pad 36 WISDM features → 561 (zeros in positions 36:561)
    X_pad  = np.zeros((len(X_w), 561))
    X_pad[:, :36] = X_w
    Xw_pad = torch.tensor(scaler.transform(X_pad), dtype=torch.float32).to(DEVICE)

    zero = {}
    for nm, mdl in feature_models:
        mdl.to(DEVICE).eval()
        with torch.no_grad():
            preds = mdl(Xw_pad).argmax(1).cpu().numpy()
        zero[nm] = accuracy_score(y_w, preds) * 100
        print(f"{nm} zero-shot: {zero[nm]:.2f}%")
    avg_zero = np.mean(list(zero.values()))
    print(f"Average zero-shot: {avg_zero:.2f}%  ← domain gap")

    # ── Fine-tuning at 3 data budgets ─────────────────────────────────────────
    X_tr_w, X_te_w, y_tr_w, y_te_w = train_test_split(
        X_w, y_w, test_size=0.2, random_state=SEED, stratify=y_w,
    )
    sc_w   = StandardScaler()
    Xwtr   = torch.tensor(sc_w.fit_transform(X_tr_w), dtype=torch.float32)
    Xwte   = torch.tensor(sc_w.transform(X_te_w),     dtype=torch.float32)
    ywtr   = torch.tensor(y_tr_w, dtype=torch.long)
    ywte   = torch.tensor(y_te_w, dtype=torch.long)

    results_ft = {}
    for pct in [0.10, 0.30, 1.00]:
        n   = max(50, int(len(Xwtr) * pct))
        idx = torch.randperm(len(Xwtr))[:n]
        ltr = DataLoader(TensorDataset(Xwtr[idx], ywtr[idx]),
                         batch_size=64, shuffle=True, drop_last=True)
        lte = DataLoader(TensorDataset(Xwte, ywte), batch_size=64)

        mdl = _WISDMNet().to(DEVICE)
        opt = optim.AdamW(mdl.parameters(), lr=1e-3)
        for _ in range(30):
            mdl.train()
            for xb, yb in ltr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                F.cross_entropy(mdl(xb), yb).backward()
                opt.step()

        mdl.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in lte:
                out      = mdl(xb.to(DEVICE))
                correct += (out.argmax(1).cpu() == yb).sum()
                total   += yb.size(0)
        acc = 100 * correct / total
        results_ft[int(pct * 100)] = float(acc)
        print(f"  {int(pct*100):>3}% data ({n:>5,} samples) → {acc:.2f}%")

    _plot_wisdm(avg_zero, results_ft, best_acc_uci)
    return {"zero_shot": avg_zero, **{f"ft_{k}": v for k, v in results_ft.items()}}


def _plot_wisdm(avg_zero: float, results_ft: dict, best_acc_uci: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Real-World Generalisation — WISDM", fontsize=12, fontweight="bold")

    conds    = ["UCI HAR\n(Lab)", "WISDM\nZero-shot",
                "WISDM\n10% FT", "WISDM\n30% FT", "WISDM\n100% FT"]
    accs_b   = [best_acc_uci, avg_zero,
                results_ft[10], results_ft[30], results_ft[100]]
    bcols    = ["#2196F3", "#F44336", "#FF9800", "#8BC34A", "#4CAF50"]

    bars = ax1.bar(conds, accs_b, color=bcols, edgecolor="white", linewidth=1.5, width=0.55)
    ax1.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Domain Gap: Lab vs Real-World", fontweight="bold")
    ax1.axhline(90, color="#E91E63", ls="--", lw=1.5, label="90% deployment threshold")
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    pts = [0, 10, 30, 100]
    la  = [avg_zero, results_ft[10], results_ft[30], results_ft[100]]
    ax2.plot(pts, la, "-", color="#1565C0", lw=2.5)
    for x, y, c in zip(pts, la, ["#F44336", "#FF9800", "#8BC34A", "#4CAF50"]):
        ax2.scatter(x, y, s=120, color=c, zorder=5, edgecolors="white", linewidth=1.5)
    ax2.axhline(best_acc_uci, color="#2196F3", ls="--", lw=1.5,
                label=f"UCI HAR best ({best_acc_uci:.1f}%)")
    ax2.axhline(90, color="#E91E63", ls=":", lw=1.5, label="90% threshold")
    ax2.set_xlabel("% WISDM Data Used")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Recovery with Fine-tuning", fontweight="bold")
    ax2.set_xlim(-5, 108)
    ax2.set_ylim(0, 108)
    ax2.set_xticks([0, 10, 30, 100])
    ax2.set_xticklabels(["0%\n(Zero-shot)", "10%", "30%", "100%"])
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Zero-shot: {avg_zero:.1f}%  |  10% FT: {results_ft[10]:.1f}%  "
          f"|  100% FT: {results_ft[100]:.1f}%")
