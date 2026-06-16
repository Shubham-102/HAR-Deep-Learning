"""
explainability/shap_analysis.py
--------------------------------
SHAP Gradient Explainer analysis for the DeepMLP model.

Three levels of analysis:
  1. Global feature importance  — top-N features by mean |SHAP| across all classes
  2. Per-sample contributions   — stability check across test samples
  3. Per-class top features     — which features drive each activity prediction
"""

import numpy as np
import matplotlib.pyplot as plt

from configs.config import (
    CLASS_NAMES, COLORS6,
    SHAP_BG_SAMPLES, SHAP_EVAL_SAMPLES, SHAP_TOP_N,
)


def run_shap_analysis(model, Xf_tr, Xf_te, save_dir=None):
    """
    Run full SHAP analysis on a feature-only model (e.g. DeepMLP).

    Parameters
    ----------
    model    : trained nn.Module (CPU, eval mode)
    Xf_tr    : (N, 561) training feature tensor  — used as SHAP background
    Xf_te    : (N, 561) test feature tensor       — samples to explain
    save_dir : Path or None — if set, saves figures to this directory

    Returns
    -------
    shap_vals : list of 6 arrays, each (n_samples, 561)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install SHAP: pip install shap")

    model.cpu().eval()

    print(f"Computing SHAP values  background={SHAP_BG_SAMPLES}  eval={SHAP_EVAL_SAMPLES}")
    explainer     = shap.GradientExplainer(model, Xf_tr[:SHAP_BG_SAMPLES])
    shap_vals_raw = explainer.shap_values(Xf_te[:SHAP_EVAL_SAMPLES])

    # Normalise shape: ensure (n_classes, n_samples, n_features)
    shap_arr = np.array(shap_vals_raw)
    if shap_arr.shape[-1] == 6:
        shap_arr = shap_arr.transpose(2, 0, 1)
    shap_vals = [shap_arr[i] for i in range(6)]
    print(f"Per-class SHAP shape: {shap_vals[0].shape}")

    # ── 1. Global top-N ───────────────────────────────────────────────────────
    shap_global = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
    top_n       = np.argsort(shap_global)[::-1][:SHAP_TOP_N]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        [f"Feature {i}" for i in top_n[::-1]],
        shap_global[top_n[::-1]],
        color="#3498db", edgecolor="white", height=0.65,
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8.5)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(
        f"Top {SHAP_TOP_N} Global Features — Mean |SHAP| (DeepMLP)",
        fontweight="bold", fontsize=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "shap_global.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── 2. Per-class top-10 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ci in range(6):
        imp = np.abs(shap_vals[ci]).mean(axis=0)
        top = np.argsort(imp)[::-1][:10]
        axes[ci].barh(
            [f"Feature {i}" for i in top[::-1]],
            imp[top[::-1]],
            color=COLORS6[ci], edgecolor="white", height=0.6,
        )
        axes[ci].set_title(CLASS_NAMES[ci], fontweight="bold")
        axes[ci].set_xlabel("Mean |SHAP|", fontsize=9)
        axes[ci].spines[["top", "right"]].set_visible(False)
        axes[ci].grid(axis="x", alpha=0.3)

    fig.suptitle("Top 10 SHAP Features per Activity — DeepMLP",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "shap_per_class.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nTop 5 global features: {top_n[:5].tolist()}")
    print("SHAP analysis complete.")
    return shap_vals
