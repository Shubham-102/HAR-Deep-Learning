"""
experiments/fall_detection.py
------------------------------
Case Study: Elderly Fall-Risk Detection.

Maps the 6 HAR activity classes to a binary fall-risk label:
  High-risk  (1) — dynamic activities: Walking, Walk Up, Walk Down
  Low-risk   (0) — static activities:  Sitting, Standing, Laying

Clinical target: ≥95% sensitivity (recall on high-risk windows).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

from har.evaluation.metrics import evaluate
from configs.config import CLASS_NAMES


FALL_MAP   = {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0}
FALL_NAMES = ["Low-Risk (Static)", "High-Risk (Dynamic)"]
CLINICAL_SENSITIVITY = 95.0   # % — threshold used in literature


def _binary_eval(model, loader, multimodal: bool = False):
    """Run 6-class inference and map predictions + probabilities to binary."""
    _, _, yp_6, probs = evaluate(model, loader, multimodal)
    yp_bin    = np.array([FALL_MAP[p] for p in yp_6])
    fall_prob = probs[:, 0] + probs[:, 1] + probs[:, 2]   # P(dynamic)
    return yp_bin, fall_prob


def run_fall_detection_case_study(model_list: list, y_te_np: np.ndarray, verbose: bool = True):
    """
    Evaluate fall-risk detection for a list of models.

    Parameters
    ----------
    model_list : list of (name, model, dataloader, multimodal_bool)
    y_te_np    : ground-truth 6-class labels as numpy array

    Returns
    -------
    fall_res : dict  {model_name: {acc, sens, spec, auc, fpr, tpr}}
    """
    y_te_bin = np.array([FALL_MAP[l] for l in y_te_np])
    fall_res  = {}

    if verbose:
        print(f"{'Model':<16} {'Acc':>8} {'Sensitivity':>14} {'Specificity':>13} {'AUC':>8}")
        print("─" * 62)

    for name, model, loader, mm in model_list:
        yp_b, fp    = _binary_eval(model, loader, mm)
        tn, fp_, fn, tp = confusion_matrix(y_te_bin, yp_b).ravel()
        sens   = 100 * tp / (tp + fn)
        spec   = 100 * tn / (tn + fp_)
        acc_b  = 100 * (y_te_bin == yp_b).mean()
        fr, tr_, _ = roc_curve(y_te_bin, fp)
        ra     = auc(fr, tr_)
        fall_res[name] = dict(acc=acc_b, sens=sens, spec=spec, auc=ra, fpr=fr, tpr=tr_)

        if verbose:
            flag = " ← ✓ clinical" if sens >= CLINICAL_SENSITIVITY else ""
            print(f"{name:<16} {acc_b:>7.2f}% {sens:>13.2f}% {spec:>12.2f}% {ra:>7.3f}{flag}")

    _plot_fall_results(fall_res)
    best = max(fall_res.items(), key=lambda x: x[1]["sens"])
    print(f"\nBest sensitivity: {best[0]} ({best[1]['sens']:.1f}%)")
    return fall_res


def _plot_fall_results(fall_res: dict, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Case Study: Elderly Fall-Risk Detection",
                 fontsize=12, fontweight="bold")
    names  = list(fall_res.keys())
    pal    = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))

    # Sensitivity bar
    ax = axes[0]
    bars = ax.barh(names, [fall_res[n]["sens"] for n in names],
                   color=pal, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8.5)
    ax.axvline(CLINICAL_SENSITIVITY, color="red", ls="--", lw=1.5,
               label=f"{CLINICAL_SENSITIVITY}% clinical threshold")
    ax.set_xlabel("Sensitivity — High-risk Recall (%)")
    ax.set_title("Sensitivity", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Accuracy vs Sensitivity scatter
    ax = axes[1]
    for i, (n, r) in enumerate(fall_res.items()):
        ax.scatter(r["acc"], r["sens"], s=90, color=pal[i], zorder=5,
                   edgecolors="white", linewidth=1.2)
        ax.annotate(n, (r["acc"], r["sens"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7.5)
    ax.axhline(CLINICAL_SENSITIVITY, color="red", ls="--", lw=1, alpha=0.7,
               label=f"{CLINICAL_SENSITIVITY}% sensitivity")
    ax.axvline(95, color="blue", ls="--", lw=1, alpha=0.7, label="95% accuracy")
    ax.set_xlabel("Overall Accuracy (%)")
    ax.set_ylabel("Sensitivity (%)")
    ax.set_title("Accuracy vs Sensitivity", fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # ROC curves
    ax = axes[2]
    for i, (n, r) in enumerate(fall_res.items()):
        ax.plot(r["fpr"], r["tpr"], lw=1.5, color=pal[i],
                label=f"{n} ({r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
