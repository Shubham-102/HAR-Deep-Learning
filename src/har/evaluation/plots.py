"""
evaluation/plots.py
-------------------
Reusable plotting functions for training curves, confusion matrices,
ROC curves, and accuracy comparison bar charts.

All functions save figures to results/ when a save_path is provided,
and always call plt.show() for interactive use.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from configs.config import CLASS_NAMES, COLORS6


def plot_history(history: dict, title: str, save_path=None):
    """Plot training loss and accuracy curves from a history dict."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["train_loss"]) + 1)

    a1.plot(ep, history["train_loss"], lw=2, color="#e74c3c")
    a1.set_title(f"{title} — Loss")
    a1.set_xlabel("Epoch")
    a1.grid(alpha=0.3)
    a1.spines[["top", "right"]].set_visible(False)

    a2.plot(ep, history["train_acc"], lw=2, color="#2ecc71")
    a2.fill_between(ep, history["train_acc"], alpha=0.15, color="#2ecc71")
    a2.set_title(f"{title} — Train Accuracy")
    a2.set_xlabel("Epoch")
    a2.grid(alpha=0.3)
    a2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cm(yt, yp, title: str, save_path=None):
    """Plot a normalised confusion matrix heatmap."""
    cm = confusion_matrix(yt, yp, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_accuracy_bar(names: list, accs: list, save_path=None):
    """
    Horizontal bar chart comparing all model accuracies.
    Color-coded by tier: feature-only / multi-modal / advanced.
    """
    bar_cols = ["#aed6f1", "#7fb3d3", "#5499c7", "#1a6fa1", "#0d3b6e", "#27ae60", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, accs, color=bar_cols[:len(names)],
                  edgecolor="white", linewidth=1.5, width=0.6)
    ax.bar_label(bars, fmt="%.2f%%", padding=5, fontsize=10, fontweight="bold")
    ax.set_ylim(min(accs) - 3, 101)
    ax.set_title(
        "All Models — Test Accuracy (Subject-Based 70/30 Split)\n"
        "Evaluated on 9 unseen subjects — honest generalisation",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Test Accuracy (%)")
    ax.axhline(max(accs), color="gold", ls="--", lw=1.5,
               label=f"Best: {max(accs):.2f}%")
    ax.tick_params(axis="x", rotation=15)
    patches = [
        mpatches.Patch(color="#7fb3d3", label="Feature-only"),
        mpatches.Patch(color="#1a6fa1", label="Multi-modal"),
        mpatches.Patch(color="#27ae60", label="Advanced"),
    ]
    ax.legend(handles=patches + [plt.Line2D([0], [0], color="gold", ls="--", lw=1.5,
              label=f"Best: {max(accs):.2f}%")], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(model_evals: list, y_true_bin, save_path=None):
    """
    Plot ROC curves for multiple models in a grid.

    Parameters
    ----------
    model_evals : list of (name, probs_array)
    y_true_bin  : binarized ground-truth labels (N, n_classes)
    """
    n = len(model_evals)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
    axes = axes.flatten()
    pal  = plt.cm.tab10(np.linspace(0, 1, 6))

    for ax, (name, probs) in zip(axes[:n], model_evals):
        mauc = []
        for i in range(6):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
            ra = auc(fpr, tpr)
            mauc.append(ra)
            ax.plot(fpr, tpr, lw=1.5, color=pal[i],
                    label=f"{CLASS_NAMES[i]} ({ra:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{name}\nmAUC={np.mean(mauc):.3f}", fontweight="bold", fontsize=10)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "ROC Curves — All Models | One-vs-Rest | 9 Unseen Subjects",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_robustness(rob: dict, save_path=None):
    """Line chart of accuracy under increasing degradation for all models."""
    conds = list(next(iter(rob.values())).keys())
    pal   = plt.cm.tab10(np.linspace(0, 1, len(rob)))
    stys  = [("-", "o"), ("--", "s"), ("-.", "v"), (":", "D"),
             ("-", "P"), ("--", "*"), ("-.", "X")]

    fig, ax = plt.subplots(figsize=(11, 5))
    for (name, res), (ls, mk), col in zip(rob.items(), stys, pal):
        ax.plot(conds, list(res.values()), linestyle=ls, marker=mk,
                label=name, lw=2, color=col, markersize=7)

    ax.set_title("Robustness Under Sensor Degradation", fontsize=13, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("\nDrop (Baseline → Severe):")
    rows = [(n, r["Baseline"] - r["Severe"], r["Baseline"]) for n, r in rob.items()]
    for name, drop, base in sorted(rows, key=lambda x: x[1]):
        print(f"  {name:<18} base={base:.2f}%  drop={drop:+.1f}pp")
