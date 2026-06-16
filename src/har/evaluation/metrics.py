"""
evaluation/metrics.py
---------------------
Model evaluation helpers: inference, classification reports, robustness.
"""

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score

from configs.config import CLASS_NAMES, SEED
from configs.config import ROB_NOISE_SIGMA, ROB_MISSING_COLS, ROB_SEVERE_SIGMA


def evaluate(model, loader, multimodal: bool = False, device=None):
    """
    Run inference on a DataLoader and return accuracy + raw arrays.

    Returns
    -------
    acc    : float        — test accuracy (%)
    labels : np.ndarray   — true labels
    preds  : np.ndarray   — predicted labels
    probs  : np.ndarray   — softmax probabilities (N, n_classes)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    preds, labels, probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            if multimodal:
                xf, xt, y = batch
                out = model(xf.to(device), xt.to(device))
            else:
                xf, y = batch[0], batch[1]
                out = model(xf.to(device))

            probs.extend(torch.softmax(out, 1).cpu().numpy())
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    p  = np.array(preds)
    l  = np.array(labels)
    pr = np.array(probs)
    return 100 * (p == l).mean(), l, p, pr


def report(model, loader, name: str, multimodal: bool = False, device=None):
    """Print a full sklearn classification report and return (acc, yt, yp)."""
    acc, yt, yp, _ = evaluate(model, loader, multimodal, device)
    print(f"\n{'='*50}")
    print(f"  {name}  |  Test Accuracy: {acc:.2f}%")
    print(f"{'='*50}")
    print(classification_report(yt, yp, target_names=CLASS_NAMES, digits=3))
    return acc, yt, yp


def robustness_eval(model, Xf, y_np, multimodal: bool = False, Xt=None, device=None):
    """
    Evaluate model accuracy under three simulated sensor degradation conditions.

    Conditions
    ----------
    Baseline      : clean test set
    Noise σ=0.1   : additive Gaussian noise on all features
    Missing 18%   : first 100 features zeroed (sensor failure)
    Severe        : both noise + missing applied simultaneously

    Returns
    -------
    dict : {condition_name: accuracy_float}
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    torch.manual_seed(SEED)

    def _acc(xf, xt=None):
        xf = xf.to(device)
        with torch.no_grad():
            if multimodal and xt is not None:
                out = model(xf, xt.to(device))
            else:
                out = model(xf)
        return 100 * (out.argmax(1).cpu().numpy() == y_np).mean()

    noisy  = Xf + torch.randn_like(Xf) * ROB_NOISE_SIGMA
    miss   = Xf.clone()
    miss[:, :ROB_MISSING_COLS] = 0.0
    severe = miss + torch.randn_like(miss) * ROB_SEVERE_SIGMA

    return {
        "Baseline"    : _acc(Xf,    Xt),
        "Noise σ=0.1" : _acc(noisy, Xt),
        "Missing 18%" : _acc(miss,  Xt),
        "Severe"      : _acc(severe, Xt),
    }
