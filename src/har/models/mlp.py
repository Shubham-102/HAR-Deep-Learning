"""
models/mlp.py
-------------
Tier 1 — Feature-Only MLP family (Models 1–3).

All three operate on the 561-dimensional handcrafted UCI HAR feature vector.
They accept an optional `ts` argument (ignored) so the same training loop
works for both feature-only and multi-modal models.
"""

import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    Model 1 — 2-hidden-layer MLP with no regularisation.
    Serves as the performance baseline / lower bound.
    Parameters: ~177K
    """
    def __init__(self, in_dim: int = 561, n_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, ts=None):
        return self.net(x)


class RegMLP(nn.Module):
    """
    Model 2 — SimpleMLP + Dropout (p=0.35) after each hidden layer.
    Isolates the effect of standard dropout regularisation.
    Parameters: ~177K
    """
    def __init__(self, in_dim: int = 561, n_classes: int = 6, p: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, ts=None):
        return self.net(x)


class DeepMLP(nn.Module):
    """
    Model 3 — 4-layer funnel MLP with BatchNorm + Dropout on every layer.
    Tests whether additional depth and normalisation extract more signal
    from the handcrafted features.
    Parameters: ~455K
    """
    def __init__(self, in_dim: int = 561, n_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, ts=None):
        return self.net(x)
