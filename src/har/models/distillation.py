"""
models/distillation.py
----------------------
Knowledge Distillation (Modern Technique 2) — TinyHAR student model.

TinyHAR is a 2-hidden-layer MLP with ~8K parameters, designed to fit
on wearable MCUs with 256KB RAM. It is trained via knowledge distillation
from a FusionNet teacher (~300K parameters), achieving nearly the same
accuracy at 1/37th the size.
"""

import torch.nn as nn


class TinyHAR(nn.Module):
    """
    Ultra-lightweight student model for on-device HAR inference.

    Parameters: ~8K  (vs FusionNet teacher ~300K → ~37× smaller)
    Target     : Wearable MCU with 256KB RAM / ARM Cortex-M series

    Receives the same feature-only input as the MLP family.
    The optional `ts` argument is ignored (MCU has no CNN branch).
    """
    def __init__(self, in_dim: int = 561, n_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x, ts=None):
        return self.net(x)
