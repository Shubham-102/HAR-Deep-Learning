"""
models/fusion.py
----------------
Tier 2 — Multi-Modal Fusion architectures (Models 4–5).

Both models process two input streams in parallel:
  - xf : (N, 561)   — handcrafted statistical/frequency features
  - xt : (N, 3, 128) — raw tri-axial body accelerometer signal

TSEncoder is the shared CNN backbone used by both FusionNet and GatedFusionNet.
"""

import torch
import torch.nn as nn


class TSEncoder(nn.Module):
    """
    Shared 1-D CNN backbone for raw time-series input.

    Architecture:
        Conv1d(3→32, k=7) → BN → ReLU
        Conv1d(32→64, k=5) → BN → ReLU
        MaxPool1d(2)
        Conv1d(64→64, k=3) → BN → ReLU
        AdaptiveAvgPool1d(8)          ← fixed output size regardless of input length
        Linear(512→128) → ReLU → Dropout(0.3)
        Linear(128→out_dim)

    Input  : (N, 3, 128)
    Output : (N, out_dim)
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, 7, padding=3),  nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8, 128), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.head(self.conv(x).view(x.size(0), -1))


class FusionNet(nn.Module):
    """
    Model 4 — Simple concatenation fusion (★ Best overall model).

    Streams:
        feat_enc : MLP branch on 561-d feature vector → feat_dim embedding
        ts_enc   : TSEncoder on (3,128) signal        → ts_dim embedding
        clf      : Linear head on concatenated [feat ‖ ts] embedding

    Key insight: the CNN branch provides a structural fallback when
    handcrafted features are corrupted — explaining the smallest
    robustness drop (−16.4 pp) across all 7 models.

    Parameters: ~300K
    """
    def __init__(self, feat_dim: int = 128, ts_dim: int = 64, n_classes: int = 6):
        super().__init__()
        self.feat_enc = nn.Sequential(
            nn.Linear(561, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, feat_dim), nn.ReLU(),
        )
        self.ts_enc = TSEncoder(out_dim=ts_dim)
        self.clf = nn.Sequential(
            nn.Linear(feat_dim + ts_dim, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, xf, xt):
        return self.clf(torch.cat([self.feat_enc(xf), self.ts_enc(xt)], dim=1))


class GatedFusionNet(nn.Module):
    """
    Model 5 — Gated fusion with learned per-sample modality weighting.

    Gate mechanism:
        α = Sigmoid(Linear([feat ‖ ts]))   ← scalar ∈ [0,1] per sample
        fused = α · proj_f(feat) + (1−α) · proj_t(ts)

    Note: trained on clean data, the gate does NOT generalise well to
    degraded inputs (largest robustness drop: −19.9 pp).
    Fix: add degradation augmentation during training.

    Parameters: ~321K
    """
    def __init__(self, feat_dim: int = 128, ts_dim: int = 64,
                 fused_dim: int = 128, n_classes: int = 6):
        super().__init__()
        self.feat_enc = nn.Sequential(
            nn.Linear(561, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, feat_dim), nn.ReLU(),
        )
        self.ts_enc  = TSEncoder(out_dim=ts_dim)
        self.proj_f  = nn.Linear(feat_dim, fused_dim)
        self.proj_t  = nn.Linear(ts_dim,   fused_dim)
        self.gate    = nn.Sequential(
            nn.Linear(feat_dim + ts_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.clf = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, xf, xt):
        f = self.feat_enc(xf)
        t = self.ts_enc(xt)
        alpha = self.gate(torch.cat([f, t], dim=1))
        fused = alpha * self.proj_f(f) + (1 - alpha) * self.proj_t(t)
        return self.clf(fused)
