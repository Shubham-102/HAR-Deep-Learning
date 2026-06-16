"""
models/sequential.py
--------------------
Tier 3 — Advanced sequential architectures (Models 6–7).

Both models use multi-modal input (xf features + xt raw signals) and
are informed by the 2024 systematic review of 226 HAR studies which
consistently ranked CNN-LSTM hybrids at the top of the literature.
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """
    Model 6 — Bidirectional LSTM + feature MLP fusion.

    The bidirectional LSTM processes the raw time-series in both
    forward (past → future) and backward (future → past) directions,
    giving richer context at every timestep.

    Reference: Ordóñez & Roggen (2016), Ullah et al. (2019)
    """
    def __init__(self, input_size: int = 3, hidden: int = 128,
                 layers: int = 2, n_classes: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, layers,
            batch_first=True, dropout=0.3, bidirectional=True,
        )
        self.feat_enc = nn.Sequential(
            nn.Linear(561, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.clf = nn.Sequential(
            nn.Linear(hidden * 2 + 128, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, feat, ts):
        # ts: (N, 3, 128) → permute to (N, 128, 3) for LSTM
        out, _ = self.lstm(ts.permute(0, 2, 1))
        return self.clf(torch.cat([out[:, -1, :], self.feat_enc(feat)], dim=1))


class CNNLSTMHybrid(nn.Module):
    """
    Model 7 — CNN-LSTM Hybrid (ranked #1 in 2024 systematic review, 226 studies).

    Pipeline:
        CNN   → extracts local motion patterns from raw signal
        LSTM  → models how those patterns evolve over time
        MLP   → encodes handcrafted features in parallel
        Head  → fuses all three streams for classification

    Used in Samsung Health, Google Fit, Apple Watch internally.
    """
    def __init__(self, in_ch: int = 3, cnn_ch: int = 64,
                 lstm_h: int = 128, n_classes: int = 6, p: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2),  nn.BatchNorm1d(32),    nn.ReLU(),
            nn.Conv1d(32, cnn_ch, 3, padding=1), nn.BatchNorm1d(cnn_ch), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(p),
        )
        self.lstm = nn.LSTM(
            cnn_ch, lstm_h, 2,
            batch_first=True, dropout=p, bidirectional=True,
        )
        self.feat_enc = nn.Sequential(
            nn.Linear(561, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p),
        )
        self.clf = nn.Sequential(
            nn.Linear(lstm_h * 2 + 128, 128), nn.ReLU(),
            nn.Dropout(p), nn.Linear(128, n_classes),
        )

    def forward(self, feat, ts):
        # ts: (N, 3, 128) → CNN → (N, cnn_ch, 64) → permute → LSTM
        x = self.cnn(ts).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.clf(torch.cat([out[:, -1, :], self.feat_enc(feat)], dim=1))
