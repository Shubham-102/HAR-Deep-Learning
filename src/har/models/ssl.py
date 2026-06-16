"""
models/ssl.py
-------------
Self-Supervised Learning (Modern Technique 1) — Masked Signal Reconstruction.

Pipeline:
    Phase 1 — Pre-train SSLPretrainModel on unlabelled raw signals:
               mask 30% of timesteps → encode → decode → minimise MSE
    Phase 2 — Fine-tune SSLClassifier using the frozen/unfrozen encoder
               on labelled data with a lightweight classification head.
"""

import torch
import torch.nn as nn


class SSLEncoder(nn.Module):
    """
    CNN encoder shared between pre-training and fine-tuning.

    Input  : (N, 3, 128) raw tri-axial signal
    Output : (N, embed_dim) embedding
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(3,   64,  7, padding=3), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.proj = nn.Linear(128 * 8, embed_dim)

    def forward(self, x):
        return self.proj(self.enc(x).view(x.size(0), -1))


class SSLPretrainModel(nn.Module):
    """
    Masked auto-encoder for self-supervised pre-training.

    Forward pass:
        1. Randomly mask `mask_ratio` fraction of timesteps (set to 0)
        2. Encode masked signal → embedding
        3. Decode embedding → reconstructed (3, 128) signal
        4. Loss: MSE between reconstruction and original (unmasked)
    """
    def __init__(self, embed_dim: int = 128, mask_ratio: float = 0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder    = SSLEncoder(embed_dim)
        self.decoder    = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 3 * 128),
        )

    def forward(self, x):
        B, C, T = x.shape
        mask = torch.rand(B, T, device=x.device) < self.mask_ratio
        xm   = x.clone()
        xm[:, :, mask[0]] = 0.0          # apply same mask across channels
        recon = self.decoder(self.encoder(xm)).view(B, C, T)
        return recon, x                  # (reconstruction, original)


class SSLClassifier(nn.Module):
    """
    Classification head placed on top of a pre-trained (or random) SSLEncoder.

    Usage:
        # After SSL pre-training:
        clf = SSLClassifier(ssl_model.encoder)
        # Fine-tune clf on labelled data.
    """
    def __init__(self, encoder: SSLEncoder, embed_dim: int = 128, n_classes: int = 6):
        super().__init__()
        self.encoder = encoder
        self.clf = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, feat, ts):          # feat is ignored — SSL uses raw signal only
        return self.clf(self.encoder(ts))
