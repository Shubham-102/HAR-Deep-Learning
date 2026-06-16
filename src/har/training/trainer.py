"""
training/trainer.py
-------------------
Two training functions shared across all experiments:
  - train_model : standard supervised training (all 7 models + SSL fine-tune)
  - train_kd    : knowledge distillation training (TinyHAR student)

Design choices:
  - AdamW optimizer with weight decay
  - CosineAnnealingLR scheduler
  - Gradient clipping (max norm 1.0)
  - Label smoothing (ε=0.05) for better calibration
  - Early stopping (patience configurable)
  - Best-model checkpoint saved in memory (no disk I/O during training)
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from configs.config import DEVICE, PATIENCE


# ── Helpers ────────────────────────────────────────────────────────────────────

def _move(batch, device, multimodal: bool):
    """Unpack a DataLoader batch and move tensors to device."""
    if multimodal:
        xf, xt, y = batch
        return xf.to(device), xt.to(device), y.to(device)
    else:
        xf, y = batch
        return xf.to(device), None, y.to(device)


def _forward(model, xf, xt, multimodal: bool):
    """Call model with the correct signature."""
    return model(xf, xt) if multimodal else model(xf)


# ── Public API ─────────────────────────────────────────────────────────────────

def train_model(
    model,
    loader,
    epochs: int      = 50,
    lr: float        = 1e-3,
    multimodal: bool = False,
    name: str        = "Model",
    patience: int    = PATIENCE,
    device           = None,
    verbose: bool    = True,
):
    """
    Standard supervised training loop.

    Parameters
    ----------
    model       : nn.Module  — any HAR model
    loader      : DataLoader — feature-only or multi-modal train loader
    epochs      : int        — maximum training epochs
    lr          : float      — initial learning rate
    multimodal  : bool       — True for FusionNet / LSTM variants
    name        : str        — model name printed in logs
    patience    : int        — early stopping patience (epochs without improvement)
    device      : torch.device or None (auto-detect)
    verbose     : bool       — print epoch logs

    Returns
    -------
    history : dict with keys 'train_loss', 'train_acc'
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    history   = {"train_loss": [], "train_acc": []}
    best_loss = float("inf")
    no_imp    = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        tr_loss = correct = total = 0

        for batch in loader:
            xf, xt, y = _move(batch, device, multimodal)
            out  = _forward(model, xf, xt, multimodal)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)

        scheduler.step()
        avg = tr_loss / len(loader)
        acc = 100 * correct / total
        history["train_loss"].append(avg)
        history["train_acc"].append(acc)

        # Early stopping
        if avg < best_loss - 1e-4:
            best_loss  = avg
            no_imp     = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  [{name:>18}] ep{epoch+1:>3}/{epochs}  "
                  f"loss={avg:.4f}  acc={acc:.2f}%")

        if no_imp >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def train_kd(
    teacher,
    student,
    loader,
    epochs:  int   = 40,
    lr:      float = 1e-3,
    T:       float = 4.0,
    alpha:   float = 0.7,
    patience: int  = PATIENCE,
    device         = None,
    verbose: bool  = True,
):
    """
    Knowledge distillation training loop.

    Loss = α · KL(student ‖ teacher; T) + (1−α) · CrossEntropy(student, hard labels)

    Parameters
    ----------
    teacher  : pre-trained FusionNet (frozen during KD)
    student  : TinyHAR (trained from scratch)
    loader   : feature-only DataLoader (student operates on features only)
    T        : temperature for soft targets (higher → softer distribution)
    alpha    : weight for KL divergence vs hard-label cross-entropy
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.eval().to(device)
    student.to(device)

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history    = {"train_loss": [], "train_acc": []}
    best_loss  = float("inf")
    no_imp     = 0
    best_state = None

    # Dummy time-series tensor for teacher forward pass (teacher needs xt)
    _dummy_ts_cache = {}

    def _kd_loss(s, t, y):
        kl   = F.kl_div(
            F.log_softmax(s / T, dim=1),
            F.softmax(t / T, dim=1),
            reduction="batchmean",
        ) * (T ** 2)
        return alpha * kl + (1 - alpha) * F.cross_entropy(s, y)

    for epoch in range(epochs):
        student.train()
        tr_loss = correct = total = 0

        for xf, y in loader:
            xf, y = xf.to(device), y.to(device)
            B = xf.size(0)

            # Create/cache dummy xt for teacher (zeros, same device)
            if B not in _dummy_ts_cache:
                _dummy_ts_cache[B] = torch.zeros(B, 3, 128, device=device)
            dummy_xt = _dummy_ts_cache[B]

            with torch.no_grad():
                t_logits = teacher(xf, dummy_xt)

            s_logits = student(xf)
            loss     = _kd_loss(s_logits, t_logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            correct += (s_logits.argmax(1) == y).sum().item()
            total   += y.size(0)

        scheduler.step()
        avg = tr_loss / len(loader)
        acc = 100 * correct / total
        history["train_loss"].append(avg)
        history["train_acc"].append(acc)

        if avg < best_loss - 1e-4:
            best_loss  = avg
            no_imp     = 0
            best_state = {k: v.clone() for k, v in student.state_dict().items()}
        else:
            no_imp += 1

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"  [TinyHAR-KD] ep{epoch+1:>3}/{epochs}  "
                  f"loss={avg:.4f}  acc={acc:.2f}%")

        if no_imp >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        student.load_state_dict(best_state)

    return history
