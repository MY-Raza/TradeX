"""
train_loop.py
=============
Core training / evaluation loop primitives shared by all DL models.

Exports
-------
- ``run_epoch``      : One pass over a DataLoader (train or eval mode).
- ``EarlyStopping``  : Callback that signals when to halt training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from TradeX.utils.common.logs import get_logger

logger = get_logger("train_loop")


# ──────────────────────────────────────────────────────────────────────────────
# Epoch runner
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimiser: Optional[Optimizer],
    device: torch.device,
    training: bool,
    model_type: str = "regressor",
) -> float:
    """
    Run a single training or evaluation epoch over ``dataloader``.

    Args:
        model      : PyTorch ``nn.Module`` to train / evaluate.
        dataloader : DataLoader yielding ``(X_batch, y_batch)`` tuples.
        criterion  : Loss function (``MSELoss`` for regression,
                     ``CrossEntropyLoss`` for classification).
        optimiser  : Adam (or other) optimiser; pass ``None`` for eval mode.
        device     : ``torch.device`` to move tensors to.
        training   : If ``True`` run forward + backward + step; else eval only.
        model_type : ``'classifier'`` or ``'regressor'`` — controls label dtype
                     passed to the criterion.

    Returns:
        Mean loss over all batches in the epoch.
    """
    model.train(training)
    total_loss  = 0.0
    n_batches   = 0

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)

            if model_type == "classifier":
                # CrossEntropyLoss expects Long labels
                # Labels in dataset are float; map { -1→0, 0→1, 1→2 }
                y_long  = _remap_labels(y_batch).to(device)
                out     = model(X_batch)          # (B, num_classes)
                loss    = criterion(out, y_long)
            else:
                y_batch = y_batch.to(device)
                out     = model(X_batch).squeeze(-1)   # (B,)
                loss    = criterion(out, y_batch)

            if training:
                optimiser.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def _remap_labels(y: torch.Tensor) -> torch.LongTensor:
    """
    Map float classification labels {-1, 0, 1} → long integers {0, 1, 2}.

    This is required because ``nn.CrossEntropyLoss`` expects class indices
    starting at 0, but the pipeline uses ``-1 / 0 / 1`` for short/neutral/long.

    Args:
        y : Float tensor of labels, values in ``{-1.0, 0.0, 1.0}``.

    Returns:
        Long tensor with values in ``{0, 1, 2}``.
    """
    return (y + 1).long()


# ──────────────────────────────────────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Early-stopping monitor.

    Tracks the monitored metric (lower is better — loss).  After
    ``patience`` epochs without improvement the ``__call__`` method returns
    ``True``, signalling that training should stop.

    Args:
        patience  : Number of epochs to wait after last improvement.
        min_delta : Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-6) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score: float          = float("inf")
        self.counter:    int            = 0
        self._stopped:   bool           = False

    @property
    def stopped(self) -> bool:
        """``True`` if early stopping has been triggered."""
        return self._stopped

    def __call__(self, metric: float) -> bool:
        """
        Update internal state and return whether training should stop.

        Args:
            metric : Current epoch's monitored loss value.

        Returns:
            ``True``  → stop training now.
            ``False`` → continue training.
        """
        if metric < self.best_score - self.min_delta:
            self.best_score = metric
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._stopped = True
                return True
        return False

    def reset(self) -> None:
        """Reset state — useful when re-using the same instance."""
        self.best_score = float("inf")
        self.counter    = 0
        self._stopped   = False