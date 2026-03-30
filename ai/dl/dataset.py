from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Sequence builder
# ──────────────────────────────────────────────────────────────────────────────

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping fixed-length windows from a time-ordered feature array.

    For each valid starting position ``i`` the window is
    ``X[i : i + seq_len]`` and the label is ``y[i + seq_len - 1]``
    (the label aligned with the **last** timestep in the window).

    This means the output has ``len(X) - seq_len + 1`` samples — the first
    ``seq_len - 1`` rows of X act as warm-up and do not appear as standalone
    predictions.

    Args:
        X       : Feature array of shape ``(n_timesteps, n_features)``.
        y       : Label array of shape ``(n_timesteps,)``.
        seq_len : Look-back window length.

    Returns:
        Tuple ``(X_seq, y_seq)`` where:
            - ``X_seq`` has shape ``(n_samples, seq_len, n_features)``
            - ``y_seq`` has shape ``(n_samples,)``

    Raises:
        ValueError : If ``seq_len > len(X)``.
    """
    n = len(X)
    if seq_len > n:
        raise ValueError(
            f"seq_len={seq_len} is larger than the number of available "
            f"timesteps={n}. Reduce seq_len or provide more data."
        )

    n_samples   = n - seq_len + 1
    n_features  = X.shape[1] if X.ndim > 1 else 1

    X_seq = np.empty((n_samples, seq_len, n_features), dtype=np.float32)
    y_seq = np.empty(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len - 1]

    return X_seq, y_seq


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Minimal PyTorch Dataset wrapping pre-built sequence arrays.

    Args:
        X_seq : Feature sequences, shape ``(n_samples, seq_len, n_features)``.
        y_seq : Labels / targets, shape ``(n_samples,)``.
    """

    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray) -> None:
        self.X = torch.from_numpy(X_seq.astype(np.float32))
        self.y = torch.from_numpy(y_seq.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]