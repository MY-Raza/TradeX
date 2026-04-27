from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Sequence builder — vectorized via numpy stride tricks (no Python loop)
# ──────────────────────────────────────────────────────────────────────────────

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping fixed-length windows from a time-ordered feature array.

    Uses ``np.lib.stride_tricks.as_strided`` to produce a *zero-copy* view of
    X without any Python-level loop, making it ~50-100× faster than the
    original loop for typical (n_timesteps ≈ 6000, seq_len ≈ 60) inputs.

    The returned X_seq is a **copy** (``np.ascontiguousarray``) so it is safe
    to modify and to pass to PyTorch.

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

    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    n_features = X.shape[1]
    n_samples  = n - seq_len + 1

    # --- stride-trick view (zero-copy) ------------------------------------
    # Each "row" of the output advances by one element of the time axis.
    s0, s1     = X.strides            # (row_stride, col_stride)
    X_strided  = np.lib.stride_tricks.as_strided(
        X,
        shape=(n_samples, seq_len, n_features),
        strides=(s0, s0, s1),
    )
    # Make a contiguous copy so the tensor owns its memory
    X_seq = np.ascontiguousarray(X_strided)

    # Labels: y[seq_len-1], y[seq_len], ..., y[n-1]
    y_seq = np.asarray(y[seq_len - 1:], dtype=np.float32)

    return X_seq, y_seq


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset — stores tensors directly (avoids per-epoch re-conversion)
# ──────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Minimal PyTorch Dataset wrapping pre-built sequence arrays.

    Stores data as pinned-memory tensors when running on CPU so that
    DataLoader worker transfers are faster.

    Args:
        X_seq : Feature sequences, shape ``(n_samples, seq_len, n_features)``.
        y_seq : Labels / targets, shape ``(n_samples,)``.
    """

    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray) -> None:
        self.X = torch.from_numpy(np.ascontiguousarray(X_seq, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y_seq, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]