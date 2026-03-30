from __future__ import annotations

import abc
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from TradeX.ai.dl.dataset import TimeSeriesDataset, build_sequences
from TradeX.ai.dl.models.train_loop import run_epoch, EarlyStopping
from TradeX.ai.dl.models.train_utils import (
    validate_and_sort,
    apply_log_diff_transform,
    split_features_labels,
    run_chunked_backtest,
)
from TradeX.utils.common.logs import get_logger

logger = get_logger("base_dl_model")


class BaseDLModel(abc.ABC):
    """
    Abstract base for all PyTorch time-series forecasting models.

    Subclasses must implement:
        - ``_build_network(input_size, **kwargs) -> nn.Module``
        - ``_predict_raw(X_tensor) -> np.ndarray``

    Parameters
    ----------
    seq_len : int
        Look-back window length fed into the network.
    horizon : int
        Number of steps ahead to forecast (currently 1).
    hidden_size : int
        Hidden / channel dimension for the network.
    num_layers : int
        Stacking depth of recurrent layers or TCN blocks.
    dropout : float
        Dropout probability applied inside the network.
    batch_size : int
        Mini-batch size for DataLoader.
    epochs : int
        Maximum training epochs.
    lr : float
        Adam learning rate.
    patience : int
        Early-stopping patience (epochs without validation improvement).
    device : str or None
        ``'cpu'``, ``'cuda'``, or ``None`` (auto-detect).
    model_type : str
        ``'classifier'`` or ``'regressor'``.
    """

    def __init__(
        self,
        seq_len: int = 60,
        horizon: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        batch_size: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
        device: Optional[str] = None,
        model_type: str = "regressor",
    ) -> None:
        self.seq_len     = seq_len
        self.horizon     = horizon
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.lr          = lr
        self.patience    = patience
        self.model_type  = model_type

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"[{self.__class__.__name__}] Using device: {self.device}")

        # Set after fit()
        self.network_: Optional[nn.Module]  = None
        self.feature_names_: list[str]      = []
        self.input_size_: int               = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_network(self, input_size: int) -> nn.Module:
        """
        Construct and return the ``nn.Module`` for this architecture.

        Args:
            input_size: Number of input features per time step.

        Returns:
            Uninitialised ``nn.Module``.
        """

    # ------------------------------------------------------------------
    # Public interface — mirrors train() in sklearn-style model files
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series]    = None,
    ) -> "BaseDLModel":
        """
        Build sequences, construct the network, and run the training loop.

        Args:
            X_train : Training feature matrix (rows = timesteps).
            y_train : Training labels / targets aligned with X_train.
            X_val   : Optional validation feature matrix.
            y_val   : Optional validation labels.

        Returns:
            self (for chaining).
        """
        self.feature_names_ = X_train.columns.tolist()
        self.input_size_    = len(self.feature_names_)

        # ── Build sequences ──────────────────────────────────────────
        X_tr_np = X_train.to_numpy(dtype=np.float32)
        y_tr_np = y_train.to_numpy(dtype=np.float32)

        X_seq_tr, y_seq_tr = build_sequences(X_tr_np, y_tr_np, self.seq_len)

        train_ds = TimeSeriesDataset(X_seq_tr, y_seq_tr)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(self.device.type == "cuda"))

        val_dl = None
        if X_val is not None and y_val is not None:
            X_v_np  = X_val.to_numpy(dtype=np.float32)
            y_v_np  = y_val.to_numpy(dtype=np.float32)
            X_seq_v, y_seq_v = build_sequences(X_v_np, y_v_np, self.seq_len)
            val_ds  = TimeSeriesDataset(X_seq_v, y_seq_v)
            val_dl  = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                 num_workers=0)

        # ── Network ──────────────────────────────────────────────────
        self.network_ = self._build_network(self.input_size_).to(self.device)
        logger.info(
            f"[{self.__class__.__name__}] Network params: "
            f"{sum(p.numel() for p in self.network_.parameters()):,}"
        )

        # ── Loss & optimiser ─────────────────────────────────────────
        if self.model_type == "classifier":
            # Multi-class cross-entropy; labels must be long integers
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimiser = Adam(self.network_.parameters(), lr=self.lr)
        stopper   = EarlyStopping(patience=self.patience)

        # ── Training loop ────────────────────────────────────────────
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            train_loss = run_epoch(
                self.network_, train_dl, criterion, optimiser,
                self.device, training=True, model_type=self.model_type,
            )
            val_loss = None
            if val_dl is not None:
                val_loss = run_epoch(
                    self.network_, val_dl, criterion, None,
                    self.device, training=False, model_type=self.model_type,
                )

            monitor = val_loss if val_loss is not None else train_loss
            if epoch % 10 == 0 or epoch == 1:
                msg = (
                    f"[{self.__class__.__name__}] Epoch {epoch}/{self.epochs} "
                    f"train_loss={train_loss:.6f}"
                )
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.6f}"
                logger.info(msg)

            if stopper(monitor):
                logger.info(
                    f"[{self.__class__.__name__}] Early stop at epoch {epoch}. "
                    f"Best val_loss={stopper.best_score:.6f}"
                )
                break

        elapsed = time.time() - t0
        logger.info(f"[{self.__class__.__name__}] Training finished in {elapsed:.1f}s")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for the given feature matrix.

        The first ``seq_len - 1`` rows are consumed as the warm-up window and
        produce no output; the returned array therefore has
        ``len(X) - seq_len + 1`` elements.

        Args:
            X : Feature matrix aligned with the test set.

        Returns:
            1-D NumPy array of predictions (probabilities for classifiers,
            continuous values for regressors).
        """
        if self.network_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_np   = X.to_numpy(dtype=np.float32)
        # Build sequences without labels — pass dummy y
        dummy_y = np.zeros(len(X_np), dtype=np.float32)
        X_seq, _ = build_sequences(X_np, dummy_y, self.seq_len)

        ds = TimeSeriesDataset(X_seq, np.zeros(len(X_seq), dtype=np.float32))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.network_.eval()
        all_preds: list[np.ndarray] = []

        with torch.no_grad():
            for X_batch, _ in dl:
                X_batch = X_batch.to(self.device)
                out     = self.network_(X_batch)  # (B, output_size)

                if self.model_type == "classifier":
                    # Return softmax probability of the positive class (index 1)
                    probs = torch.softmax(out, dim=-1)
                    # Handle both binary (2 classes) and multi-class
                    if probs.shape[-1] >= 2:
                        preds_np = probs[:, 1].cpu().numpy()
                    else:
                        preds_np = probs[:, 0].cpu().numpy()
                else:
                    preds_np = out.squeeze(-1).cpu().numpy()

                all_preds.append(preds_np)

        result = np.concatenate(all_preds, axis=0)
        logger.info(
            f"[{self.__class__.__name__}] predict() → "
            f"min={result.min():.4f}  max={result.max():.4f}  mean={result.mean():.4f}"
        )
        return result

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """
        Compute evaluation metrics on the test set.

        For regressors: MSE and MAE.
        For classifiers: accuracy.

        Args:
            X_test : Test feature matrix.
            y_test : Ground-truth labels / targets.

        Returns:
            Dict of metric name → value.
        """
        preds = self.predict(X_test)
        # Align lengths (seq_len warm-up removes some leading rows)
        y_aligned = y_test.to_numpy(dtype=np.float32)[-len(preds):]

        if self.model_type == "classifier":
            pred_classes = (preds > 0.5).astype(int)
            accuracy = float(np.mean(pred_classes == y_aligned.astype(int)))
            logger.info(f"[{self.__class__.__name__}] Test accuracy: {accuracy:.4f}")
            return {"accuracy": accuracy}
        else:
            mse = float(np.mean((preds - y_aligned) ** 2))
            mae = float(np.mean(np.abs(preds - y_aligned)))
            logger.info(f"[{self.__class__.__name__}] MSE={mse:.6f}  MAE={mae:.6f}")
            return {"mse": mse, "mae": mae}