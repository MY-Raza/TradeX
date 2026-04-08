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


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

class DirectionalLoss(nn.Module):
    """
    Directional loss for regression models.

    Encourages predictions to have the correct sign (direction) relative to
    the target return, and rewards larger magnitude predictions when correct.

    Loss = -mean(sign(targets) * preds) + λ * mean(preds²)

    The L2 magnitude penalty (λ, default 0.01) prevents the degenerate
    constant-negative local minimum that pure directional loss falls into
    when targets are imbalanced.  Without it, the optimiser discovers that
    predicting a single large-magnitude constant in the majority-class
    direction achieves a low loss and never escapes.  This was the root cause
    of the constant-negative collapse observed in TFT, GRU, and LSTM, and the
    magnitude explosion (float32 overflow) observed in TCN.

    Args:
        magnitude_penalty : L2 coefficient λ.  Set to 0.0 to recover the
                            original pure directional loss (not recommended).
    """

    def __init__(self, magnitude_penalty: float = 0.01) -> None:
        super().__init__()
        self.lam = magnitude_penalty

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        directional   = -torch.mean(torch.sign(targets) * preds)
        magnitude_reg = self.lam * torch.mean(preds ** 2)
        return directional + magnitude_reg


class ConfidenceWeightedLoss(nn.Module):
    """
    Confidence-weighted directional loss for regression models.

    Extends DirectionalLoss by scaling each term by the absolute prediction
    magnitude (confidence):

        Loss = -mean(sign(targets) * preds * |preds|) + λ * mean(preds²)

    The same L2 penalty is applied to prevent constant-prediction collapse.

    Args:
        magnitude_penalty : L2 coefficient λ.
    """

    def __init__(self, magnitude_penalty: float = 0.01) -> None:
        super().__init__()
        self.lam = magnitude_penalty

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        direction     = torch.sign(targets)
        confidence    = torch.abs(preds)
        directional   = -torch.mean(direction * preds * confidence)
        magnitude_reg = self.lam * torch.mean(preds ** 2)
        return directional + magnitude_reg


class BaseDLModel(abc.ABC):
    """
    Abstract base for all PyTorch time-series forecasting models.

    Subclasses must implement:
        - ``_build_network(input_size) -> nn.Module``

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
    loss_fn : str
        Loss function selector for regressors.
        ``'directional'`` (default) or ``'confidence_weighted'``.
        Classifiers always use ``CrossEntropyLoss`` regardless of this setting.
    magnitude_penalty : float
        L2 coefficient λ applied inside DirectionalLoss / ConfidenceWeightedLoss
        to prevent constant-prediction collapse.  Default 0.01.
    max_grad_norm : float
        Gradient clipping norm applied after every backward pass.
        Prevents the float32 magnitude explosion observed in TCN with
        weight_norm layers, and slows collapse speed in TFT/GRU/LSTM.
        Default 1.0.  Set to 0.0 to disable (not recommended).
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
        loss_fn: str = "directional",
        magnitude_penalty: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.seq_len           = seq_len
        self.horizon           = horizon
        self.hidden_size       = hidden_size
        self.num_layers        = num_layers
        self.dropout           = dropout
        self.batch_size        = batch_size
        self.epochs            = epochs
        self.lr                = lr
        self.patience          = patience
        self.model_type        = model_type
        self.loss_fn           = loss_fn
        self.magnitude_penalty = magnitude_penalty
        self.max_grad_norm     = max_grad_norm

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"[{self.__class__.__name__}] Using device: {self.device}")

        # Set after fit()
        self.network_: Optional[nn.Module] = None
        self.feature_names_: list[str]     = []
        self.input_size_: int              = 0

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
    # Loss factory
    # ------------------------------------------------------------------

    def _build_criterion(self, y_seq_tr: np.ndarray) -> nn.Module:
        """
        Construct the appropriate loss function.

        Classifiers: ``CrossEntropyLoss`` with inverse-frequency class weights.
        Regressors:  ``DirectionalLoss`` or ``ConfidenceWeightedLoss``, both
                     with L2 magnitude penalty to prevent constant collapse.

        Also logs the target sign balance to surface imbalance early — a
        positive fraction below 40% or above 60% is a warning that constant-
        prediction collapse risk is high even with the penalty active.

        Args:
            y_seq_tr : Training label array used for class-weight computation
                       and sign-balance logging.

        Returns:
            Configured ``nn.Module`` loss criterion.
        """
        if self.model_type == "classifier":
            remapped = (torch.from_numpy(y_seq_tr) + 1).long()
            counts   = torch.bincount(remapped, minlength=3).float().clamp(min=1.0)
            weights  = (1.0 / counts)
            weights  = (weights / weights.sum() * 3).to(self.device)
            logger.info(
                f"[{self.__class__.__name__}] Class weights: "
                f"short={weights[0]:.3f} neutral={weights[1]:.3f} long={weights[2]:.3f}"
            )
            return nn.CrossEntropyLoss(weight=weights)

        # Log sign balance — surfaces imbalanced target distributions early
        pos_frac = float((y_seq_tr > 0).mean())
        logger.info(
            f"[{self.__class__.__name__}] Target sign balance — "
            f"{pos_frac:.1%} positive / {1 - pos_frac:.1%} negative"
        )
        if pos_frac < 0.40 or pos_frac > 0.60:
            logger.warning(
                f"[{self.__class__.__name__}] Target is imbalanced "
                f"({pos_frac:.1%} positive). Constant-prediction collapse risk "
                f"is elevated. magnitude_penalty={self.magnitude_penalty} is active."
            )

        if self.loss_fn == "confidence_weighted":
            logger.info(
                f"[{self.__class__.__name__}] Using ConfidenceWeightedLoss "
                f"(λ={self.magnitude_penalty})."
            )
            return ConfidenceWeightedLoss(magnitude_penalty=self.magnitude_penalty)

        logger.info(
            f"[{self.__class__.__name__}] Using DirectionalLoss "
            f"(λ={self.magnitude_penalty})."
        )
        return DirectionalLoss(magnitude_penalty=self.magnitude_penalty)

    # ------------------------------------------------------------------
    # Public interface
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

        Gradient clipping (``max_grad_norm``) is applied after every backward
        pass to prevent the magnitude explosion seen in TCN and the rapid
        collapse seen in TFT/GRU/LSTM.

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
        train_dl = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=(self.device.type == "cuda"),
        )

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
        criterion = self._build_criterion(y_seq_tr)
        optimiser = Adam(self.network_.parameters(), lr=self.lr)
        stopper   = EarlyStopping(patience=self.patience)

        # ── Training loop ────────────────────────────────────────────
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            train_loss = run_epoch(
                self.network_, train_dl, criterion, optimiser,
                self.device, training=True, model_type=self.model_type,
                max_grad_norm=self.max_grad_norm,   # FIX: gradient clipping
            )
            val_loss = None
            if val_dl is not None:
                val_loss = run_epoch(
                    self.network_, val_dl, criterion, None,
                    self.device, training=False, model_type=self.model_type,
                    max_grad_norm=self.max_grad_norm,
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

        The first ``seq_len - 1`` rows are consumed as the warm-up window.
        Returns ``len(X) - seq_len + 1`` elements.

        For classifiers: integer signals in ``{-1, 0, 1}`` via argmax.
        For regressors:  raw continuous values from the linear head.
        """
        if self.network_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_np    = X.to_numpy(dtype=np.float32)
        dummy_y = np.zeros(len(X_np), dtype=np.float32)
        X_seq, _ = build_sequences(X_np, dummy_y, self.seq_len)

        ds = TimeSeriesDataset(X_seq, np.zeros(len(X_seq), dtype=np.float32))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.network_.eval()
        all_preds: list[np.ndarray] = []

        with torch.no_grad():
            for X_batch, _ in dl:
                X_batch = X_batch.to(self.device)
                out     = self.network_(X_batch)

                if self.model_type == "classifier":
                    pred_cls = torch.argmax(out, dim=-1)
                    preds_np = (pred_cls - 1).cpu().numpy().astype(np.float32)
                else:
                    preds_np = out.squeeze(-1).cpu().numpy()

                all_preds.append(preds_np)

        result = np.concatenate(all_preds, axis=0)
        logger.info(
            f"[{self.__class__.__name__}] predict() → "
            f"min={result.min():.4f}  max={result.max():.4f}  mean={result.mean():.4f}"
        )
        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class probability estimates (classifiers) or raw scalar outputs
        (regressors).

        For classifiers: softmax over logits → shape ``(N, 3)``,
                         columns = [P(short), P(neutral), P(long)].
        For regressors:  identical to ``predict()`` → shape ``(N,)``.
        """
        if self.network_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_np    = X.to_numpy(dtype=np.float32)
        dummy_y = np.zeros(len(X_np), dtype=np.float32)
        X_seq, _ = build_sequences(X_np, dummy_y, self.seq_len)

        ds = TimeSeriesDataset(X_seq, np.zeros(len(X_seq), dtype=np.float32))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.network_.eval()
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for X_batch, _ in dl:
                X_batch = X_batch.to(self.device)
                out     = self.network_(X_batch)

                if self.model_type == "classifier":
                    probs = torch.softmax(out, dim=-1)
                    all_probs.append(probs.cpu().numpy())
                else:
                    all_probs.append(out.squeeze(-1).cpu().numpy())

        result = np.concatenate(all_probs, axis=0)
        logger.info(
            f"[{self.__class__.__name__}] predict_proba() → shape={result.shape}"
        )
        return result

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """
        Compute evaluation metrics on the test set.

        For regressors: MSE, MAE, and directional accuracy.
        For classifiers: accuracy.
        """
        preds     = self.predict(X_test)
        y_aligned = y_test.to_numpy(dtype=np.float32)[-len(preds):]

        if self.model_type == "classifier":
            accuracy = float(np.mean(preds == y_aligned.astype(np.float32)))
            logger.info(f"[{self.__class__.__name__}] Test accuracy: {accuracy:.4f}")
            return {"accuracy": accuracy}

        mse     = float(np.mean((preds - y_aligned) ** 2))
        mae     = float(np.mean(np.abs(preds - y_aligned)))
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y_aligned)))
        logger.info(
            f"[{self.__class__.__name__}] MSE={mse:.6f}  MAE={mae:.6f}  "
            f"DirectionalAcc={dir_acc:.4f}"
        )
        return {"mse": mse, "mae": mae, "directional_accuracy": dir_acc}