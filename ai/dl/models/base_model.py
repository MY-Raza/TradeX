from __future__ import annotations

import abc
import multiprocessing
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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

_NUM_WORKERS = min(4, max(1, multiprocessing.cpu_count() // 2))


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

class DirectionalLoss(nn.Module):
    """
    Directional loss for regression models.

    Loss = -mean(sign(targets) * preds)

    CRITICAL DESIGN REQUIREMENT: The network forward() MUST apply tanh() to
    its scalar output before this loss sees it. Without tanh bounding the
    output to (-1, 1), the gradient is constant w.r.t. magnitude, so the
    optimizer can lower this loss indefinitely by growing weights in whichever
    sign wins the majority of training steps. The result is a collapsed model
    that predicts one large constant (as observed: preds ~ -11819 for all
    inputs). With tanh, the gradient vanishes as |pred| → 1, which is the
    correct saturation behaviour for a directional signal.
    """

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.sign(targets) * preds)


class ConfidenceWeightedLoss(nn.Module):
    """
    Confidence-weighted directional loss.

    Loss = -mean(sign(targets) * preds * |preds|)

    Correct high-confidence predictions (large |pred|, right sign) contribute
    a large negative term; wrong high-confidence predictions contribute a large
    positive term. Requires tanh-bounded output for the same reason as
    DirectionalLoss.
    """

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return -torch.mean(torch.sign(targets) * preds * torch.abs(preds))


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseDLModel(abc.ABC):
    """
    Abstract base for all PyTorch time-series forecasting models.

    Subclasses must implement ``_build_network(input_size) -> nn.Module``.

    CONTRACT FOR REGRESSOR SUBCLASSES:
        The network's forward() method MUST apply ``torch.tanh`` to its scalar
        output.  This bounds predictions to (-1, 1), which is required for
        DirectionalLoss / ConfidenceWeightedLoss to be stable. Without tanh
        the loss gradient has constant magnitude with respect to pred, so the
        model learns one dominant sign and makes weights grow unboundedly in
        that direction — producing the symptom where all predictions are a
        large negative constant.

    Parameters
    ----------
    seq_len : int
        Look-back window length.
    hidden_size : int
        Hidden / channel dimension.
    num_layers : int
        Stacking depth.
    dropout : float
        Dropout probability.
    batch_size : int
        Mini-batch size.
    epochs : int
        Maximum training epochs.
    lr : float
        AdamW learning rate.
    patience : int
        Early-stopping patience.
    device : str or None
        ``'cpu'``, ``'cuda'``, or ``None`` (auto-detect).
    model_type : str
        ``'classifier'`` or ``'regressor'``.
    loss_fn : str
        ``'directional'`` (default) or ``'confidence_weighted'`` for regressors.
        Classifiers always use CrossEntropyLoss.
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
        self.loss_fn     = loss_fn

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"[{self.__class__.__name__}] Using device: {self.device}")

        self.network_: Optional[nn.Module] = None
        self.feature_names_: list[str]     = []
        self.input_size_: int              = 0

    @abc.abstractmethod
    def _build_network(self, input_size: int) -> nn.Module:
        """Return the nn.Module. Regressor forward() must end with tanh()."""

    def _build_criterion(self, y_seq_tr: np.ndarray) -> nn.Module:
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
        else:
            if self.loss_fn == "confidence_weighted":
                logger.info(f"[{self.__class__.__name__}] Using ConfidenceWeightedLoss.")
                return ConfidenceWeightedLoss()
            else:
                logger.info(f"[{self.__class__.__name__}] Using DirectionalLoss.")
                return DirectionalLoss()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series]    = None,
    ) -> "BaseDLModel":
        self.feature_names_ = X_train.columns.tolist()
        self.input_size_    = len(self.feature_names_)

        X_tr_np  = X_train.to_numpy(dtype=np.float32)
        y_tr_np  = y_train.to_numpy(dtype=np.float32)
        X_seq_tr, y_seq_tr = build_sequences(X_tr_np, y_tr_np, self.seq_len)

        _pw      = _NUM_WORKERS > 0
        train_ds = TimeSeriesDataset(X_seq_tr, y_seq_tr)
        train_dl = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=_NUM_WORKERS, pin_memory=False,
            persistent_workers=_pw, prefetch_factor=2 if _pw else None,
        )

        val_dl = None
        if X_val is not None and y_val is not None:
            X_v_np  = X_val.to_numpy(dtype=np.float32)
            y_v_np  = y_val.to_numpy(dtype=np.float32)
            X_seq_v, y_seq_v = build_sequences(X_v_np, y_v_np, self.seq_len)
            val_ds  = TimeSeriesDataset(X_seq_v, y_seq_v)
            val_dl  = DataLoader(
                val_ds, batch_size=self.batch_size * 2, shuffle=False,
                num_workers=_NUM_WORKERS, pin_memory=False,
                persistent_workers=_pw, prefetch_factor=2 if _pw else None,
            )

        self.network_ = self._build_network(self.input_size_).to(self.device)
        logger.info(
            f"[{self.__class__.__name__}] Network params: "
            f"{sum(p.numel() for p in self.network_.parameters()):,}"
        )

        criterion = self._build_criterion(y_seq_tr)
        optimiser = AdamW(self.network_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimiser, T_max=self.epochs, eta_min=self.lr * 0.01)
        stopper   = EarlyStopping(patience=self.patience)

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

            scheduler.step()

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
        Generate predictions.

        Regressors: tanh-bounded floats in (-1, 1).
            Positive → bullish signal (long).
            Negative → bearish signal (short).
            Magnitude → confidence.
        Classifiers: integers in {-1, 0, 1}.
        """
        if self.network_ is None:
            raise RuntimeError("Call fit() before predict().")

        X_np     = X.to_numpy(dtype=np.float32)
        dummy_y  = np.zeros(len(X_np), dtype=np.float32)
        X_seq, _ = build_sequences(X_np, dummy_y, self.seq_len)

        ds = TimeSeriesDataset(X_seq, np.zeros(len(X_seq), dtype=np.float32))
        dl = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False, num_workers=0)

        self.network_.eval()
        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch, _ in dl:
                out = self.network_(X_batch.to(self.device))
                if self.model_type == "classifier":
                    preds_np = (torch.argmax(out, dim=-1) - 1).cpu().numpy().astype(np.float32)
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
        if self.network_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        X_np     = X.to_numpy(dtype=np.float32)
        dummy_y  = np.zeros(len(X_np), dtype=np.float32)
        X_seq, _ = build_sequences(X_np, dummy_y, self.seq_len)

        ds = TimeSeriesDataset(X_seq, np.zeros(len(X_seq), dtype=np.float32))
        dl = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False, num_workers=0)

        self.network_.eval()
        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch, _ in dl:
                out = self.network_(X_batch.to(self.device))
                if self.model_type == "classifier":
                    all_probs.append(torch.softmax(out, dim=-1).cpu().numpy())
                else:
                    all_probs.append(out.squeeze(-1).cpu().numpy())

        result = np.concatenate(all_probs, axis=0)
        logger.info(f"[{self.__class__.__name__}] predict_proba() → shape={result.shape}")
        return result

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        preds     = self.predict(X_test)
        y_aligned = y_test.to_numpy(dtype=np.float32)[-len(preds):]
        if self.model_type == "classifier":
            acc = float(np.mean(preds == y_aligned.astype(np.float32)))
            logger.info(f"[{self.__class__.__name__}] Test accuracy: {acc:.4f}")
            return {"accuracy": acc}
        else:
            mse     = float(np.mean((preds - y_aligned) ** 2))
            mae     = float(np.mean(np.abs(preds - y_aligned)))
            dir_acc = float(np.mean(np.sign(preds) == np.sign(y_aligned)))
            logger.info(
                f"[{self.__class__.__name__}] MSE={mse:.6f}  MAE={mae:.6f}  "
                f"DirectionalAcc={dir_acc:.4f}"
            )
            return {"mse": mse, "mae": mae, "directional_accuracy": dir_acc}