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
from TradeX.utils.common.logs import get_logger

logger = get_logger("base_dl_model")


# ─────────────────────────────────────────────────────────────
# ✅ LOSSES (unchanged)
# ─────────────────────────────────────────────────────────────

class DirectionalLoss(nn.Module):
    def forward(self, preds, targets):
        return -torch.mean(torch.sign(targets) * preds)


class ConfidenceWeightedLoss(nn.Module):
    def forward(self, preds, targets):
        return -torch.mean(torch.sign(targets) * preds * torch.abs(preds))


# ─────────────────────────────────────────────────────────────
# BASE MODEL
# ─────────────────────────────────────────────────────────────

class BaseDLModel(abc.ABC):

    def __init__(
        self,
        seq_len: int = 60,
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

        # ✅ NEW
        temperature: float = 1.5,
        confidence_threshold: float = 0.4,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.model_type = model_type
        self.loss_fn = loss_fn

        # ✅ NEW
        self.temperature = temperature
        self.conf_threshold = confidence_threshold

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.network_ = None
        self.feature_names_ = []
        self.input_size_ = 0

        logger.info(f"[{self.__class__.__name__}] Using device: {self.device}")

    # ─────────────────────────────────────────────

    @abc.abstractmethod
    def _build_network(self, input_size: int) -> nn.Module:
        pass

    # ─────────────────────────────────────────────

    def _build_criterion(self, y_seq_tr):
        if self.model_type == "classifier":
            remapped = (torch.from_numpy(y_seq_tr) + 1).long()
            counts = torch.bincount(remapped, minlength=3).float().clamp(min=1)
            weights = (1.0 / counts)
            weights = (weights / weights.sum() * 3).to(self.device)

            return nn.CrossEntropyLoss(weight=weights)

        return (
            ConfidenceWeightedLoss()
            if self.loss_fn == "confidence_weighted"
            else DirectionalLoss()
        )

    # ─────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        self.feature_names_ = X_train.columns.tolist()
        self.input_size_ = len(self.feature_names_)

        X_tr = X_train.to_numpy(np.float32)
        y_tr = y_train.to_numpy(np.float32)

        X_seq, y_seq = build_sequences(X_tr, y_tr, self.seq_len)

        train_dl = DataLoader(
            TimeSeriesDataset(X_seq, y_seq),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_dl = None
        if X_val is not None:
            X_v = X_val.to_numpy(np.float32)
            y_v = y_val.to_numpy(np.float32)

            X_seq_v, y_seq_v = build_sequences(X_v, y_v, self.seq_len)

            val_dl = DataLoader(
                TimeSeriesDataset(X_seq_v, y_seq_v),
                batch_size=self.batch_size,
                shuffle=False,
            )

        self.network_ = self._build_network(self.input_size_).to(self.device)

        criterion = self._build_criterion(y_seq)
        optimizer = Adam(self.network_.parameters(), lr=self.lr)

        stopper = EarlyStopping(self.patience)

        for epoch in range(self.epochs):

            train_loss = run_epoch(
                self.network_, train_dl, criterion, optimizer,
                self.device, True, self.model_type
            )

            val_loss = None
            if val_dl:
                val_loss = run_epoch(
                    self.network_, val_dl, criterion, None,
                    self.device, False, self.model_type
                )

            monitor = val_loss if val_loss else train_loss

            if stopper(monitor):
                break

        return self

    # ─────────────────────────────────────────────
    # ✅ FIXED PREDICT (adds threshold)
    # ─────────────────────────────────────────────

    def predict(self, X):

        probs = self.predict_proba(X)

        if self.model_type == "classifier":

            preds = np.argmax(probs, axis=1)

            # ✅ confidence filter
            confidence = np.max(probs, axis=1)
            preds = np.where(confidence < self.conf_threshold, 1, preds)

            return (preds - 1).astype(np.float32)

        return probs

    # ─────────────────────────────────────────────
    # ✅ FIXED PROBA (main fix here)
    # ─────────────────────────────────────────────

    def predict_proba(self, X):

        X_np = X.to_numpy(np.float32)
        dummy = np.zeros(len(X_np), dtype=np.float32)

        X_seq, _ = build_sequences(X_np, dummy, self.seq_len)

        dl = DataLoader(
            TimeSeriesDataset(X_seq, dummy[:len(X_seq)]),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.network_.eval()
        outputs = []

        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(self.device)
                logits = self.network_(xb)

                if self.model_type == "classifier":

                    # ✅ KEY FIX 1: normalize logits
                    logits = logits - logits.mean(dim=1, keepdim=True)

                    # ✅ KEY FIX 2: temperature scaling
                    logits = logits / self.temperature

                    probs = torch.softmax(logits, dim=-1)

                    outputs.append(probs.cpu().numpy())
                else:
                    outputs.append(logits.squeeze(-1).cpu().numpy())

        result = np.concatenate(outputs, axis=0)

        logger.info(f"[predict_proba] shape={result.shape}")
        return result