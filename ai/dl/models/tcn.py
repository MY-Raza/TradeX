from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from TradeX.ai.dl.models.base_model import BaseDLModel
from TradeX.ai.dl.models.train_utils import (
    validate_and_sort,
    apply_log_diff_transform,
    normalize_regression_target,
    split_features_labels,
    run_chunked_backtest,
)
from TradeX.utils.common.logs import get_logger

logger = get_logger("tcn")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# 🔥 NEW — Signal Generation (FIXES ZERO SIGNAL ISSUE)
# ──────────────────────────────────────────────────────────────────────────────

def generate_signals_from_probs(probs: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Convert probabilities → trading signals using margin vs neutral.

    probs: (N,3) → [short, neutral, long]
    """

    short_p   = probs[:, 0]
    neutral_p = probs[:, 1]
    long_p    = probs[:, 2]

    long_score  = long_p  - neutral_p
    short_score = short_p - neutral_p

    signals = np.zeros(len(probs), dtype=np.float32)

    signals[long_score > threshold] = 1
    signals[short_score > threshold] = -1

    return signals


# ──────────────────────────────────────────────────────────────────────────────
# TCN building blocks
# ──────────────────────────────────────────────────────────────────────────────

class _CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
        )
        nn.init.kaiming_normal_(self.conv.weight_v, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, : -self.padding] if self.padding > 0 else out


class _TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.dropout(out)
        out = self.activation(self.conv2(out))
        out = self.dropout(out)

        residual = x if self.downsample is None else self.downsample(x)
        return out + residual


class _TCNNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        num_channels,
        num_layers,
        kernel_size,
        dropout,
        output_size,
        model_type="regressor",
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels
            layers.append(_TCNBlock(in_ch, num_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        last = out[:, :, -1]
        logits = self.fc(last)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Model Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TCNModel(BaseDLModel):
    def __init__(self, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def _build_network(self, input_size: int) -> nn.Module:
        output_size = 3 if self.model_type == "classifier" else 1
        return _TCNNetwork(
            input_size=input_size,
            num_channels=self.hidden_size,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            output_size=output_size,
            model_type=self.model_type,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    df_1m: pd.DataFrame,
    target_col: str = "target",
    split_date: str = "2024-01-01 00:00",
    n_trials: int = 10,
    k: float = 0.35,   # 🔥 FIXED
    transform_features: bool = True,
    model_type: str = "regressor",
    loss_fn: str = "directional",
) -> tuple:

    df = validate_and_sort(df, target_col)

    if transform_features:
        df = apply_log_diff_transform(df)

    if model_type == "regressor":
        df = normalize_regression_target(df, target_col)

    df_normalised = df.copy()

    X_train, y_train, X_test, y_test = split_features_labels(df, target_col, split_date)

    logger.info(f"[train] Label distribution:\n{y_train.value_counts()}")

    def objective(trial):

        model = TCNModel(
            seq_len=trial.suggest_int("seq_len", 20, 120),
            hidden_size=trial.suggest_int("num_channels", 32, 256),
            num_layers=trial.suggest_int("num_layers", 2, 6),
            kernel_size=trial.suggest_int("kernel_size", 2, 8),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            epochs=30,
            patience=5,
            model_type=model_type,
            loss_fn=loss_fn,
        )

        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        if model_type == "classifier":
            probs = model.predict_proba(X_test)
            preds = generate_signals_from_probs(probs)
        else:
            preds = model.predict(X_test)

        if np.std(preds) < 1e-8:
            raise optuna.TrialPruned()

        aligned_index = X_test.index[-len(preds):]

        return run_chunked_backtest(
            trial,
            df,
            preds,
            aligned_index,
            df_1m,
            model_type=model_type,
            k=k,
            lookback=model.seq_len,
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params

    final_model = TCNModel(
        seq_len=best["seq_len"],
        hidden_size=best["num_channels"],
        num_layers=best["num_layers"],
        kernel_size=best["kernel_size"],
        dropout=best["dropout"],
        lr=best["lr"],
        batch_size=best["batch_size"],
        epochs=50,
        patience=10,
        model_type=model_type,
        loss_fn=loss_fn,
    )

    final_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    if model_type == "classifier":
        probs = final_model.predict_proba(X_test)
        final_preds = generate_signals_from_probs(probs)
    else:
        final_preds = final_model.predict(X_test)

    logger.info(
        f"[train] Final preds — min={final_preds.min():.4f}, "
        f"max={final_preds.max():.4f}, mean={final_preds.mean():.4f}"
    )

    aligned_index = X_test.index[-len(final_preds):]
    X_test_aligned = X_test.loc[aligned_index]
    df_test_norm = df.loc[aligned_index].reset_index(drop=True)

    return final_model, final_preds, aligned_index, X_test_aligned, df_test_norm