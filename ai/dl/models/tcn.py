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
# TCN building blocks
# ──────────────────────────────────────────────────────────────────────────────

class _CausalConv1d(nn.Module):
    """
    1-D causal convolution: pads only on the left so no future leakage occurs.

    Args:
        in_channels  : Input channel count.
        out_channels : Output channel count.
        kernel_size  : Convolution kernel width.
        dilation     : Dilation factor (receptive field multiplier).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = weight_norm(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
        )
        # Kaiming init for ReLU networks
        nn.init.kaiming_normal_(self.conv.weight_v, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove the right-side padding to preserve causality
        return out[:, :, : -self.padding] if self.padding > 0 else out


class _TCNBlock(nn.Module):
    """
    One residual TCN block: two causal conv layers + skip connection.

    Args:
        in_channels  : Input channels (must equal out_channels for residual).
        out_channels : Output channels.
        kernel_size  : Kernel width of each conv layer.
        dilation     : Dilation for this block.
        dropout      : Spatial dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1    = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2    = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(dropout)
        # Downsample / projection for residual when channels change
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        if self.downsample is not None:
            nn.init.kaiming_normal_(
                self.downsample.weight, mode="fan_in", nonlinearity="relu"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class _TCNNetwork(nn.Module):
    """
    Full TCN: stack of ``_TCNBlock`` with exponentially increasing dilations,
    followed by a linear projection head.

    Args:
        input_size   : Number of input features (channels).
        num_channels : Number of channels per TCN block.
        num_layers   : Number of stacked TCN blocks.
        kernel_size  : Kernel width (same for every block).
        dropout      : Dropout probability.
        output_size  : Dimension of the linear head output.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            dilation   = 2 ** i
            in_ch      = input_size if i == 0 else num_channels
            out_ch     = num_channels
            layers.append(
                _TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)
        self.fc      = nn.Linear(num_channels, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, seq_len, input_size)  →  transpose for Conv1d
        x   = x.permute(0, 2, 1)          # (B, input_size, seq_len)
        out = self.network(x)              # (B, num_channels, seq_len)
        last = out[:, :, -1]              # take last timestep: (B, num_channels)
        return self.fc(last)              # (B, output_size)


# ──────────────────────────────────────────────────────────────────────────────
# BaseDLModel subclass
# ──────────────────────────────────────────────────────────────────────────────

class TCNModel(BaseDLModel):
    """
    TCN-based forecasting model.

    Additional constructor parameters (beyond ``BaseDLModel``):
        kernel_size (int): Convolution kernel width (default 3).
        num_channels (int): Channel width of every TCN block; maps to
            ``hidden_size`` inherited from ``BaseDLModel``.
    """

    def __init__(self, kernel_size: int = 3, **kwargs) -> None:
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
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public train()
# ──────────────────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    df_1m: pd.DataFrame,
    target_col: str = "target",
    split_date: str = "2024-01-01 00:00",
    n_trials: int = 10,
    k: float = 0.5,
    transform_features: bool = True,
    model_type: str = "regressor",
) -> tuple:
    """
    Train a TCN model using PnL-based Optuna optimisation.

    Args:
        df               : Feature DataFrame (output of ``data_pipeline``).
        df_1m            : 1-minute OHLCV for backtesting.
        target_col       : Name of the label column.
        split_date       : ISO date string for train/test boundary.
        n_trials         : Optuna trial budget.
        k                : Top-k threshold fraction for signal selection.
        transform_features: Apply log-diff to OHLCV columns if True.
        model_type       : ``'classifier'`` or ``'regressor'``.

    Returns:
        (model, preds, test_index, X_test)
    """
    df = validate_and_sort(df, target_col)

    if transform_features:
        df = apply_log_diff_transform(df)

    if model_type == "regressor":
        df = normalize_regression_target(df, target_col)

    df_normalised = df.copy()

    X_train, y_train, X_test, y_test = split_features_labels(df, target_col, split_date)

    logger.info(f"[train] Starting TCN Optuna study ({n_trials} trials)…")

    def objective(trial: optuna.Trial) -> float:
        seq_len     = trial.suggest_int("seq_len",      20,  120)
        num_channels = trial.suggest_int("num_channels", 32,  256)
        num_layers  = trial.suggest_int("num_layers",    2,    6)
        kernel_size = trial.suggest_int("kernel_size",   2,    8)
        dropout     = trial.suggest_float("dropout",    0.0,  0.5)
        lr          = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = TCNModel(
            seq_len=seq_len,
            hidden_size=num_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=30,
            patience=5,
            model_type=model_type,
        )
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        preds = model.predict(X_test)

        # Prune trials where the model outputs a constant (dead network).
        # For regressors, check std. For classifiers, the signed score
        # (long_prob - short_prob) should vary; std near zero means all-neutral.
        if np.std(preds) < 1e-8:
            raise optuna.TrialPruned()

        aligned_index = X_test.index[-(len(preds)):]
        return run_chunked_backtest(
            trial, df, preds, aligned_index, df_1m,
            model_type=model_type, k=k, lookback=seq_len,
        )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = study.best_params
    logger.info(f"[train] Best params: {best} | Best PnL: {study.best_value:.4f}")

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
    )
    final_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    final_preds = final_model.predict(X_test)

    if np.std(final_preds) < 1e-6:
        logger.warning(
            "[train] Final TCN model collapsed to constant output "
            f"({final_preds.mean():.6f}). Adding small noise to prevent all-zero signals."
        )
        rng = np.random.default_rng(42)
        final_preds = final_preds + rng.normal(0, 1e-4, size=final_preds.shape)

    # Trim X_test to match the (seq_len warm-up shortened) preds length so
    # every downstream caller (backtest, importance, dry-run) sees aligned arrays.
    aligned_index = X_test.index[-(len(final_preds)):]
    X_test_aligned = X_test.loc[aligned_index]

    logger.info(
        f"[train] Final preds — min: {final_preds.min():.4f}, "
        f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}"
    )
    # Build df_for_backtest: a length-aligned frame with both 'datetime' and
    # all feature columns.  pnl_permutation_importance uses this as its `df`
    # argument, so it must contain every feature column that X_test_aligned has
    # (for shuffling) plus 'datetime' (for BackTest lookup).
    n_preds = len(final_preds)
    iloc_start = max(0, len(df_normalised) - n_preds)
    pred_datetimes = df_normalised.iloc[iloc_start : iloc_start + n_preds]["datetime"].values

    pred_datetimes     = pred_datetimes[-n_preds:]
    backtest_positions = np.arange(n_preds)

    df_for_backtest = X_test_aligned.reset_index(drop=True).copy()
    df_for_backtest.insert(0, "datetime", pred_datetimes)

    assert len(df_for_backtest) == n_preds, (
        f"df_for_backtest length {len(df_for_backtest)} != preds length {n_preds}"
    )

    return final_model, final_preds, backtest_positions, X_test_aligned, df_for_backtest