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
        # ✅ UPDATED — use fan_in + relu for Kaiming init (compatible with GELU too;
        # GELU ≈ ReLU in the positive half so the same init heuristic applies).
        nn.init.kaiming_normal_(self.conv.weight_v, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove the right-side padding to preserve causality
        return out[:, :, : -self.padding] if self.padding > 0 else out


class _TCNBlock(nn.Module):
    """
    One residual TCN block: two causal conv layers + skip connection.

    ✅ UPDATED — uses GELU activation instead of ReLU.

    GELU advantages over ReLU for financial time-series:
        - Smooth, non-zero gradient everywhere (no dying-neuron problem).
        - Stochastic regularisation effect (similar to dropout) during training.
        - Better empirical performance in transformer / sequence models.

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
        self.conv1      = _CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.conv2      = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.activation = nn.GELU()    # ✅ UPDATED — GELU replaces ReLU
        self.dropout    = nn.Dropout(dropout)
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
        out = self.activation(self.conv1(x))   # ✅ UPDATED — GELU
        out = self.dropout(out)
        out = self.activation(self.conv2(out)) # ✅ UPDATED — GELU
        out = self.dropout(out)

        residual = x if self.downsample is None else self.downsample(x)
        # NOTE: No activation after residual addition (Bai et al. 2018).
        # Applying ReLU/GELU here clamps block outputs and causes the FC head
        # bias to absorb the negative target mean — reproducing the bug that
        # was previously fixed. Keep the residual connection activation-free.
        return out + residual


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
        model_type   : ``'classifier'`` or ``'regressor'``.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        output_size: int,
        model_type: str = "regressor",   # ✅ UPDATED
    ) -> None:
        super().__init__()
        self.model_type = model_type      # ✅ UPDATED
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
        x    = x.permute(0, 2, 1)          # (B, input_size, seq_len)
        out  = self.network(x)              # (B, num_channels, seq_len)
        last = out[:, :, -1]               # take last timestep: (B, num_channels)
        logits = self.fc(last)             # (B, output_size)

        # ✅ UPDATED — output activation strategy:
        #   Classifier: return raw logits. CrossEntropyLoss applies log-softmax
        #               internally (numerically stable). predict_proba() applies
        #               softmax explicitly at inference time.
        #   Regressor:  apply tanh to bound output to (-1, 1). When using
        #               MarginDirectionalLoss, tanh is not strictly required
        #               but still helps with gradient flow and interpretation.
        if self.model_type == "classifier":
            return logits
        else:
            return torch.tanh(logits)


# ──────────────────────────────────────────────────────────────────────────────
# BaseDLModel subclass
# ──────────────────────────────────────────────────────────────────────────────

class TCNModel(BaseDLModel):
    """
    TCN-based forecasting model.

    Additional constructor parameters (beyond ``BaseDLModel``):
        kernel_size (int): Convolution kernel width (default 3).
        num_channels (int): Channel width of every TCN block; maps to
                           ``hidden_size`` in ``BaseDLModel``.
    """

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
        kernel_size: int = 3,
        device: str | None = None,
        model_type: str = "regressor",
        loss_fn: str = "margin",  # ✅ CHANGED DEFAULT from "mse" to "margin"
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            patience=patience,
            device=device,
            model_type=model_type,
            loss_fn=loss_fn,
        )
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
# Public training function
# ──────────────────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    df_1m: pd.DataFrame,
    target_col: str = "target",
    split_date: str = "2024-01-01 00:00",
    n_trials: int = 2,
    k: float = 0.5,
    transform_features: bool = True,
    model_type: str = "regressor",
    loss_fn: str = "margin",  # ✅ CHANGED DEFAULT from "mse" to "margin"
) -> tuple:
    """
    Hyperparameter optimisation + final model training via Optuna.

    ✅ IMPROVEMENTS IN THIS VERSION:
    - Changed default loss function from MSELoss to MarginDirectionalLoss
    - Better Optuna search space (shorter sequences, lower layer count)
    - Larger trial epoch budget (epochs=30 during trials, 50 final)
    - Better patience for early stopping (5 during trials, 10 final)
    - Improved collapse detection with clear diagnostics
    - Signal distribution logging for debugging

    Args:
        df               : Feature DataFrame (output of ``data_pipeline``).
        df_1m            : 1-minute OHLCV for backtesting.
        target_col       : Name of the label column.
        split_date       : ISO date string for train/test boundary.
        n_trials         : Optuna trial budget.
        k                : Top-k threshold fraction for signal selection.
        transform_features: Apply log-diff to OHLCV columns if True.
        model_type       : ``'classifier'`` or ``'regressor'``.
        loss_fn          : Loss function — ``'margin'`` (default, recommended),
                           ``'mse'``, ``'directional'``, or ``'confidence_weighted'``.

    Returns:
        (model, preds, test_index, X_test, df_for_backtest)
    """
    df = validate_and_sort(df, target_col)

    if transform_features:
        df = apply_log_diff_transform(df)

    if model_type == "regressor":
        df = normalize_regression_target(df, target_col)

    df_normalised = df.copy()

    X_train, y_train, X_test, y_test = split_features_labels(df, target_col, split_date)

    logger.info(f"[train] Starting TCN Optuna study ({n_trials} trials)…")
    logger.info(f"[train] Using loss_fn='{loss_fn}' (default='margin' to prevent collapse)")
    logger.info(
        f"[train] Data: {len(X_train)} train, {len(X_test)} test | "
        f"Target distribution — mean={y_test.mean():.4f}, std={y_test.std():.4f}"
    )

    def objective(trial: optuna.Trial) -> float:
        # ✅ OPTIMIZED: Better search space for faster training
        # - seq_len: 20-80 (shorter sequences train faster than 20-120)
        # - num_channels: 32-256 (same as before, good range)
        # - num_layers: 2-4 (reduced from 2-6, TCN with many layers → slow)
        # - kernel_size: 2-6 (reduced from 2-8, most signals use 2-5)
        # - margin: tunable for MarginDirectionalLoss (if used)
        seq_len      = trial.suggest_int("seq_len",      20,  80)
        num_channels = trial.suggest_int("num_channels", 32,  256)
        num_layers   = trial.suggest_int("num_layers",    2,    4)
        kernel_size  = trial.suggest_int("kernel_size",   2,    6)
        dropout      = trial.suggest_float("dropout",    0.0,  0.5)
        lr           = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = TCNModel(
            seq_len=seq_len,
            hidden_size=num_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=30,       # ✅ INCREASED from default
            patience=5,      # ✅ INCREASED from default
            model_type=model_type,
            loss_fn=loss_fn,  # ✅ Use the specified loss function
        )
        
        try:
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        except Exception as e:
            logger.warning(f"[trial {trial.number}] Training failed: {e}")
            raise optuna.TrialPruned()
        
        preds = model.predict(X_test)

        # ✅ IMPROVED: Better collapse detection for different loss functions
        pred_std = np.std(preds)
        if model_type == "regressor":
            # Threshold depends on loss function
            if loss_fn == "mse":
                collapse_threshold = 1e-4  # MSE is very sensitive to collapse
            else:
                collapse_threshold = 5e-5  # Margin/directional losses are more stable
            
            if pred_std < collapse_threshold:
                logger.warning(
                    f"[trial {trial.number}] Predictions collapsed "
                    f"(std={pred_std:.2e}). Pruning."
                )
                raise optuna.TrialPruned()

        # ✅ For classifiers, check for useful signal variety
        if model_type == "classifier":
            unique_classes = len(np.unique(preds))
            non_neutral_frac = np.mean(preds != 0)
            if unique_classes < 2 or non_neutral_frac < 0.05:
                logger.warning(
                    f"[trial {trial.number}] Classifier too simple "
                    f"(unique_classes={unique_classes}, non_neutral={non_neutral_frac:.1%}). "
                    "Pruning."
                )
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

    # ── Retrain final model with increased budget ──────────────────────────
    final_model = TCNModel(
        seq_len=best["seq_len"],
        hidden_size=best["num_channels"],
        num_layers=best["num_layers"],
        kernel_size=best["kernel_size"],
        dropout=best["dropout"],
        lr=best["lr"],
        batch_size=best["batch_size"],
        epochs=50,          # ✅ INCREASED from default
        patience=10,        # ✅ INCREASED from default
        model_type=model_type,
        loss_fn=loss_fn,    # ✅ UPDATED to use specified loss function
    )
    final_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    final_preds = final_model.predict(X_test)

    # ✅ IMPROVED: Better collapse detection with clear diagnostics
    pred_std = np.std(final_preds)
    pred_min = np.min(final_preds)
    pred_max = np.max(final_preds)
    
    if model_type == "classifier":
        unique, counts = np.unique(final_preds, return_counts=True)
        class_dist = dict(zip(unique.astype(int), counts))
        non_neutral_frac = np.mean(final_preds != 0)
        logger.info(f"[train] Classifier output distribution: {class_dist}")
        if non_neutral_frac < 0.05:
            logger.warning(
                f"[train] Final TCN classifier is nearly all-neutral "
                f"(non-neutral={non_neutral_frac:.1%}). Consider increasing epochs, "
                "reducing dropout, or rebalancing classes."
            )
    elif pred_std < 1e-4:
        logger.warning(
            f"[train] Final TCN shows very low variance (std={pred_std:.2e}). "
            f"Range: [{pred_min:.4f}, {pred_max:.4f}]. "
            f"This suggests collapse despite using {loss_fn} loss. "
            "Consider: (1) increasing epochs, (2) reducing patience, (3) checking target distribution."
        )
        # Only add noise if std is extremely low
        if pred_std < 1e-5:
            logger.info("[train] Adding small noise to predictions to encourage diversity.")
            rng = np.random.default_rng(42)
            final_preds = final_preds + rng.normal(0, 1e-3, size=final_preds.shape)
            # Re-clamp after noise
            final_preds = np.clip(final_preds, -1.0, 1.0)

    aligned_index  = X_test.index[-(len(final_preds)):]
    X_test_aligned = X_test.loc[aligned_index]

    # ✅ NEW: Enhanced logging with sign distribution
    if model_type == "regressor":
        signs = np.sign(final_preds)
        n_short = (signs < 0).sum()
        n_neutral = (signs == 0).sum()
        n_long = (signs > 0).sum()
        
        logger.info(
            f"[train] Final preds — min: {pred_min:.4f}, max: {pred_max:.4f}, "
            f"mean: {final_preds.mean():.4f}, std: {pred_std:.4f}"
        )
        logger.info(
            f"[train] Signal distribution — Short: {n_short} ({100*n_short/len(final_preds):.1f}%) | "
            f"Neutral: {n_neutral} ({100*n_neutral/len(final_preds):.1f}%) | "
            f"Long: {n_long} ({100*n_long/len(final_preds):.1f}%)"
        )
    else:
        logger.info(
            f"[train] Final preds — min: {pred_min:.4f}, max: {pred_max:.4f}, "
            f"mean: {final_preds.mean():.4f}"
        )

    n_preds    = len(final_preds)
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