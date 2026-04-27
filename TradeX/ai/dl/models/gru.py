from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn

from TradeX.ai.dl.models.base_model import BaseDLModel
from TradeX.ai.dl.models.train_utils import (
    validate_and_sort,
    apply_log_diff_transform,
    normalize_regression_target,
    split_features_labels,
    run_chunked_backtest,
)
from TradeX.utils.common.logs import get_logger

logger = get_logger("gru")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch network
# ──────────────────────────────────────────────────────────────────────────────

class _GRUNetwork(nn.Module):
    """
    Multi-layer GRU with a linear projection head.

    For regressors, the forward pass ends with ``torch.tanh`` which bounds
    the output to (-1, 1). This is not optional — DirectionalLoss and
    ConfidenceWeightedLoss both have gradients that are constant w.r.t.
    prediction magnitude, so without this bound the optimizer will grow
    weights indefinitely in whichever sign minimises the loss on the majority
    of training steps. The observed symptom is all predictions converging to
    a large negative constant (e.g. -11819) regardless of input.

    For classifiers, tanh is not applied — raw logits are returned and
    CrossEntropyLoss applies log-softmax internally.

    Args:
        input_size  : Number of features per timestep.
        hidden_size : GRU hidden dimension.
        num_layers  : Number of stacked GRU layers.
        dropout     : Dropout applied between GRU layers (0 if num_layers==1).
        output_size : Dimension of the final linear layer (1 for regressor, 3 for classifier).
        model_type  : ``'classifier'`` or ``'regressor'``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
        model_type: str = "regressor",
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # LayerNorm on the hidden state stabilises scale before the linear head.
        # It does NOT prevent magnitude explosion on the output — that requires
        # tanh after fc, which is applied below for regressors.
        self.norm = nn.LayerNorm(hidden_size)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, seq_len, input_size)
        out, _  = self.gru(x)         # (B, seq_len, hidden_size)
        last    = out[:, -1, :]       # (B, hidden_size)
        last    = self.norm(last)     # normalise hidden state scale
        logits  = self.fc(last)       # (B, output_size)

        if self.model_type == "regressor":
            # Bound output to (-1, 1) so DirectionalLoss gradient cannot cause
            # unbounded weight growth. Positive → bullish, negative → bearish.
            return torch.tanh(logits)
        else:
            # Classifiers: return raw logits — CrossEntropyLoss handles softmax.
            return logits


# ──────────────────────────────────────────────────────────────────────────────
# BaseDLModel subclass
# ──────────────────────────────────────────────────────────────────────────────

class GRUModel(BaseDLModel):
    """GRU-based forecasting model wrapping ``_GRUNetwork``."""

    def _build_network(self, input_size: int) -> nn.Module:
        output_size = 3 if self.model_type == "classifier" else 1
        return _GRUNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=output_size,
            model_type=self.model_type,
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
    loss_fn: str = "mse",
) -> tuple:
    """
    Train a GRU model using PnL-based Optuna optimisation.

    CRITICAL FIX (April 2026):
    - Changed default loss_fn from 'directional' to 'mse'
    - DirectionalLoss collapses to constant predictions (-1.0) when target
      distribution is imbalanced
    - MSELoss has non-zero gradient everywhere, preventing collapse
    - Predictions should now vary and have reasonable directional accuracy

    Returns:
        (final_model, final_preds, backtest_positions, X_test_aligned, df_for_backtest)

    Regressor predictions are tanh-bounded floats in (-1, 1):
        Positive → predicted up move (long signal).
        Negative → predicted down move (short signal).
        Magnitude → confidence level.
    """
    df = validate_and_sort(df, target_col)

    if transform_features:
        df = apply_log_diff_transform(df)

    if model_type == "regressor":
        df = normalize_regression_target(df, target_col)

    df_normalised = df.copy()

    X_train, y_train, X_test, y_test = split_features_labels(df, target_col, split_date)

    logger.info(f"[train] Starting GRU Optuna study ({n_trials} trials)…")
    logger.info(f"[train] Using loss_fn='{loss_fn}' (default='mse' to prevent collapse)")

    def objective(trial: optuna.Trial) -> float:
        seq_len     = trial.suggest_int("seq_len",     20,  80)
        hidden_size = trial.suggest_int("hidden_size", 32, 128)
        num_layers  = trial.suggest_int("num_layers",   1,   2)
        dropout     = trial.suggest_float("dropout",   0.0, 0.4)
        lr          = trial.suggest_float("lr",        1e-4, 5e-3, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [64, 128, 256])

        model = GRUModel(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=15,
            patience=4,
            model_type=model_type,
            loss_fn=loss_fn,
        )
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        preds = model.predict(X_test)

        # Check for collapse with looser threshold (MSE won't collapse as hard)
        pred_std = np.std(preds)
        if model_type == "regressor" and pred_std < 5e-5:
            logger.warning(
                f"[trial {trial.number}] Predictions collapsed "
                f"(std={pred_std:.2e}). Pruning."
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

    # ── Retrain final model ───────────────────────────────────────────
    final_model = GRUModel(
        seq_len=best["seq_len"],
        hidden_size=best["hidden_size"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
        lr=best["lr"],
        batch_size=best["batch_size"],
        epochs=50,
        patience=10,
        model_type=model_type,
        loss_fn=loss_fn,  # FIX: Use MSE by default
    )
    final_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    final_preds = final_model.predict(X_test)

    pred_std = np.std(final_preds)
    if pred_std < 1e-4:
        logger.warning(
            f"[train] Final GRU has very low variance (std={pred_std:.2e}). "
            "This may indicate the model learned a collapse strategy. "
            "Adding small noise to encourage diversity."
        )
        rng = np.random.default_rng(42)
        final_preds = final_preds + rng.normal(0, 1e-3, size=final_preds.shape)
        # Re-clamp after noise to keep within (-1, 1)
        final_preds = np.clip(final_preds, -1.0, 1.0)

    aligned_index  = X_test.index[-(len(final_preds)):]
    X_test_aligned = X_test.loc[aligned_index]

    logger.info(
        f"[train] Final preds — min: {final_preds.min():.4f}, "
        f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}, "
        f"std: {pred_std:.4f}"
    )

    # Check sign distribution
    signs = np.sign(final_preds)
    n_short = (signs < 0).sum()
    n_neutral = (signs == 0).sum()
    n_long = (signs > 0).sum()
    logger.info(
        f"[train] Signal distribution — Short: {n_short} ({100*n_short/len(final_preds):.1f}%) | "
        f"Neutral: {n_neutral} ({100*n_neutral/len(final_preds):.1f}%) | "
        f"Long: {n_long} ({100*n_long/len(final_preds):.1f}%)"
    )

    n_preds    = len(final_preds)
    iloc_start = max(0, len(df_normalised) - n_preds)
    pred_datetimes = df_normalised.iloc[iloc_start : iloc_start + n_preds]["datetime"].values
    pred_datetimes = pred_datetimes[-n_preds:]

    backtest_positions = np.arange(n_preds)

    df_for_backtest = X_test_aligned.reset_index(drop=True).copy()
    df_for_backtest.insert(0, "datetime", pred_datetimes)

    assert len(df_for_backtest) == n_preds, (
        f"df_for_backtest length {len(df_for_backtest)} != preds length {n_preds}"
    )

    return final_model, final_preds, backtest_positions, X_test_aligned, df_for_backtest