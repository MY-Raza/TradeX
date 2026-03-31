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

    Args:
        input_size  : Number of features per timestep.
        hidden_size : GRU hidden dimension.
        num_layers  : Number of stacked GRU layers.
        dropout     : Dropout applied between GRU layers (ignored if num_layers==1).
        output_size : Dimension of the final linear layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, seq_len, input_size)
        out, _ = self.gru(x)          # (B, seq_len, hidden_size)
        last    = out[:, -1, :]       # (B, hidden_size)
        return self.fc(last)          # (B, output_size)


# ──────────────────────────────────────────────────────────────────────────────
# BaseDLModel subclass
# ──────────────────────────────────────────────────────────────────────────────

class GRUModel(BaseDLModel):
    """
    GRU-based forecasting model wrapping ``_GRUNetwork``.

    All hyper-parameters are inherited from ``BaseDLModel`` and can be
    overridden at construction time or via Optuna.
    """

    def _build_network(self, input_size: int) -> nn.Module:
        output_size = 3 if self.model_type == "classifier" else 1
        return _GRUNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=output_size,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public train() — identical signature to RF/XGB counterparts
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
    Train a GRU model using PnL-based Optuna optimisation.

    Mirrors the ``train()`` contract in ``randomforest_clf.py`` and
    ``xgboost_reg.py``:
        - Validates & sorts the input DataFrame.
        - Optionally log-diff transforms OHLCV columns.
        - Splits into train / test sets via a temporal boundary.
        - Runs an Optuna study maximising backtested PnL.
        - Retrains the best model on the full training set.

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
        - model      : Fitted ``GRUModel`` instance.
        - preds      : 1-D np.ndarray of predictions on the test set.
        - test_index : pd.Index of test-set rows in the original DataFrame.
        - X_test     : Test feature matrix.
    """
    df = validate_and_sort(df, target_col)

    if transform_features:
        df = apply_log_diff_transform(df)

    # Capture the normalised df NOW — its RangeIndex matches the idx_arr
    # integers that aligned_index will contain after the train/test split.
    df_normalised = df.copy()

    X_train, y_train, X_test, y_test = split_features_labels(df, target_col, split_date)

    logger.info(f"[train] Starting GRU Optuna study ({n_trials} trials)…")

    def objective(trial: optuna.Trial) -> float:
        seq_len     = trial.suggest_int("seq_len",     20,  120)
        hidden_size = trial.suggest_int("hidden_size", 32,  256)
        num_layers  = trial.suggest_int("num_layers",   1,    4)
        dropout     = trial.suggest_float("dropout",   0.0,  0.5)
        lr          = trial.suggest_float("lr",        1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = GRUModel(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=30,
            patience=5,
            model_type=model_type,
        )
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        preds = model.predict(X_test)

        if model_type == "regressor" and np.std(preds) < 1e-8:
            raise optuna.TrialPruned()

        # Align test_index with the (potentially shorter) preds array
        aligned_index = X_test.index[-(len(preds)):]

        return run_chunked_backtest(
            trial, df, preds, aligned_index, df_1m,
            model_type=model_type, k=k,
        )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = study.best_params
    logger.info(f"[train] Best params: {best} | Best PnL: {study.best_value:.4f}")

    # ── Retrain final model with best hyper-parameters ───────────────
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
    )
    final_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    final_preds = final_model.predict(X_test)

    # Trim X_test to match the (seq_len warm-up shortened) preds length so
    # every downstream caller (backtest, importance, dry-run) sees aligned arrays.
    aligned_index = X_test.index[-(len(final_preds)):]
    X_test_aligned = X_test.loc[aligned_index]

    logger.info(
        f"[train] Final preds — min: {final_preds.min():.4f}, "
        f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}"
    )

    # Build a minimal DataFrame for prepare_predictions that is guaranteed to
    # be length-aligned with final_preds regardless of index gaps or seq_len
    # warm-up effects.
    #
    # prepare_predictions does:  df.iloc[test_index]['datetime']
    # so we construct df_for_backtest with RangeIndex 0..P-1 containing only
    # the datetimes that correspond to each prediction, extracted directly from
    # df_normalised using aligned_index (valid .loc labels into df_normalised).
    # test_index is then simply np.arange(P) — iloc[0..P-1] on a P-row frame
    # is always valid and always returns exactly P rows.
    # Use .iloc with a clipped range so pred_datetimes is *always* exactly
    # len(final_preds) rows, regardless of index gaps introduced by
    # log-diff dropping or split sanitisation.
    n_preds = len(final_preds)
    iloc_end   = min(len(df_normalised), len(df_normalised))   # full length
    iloc_start = max(0, len(df_normalised) - n_preds)
    pred_datetimes = df_normalised.iloc[iloc_start : iloc_start + n_preds]["datetime"].values

    # Guarantee exact length match (defensive clamp)
    pred_datetimes     = pred_datetimes[-n_preds:]
    backtest_positions = np.arange(n_preds)
    df_for_backtest    = pd.DataFrame({"datetime": pred_datetimes})  # RangeIndex 0..P-1

    assert len(df_for_backtest) == n_preds, (
        f"df_for_backtest length {len(df_for_backtest)} != preds length {n_preds}"
    )

    return final_model, final_preds, backtest_positions, X_test_aligned, df_for_backtest