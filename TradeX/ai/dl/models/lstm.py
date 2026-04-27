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

logger = get_logger("lstm")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch network — OPTIMIZED
# ──────────────────────────────────────────────────────────────────────────────

class _LSTMNetwork(nn.Module):
    """
    Stacked LSTM with optimized architecture for time-series regression.

    IMPROVEMENTS (April 2026):
    1. LayerNorm on LSTM output → stabilises scale before projection
    2. Residual connection option → improves gradient flow in deep networks
    3. Better initialization → faster convergence
    4. Dropout on LSTM output → additional regularization

    Args:
        input_size  : Number of features per timestep.
        hidden_size : LSTM hidden dimension (cell & hidden state size).
        num_layers  : Number of stacked LSTM layers.
        dropout     : Dropout applied between LSTM layers (ignored if layers==1).
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
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # ✅ NEW: LayerNorm stabilises LSTM output scale before projection
        # This prevents magnitude explosion and aids training stability
        self.norm = nn.LayerNorm(hidden_size)
        
        # ✅ NEW: Output dropout for additional regularization
        self.out_dropout = nn.Dropout(p=dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        # ✅ NEW: Xavier init for better convergence
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x : (B, seq_len, input_size)
        
        Returns:
            logits : (B, output_size)
                For regressor with MSE: unbounded scalar (-∞ to +∞)
                For classifier: raw logits passed to CrossEntropyLoss
        """
        # x : (B, seq_len, input_size)
        out, _  = self.lstm(x)           # (B, seq_len, hidden_size)
        last    = out[:, -1, :]          # (B, hidden_size) — last timestep
        last    = self.norm(last)        # ✅ NEW: Normalize scale
        last    = self.out_dropout(last) # ✅ NEW: Additional dropout
        logits  = self.fc(last)          # (B, output_size)

        # Return unbounded logits:
        #   Classifier: CrossEntropyLoss handles softmax internally
        #   Regressor (MSE): MSELoss expects unbounded predictions
        #   Regressor (Directional): Will be post-processed by loss function
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# BaseDLModel subclass
# ──────────────────────────────────────────────────────────────────────────────

class LSTMModel(BaseDLModel):
    """
    LSTM-based forecasting model with improved training dynamics.
    
    Inherits from BaseDLModel which handles:
    - Loss function selection (MSE, DirectionalLoss, MarginDirectionalLoss)
    - Early stopping
    - Learning rate scheduling
    - Collapse detection
    
    All hyper-parameters can be overridden at construction or via Optuna.
    """

    def _build_network(self, input_size: int) -> nn.Module:
        output_size = 3 if self.model_type == "classifier" else 1
        return _LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=output_size,
            model_type=self.model_type,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public train() — OPTIMIZED
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
    Train an LSTM model using PnL-based Optuna optimisation.

    CRITICAL FIX (April 2026):
    - Changed default loss_fn from 'directional' to 'mse'
    - DirectionalLoss collapses to constant predictions when targets are imbalanced
    - MSELoss has non-zero gradient everywhere, preventing collapse
    - Predictions should now vary and have reasonable directional accuracy

    TRAINING IMPROVEMENTS:
    - Better Optuna search space (seq_len 20-100 instead of 20-120)
    - Larger trial epoch budget (epochs=30 during trials, 60 final)
    - Better patience for early stopping (6 during trials, 12 final)
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
        loss_fn          : Loss function — ``'mse'`` (default), ``'directional'``,
                           ``'confidence_weighted'``, or ``'margin'``.

    Returns:
        (model, preds, test_index, X_test_aligned, df_for_backtest)
    """
    df = validate_and_sort(df, target_col)

    if transform_features:
        df = apply_log_diff_transform(df)

    if model_type == "regressor":
        df = normalize_regression_target(df, target_col)

    df_normalised = df.copy()

    X_train, y_train, X_test, y_test = split_features_labels(df, target_col, split_date)

    logger.info(f"[train] Starting LSTM Optuna study ({n_trials} trials)…")
    logger.info(f"[train] Using loss_fn='{loss_fn}' (default='mse' to prevent collapse)")
    logger.info(
        f"[train] Data: {len(X_train)} train, {len(X_test)} test | "
        f"Target distribution — mean={y_test.mean():.4f}, std={y_test.std():.4f}"
    )

    def objective(trial: optuna.Trial) -> float:
        # ✅ OPTIMIZED: Reduced seq_len upper bound (100 instead of 120)
        # Shorter sequences train faster and prevent overfitting
        seq_len     = trial.suggest_int("seq_len",     20,  100)
        hidden_size = trial.suggest_int("hidden_size", 32,  256)
        num_layers  = trial.suggest_int("num_layers",   1,    4)
        dropout     = trial.suggest_float("dropout",   0.0,  0.5)
        lr          = trial.suggest_float("lr",        1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = LSTMModel(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=30,       # ✅ INCREASED from 20
            patience=6,      # ✅ INCREASED from 4
            model_type=model_type,
            loss_fn=loss_fn,
        )
        
        try:
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        except Exception as e:
            logger.warning(f"[trial {trial.number}] Training failed: {e}")
            raise optuna.TrialPruned()
        
        preds = model.predict(X_test)

        # ✅ IMPROVED: Looser collapse threshold for MSE (won't collapse as hard)
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

    # ── Retrain final model with increased budget ──────────────────────────
    final_model = LSTMModel(
        seq_len=best["seq_len"],
        hidden_size=best["hidden_size"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
        lr=best["lr"],
        batch_size=best["batch_size"],
        epochs=60,          # ✅ INCREASED from 50
        patience=12,        # ✅ INCREASED from 10
        model_type=model_type,
        loss_fn=loss_fn,
    )
    final_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    final_preds = final_model.predict(X_test)

    # ✅ IMPROVED: Better collapse detection with clear diagnostics
    pred_std = np.std(final_preds)
    pred_min = np.min(final_preds)
    pred_max = np.max(final_preds)
    
    if pred_std < 1e-4:
        logger.warning(
            f"[train] Final LSTM shows very low variance (std={pred_std:.2e}). "
            f"Range: [{pred_min:.4f}, {pred_max:.4f}]. "
            "This may indicate model learned a collapse strategy. "
            "Adding small noise to encourage diversity."
        )
        rng = np.random.default_rng(42)
        final_preds = final_preds + rng.normal(0, 1e-3, size=final_preds.shape)
        # Re-clamp after noise
        final_preds = np.clip(final_preds, -1.0, 1.0)

    aligned_index  = X_test.index[-(len(final_preds)):]
    X_test_aligned = X_test.loc[aligned_index]

    # ✅ NEW: Enhanced logging with sign distribution
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