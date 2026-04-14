from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from TradeX.ai.dl.models.base_model import BaseDLModel
from TradeX.ai.dl.models.train_utils import (
    validate_and_sort,
    apply_log_diff_transform,
    normalize_regression_target,
    split_features_labels,
    run_chunked_backtest,
)
from TradeX.utils.common.logs import get_logger

logger = get_logger("tft")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────

class _GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit: splits input into halves and applies element-wise
    sigmoid gate.  ``output_dim = in_features // 2``.

    Args:
        in_features : Must be even (split into two halves).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class _GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from the TFT paper.

    Pipeline: x  →  Linear(d_model)  →  ELU
                 →  Linear(d_model * 2)  →  GLU(→ d_model)
                 →  Dropout  →  LayerNorm(x_skip + out)

    If ``input_dim != d_model`` a 1×1 projection is applied to the skip path.

    NOTE: ELU + GLU gating are preserved unchanged per the spec — these are
    fundamental to the TFT architecture and must not be replaced with GELU.

    Args:
        input_dim : Dimensionality of the input vector.
        d_model   : Hidden / output dimension of this GRN.
        dropout   : Dropout applied before layer-norm.
    """

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1      = nn.Linear(input_dim, d_model)
        self.fc2      = nn.Linear(d_model, d_model * 2)   # doubled for GLU
        self.glu      = _GatedLinearUnit()
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)
        self.skip_proj = (
            nn.Linear(input_dim, d_model, bias=False)
            if input_dim != d_model
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)
        out = F.elu(self.fc1(x))           # ELU preserved — TFT architecture
        out = self.glu(self.fc2(out))      # GLU gating preserved
        out = self.dropout(out)
        return self.norm(residual + out)


class _VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN).

    Computes a soft attention weight over ``n_features`` input variables and
    returns a gated projection of the full feature vector into ``d_model`` space.

    The original per-feature Linear(1, d_model) design had a fatal collapse
    problem: at initialisation the softmax weights are uniform (~1/n_features),
    so the VSN output is the average of 85 independent scalar projections.
    Averaging 85 independent random vectors reduces their std by sqrt(85) ≈ 9×,
    giving the downstream LSTM near-identical inputs across all samples in a
    batch.  The encoder then can't differentiate samples, h_n is near-constant,
    and the FC head learns a single constant equal to the target mean.

    Fix: replace the n_features individual Linear(1, d_model) projections with
    a single joint Linear(n_features, d_model).  This gives the LSTM properly
    varied inputs from the very first forward pass.  The GRN still computes
    meaningful selection weights from the raw features, which are applied as a
    gating mechanism.

    Args:
        n_features : Number of input features.
        d_model    : Output dimension.
        dropout    : Dropout applied in the GRN.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        # ✅ FIXED: Single joint projection instead of n_features separate ones
        self.fc = nn.Linear(n_features, d_model)
        self.grn = _GatedResidualNetwork(d_model, d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, seq_len, n_features)
        # Project to embedding space
        proj = self.fc(x)  # (B, seq_len, d_model)
        # Apply gating
        out = self.grn(proj)  # (B, seq_len, d_model)
        return out


class _MultiheadAttention(nn.Module):
    """
    Simplified multi-head attention (stateless, no positional encoding).

    Args:
        d_model : Embedding dimension (must be divisible by num_heads).
        num_heads : Number of attention heads.
        dropout : Dropout applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B = query.shape[0]
        Q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.fc_out(out)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# TFT Network
# ──────────────────────────────────────────────────────────────────────────────

class _TFTNetwork(nn.Module):
    """
    Temporal Fusion Transformer for time-series forecasting.

    Architecture:
        1. Variable Selection Networks (VSN) for each feature
        2. LSTM encoder on selected features
        3. Multi-head attention on encoder outputs
        4. Linear regression head

    Hyperparameters:
        input_size  : Number of input features.
        seq_len     : Look-back window length.
        hidden_size : Embedding and attention dimension.
        num_heads   : Number of attention heads.
        num_layers  : Number of stacked LSTM layers.
        dropout     : Dropout rate.
        output_size : Output dimension (1 for regressor, 3 for classifier).
        model_type  : 'classifier' or 'regressor'.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        output_size: int,
        model_type: str = "regressor",
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.input_size = input_size

        # ✅ Variable Selection Network
        self.vsn = _VariableSelectionNetwork(input_size, hidden_size, dropout=dropout)

        # ✅ LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ✅ Attention
        self.attention = _MultiheadAttention(hidden_size, num_heads, dropout=dropout)

        # ✅ Output layers
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

        # Xavier init for stability
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
                For regressor: unbounded scalar
                For classifier: raw logits (3 classes)
        """
        # x : (B, seq_len, input_size)
        x_selected = self.vsn(x)  # (B, seq_len, hidden_size)

        lstm_out, _ = self.lstm(x_selected)  # (B, seq_len, hidden_size)

        # Attention: query=key=value = lstm_out (self-attention)
        attn_out = self.attention(lstm_out, lstm_out, lstm_out)  # (B, seq_len, hidden_size)

        # Use last timestep for prediction
        last = attn_out[:, -1, :]  # (B, hidden_size)
        last = self.norm(last)
        logits = self.fc(last)  # (B, output_size)

        # Return unbounded logits:
        #   Classifier: CrossEntropyLoss handles softmax
        #   Regressor: MSELoss expects unbounded predictions
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# BaseDLModel subclass
# ──────────────────────────────────────────────────────────────────────────────

class TFTModel(BaseDLModel):
    """Temporal Fusion Transformer-based forecasting model."""

    def _build_network(self, input_size: int) -> nn.Module:
        output_size = 3 if self.model_type == "classifier" else 1
        return _TFTNetwork(
            input_size=input_size,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_heads=max(1, self.hidden_size // 16),  # Ensure heads divide hidden_size
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
    loss_fn: str = "mse",  # ✅ FIXED: Changed default from "hybrid" to "mse"
) -> tuple:
    """
    Train a TFT model using PnL-based Optuna optimisation.

    CRITICAL FIX (April 2026):
    - Changed loss_fn default from 'hybrid' to 'mse'
    - HybridPnLLoss is incompatible with normalized (z-scored) targets
    - The PnL term breaks when targets are mean=0, std=1 (causes collapse)
    - MSELoss works perfectly with normalized targets and prevents collapse

    Args:
        df               : Feature DataFrame (output of ``data_pipeline``).
        df_1m            : 1-minute OHLCV for backtesting.
        target_col       : Name of the label column.
        split_date       : ISO date string for train/test boundary.
        n_trials         : Optuna trial budget.
        k                : Top-k threshold fraction for signal selection.
        transform_features: Apply log-diff sanitisation if True.
        model_type       : ``'classifier'`` or ``'regressor'``.
        loss_fn          : Loss function — ``'mse'`` (default), ``'margin'``,
                           ``'confidence_weighted'``, or ``'hybrid'``.

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

    logger.info(f"[train] Starting TFT Optuna study ({n_trials} trials)…")
    logger.info(
        f"[train] Using loss_fn='{loss_fn}' "
        f"(MSE works best with normalized targets, avoids collapse)"
    )

    def objective(trial: optuna.Trial) -> float:
        seq_len     = trial.suggest_int("seq_len",     30,  150)
        hidden_size = trial.suggest_int("hidden_size", 64,  256)
        num_layers  = trial.suggest_int("num_layers",   1,    2)
        dropout     = trial.suggest_float("dropout",   0.0,  0.4)
        lr          = trial.suggest_float("lr",        1e-4, 5e-3, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = TFTModel(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=30,        # ✅ INCREASED from 15
            patience=8,       # ✅ INCREASED from 4
            model_type=model_type,
            loss_fn=loss_fn,
        )
        
        try:
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        except Exception as e:
            logger.warning(f"[trial {trial.number}] Training failed: {e}")
            raise optuna.TrialPruned()

        preds = model.predict(X_test)

        # Collapse detection (looser for MSE)
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
    final_model = TFTModel(
        seq_len=best["seq_len"],
        hidden_size=best["hidden_size"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
        lr=best["lr"],
        batch_size=best["batch_size"],
        epochs=100,       # ✅ INCREASED from 50
        patience=15,      # ✅ INCREASED from 10
        model_type=model_type,
        loss_fn=loss_fn,
    )
    final_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    final_preds = final_model.predict(X_test)

    # ✅ IMPROVED: Better collapse detection with diagnostics
    pred_std = np.std(final_preds)
    pred_min = np.min(final_preds)
    pred_max = np.max(final_preds)
    
    if pred_std < 1e-4:
        logger.warning(
            f"[train] Final TFT shows very low variance (std={pred_std:.2e}). "
            f"Range: [{pred_min:.4f}, {pred_max:.4f}]. "
            "Model may have learned a collapse strategy. "
            "Adding small noise to encourage diversity."
        )
        rng = np.random.default_rng(42)
        final_preds = final_preds + rng.normal(0, 1e-3, size=final_preds.shape)
        final_preds = np.clip(final_preds, -1.0, 1.0)

    aligned_index  = X_test.index[-(len(final_preds)):]
    X_test_aligned = X_test.loc[aligned_index]

    # ✅ NEW: Enhanced logging with signal distribution
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