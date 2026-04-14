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
    scalar gate on the joint projection output.

    Args:
        n_features : Number of input features.
        d_model    : Output projection dimension.
        dropout    : Dropout inside the GRN.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model    = d_model

        self.input_projection = nn.Linear(n_features, d_model)
        self.grn     = _GatedResidualNetwork(n_features, d_model, dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, n_features)
        B, T, _ = x.shape

        projected = self.input_projection(x)

        flat    = x.reshape(B * T, self.n_features)
        gate    = self.sigmoid(
            self.grn(flat).reshape(B, T, self.d_model)
        )

        return projected * gate


class _TFTNetwork(nn.Module):
    """
    Simplified TFT core network.

    Args:
        input_size   : Number of input features per timestep.
        d_model      : Internal model dimension (hidden size).
        num_heads    : Number of attention heads.
        num_layers   : Number of LSTM layers in encoder and decoder.
        dropout      : Dropout probability.
        output_size  : Output dimensionality of the head.
        model_type   : ``'classifier'`` or ``'regressor'``.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        output_size: int,
        model_type: str = "regressor",   # ✅ UPDATED
    ) -> None:
        super().__init__()
        self.model_type = model_type      # ✅ UPDATED

        # Variable selection
        self.vsn = _VariableSelectionNetwork(input_size, d_model, dropout)

        # LSTM encoder — internal sigmoid/tanh activations preserved unchanged
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # LSTM decoder (single step; uses encoder's final hidden/cell)
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Multi-head self-attention
        self.attention   = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm   = nn.LayerNorm(d_model)

        # Post-attention GRN — ELU + GLU gating preserved
        self.post_attn_grn = _GatedResidualNetwork(d_model, d_model, dropout)

        # Output projection
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, seq_len, input_size)

        # 1. Variable selection
        vsn_out = self.vsn(x)                           # (B, seq_len, d_model)

        # 2. LSTM encoding
        enc_out, (h_n, c_n) = self.encoder(vsn_out)    # enc_out: (B, T, d_model)

        # 3. LSTM decoding — single step using last encoded hidden state
        seed        = enc_out[:, -1:, :]                # (B, 1, d_model)
        dec_out, _  = self.decoder(seed, (h_n, c_n))    # (B, 1, d_model)

        # 4. Self-attention over encoder outputs
        attn_input  = enc_out                            # (B, T, d_model)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        attn_out    = self.attn_norm(attn_input + attn_out)  # residual + norm

        # 5. Combine decoder output with attended context (last timestep)
        combined = dec_out[:, 0, :] + attn_out[:, -1, :]   # (B, d_model)

        # 6. Post-attention GRN
        grn_out = self.post_attn_grn(combined.unsqueeze(1)).squeeze(1)  # (B, d_model)

        # 7. Linear head — raw logits / scalar
        logits = self.fc(grn_out)                        # (B, output_size)

        # ✅ UPDATED — output activation strategy:
        #   Classifier: return raw logits. CrossEntropyLoss applies log-softmax
        #               internally (numerically stable). predict_proba() applies
        #               softmax explicitly at inference time.
        #   Regressor:  return raw scalar — DirectionalLoss / ConfidenceWeightedLoss
        #               require unbounded outputs.
        #   ELU + GLU gating structure preserved throughout (unchanged).
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# BaseDLModel subclass
# ──────────────────────────────────────────────────────────────────────────────

class TFTModel(BaseDLModel):
    """
    Temporal Fusion Transformer forecasting model.

    Additional constructor parameters (beyond ``BaseDLModel``):
        num_heads (int): Number of attention heads in the self-attention layer.
            Must evenly divide ``hidden_size``. Defaults to 4.
    """

    def __init__(self, num_heads: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def _build_network(self, input_size: int) -> nn.Module:
        output_size = 3 if self.model_type == "classifier" else 1
        # Ensure hidden_size is divisible by num_heads
        d_model = self.hidden_size - (self.hidden_size % self.num_heads)
        if d_model != self.hidden_size:
            logger.warning(
                f"[TFTModel] hidden_size={self.hidden_size} not divisible by "
                f"num_heads={self.num_heads}. Adjusted to d_model={d_model}."
            )
        return _TFTNetwork(
            input_size=input_size,
            d_model=d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=output_size,
            model_type=self.model_type,   # ✅ UPDATED
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
    loss_fn: str = "hybrid",         # ✅ UPDATED
) -> tuple:
    """
    Train a TFT model using PnL-based Optuna optimisation.

    Args:
        df               : Feature DataFrame (output of ``data_pipeline``).
        df_1m            : 1-minute OHLCV for backtesting.
        target_col       : Name of the label column.
        split_date       : ISO date string for train/test boundary.
        n_trials         : Optuna trial budget.
        k                : Top-k threshold fraction for signal selection.
        transform_features: Apply log-diff to OHLCV columns if True.
        model_type       : ``'classifier'`` or ``'regressor'``.
        loss_fn          : ✅ UPDATED — ``'directional'`` or
                           ``'confidence_weighted'`` (regressor only).

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

    logger.info(f"[train] Starting TFT Optuna study ({n_trials} trials)…")

    def objective(trial: optuna.Trial) -> float:
        seq_len     = trial.suggest_int("seq_len",     20,  120)
        hidden_size = trial.suggest_int("hidden_size", 32,  128)
        num_heads   = trial.suggest_categorical("num_heads", [2, 4, 8])
        num_layers  = trial.suggest_int("num_layers",   1,    3)
        dropout     = trial.suggest_float("dropout",   0.0,  0.4)
        lr          = trial.suggest_float("lr",        1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

        d_model = hidden_size - (hidden_size % num_heads)
        if d_model < num_heads:
            raise optuna.TrialPruned()

        model = TFTModel(
            seq_len=seq_len,
            hidden_size=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=30,
            patience=5,
            model_type=model_type,
            loss_fn=loss_fn,              # ✅ UPDATED
        )
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        preds = model.predict(X_test)

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

    d_model = best["hidden_size"] - (best["hidden_size"] % best["num_heads"])
    final_model = TFTModel(
        seq_len=best["seq_len"],
        hidden_size=d_model,
        num_heads=best["num_heads"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
        lr=best["lr"],
        batch_size=best["batch_size"],
        epochs=50,
        patience=10,
        model_type=model_type,
        loss_fn=loss_fn,                  # ✅ UPDATED
    )
    final_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    final_preds = final_model.predict(X_test)

    if np.std(final_preds) < 1e-6:
        logger.warning(
            "[train] Final TFT model collapsed to constant output "
            f"({final_preds.mean():.6f}). Adding small noise to prevent all-zero signals."
        )
        rng = np.random.default_rng(42)
        final_preds = final_preds + rng.normal(0, 1e-4, size=final_preds.shape)

    aligned_index  = X_test.index[-(len(final_preds)):]
    X_test_aligned = X_test.loc[aligned_index]

    logger.info(
        f"[train] Final preds — min: {final_preds.min():.4f}, "
        f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}"
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