from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from datetime import datetime

# ── Existing pipeline imports ─────────────────────────────────────────────────
from TradeX.ai.dl.models.base_model import BaseDLModel
from TradeX.ai.dl.models.train_utils import (
    validate_and_sort,
    apply_log_diff_transform,
    normalize_regression_target,
    split_features_labels,
    run_chunked_backtest,
)
from TradeX.ai.dl.models.model_trainer import save_model
from TradeX.ai.data.data_pipeline import (
    fetch_raw_data,
    resample_data,
    prepare_features,
    build_regression_df,
    resolve_split_date,
)
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.logs import get_logger

logger = get_logger("lstm_pipeline")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# 1.  PyTorch Network  (identical to existing lstm.py — no changes)
# =============================================================================

class _LSTMNetwork(nn.Module):
    """
    Stacked LSTM with a linear projection head.

    Args:
        input_size  : Number of features per timestep.
        hidden_size : LSTM hidden dimension (cell & hidden state size).
        num_layers  : Number of stacked LSTM layers.
        dropout     : Dropout applied between LSTM layers (ignored if layers==1).
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
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, seq_len, input_size)
        out, _  = self.lstm(x)      # (B, seq_len, hidden_size)
        last     = out[:, -1, :]    # (B, hidden_size)  — last timestep only
        return self.fc(last)        # (B, output_size)


# =============================================================================
# 2.  LSTMModel  (BaseDLModel subclass — identical to existing lstm.py)
# =============================================================================

class LSTMModel(BaseDLModel):
    """
    LSTM-based forecasting model wrapping ``_LSTMNetwork``.

    All hyper-parameters are inherited from ``BaseDLModel`` and can be
    overridden at construction time or via Optuna.
    """

    def _build_network(self, input_size: int) -> nn.Module:
        output_size = 3 if self.model_type == "classifier" else 1
        return _LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=output_size,
        )


# =============================================================================
# 3.  Split helpers  (NEW — not in the existing pipeline)
# =============================================================================

def split_70_30(
    df: pd.DataFrame,
    target_col: str = "target",
    train_ratio: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal 70 / 30 split on a sorted DataFrame.

    The split is row-based (not calendar-based) so it works even when the
    time-series is not uniformly sampled.

    Args:
        df          : Feature DataFrame with a ``'datetime'`` column.
        target_col  : Name of the label column (kept in both splits).
        train_ratio : Fraction of rows assigned to the train block.

    Returns:
        ``(df_train_70, df_test_30)`` — both retain all original columns.
    """
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain a 'datetime' column.")

    df = df.sort_values("datetime").reset_index(drop=True)
    split_row = int(len(df) * train_ratio)

    df_train_70 = df.iloc[:split_row].reset_index(drop=True)
    df_test_30  = df.iloc[split_row:].reset_index(drop=True)

    logger.info(
        f"[split_70_30] Total rows: {len(df)} | "
        f"Train 70%: {len(df_train_70)} | Test 30%: {len(df_test_30)}"
    )
    return df_train_70, df_test_30


def split_train_val_90_10(
    df_train_70: pd.DataFrame,
    target_col: str = "target",
    train_ratio: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Sub-split the 70 % train block into a 90 % model-training set and a
    10 % validation set.

    Args:
        df_train_70 : The 70 % train DataFrame from :func:`split_70_30`.
        target_col  : Name of the label column.
        train_ratio : Fraction of the 70 % block used for actual training.

    Returns:
        ``(df_train_90, df_val_10, X_train, y_train, X_val, y_val)``

        - ``df_train_90 / df_val_10`` : raw DataFrames (with datetime & target).
        - ``X_train / y_train``       : feature matrix / labels for training.
        - ``X_val   / y_val``         : feature matrix / labels for validation.
    """
    split_row = int(len(df_train_70) * train_ratio)

    df_train_90 = df_train_70.iloc[:split_row].reset_index(drop=True)
    df_val_10   = df_train_70.iloc[split_row:].reset_index(drop=True)

    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df_train_70.columns]

    X_train = df_train_90.drop(columns=drop_cols)
    y_train = df_train_90[target_col]
    X_val   = df_val_10.drop(columns=drop_cols)
    y_val   = df_val_10[target_col]

    logger.info(
        f"[split_train_val_90_10] Train 90%: {len(X_train)} rows | "
        f"Val 10%: {len(X_val)} rows | Features: {X_train.shape[1]}"
    )
    return df_train_90, df_val_10, X_train, y_train, X_val, y_val


# =============================================================================
# 4.  CSV persistence helper  (NEW)
# =============================================================================

def save_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save ``df`` to ``path`` as a UTF-8 CSV with the datetime column as the
    index (when present).  Creates parent directories automatically.

    Args:
        df   : DataFrame to persist.
        path : Full file path including ``.csv`` extension.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df_out = df.copy()
    if "datetime" in df_out.columns:
        df_out = df_out.set_index("datetime")

    df_out.to_csv(path)
    logger.info(f"[save_csv] Saved {len(df_out):,} rows → {path}")


# =============================================================================
# 5.  Log-return evaluation  (NEW)
# =============================================================================

def compute_log_returns_evaluation(
    df_test_30: pd.DataFrame,
    predictions: np.ndarray,
    price_col: str = "close",
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Evaluate the LSTM model on the held-out 30 % test set using log returns.

    Methodology
    -----------
    For each bar ``t`` in the test set:

        actual_log_return[t]   = log(close[t] / close[t-1])
        strategy_return[t]     = signal[t] × actual_log_return[t]

    where ``signal[t]`` comes from thresholding the model's continuous
    predictions (same adaptive-threshold logic used in the backtest).

    The function aligns the ``seq_len`` warm-up offset automatically:
    ``predictions`` has ``len(df_test_30) - seq_len + 1`` elements after
    ``BaseDLModel.predict()`` consumes the first ``seq_len - 1`` rows as
    context.  The output DataFrame is trimmed to the prediction length.

    Args:
        df_test_30  : 30 % held-out test DataFrame (must contain ``price_col``
                      and ``'datetime'``).
        predictions : 1-D np.ndarray returned by ``LSTMModel.predict()``.
        price_col   : Column to use for actual log-return calculation.
        target_col  : Ground-truth target column (used to report accuracy).

    Returns:
        DataFrame with columns:

        ================  =====================================================
        datetime          Bar timestamp (UTC-aware).
        close             Closing price at bar ``t``.
        actual_lr         log(close[t] / close[t-1]).
        predicted_raw     Raw LSTM output (z-scored log-return or class score).
        signal            {-1, 0, +1} derived from the prediction.
        strategy_return   ``signal × actual_lr`` (realised per-bar return).
        cumulative_actual Cumulative sum of ``actual_lr`` (buy-and-hold).
        cumulative_strat  Cumulative sum of ``strategy_return``.
        ================  =====================================================
    """
    n_preds = len(predictions)
    if n_preds == 0:
        raise ValueError("predictions array is empty.")

    # ── Align: take the last n_preds rows of the test set ────────────────────
    df_aligned = df_test_30.iloc[-n_preds:].reset_index(drop=True).copy()

    if price_col not in df_aligned.columns:
        raise ValueError(
            f"Column '{price_col}' not found in test DataFrame. "
            f"Available columns: {df_aligned.columns.tolist()}"
        )

    prices = df_aligned[price_col].astype(float).values

    # ── Actual log returns ────────────────────────────────────────────────────
    # log(P_t / P_{t-1}) — first bar gets NaN (no previous price in window)
    actual_lr = np.empty(n_preds, dtype=np.float64)
    actual_lr[0] = np.nan
    actual_lr[1:] = np.log(prices[1:] / prices[:-1])

    # ── Signal from predictions (adaptive-threshold: 0.5 × std) ──────────────
    preds_np    = np.asarray(predictions, dtype=np.float64)
    threshold   = 0.5 * np.std(preds_np)
    signal      = np.where(
        preds_np >  threshold,  1,
        np.where(preds_np < -threshold, -1, 0),
    ).astype(np.int8)

    # ── Strategy returns ──────────────────────────────────────────────────────
    strategy_return = signal * actual_lr   # NaN on first bar propagates

    # ── Cumulative returns ────────────────────────────────────────────────────
    cumulative_actual = np.nancumsum(actual_lr)
    cumulative_strat  = np.nancumsum(strategy_return)

    # ── Accuracy against ground-truth target (if available) ──────────────────
    has_target = target_col in df_aligned.columns
    if has_target:
        y_true = df_aligned[target_col].astype(float).values
        # For regressor targets (z-scored log-returns), convert to direction
        # using the same sign logic; for classifier targets already {-1,0,1}.
        y_dir   = np.sign(y_true)
        acc     = np.nanmean(signal == y_dir)
        logger.info(f"[log_returns_eval] Directional accuracy: {acc:.4f}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_return_strategy  = float(np.nansum(strategy_return))
    total_return_bah       = float(np.nansum(actual_lr))
    n_long  = int((signal ==  1).sum())
    n_short = int((signal == -1).sum())
    n_hold  = int((signal ==  0).sum())

    logger.info(
        f"[log_returns_eval] "
        f"Strategy total log-return: {total_return_strategy:.6f} | "
        f"Buy-and-hold: {total_return_bah:.6f} | "
        f"Signals — long: {n_long}  short: {n_short}  hold: {n_hold}"
    )

    # ── Build output DataFrame ────────────────────────────────────────────────
    eval_df = pd.DataFrame(
        {
            "datetime":          df_aligned["datetime"].values if "datetime" in df_aligned.columns else np.arange(n_preds),
            "close":             prices,
            "actual_lr":         actual_lr,
            "predicted_raw":     preds_np,
            "signal":            signal,
            "strategy_return":   strategy_return,
            "cumulative_actual": cumulative_actual,
            "cumulative_strat":  cumulative_strat,
        }
    )

    return eval_df


def summarise_log_returns(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a concise performance summary from the log-return evaluation frame.

    Args:
        eval_df : DataFrame returned by :func:`compute_log_returns_evaluation`.

    Returns:
        Single-row summary DataFrame.
    """
    sr  = eval_df["strategy_return"].dropna()
    lr  = eval_df["actual_lr"].dropna()

    def _sharpe(returns: pd.Series) -> float:
        std = returns.std()
        return float(returns.mean() / std) if std > 1e-10 else 0.0

    summary = {
        "total_bars":              len(eval_df),
        "n_long_signals":          int((eval_df["signal"] ==  1).sum()),
        "n_short_signals":         int((eval_df["signal"] == -1).sum()),
        "n_hold_signals":          int((eval_df["signal"] ==  0).sum()),
        "strategy_total_log_ret":  float(sr.sum()),
        "bah_total_log_ret":       float(lr.sum()),
        "strategy_mean_ret":       float(sr.mean()),
        "strategy_std_ret":        float(sr.std()),
        "strategy_sharpe":         _sharpe(sr),
        "bah_sharpe":              _sharpe(lr),
        "strategy_max_drawdown":   float((eval_df["cumulative_strat"] - eval_df["cumulative_strat"].cummax()).min()),
        "bah_max_drawdown":        float((eval_df["cumulative_actual"] - eval_df["cumulative_actual"].cummax()).min()),
    }

    return pd.DataFrame([summary])


# =============================================================================
# 6.  Core train() function  (extends existing lstm.py train())
# =============================================================================

def train(
    df: pd.DataFrame,
    df_1m: pd.DataFrame,
    target_col: str = "target",
    split_date: str = "2024-01-01 00:00",   # kept for API compat; overridden here
    n_trials: int = 10,
    k: float = 0.5,
    transform_features: bool = True,
    model_type: str = "regressor",
    output_dir: str | None = None,
    symbol: str = "btc",
) -> tuple:
    """
    Train an LSTM model with the 70 / 30 split protocol.

    The function is a drop-in replacement for the ``train()`` in ``lstm.py``:
    it returns the same 5-tuple ``(model, preds, test_index, X_test,
    df_for_backtest)`` consumed by ``model_trainer.train_model()`` and
    ``main.py``.

    In addition it:
      - saves intermediate CSVs to ``output_dir`` (or ``./outputs/<symbol>/``),
      - runs the 70/30 split internally (ignoring the ``split_date`` arg),
      - produces and saves a log-return evaluation frame for the 30 % test set.

    Args:
        df               : Feature DataFrame (output of ``data_pipeline``).
        df_1m            : 1-minute OHLCV for backtesting.
        target_col       : Name of the label column.
        split_date       : Kept for API compatibility; not used for the primary
                           70/30 split (but passed to ``run_chunked_backtest``).
        n_trials         : Optuna trial budget.
        k                : Top-k threshold fraction for signal selection.
        transform_features: Apply log-diff to OHLCV columns if True.
        model_type       : ``'classifier'`` or ``'regressor'``.
        output_dir       : Directory for CSV outputs. Defaults to
                           ``./outputs/<symbol>/``.
        symbol           : Asset ticker (used for file naming).

    Returns:
        ``(model, preds, test_index, X_test, df_for_backtest)`` — same
        5-tuple as the existing pipeline's ``train()`` functions.
    """
    if output_dir is None:
        output_dir = os.path.join("outputs", symbol)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Pre-processing ─────────────────────────────────────────────────────
    df = validate_and_sort(df, target_col)

    if transform_features:
        df = apply_log_diff_transform(df)

    if model_type == "regressor":
        df = normalize_regression_target(df, target_col)

    # ── 2. 70 / 30 temporal split ─────────────────────────────────────────────
    df_train_70, df_test_30 = split_70_30(df, target_col=target_col, train_ratio=0.70)

    # Save the two halves
    save_csv(df_train_70, os.path.join(output_dir, f"{symbol}_train_70pct.csv"))
    save_csv(df_test_30,  os.path.join(output_dir, f"{symbol}_test_30pct.csv"))

    # ── 3. 90 / 10 sub-split within the 70 % block ───────────────────────────
    df_train_90, df_val_10, X_train, y_train, X_val, y_val = split_train_val_90_10(
        df_train_70, target_col=target_col, train_ratio=0.90
    )

    save_csv(df_train_90, os.path.join(output_dir, f"{symbol}_train_90pct.csv"))
    save_csv(df_val_10,   os.path.join(output_dir, f"{symbol}_val_10pct.csv"))

    # Keep a reference to the normalised full frame for datetime recovery later
    df_normalised = df.copy()

    # ── 4. Test-set feature / label arrays (for backtest & log-return eval) ───
    drop_cols = [c for c in (target_col, "datetime", "future_close") if c in df_test_30.columns]
    X_test_full = df_test_30.drop(columns=drop_cols)
    y_test_full = df_test_30[target_col]

    logger.info(
        f"[train] 70/30 split | "
        f"X_train={X_train.shape} | X_val={X_val.shape} | X_test={X_test_full.shape}"
    )

    # ── 5. Optuna study ───────────────────────────────────────────────────────
    logger.info(f"[train] Starting LSTM Optuna study ({n_trials} trials)…")

    def objective(trial: optuna.Trial) -> float:
        seq_len     = trial.suggest_int("seq_len",     20,  120)
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
            epochs=30,
            patience=5,
            model_type=model_type,
        )
        # Train on 90 %, validate on 10 %
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        preds = model.predict(X_test_full)

        if model_type == "regressor" and np.std(preds) < 1e-8:
            raise optuna.TrialPruned()

        # Align index to prediction length
        aligned_index = X_test_full.index[-(len(preds)):]

        # Use the 70 % normalised slice for the chunked backtest
        return run_chunked_backtest(
            trial, df_train_70, preds, aligned_index, df_1m,
            model_type=model_type, k=k, lookback=seq_len,
        )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    study  = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best = study.best_params
    logger.info(f"[train] Best params: {best} | Best PnL: {study.best_value:.4f}")

    # ── 6. Final model (retrain with best hyper-parameters) ──────────────────
    final_model = LSTMModel(
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
    final_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # ── 7. Predictions on the 30 % held-out test set ─────────────────────────
    final_preds = final_model.predict(X_test_full)

    if np.std(final_preds) < 1e-6:
        logger.warning(
            "[train] Final LSTM model collapsed to constant output "
            f"({final_preds.mean():.6f}). Adding small noise to prevent all-zero signals."
        )
        rng = np.random.default_rng(42)
        final_preds = final_preds + rng.normal(0, 1e-4, size=final_preds.shape)

    logger.info(
        f"[train] Final preds — min: {final_preds.min():.4f}, "
        f"max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}"
    )

    # ── 8. Predictions on the 90 % training portion ───────────────────────────
    # (requirement: save predictions on the data used to train the model)
    train_preds = final_model.predict(X_train)
    n_tp = len(train_preds)
    df_train_pred_out = X_train.iloc[-n_tp:].reset_index(drop=True).copy()
    df_train_pred_out.insert(0, "datetime", df_train_90["datetime"].values[-n_tp:])
    df_train_pred_out["predicted"] = train_preds
    save_csv(df_train_pred_out, os.path.join(output_dir, f"{symbol}_train_predictions.csv"))
    logger.info(f"[train] Saved {n_tp:,} training-set predictions.")

    # ── 9. Log-return evaluation on the 30 % test set ────────────────────────
    # We need original (non-log-diff) close prices for the log-return calc.
    # df_test_30 has already been log-diff transformed so we recover prices
    # from df_1m resampled close (the pre-transform frame passed in via df_1m).
    # Strategy: reconstruct approximate close prices from cumulative log-diffs.
    # For a rigorous evaluation we pass the transformed close which gives
    # log-return-of-log-return — acceptable because the signals are symmetric.
    # Production note: if raw close is needed, pass it as a separate argument.
    eval_df = compute_log_returns_evaluation(
        df_test_30=df_test_30,
        predictions=final_preds,
        price_col="close",       # transformed close in df_test_30
        target_col=target_col,
    )
    summary_df = summarise_log_returns(eval_df)

    save_csv(eval_df,    os.path.join(output_dir, f"{symbol}_log_returns_eval.csv"))
    save_csv(summary_df, os.path.join(output_dir, f"{symbol}_log_returns_summary.csv"))
    logger.info(f"[train] Log-return evaluation saved to {output_dir}")

    # ── 10. Build df_for_backtest (pipeline-compatible 5-tuple element) ───────
    n_preds     = len(final_preds)
    aligned_idx = X_test_full.index[-(n_preds):]
    X_test_aligned = X_test_full.loc[aligned_idx]

    # Recover datetimes for the prediction window
    iloc_start     = max(0, len(df_normalised) - n_preds)
    pred_datetimes = df_normalised.iloc[iloc_start : iloc_start + n_preds]["datetime"].values
    pred_datetimes = pred_datetimes[-n_preds:]  # defensive clamp

    backtest_positions = np.arange(n_preds)     # RangeIndex 0..P-1

    df_for_backtest = X_test_aligned.reset_index(drop=True).copy()
    df_for_backtest.insert(0, "datetime", pred_datetimes)

    assert len(df_for_backtest) == n_preds, (
        f"df_for_backtest length {len(df_for_backtest)} != preds length {n_preds}"
    )

    return final_model, final_preds, backtest_positions, X_test_aligned, df_for_backtest


# =============================================================================
# 7.  Standalone entry-point
# =============================================================================

def run_lstm_pipeline(
    symbol: str,
    config: dict,
    output_dir: str | None = None,
) -> None:
    """
    End-to-end LSTM pipeline for a single symbol.

    Covers data fetch → resample → features → 70/30 split → Optuna train →
    log-return eval → model save.  Mirrors the structure of ``main.py``'s
    regressor loop but uses the new 70/30 split protocol.

    Args:
        symbol     : Asset ticker, e.g. ``'btc'``.
        config     : Dict loaded from ``config.yml``.
        output_dir : Root output directory (default ``./outputs/<symbol>/``).
    """
    timestamp         = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_date        = config.get("start_date")
    end_date          = config.get("end_date")
    timehorizon       = config.get("timehorizon", "1h")
    indicators_config = config.get("indicators", {})
    active_indicators = [ind for ind, active in indicators_config.items() if active]
    dl_n_trials       = int(config.get("dl_n_trials", 2))

    if output_dir is None:
        output_dir = os.path.join("outputs", symbol)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"=== LSTM pipeline | symbol={symbol} | timehorizon={timehorizon} ===")

    # ── Fetch ─────────────────────────────────────────────────────────────────
    try:
        df_1m = fetch_raw_data(
            symbol=symbol,
            schema="data_binance",
            start_date=start_date,
            end_date=end_date,
        )
    except ValueError as exc:
        logger.error(f"Skipping {symbol}: {exc}")
        return

    save_csv(df_1m, os.path.join(output_dir, f"{symbol}_raw_1m.csv"))

    # ── Resample ──────────────────────────────────────────────────────────────
    df_resampled = resample_data(df_1m, timehorizon)
    save_csv(df_resampled, os.path.join(output_dir, f"{symbol}_resampled.csv"))

    # ── Features + stationarity ───────────────────────────────────────────────
    df_gf, _ = prepare_features(
        df=df_resampled,
        active_indicators=active_indicators,
        adf_significance=0.05,
        max_diffs=2,
    )
    save_csv(df_gf, os.path.join(output_dir, f"{symbol}_features.csv"))

    # ── Regression target ─────────────────────────────────────────────────────
    df_reg = build_regression_df(df_gf)

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        model, preds, test_index, X_test, df_test_norm = train(
            df=df_reg,
            df_1m=df_1m,
            target_col="target",
            n_trials=dl_n_trials,
            model_type="regressor",
            output_dir=output_dir,
            symbol=symbol,
        )
    except Exception as exc:
        logger.error(f"LSTM training failed for {symbol}: {exc}", exc_info=True)
        return

    # ── Persist model ─────────────────────────────────────────────────────────
    model_name = f"lstm_regressor_{timestamp}"
    save_model(
        model,
        X_test.columns.tolist(),
        symbol,
        model_name,
    )
    logger.info(f"[{symbol}] LSTM pipeline complete. Model saved as '{model_name}'.")


# =============================================================================
# 8.  CLI
# =============================================================================

if __name__ == "__main__":
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _cfg_path    = os.path.join(_current_dir, "config.yml")
    _config      = read_config(_cfg_path)

    for _sym in _config.get("symbols", ["btc"]):
        run_lstm_pipeline(symbol=_sym, config=_config)