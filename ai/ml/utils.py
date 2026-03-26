from __future__ import annotations
import pandas as pd
import numpy as np
from TradeX.backtest.backtest import BackTest


def compute_trade_statistics(ledger: pd.DataFrame) -> pd.DataFrame:

    df = ledger  
    stats = {}

    # ------------------------
    # Basic Counts
    # ------------------------
    stats["total_trades"] = len(df)
    stats["long_trades"] = (df["predicted_direction"] == "long").sum()
    stats["short_trades"] = (df["predicted_direction"] == "short").sum()

    # ------------------------
    # Win / Loss
    # ------------------------
    stats["win_trades"] = (df["pnl"] > 0).sum()
    stats["loss_trades"] = (df["pnl"] < 0).sum()
    stats["breakeven_trades"] = (df["pnl"] == 0).sum()

    stats["win_rate"] = stats["win_trades"] / stats["total_trades"] if stats["total_trades"] else 0
    stats["loss_rate"] = stats["loss_trades"] / stats["total_trades"] if stats["total_trades"] else 0

    # ------------------------
    # Profit / Loss
    # ------------------------
    gross_profit = df.loc[df["pnl"] > 0, "pnl"].sum()
    gross_loss = df.loc[df["pnl"] < 0, "pnl"].sum()

    stats["gross_profit"] = gross_profit
    stats["gross_loss"] = gross_loss
    stats["net_profit"] = df["pnl"].sum()

    # ------------------------
    # Averages
    # ------------------------
    stats["avg_trade_pnl"] = df["pnl"].mean()

    avg_win = df.loc[df["pnl"] > 0, "pnl"].mean()
    avg_loss = df.loc[df["pnl"] < 0, "pnl"].mean()

    stats["avg_win"] = avg_win
    stats["avg_loss"] = avg_loss

    # ------------------------
    # Risk Ratios
    # ------------------------
    stats["risk_reward_ratio"] = abs(avg_win / avg_loss) if avg_loss and not np.isnan(avg_loss) else np.nan
    stats["profit_factor"] = gross_profit / abs(gross_loss) if gross_loss != 0 else np.nan

    # ------------------------
    # Drawdown
    # ------------------------
    equity = df["balance"]

    rolling_max = equity.cummax()
    drawdown = equity - rolling_max

    stats["max_drawdown"] = drawdown.min()
    stats["max_drawdown_pct"] = (drawdown / rolling_max).min()

    # ------------------------
    # Sharpe Ratio
    # ------------------------
    returns = df["pnl"]

    if returns.std() and not np.isnan(returns.std()):
        stats["sharpe_ratio"] = returns.mean() / returns.std()
    else:
        stats["sharpe_ratio"] = 0

    # ------------------------
    # Sortino Ratio
    # ------------------------
    downside = returns[returns < 0]

    if downside.std() and not np.isnan(downside.std()):
        stats["sortino_ratio"] = returns.mean() / downside.std()
    else:
        stats["sortino_ratio"] = 0

    # ------------------------
    # Consecutive Wins / Losses
    # ------------------------
    wins = df["pnl"] > 0
    losses = df["pnl"] < 0

    win_streak = wins.astype(int).groupby((wins != wins.shift()).cumsum()).cumsum()
    loss_streak = losses.astype(int).groupby((losses != losses.shift()).cumsum()).cumsum()

    stats["max_consecutive_wins"] = win_streak.max()
    stats["max_consecutive_losses"] = loss_streak.max()

    # ------------------------
    # Return DataFrame
    # ------------------------
    stats_df = pd.DataFrame([stats])

    return stats_df

def _is_log_return(preds_np: np.ndarray) -> bool:
    """
    Heuristic: if the median absolute prediction is < 0.05 the model is
    predicting log-returns (order ~1e-4 to 1e-2), not price levels (~1e3+).
    """
    return float(np.median(np.abs(preds_np))) < 0.05
 
 
def prepare_predictions(
    df,
    preds,
    test_index,
    model_type: str,
    threshold=None,
    k: float = 0.5,
    lookback: int | None = None,
    last_train_value=None,
) -> pd.DataFrame:
    """
    Prepare a predictions DataFrame for backtesting.

    Parameters
    ----------
    df : pd.DataFrame
        Original price DataFrame with a 'datetime' column or DatetimeIndex.
    preds : np.ndarray | darts.TimeSeries
        Model predictions.
    test_index : array-like
        Integer positions of the test rows in df.
    model_type : str
        ``'classifier'``, ``'regressor'``, ``'dl'`` (LSTM-style, requires
        lookback), or ``'dl_darts'`` (Darts models — no lookback needed).
    threshold : float or None
        Signal threshold.  For log_return models this is the dead-band
        (|pred| must exceed threshold to generate a trade).
        If None, derived as ``k * std(preds_np)``.

        BUG-FIX: a caller-supplied threshold is now sanity-checked against the
        actual prediction scale.  If the threshold exceeds the 90th-percentile
        of |preds_np| (meaning it would silence >90 % of signals), the
        threshold is automatically relaxed to ``k * std(preds_np)`` and a
        warning is logged.  This prevents the common case where a threshold
        calibrated for one model (e.g. Transformer, pred std ~3e-4) is passed
        unchanged to a model whose predictions are 10-30× smaller (e.g. ARIMA,
        pred std ~1e-5).
    k : float
        Std multiplier when threshold is derived automatically.
    lookback : int or None
        LSTM warm-up steps (only used for ``model_type='dl'``).
    last_train_value : float or None
        Last training value for inverse log-diff (reserved, unused here).

    Returns
    -------
    pd.DataFrame
        Columns: ['datetime', 'signals']  where signals ∈ {-1, 0, 1}.

    Bugs fixed in this version
    --------------------------
    BUG-1 (ARIMA all-zero signals):
        signal_threshold=3e-4 is calibrated for Transformer/NBEATS whose
        log_return predictions have std ~3e-4.  ARIMA log_return predictions
        have std ~1e-5 — 10-30× smaller — so the fixed threshold silenced
        every single signal.  Fix: auto-relax the threshold when it exceeds
        the 90th-percentile of |preds_np|.

    BUG-2 (VARIMA wrong column extracted):
        VARIMA jointly models [open_lr, high_lr, low_lr, close_lr] so
        preds.values() has shape (n_steps, 4).  The old ``ravel()`` call
        flattened all four columns into one vector, mixing open/high/low
        log-returns into the signal computation.  Fix: for multivariate
        Darts predictions, extract the last component column (close_lr,
        index -1) before ravel().

    BUG-3 (debug print statements left in production code):
        Removed all ``print()`` calls that were leaking prediction values
        and DataFrames to stdout.
    """

    # ------------------------------------------------------------------
    # Normalise datetime column to UTC-aware (guard against re-localising)
    # ------------------------------------------------------------------
    df = df.copy()  # never mutate caller's frame
    if "datetime" in df.columns:
        dt_col = pd.to_datetime(df["datetime"])
        if dt_col.dt.tz is None:
            dt_col = dt_col.dt.tz_localize("UTC")
        df["datetime"] = dt_col
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "datetime"})

    # ------------------------------------------------------------------
    # Branch: classifier
    # ------------------------------------------------------------------
    if model_type == "classifier":
        upper = preds.mean() + 0.25 * preds.std()
        lower = preds.mean() - 0.25 * preds.std()
        signals = np.where(preds > upper, 1, np.where(preds < lower, -1, 0))

    # ------------------------------------------------------------------
    # Branch: regressor
    # ------------------------------------------------------------------
    elif model_type == "regressor":
        if threshold is None:
            preds_series = pd.Series(preds)
            rolling_mean = preds_series.rolling(window=20, min_periods=1).mean()
            rolling_std  = preds_series.rolling(window=20, min_periods=1).std().fillna(0)
            rolling_std  = rolling_std.replace(0, 1e-8)
            threshold    = k * rolling_std
            signals = np.where(preds_series > rolling_mean + threshold,  1,
                       np.where(preds_series < rolling_mean - threshold, -1, 0))

    # ------------------------------------------------------------------
    # Branch: dl_darts  (ARIMA / VARIMA / NBEATS / Transformer via Darts)
    # ------------------------------------------------------------------
    elif model_type == "dl_darts":
        # ── BUG-2 FIX: extract numpy array, handling multivariate output ──
        # VARIMA predicts [open_lr, high_lr, low_lr, close_lr] jointly.
        # preds.values() returns shape (n_steps, n_components).
        # We want only the LAST component (close_lr) for signal generation.
        # Univariate models (ARIMA, NBEATS, Transformer) have n_components=1
        # so [:, -1] is equivalent to ravel() for them — no behaviour change.
        if hasattr(preds, "values"):
            raw = preds.values()               # shape: (n_steps,) or (n_steps, n_components)
            if raw.ndim == 2:
                # Multivariate (VARIMA): take last column = close_lr
                preds_np = raw[:, -1].astype(np.float64)
            else:
                preds_np = raw.ravel().astype(np.float64)
        else:
            preds_np = np.asarray(preds, dtype=np.float64).ravel()

        # Align lengths
        test_index = np.asarray(test_index)
        min_len    = min(len(preds_np), len(test_index))
        preds_np   = preds_np[:min_len]
        test_index = test_index[:min_len]

        if _is_log_return(preds_np):
            # -----------------------------------------------------------------
            # LOG-RETURN PATH
            #
            # preds_np values are log_returns, e.g. +0.0005 means +0.05% move.
            # Signal direction comes directly from the sign of the prediction.
            #   pred > +threshold  → 1  (BUY)
            #   pred < -threshold  → -1 (SELL)
            #   |pred| <= threshold → 0  (NO TRADE — within dead-band)
            #
            # ── BUG-1 FIX: adaptive threshold sanity-check ──────────────────
            # Different models produce predictions at very different scales:
            #   ARIMA       : std ~1e-5  (pure AR on log_return)
            #   VARIMA      : std ~1e-5  (VAR on OHLC log_returns)
            #   NBEATS      : std ~1e-4  (deep network, more expressive)
            #   Transformer : std ~3e-4  (largest model, widest range)
            #
            # A fixed threshold=3e-4 (calibrated for Transformer) silences
            # ALL ARIMA/VARIMA signals because their predictions never reach
            # that magnitude.  We auto-relax to k*std when the fixed threshold
            # would suppress more than 90% of predictions.
            # -----------------------------------------------------------------
            pred_abs   = np.abs(preds_np)
            auto_thresh = k * np.std(preds_np)

            if threshold is None:
                effective_threshold = auto_thresh
            else:
                # Check: would this threshold silence >90% of predictions?
                pct_silenced = np.mean(pred_abs <= threshold)
                if pct_silenced > 0.90:
                    import warnings
                    warnings.warn(
                        f"prepare_predictions: supplied threshold={threshold:.2e} "
                        f"would silence {pct_silenced*100:.1f}% of predictions "
                        f"(pred std={np.std(preds_np):.2e}). "
                        f"Auto-relaxing to k*std={auto_thresh:.2e}. "
                        f"Consider lowering signal_threshold for this model.",
                        UserWarning,
                        stacklevel=2,
                    )
                    effective_threshold = auto_thresh
                else:
                    effective_threshold = threshold

            signals = np.where(
                preds_np >  effective_threshold,  1,
                np.where(
                preds_np < -effective_threshold, -1, 0)
            )

        else:
            # -----------------------------------------------------------------
            # PRICE LEVEL PATH (legacy: models predicting raw close values)
            # -----------------------------------------------------------------
            if "close" in df.columns:
                actual = df.iloc[test_index]["close"].values.astype(np.float64)
                errors = preds_np - actual
                if threshold is None:
                    threshold = k * np.std(errors)
                signals = np.where(errors >  threshold,  1,
                          np.where(errors < -threshold, -1, 0))
            else:
                if threshold is None:
                    threshold = k * np.std(preds_np)
                signals = np.where(preds_np >  threshold,  1,
                          np.where(preds_np < -threshold, -1, 0))

    # ------------------------------------------------------------------
    # Branch: dl  (legacy LSTM / sequence models — lookback required)
    # ------------------------------------------------------------------
    elif model_type == "dl":
        if lookback is None:
            raise ValueError(
                "lookback must be provided for model_type='dl' (LSTM-style). "
                "For Darts models use model_type='dl_darts'."
            )

        test_index_aligned = np.asarray(test_index)[lookback:]
        preds_aligned      = np.asarray(preds, dtype=np.float64).ravel()
        preds_aligned      = preds_aligned[:len(test_index_aligned)]

        if "close" in df.columns:
            actual = df.iloc[test_index_aligned]["close"].values.astype(np.float64)
            errors = preds_aligned - actual
            if threshold is None:
                threshold = k * np.std(errors)
        else:
            errors    = preds_aligned
            threshold = threshold or k * np.std(preds_aligned)

        signals    = np.where(errors >  threshold,  1,
                              np.where(errors < -threshold, -1, 0))
        test_index = test_index_aligned

    else:
        raise ValueError(
            f"model_type must be one of: 'classifier', 'regressor', "
            f"'dl', 'dl_darts'.  Got: '{model_type}'"
        )

    # ------------------------------------------------------------------
    # Build output DataFrame — columns always: ['datetime', 'signals']
    # ------------------------------------------------------------------
    datetimes = df.iloc[np.asarray(test_index)]["datetime"].values

    df_predictions = pd.DataFrame({
        "datetime": datetimes,
        "signals":  signals,
    })

    return df_predictions

def pnl_permutation_importance(
    model,
    X_test,
    df,
    df_1m,
    base_pnl,
    model_type="classifier",
    k=0.5,
    threshold=None,
    n_repeats=3
):

    results = []


    for col in X_test.columns:
        pnl_scores = []

        for _ in range(n_repeats):
            X_perm = X_test  
            X_perm[col] = np.random.permutation(X_perm[col].values)

            # ----------------------------
            # Generate predictions
            # ----------------------------
            if model_type == "classifier":
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_perm)[:, 1]
                else:
                    preds = model.predict(X_perm)
            else:
                preds = model.predict(X_perm)

            # ----------------------------
            # Convert → trades
            # ----------------------------
            df_preds = prepare_predictions(
                df,
                preds,
                X_perm.index,
                model_type=model_type,
                threshold=threshold,
                k=k
            )

            df_preds["datetime"] = pd.to_datetime(
                df_preds["datetime"], utc=True
            )

            bt = BackTest(
                df_1m,
                df_preds,
                take_profit=3,
                stop_loss=1
            )
            _, _, pnl = bt.run()

            pnl_scores.append(pnl)

        pnl_drop = base_pnl - np.mean(pnl_scores)

        results.append({
            "feature": col,
            "pnl_drop": pnl_drop
        })

    return pd.DataFrame(results).sort_values("pnl_drop", ascending=False)

def extract_important_features(pnl_importance_wide: pd.DataFrame, model_name: str):
    # Drop pnl column
    feature_row = pnl_importance_wide.drop(columns=["pnl"], errors="ignore").iloc[0]

    # Keep only features with value > 0
    important_features = feature_row[feature_row > 0].index.tolist()

    return pd.DataFrame({
        "model_name": [model_name],
        "important_features": [important_features]
    })