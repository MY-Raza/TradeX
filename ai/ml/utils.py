from __future__ import annotations
import pandas as pd
import numpy as np
from TradeX.backtest.backtest import BackTest


def compute_trade_statistics(ledger: pd.DataFrame) -> pd.DataFrame:

    df = ledger.copy()
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
        Signal threshold.  If None, derived as ``k * std(...)``.
    k : float
        Std multiplier when threshold is None.
    lookback : int or None
        LSTM warm-up steps (only used for ``model_type='dl'``).
    last_train_value : float or None
        Last training value for inverse log-diff (reserved, unused here).

    Returns
    -------
    pd.DataFrame
        Columns: ['datetime', 'signals']  where signals ∈ {-1, 0, 1}.
    """

    # ------------------------------------------------------------------
    # Normalise datetime column to UTC-aware (guard against re-localising)
    # ------------------------------------------------------------------
       # never mutate caller's frame
    if "datetime" in df.columns:
        dt_col = pd.to_datetime(df["datetime"])
        if dt_col.dt.tz is None:
            dt_col = dt_col.dt.tz_localize("UTC")
        df["datetime"] = dt_col
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.reset_index()          # bring DatetimeIndex → "datetime" column
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
            rolling_std = preds_series.rolling(window=20, min_periods=1).std().fillna(0)
    
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, 1e-8)
    
            threshold = k * rolling_std
    
            signals = np.where(preds_series > rolling_mean + threshold, 1,
                       np.where(preds_series < rolling_mean - threshold, -1, 0))

    # ------------------------------------------------------------------
    # Branch: dl_darts  (ARIMA / VARIMA / NBEATS / Transformer via Darts)
    # Darts predicts the full test window — no lookback warm-up gap.
    # ------------------------------------------------------------------
    elif model_type == "dl_darts":
        # Extract numpy array from Darts TimeSeries if needed
        if hasattr(preds, "values"):
            preds_np = preds.values().ravel().astype(np.float64)
        else:
            preds_np = np.asarray(preds, dtype=np.float64).ravel()

        # test_index and preds must be the same length
        test_index = np.asarray(test_index)
        min_len    = min(len(preds_np), len(test_index))
        preds_np   = preds_np[:min_len]
        test_index = test_index[:min_len]

        if "close" in df.columns:
            actual = df.iloc[test_index]["close"].values.astype(np.float64)
            errors = preds_np - actual
            if threshold is None:
                threshold = k * np.std(errors)
            signals = np.where(errors > threshold, 1,
                               np.where(errors < -threshold, -1, 0))
        else:
            if threshold is None:
                threshold = k * np.std(preds_np)
            signals = np.where(preds_np > threshold, 1,
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

        signals    = np.where(errors > threshold, 1,
                              np.where(errors < -threshold, -1, 0))
        test_index = test_index_aligned

    else:
        raise ValueError(
            f"model_type must be one of: 'classifier', 'regressor', "
            f"'dl', 'dl_darts'.  Got: '{model_type}'"
        )

    # ------------------------------------------------------------------
    # Build output DataFrame using iloc (safe for any index type)
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
            X_perm = X_test.copy()
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