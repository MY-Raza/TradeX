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


def prepare_predictions(df, preds, test_index, model_type, threshold=None, k=0.5):
    """
    Prepare a predictions DataFrame for backtesting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original price DataFrame with 'datetime' column.
        
    preds : np.ndarray
        Model predictions (classifier or regressor outputs)
        
    test_index : array-like
        Indices of the test set in df
        
    model_type : str
        'classifier' or 'regressor'
        
    threshold : float or None
        If None and model_type='regressor', automatically computed as k*std(preds)
        
    k : float
        Multiplier for standard deviation when auto thresholding (default 0.5)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'signals'] with -1, 0, 1 signals
    """
    
    # Ensure datetime is UTC-aware
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    if model_type == "classifier":
        # Convert probability into 3 trading zones
        upper = 0.55
        lower = 0.45

        signals = np.where(preds > upper, 1,
              np.where(preds < lower, -1, 0))

    elif model_type == "regressor":
        # Auto-compute threshold if not provided
        if threshold is None:
            threshold = k * np.std(preds)
        # Convert continuous predictions into discrete signals
        signals = np.where(preds > threshold, 1,
                  np.where(preds < -threshold, -1, 0))
        

    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    # Build prediction DataFrame
    df_predictions = pd.DataFrame({
        "datetime": df.loc[test_index, "datetime"].values,
        "signals": signals
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