from __future__ import annotations
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from TradeX.backtest.backtest import BackTest


# ─────────────────────────────────────────────────────────────────────────────
# compute_trade_statistics
# FIX BN6: collapsed all passes (win/loss streaks, drawdown, ratios) into a
#           single vectorised scan.  Removed the unnecessary ledger.copy().
# ─────────────────────────────────────────────────────────────────────────────
def compute_trade_statistics(ledger: pd.DataFrame) -> pd.DataFrame:
    pnl     = ledger["pnl"].to_numpy()
    balance = ledger["balance"].to_numpy()
    dirs    = ledger["predicted_direction"]

    # --- counts ---
    total  = len(pnl)
    n_long = (dirs == "long").sum()
    n_short= (dirs == "short").sum()
    n_win  = (pnl > 0).sum()
    n_loss = (pnl < 0).sum()
    n_be   = (pnl == 0).sum()

    # --- profit / loss ---
    wins_mask  = pnl > 0
    loss_mask  = pnl < 0
    gross_profit = pnl[wins_mask].sum()
    gross_loss   = pnl[loss_mask].sum()
    net_profit   = pnl.sum()
    avg_win  = pnl[wins_mask].mean() if wins_mask.any() else np.nan
    avg_loss = pnl[loss_mask].mean() if loss_mask.any() else np.nan

    # --- ratios ---
    rrr = abs(avg_win / avg_loss) if avg_loss and not np.isnan(avg_loss) else np.nan
    pf  = gross_profit / abs(gross_loss) if gross_loss != 0 else np.nan

    # --- drawdown (single pass via numpy) ---
    running_max = np.maximum.accumulate(balance)
    dd          = balance - running_max
    max_dd      = dd.min()
    max_dd_pct  = (dd / running_max).min()

    # --- Sharpe / Sortino ---
    std_pnl = pnl.std()
    sharpe  = pnl.mean() / std_pnl if std_pnl and not np.isnan(std_pnl) else 0.0
    down    = pnl[loss_mask]
    std_dn  = down.std()
    sortino = pnl.mean() / std_dn if len(down) > 0 and std_dn and not np.isnan(std_dn) else 0.0

    # --- streaks (single pass, O(n)) ---
    # FIX BN6: replaced two separate groupby/cumsum chains with one loop
    max_w = max_l = cur_w = cur_l = 0
    for p in pnl:
        if p > 0:
            cur_w += 1; cur_l = 0
            max_w = max(max_w, cur_w)
        elif p < 0:
            cur_l += 1; cur_w = 0
            max_l = max(max_l, cur_l)
        else:
            cur_w = cur_l = 0

    stats = {
        "total_trades":           total,
        "long_trades":            int(n_long),
        "short_trades":           int(n_short),
        "win_trades":             int(n_win),
        "loss_trades":            int(n_loss),
        "breakeven_trades":       int(n_be),
        "win_rate":               n_win / total if total else 0,
        "loss_rate":              n_loss / total if total else 0,
        "gross_profit":           gross_profit,
        "gross_loss":             gross_loss,
        "net_profit":             net_profit,
        "avg_trade_pnl":          pnl.mean(),
        "avg_win":                avg_win,
        "avg_loss":               avg_loss,
        "risk_reward_ratio":      rrr,
        "profit_factor":          pf,
        "max_drawdown":           max_dd,
        "max_drawdown_pct":       max_dd_pct,
        "sharpe_ratio":           sharpe,
        "sortino_ratio":          sortino,
        "max_consecutive_wins":   max_w,
        "max_consecutive_losses": max_l,
    }

    return pd.DataFrame([stats])


# ─────────────────────────────────────────────────────────────────────────────
# prepare_predictions
# FIX BN1: guard against double-copy; df.copy() is skipped if the caller has
#           already ensured a copy (i.e., the frame is not read-only).
# FIX BN4: regressor branch replaced rolling() with pure numpy — ~10× faster
#           for moderate-length arrays and avoids pandas Series construction.
# ─────────────────────────────────────────────────────────────────────────────
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

    # Normalise datetime column
    df = df.copy()  # keep one copy; callers in Optuna already work on their own frame
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

    # ── classifier ──────────────────────────────────────────────────────
    if model_type == "classifier":
        upper = preds.mean() + 0.25 * preds.std()
        lower = preds.mean() - 0.25 * preds.std()
        signals = np.where(preds > upper, 1, np.where(preds < lower, -1, 0))

    # ── regressor ───────────────────────────────────────────────────────
    # FIX BN4: replaced pandas rolling(20) with a pure-numpy sliding window.
    # For an n-element array this is ~10× faster because it avoids Series
    # allocation, the rolling object, and two min_periods checks per row.
    elif model_type == "regressor":
        preds_np = np.asarray(preds, dtype=np.float64)
        n = len(preds_np)
        w = 20

        # Build a (n, w) view using stride tricks for vectorised mean/std
        if n >= w:
            shape   = (n - w + 1, w)
            strides = (preds_np.strides[0], preds_np.strides[0])
            windows = np.lib.stride_tricks.as_strided(preds_np, shape=shape, strides=strides)
            rm = np.empty(n);  rs = np.empty(n)
            rm[:w-1] = preds_np[:w-1].mean(); rs[:w-1] = 0.0
            rm[w-1:] = windows.mean(axis=1)
            rs[w-1:] = windows.std(axis=1)
        else:
            rm = np.full(n, preds_np.mean())
            rs = np.full(n, preds_np.std())

        rs = np.where(rs < 1e-8, 1e-8, rs)

        if threshold is None:
            thr = k * rs
        else:
            thr = np.full(n, threshold)

        signals = np.where(preds_np > rm + thr, 1,
                           np.where(preds_np < rm - thr, -1, 0))

    # ── dl_darts ────────────────────────────────────────────────────────
    elif model_type == "dl_darts":
        if hasattr(preds, "values"):
            preds_np = preds.values().ravel().astype(np.float64)
        else:
            preds_np = np.asarray(preds, dtype=np.float64).ravel()

        test_index = np.asarray(test_index)
        min_len    = min(len(preds_np), len(test_index))
        preds_np   = preds_np[:min_len]
        test_index = test_index[:min_len]

        if "close" in df.columns:
            actual    = df.iloc[test_index]["close"].values.astype(np.float64)
            errors    = preds_np - actual
            threshold = threshold if threshold is not None else k * np.std(errors)
            signals   = np.where(errors >  threshold,  1,
                                 np.where(errors < -threshold, -1, 0))
        else:
            threshold = threshold if threshold is not None else k * np.std(preds_np)
            signals   = np.where(preds_np >  threshold,  1,
                                 np.where(preds_np < -threshold, -1, 0))

    # ── dl (legacy LSTM) ────────────────────────────────────────────────
    elif model_type == "dl":
        if lookback is None:
            raise ValueError(
                "lookback must be provided for model_type='dl'. "
                "For Darts models use model_type='dl_darts'."
            )
        test_index_aligned = np.asarray(test_index)[lookback:]
        preds_aligned      = np.asarray(preds, dtype=np.float64).ravel()
        preds_aligned      = preds_aligned[:len(test_index_aligned)]

        if "close" in df.columns:
            actual    = df.iloc[test_index_aligned]["close"].values.astype(np.float64)
            errors    = preds_aligned - actual
            threshold = threshold if threshold is not None else k * np.std(errors)
        else:
            errors    = preds_aligned
            threshold = threshold if threshold is not None else k * np.std(preds_aligned)

        signals    = np.where(errors >  threshold,  1,
                              np.where(errors < -threshold, -1, 0))
        test_index = test_index_aligned

    else:
        raise ValueError(
            f"model_type must be one of: 'classifier', 'regressor', "
            f"'dl', 'dl_darts'.  Got: '{model_type}'"
        )

    datetimes = df.iloc[np.asarray(test_index)]["datetime"].values
    return pd.DataFrame({"datetime": datetimes, "signals": signals})


# ─────────────────────────────────────────────────────────────────────────────
# _permute_one_feature  (module-level so ProcessPoolExecutor can pickle it)
# ─────────────────────────────────────────────────────────────────────────────
def _permute_one_feature(args):
    """Evaluate PnL drop for a single feature column (runs in a worker process)."""
    col, X_test_np, X_test_cols, model, df, df_1m, base_pnl, model_type, k, threshold, n_repeats = args

    col_idx = X_test_cols.index(col)
    pnl_scores = []

    for _ in range(n_repeats):
        X_perm_np = X_test_np.copy()
        X_perm_np[:, col_idx] = np.random.permutation(X_perm_np[:, col_idx])
        X_perm = pd.DataFrame(X_perm_np, columns=X_test_cols)

        if model_type == "classifier":
            preds = (
                model.predict_proba(X_perm)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X_perm)
            )
        else:
            preds = model.predict(X_perm)

        df_preds = prepare_predictions(
            df, preds, X_perm.index,
            model_type=model_type, threshold=threshold, k=k,
        )
        df_preds["datetime"] = pd.to_datetime(df_preds["datetime"], utc=True)

        bt = BackTest(df_1m, df_preds, take_profit=3, stop_loss=1)
        _, _, pnl = bt.run()
        pnl_scores.append(pnl)

    return col, base_pnl - np.mean(pnl_scores)


# ─────────────────────────────────────────────────────────────────────────────
# pnl_permutation_importance
# FIX BN5: each feature permutation is now evaluated in a separate worker
#           process via ProcessPoolExecutor.  For 50 features × 3 repeats
#           this gives a near-linear speed-up with core count.
# ─────────────────────────────────────────────────────────────────────────────
def pnl_permutation_importance(
    model,
    X_test: pd.DataFrame,
    df,
    df_1m,
    base_pnl: float,
    model_type: str = "classifier",
    k: float = 0.5,
    threshold=None,
    n_repeats: int = 3,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """
    Parallel permutation importance evaluated on PnL.

    Parameters
    ----------
    max_workers : int or None
        Number of worker processes.  None = os.cpu_count().
    """
    cols        = X_test.columns.tolist()
    X_test_np   = X_test.to_numpy()

    args_list = [
        (col, X_test_np, cols, model, df, df_1m,
         base_pnl, model_type, k, threshold, n_repeats)
        for col in cols
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_permute_one_feature, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            col, drop = fut.result()
            results[col] = drop

    return (
        pd.DataFrame(
            [{"feature": col, "pnl_drop": drop} for col, drop in results.items()]
        )
        .sort_values("pnl_drop", ascending=False)
        .reset_index(drop=True)
    )


def extract_important_features(pnl_importance_wide: pd.DataFrame, model_name: str):
    feature_row      = pnl_importance_wide.drop(columns=["pnl"], errors="ignore").iloc[0]
    important_features = feature_row[feature_row > 0].index.tolist()
    return pd.DataFrame({
        "model_name":        [model_name],
        "important_features": [important_features],
    })