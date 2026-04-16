"""
backtest_runner.py
==================
Wires the BackTest engine to the ML signal pipeline.

Responsibilities
----------------
* run_backtest()  — loads minute OHLCV spanning the test set, aligns
                    signal timestamps to the BackTest engine's expectations,
                    executes BackTest.run(), and returns structured results.

TIMING ALIGNMENT CONTRACT
--------------------------
Feature rows carry hourly timestamps.  Each feature timestamp T represents
a candle that CLOSES at T (activity window [T-1h, T)).  The signal for that
bar is therefore known ONLY at time T (bar-close).

BackTest.run() interprets prediction timestamps as the moment the signal
is received and uses buy_after_minutes to delay actual entry.  Setting
buy_after_minutes=0 enters on the OPEN of the NEXT minute bar after T,
which is correctly forward in time and free of look-ahead bias.
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass

from TradeX.utils.common.logs import get_logger
from backtest import BackTest          # provided class
from data_loader import load_price_data
from config import BACKTEST_PARAMS, DATETIME_COL

logger = get_logger("backtest_runner")


# =========================================================
# RESULT CONTAINER
# =========================================================

@dataclass
class BacktestResult:
    ledger:        pd.DataFrame
    final_balance: float
    total_pnl_pct: float
    summary:       dict


# =========================================================
# PUBLIC API
# =========================================================

def run_backtest(df_signals: pd.DataFrame) -> BacktestResult:
    """
    Execute the backtest using ML-generated signals on minute OHLCV data.

    Parameters
    ----------
    df_signals : pd.DataFrame
        Columns: [datetime, signals]
        datetime — UTC-aware hourly timestamps.
        signals  — {1 (long), -1 (short), 0 (neutral)}.

    Returns
    -------
    BacktestResult
        ledger        — full trade-by-trade log as pd.DataFrame
        final_balance — ending account balance
        total_pnl_pct — cumulative PnL % (sum of individual trade PnLs)
        summary       — dict suitable for DB insertion
    """
    if df_signals.empty:
        raise ValueError("df_signals is empty — no signals to backtest.")

    # ── 1. Validate / normalise signals DataFrame ─────────
    df_signals = df_signals[[DATETIME_COL, "signals"]].copy()
    df_signals[DATETIME_COL] = pd.to_datetime(df_signals[DATETIME_COL], utc=True)
    df_signals = df_signals.sort_values(DATETIME_COL).reset_index(drop=True)

    # BackTest expects column name 'datetime'
    df_predictions = df_signals.rename(columns={DATETIME_COL: "datetime"})
    df_predictions["signals"] = df_predictions["signals"].astype(int)

    # ── 2. Determine price data range ─────────────────────
    #   Expand by 2 h on each side to give the engine room at boundaries.
    signal_start = df_predictions["datetime"].min() - pd.Timedelta(hours=2)
    signal_end   = df_predictions["datetime"].max() + pd.Timedelta(hours=2)

    logger.info(
        f"Fetching price data for backtest window: "
        f"{signal_start} → {signal_end}"
    )

    df_price = load_price_data(start_date=signal_start, end_date=signal_end)

    # BackTest engine requires column 'datetime' (not UTC-aware is fine if consistent)
    df_price = df_price.rename(columns={DATETIME_COL: "datetime"})
    df_price["datetime"] = pd.to_datetime(df_price["datetime"], utc=True)

    # ── 3. Log pre-flight diagnostics ─────────────────────
    n_long    = int((df_predictions["signals"] ==  1).sum())
    n_short   = int((df_predictions["signals"] == -1).sum())
    n_neutral = int((df_predictions["signals"] ==  0).sum())

    logger.info(
        f"Backtest inputs → "
        f"signal rows: {len(df_predictions)} | "
        f"price rows: {len(df_price)} | "
        f"LONG: {n_long} | SHORT: {n_short} | NEUTRAL: {n_neutral}"
    )

    # ── 4. Instantiate and run BackTest ───────────────────
    engine = BackTest(
        df_price=df_price,
        df_predictions=df_predictions,
        **BACKTEST_PARAMS,
    )

    ledger, final_balance, total_pnl_pct = engine.run()

    # ── 5. Parse results ──────────────────────────────────
    n_trades = 0
    if not ledger.empty:
        # Count only sell actions (each complete trade = 1 buy + 1 sell)
        n_trades = int(ledger["action"].str.startswith("sell").sum())

    logger.info("=" * 64)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 64)
    logger.info(f"  Starting balance : {BACKTEST_PARAMS['starting_balance']:.2f}")
    logger.info(f"  Final balance    : {final_balance:.2f}")
    logger.info(f"  Total PnL %      : {total_pnl_pct:.2f}%")
    logger.info(f"  Completed trades : {n_trades}")
    if n_trades > 0 and not ledger.empty:
        sell_rows = ledger[ledger["action"].str.startswith("sell")]
        if not sell_rows.empty:
            win_rate = (sell_rows["pnl"] > 0).mean()
            avg_pnl  = sell_rows["pnl"].mean()
            max_dd   = _compute_max_drawdown(ledger)
            logger.info(f"  Win rate         : {win_rate:.1%}")
            logger.info(f"  Avg trade PnL    : {avg_pnl:.4f}%")
            logger.info(f"  Max drawdown     : {max_dd:.2f}%")
    logger.info("=" * 64)

    # ── 6. Build summary dict ─────────────────────────────
    summary = _build_summary(ledger, final_balance, total_pnl_pct, n_trades)

    return BacktestResult(
        ledger=ledger,
        final_balance=final_balance,
        total_pnl_pct=total_pnl_pct,
        summary=summary,
    )


# =========================================================
# HELPERS
# =========================================================

def _compute_max_drawdown(ledger: pd.DataFrame) -> float:
    """Maximum peak-to-trough drawdown as a percentage of peak balance."""
    if ledger.empty or "balance" not in ledger.columns:
        return 0.0
    balance = ledger["balance"].values
    peak    = balance[0]
    max_dd  = 0.0
    for b in balance:
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


def _build_summary(
    ledger:        pd.DataFrame,
    final_balance: float,
    total_pnl_pct: float,
    n_trades:      int,
) -> dict:
    """Construct a flat summary dict for DB insertion."""
    summary: dict = {
        "final_balance":     final_balance,
        "total_pnl_percent": total_pnl_pct,
        "number_of_trades":  n_trades,
        "starting_balance":  BACKTEST_PARAMS["starting_balance"],
    }

    if not ledger.empty:
        sell_rows = ledger[ledger["action"].str.startswith("sell")]
        if not sell_rows.empty:
            summary["win_rate"]      = round(float((sell_rows["pnl"] > 0).mean()), 4)
            summary["avg_trade_pnl"] = round(float(sell_rows["pnl"].mean()), 4)
            summary["max_drawdown"]  = _compute_max_drawdown(ledger)

            # Reason breakdown
            for reason in ["take_profit", "stop_loss", "direction_change", "end_of_backtest"]:
                key = f"exits_{reason}"
                summary[key] = int(
                    sell_rows["action"].str.contains(reason).sum()
                )

    return summary