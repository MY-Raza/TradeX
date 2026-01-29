# backtester.py
import pandas as pd
import numpy as np
from TradeX.utils.common.logs import get_logger

logger = get_logger("backtester")


class Backtester:
    def __init__(self, price_df: pd.DataFrame, signal_df: pd.DataFrame, tp: float = 3, sl: float = 1):
        """
        Backtester for simple take-profit / stop-loss trades.
        """
        self.price_df = price_df.sort_values("timestamp").reset_index(drop=True)
        self.signal_df = signal_df.sort_values("timestamp").reset_index(drop=True)
        self.tp = tp
        self.sl = sl
        self.trades = []
        self.last_trade_direction = None  # Track last closed trade direction

    def run_backtest(self):
        merged_df = pd.merge_asof(
            self.price_df,
            self.signal_df.rename(columns={"timestamp": "signal_timestamp"}),
            left_on="timestamp",
            right_on="signal_timestamp",
            direction="backward"
        )

        open_trade = None

        for i, row in merged_df.iterrows():
            signal = row.get("bbands_signal", 0)

            # If a trade is already open
            if open_trade is not None:
                high = row["high"]
                low = row["low"]
                tp_sl_hit = None
                exit_price = row["close"]  # default exit price if signal flips

                # Check TP/SL for the current open trade
                if open_trade["direction"] == "buy":
                    if high >= open_trade["tp_price"]:
                        tp_sl_hit = "TP"
                        exit_price = open_trade["tp_price"]
                    elif low <= open_trade["sl_price"]:
                        tp_sl_hit = "SL"
                        exit_price = open_trade["sl_price"]
                else:  # sell
                    if low <= open_trade["tp_price"]:
                        tp_sl_hit = "TP"
                        exit_price = open_trade["tp_price"]
                    elif high >= open_trade["sl_price"]:
                        tp_sl_hit = "SL"
                        exit_price = open_trade["sl_price"]

                # Close the trade if TP/SL hit or signal changed direction
                if tp_sl_hit or (signal != 0 and signal != open_trade["signal"]):
                    pnl = (
                        (exit_price - open_trade["entryprice"]) if open_trade["direction"] == "buy" else
                        (open_trade["entryprice"] - exit_price)
                    )

                    self.trades.append({
                        "timestamp": open_trade["timestamp"],
                        "entryprice": open_trade["entryprice"],
                        "exitprice": exit_price,
                        "tp/sl": tp_sl_hit,
                        "direction": open_trade["direction"],
                        "pnl": pnl,
                    })

                    # Track last closed trade direction to block consecutive same trades
                    self.last_trade_direction = open_trade["direction"]
                    open_trade = None  # trade closed

            # Open a new trade if no trade is currently open
            if open_trade is None and signal in [1, -1]:
                direction = "buy" if signal == 1 else "sell"

                # Skip if last trade was same direction
                if self.last_trade_direction == direction:
                    continue

                entryprice = row["close"]
                open_trade = {
                    "timestamp": row["timestamp"],
                    "entryprice": entryprice,
                    "direction": direction,
                    "signal": signal,
                    "tp_price": entryprice * (1 + self.tp / 100) if signal == 1 else entryprice * (1 - self.tp / 100),
                    "sl_price": entryprice * (1 - self.sl / 100) if signal == 1 else entryprice * (1 + self.sl / 100),
                }

        logger.info(f"Backtesting completed. Total trades recorded: {len(self.trades)}")

    def get_results(self) -> pd.DataFrame:
        """
        Return a DataFrame with all trade entries.
        """
        return pd.DataFrame(self.trades)
