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

                # Check TP/SL for the current open trade
                if open_trade["direction"] == "buy":
                    if high >= open_trade["tp_price"]:
                        tp_sl_hit = "TP"
                    elif low <= open_trade["sl_price"]:
                        tp_sl_hit = "SL"
                else:  # sell
                    if low <= open_trade["tp_price"]:
                        tp_sl_hit = "TP"
                    elif high >= open_trade["sl_price"]:
                        tp_sl_hit = "SL"

                # Close the trade if TP/SL hit or signal changed direction
                if tp_sl_hit or (signal != 0 and signal != open_trade["signal"]):
                    pnl = (
                        (open_trade["tp_price"] - open_trade["buyprice"]) if open_trade["direction"] == "buy" else
                        (open_trade["buyprice"] - open_trade["tp_price"])
                    ) if tp_sl_hit == "TP" else (
                        (open_trade["sl_price"] - open_trade["buyprice"]) if open_trade["direction"] == "buy" else
                        (open_trade["buyprice"] - open_trade["sl_price"])
                    ) if tp_sl_hit == "SL" else (
                        (row["close"] - open_trade["buyprice"]) if open_trade["direction"] == "buy" else
                        (open_trade["buyprice"] - row["close"])
                    )

                    self.trades.append({
                        "timestamp": open_trade["timestamp"],
                        "buyprice": open_trade["buyprice"],
                        "tp/sl": tp_sl_hit,
                        "direction": open_trade["direction"],
                        "pnl": pnl,
                    })
                    open_trade = None  # trade closed

            # Open a new trade if no trade is currently open
            if open_trade is None and signal in [1, -1]:
                buy_price = row["close"]
                direction = "buy" if signal == 1 else "sell"  # <-- changed here
                open_trade = {
                    "timestamp": row["timestamp"],
                    "buyprice": buy_price,
                    "direction": direction,
                    "signal": signal,
                    "tp_price": buy_price * (1 + self.tp / 100) if signal == 1 else buy_price * (1 - self.tp / 100),
                    "sl_price": buy_price * (1 - self.sl / 100) if signal == 1 else buy_price * (1 + self.sl / 100),
                }

        logger.info(f"Backtesting completed. Total trades recorded: {len(self.trades)}")

    def get_results(self) -> pd.DataFrame:
        """
        Return a DataFrame with all trade entries.
        """
        return pd.DataFrame(self.trades)
