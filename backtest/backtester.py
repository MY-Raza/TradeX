# backtester.py
import pandas as pd
import numpy as np
from TradeX.utils.common.logs import get_logger

logger = get_logger("backtester")

class Backtester:
    def __init__(self, price_df: pd.DataFrame, signal_df: pd.DataFrame, tp: float = 3, sl: float = 1):
        """
        Backtester for simple take-profit / stop-loss trades.

        :param price_df: OHLCV DataFrame (1m candles) with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        :param signal_df: Signals DataFrame (1h candles) with columns: ['timestamp', 'signal'] (1 for buy, -1 for sell)
        :param tp: Take-profit multiplier
        :param sl: Stop-loss multiplier
        """
        self.price_df = price_df.sort_values("timestamp").reset_index(drop=True)
        self.signal_df = signal_df.sort_values("timestamp").reset_index(drop=True)
        self.tp = tp
        self.sl = sl
        self.trades = []

    def run_backtest(self):
        """
        Iterate over 1m price candles and execute trades based on signals.
        Signals are matched to the nearest previous 1h timestamp.
        """
        # Merge price and signals using nearest previous timestamp
        merged_df = pd.merge_asof(
            self.price_df,
            self.signal_df.rename(columns={"timestamp": "signal_timestamp"}),
            left_on="timestamp",
            right_on="signal_timestamp",
            direction="backward"
        )

        for i, row in merged_df.iterrows():
            signal = row.get("signal")
            if signal not in [1, -1]:  # skip if no signal
                continue

            buy_price = row["close"]
            direction = "bullish" if signal == 1 else "bearish"

            # Calculate TP/SL prices
            if signal == 1:
                tp_price = buy_price * (1 + self.tp / 100)
                sl_price = buy_price * (1 - self.sl / 100)
            else:
                tp_price = buy_price * (1 - self.tp / 100)
                sl_price = buy_price * (1 + self.sl / 100)

            pnl = None
            exit_ts = None

            # Loop over future candles to check if TP or SL hit
            for j in range(i + 1, len(merged_df)):
                future = merged_df.iloc[j]
                high = future["high"]
                low = future["low"]

                if signal == 1:
                    if high >= tp_price:
                        pnl = tp_price - buy_price
                        exit_ts = future["timestamp"]
                        break
                    elif low <= sl_price:
                        pnl = sl_price - buy_price
                        exit_ts = future["timestamp"]
                        break
                else:
                    if low <= tp_price:
                        pnl = buy_price - tp_price
                        exit_ts = future["timestamp"]
                        break
                    elif high >= sl_price:
                        pnl = buy_price - sl_price
                        exit_ts = future["timestamp"]
                        break

            # If neither TP nor SL hit, exit at last available close
            if pnl is None:
                pnl = (merged_df.iloc[-1]["close"] - buy_price) if signal == 1 else (buy_price - merged_df.iloc[-1]["close"])
                exit_ts = merged_df.iloc[-1]["timestamp"]

            # Store trade result
            self.trades.append({
                "timestamp": row["timestamp"],
                "buyprice": buy_price,
                "tp": tp_price,
                "sl": sl_price,
                "direction": direction,
                "pnl": pnl,
                "exit_ts": exit_ts
            })

        logger.info(f"Backtesting completed. Total trades: {len(self.trades)}")

    def get_results(self) -> pd.DataFrame:
        """
        Return a DataFrame of all trades.
        """
        return pd.DataFrame(self.trades)
