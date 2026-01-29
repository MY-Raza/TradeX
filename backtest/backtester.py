# backtester.py
import pandas as pd
import numpy as np
from TradeX.utils.common.logs import get_logger

logger = get_logger("backtester")


class Backtester:
    """
    A simple backtester for take-profit (TP) / stop-loss (SL) trading strategies.

    Attributes:
        price_df (pd.DataFrame): OHLCV price data with 'timestamp', 'high', 'low', 'close' columns.
        signal_df (pd.DataFrame): Trading signals with 'timestamp' and 'signals' columns.
            Signals: 1 = Buy, -1 = Sell, 0 = No action.
        tp (float): Take-profit percentage (default 3%).
        sl (float): Stop-loss percentage (default 1%).
        trades (list): List of trade dictionaries containing trade details.
        last_trade_direction (str | None): Tracks the direction of the last closed trade to prevent consecutive same-direction trades.
    """

    def __init__(self, price_df: pd.DataFrame, signal_df: pd.DataFrame, tp: float = 3, sl: float = 1):
        """
        Initialize the Backtester with price and signal data.

        Args:
            price_df (pd.DataFrame): OHLCV price data.
            signal_df (pd.DataFrame): Signal data.
            tp (float, optional): Take-profit percentage. Defaults to 3.
            sl (float, optional): Stop-loss percentage. Defaults to 1.
        """
        # Sort data by timestamp to ensure chronological order
        self.price_df = price_df.sort_values("timestamp").reset_index(drop=True)
        self.signal_df = signal_df.sort_values("timestamp").reset_index(drop=True)
        self.tp = tp
        self.sl = sl
        self.trades = []
        self.last_trade_direction = None  # Prevent consecutive same-direction trades

    def run_backtest(self):
        """
        Run the backtesting process.

        Steps:
            1. Merge price and signal data using simple merge (timestamps must match).
            2. Iterate through each row to simulate trades.
            3. Open trades on new signals if no trade is open and direction is different from last trade.
            4. Close trades if TP/SL is hit or signal flips.
            5. Record trade details including entry price, exit price, TP/SL status, direction, and PnL.
        """
        # -------------------------------
        # Merge signals into price data
        # -------------------------------
        merged_df = pd.merge(
            self.price_df,
            self.signal_df.rename(columns={"timestamp": "signal_timestamp"}),
            left_on="timestamp",
            right_on="signal_timestamp",
            how="left"  # keep all price rows; signals will be NaN if not available
        )

        # Fill missing signals with 0 (no action)
        merged_df["signals"] = merged_df["signals"].fillna(0)

        open_trade = None  # Track current open trade

        for i, row in merged_df.iterrows():
            signal = row.get("signals", 0)

            # -------------------------------
            # Handle existing open trade
            # -------------------------------
            if open_trade is not None:
                high = row["high"]
                low = row["low"]
                tp_sl_hit = None
                exit_price = row["close"]  # Default exit price if trade closes due to signal flip

                # Check if TP/SL has been hit
                if open_trade["direction"] == "buy":
                    if high >= open_trade["tp_price"]:
                        tp_sl_hit = "TP"
                        exit_price = open_trade["tp_price"]
                    elif low <= open_trade["sl_price"]:
                        tp_sl_hit = "SL"
                        exit_price = open_trade["sl_price"]
                else:  # sell trade
                    if low <= open_trade["tp_price"]:
                        tp_sl_hit = "TP"
                        exit_price = open_trade["tp_price"]
                    elif high >= open_trade["sl_price"]:
                        tp_sl_hit = "SL"
                        exit_price = open_trade["sl_price"]

                # Close trade if TP/SL hit or signal direction changes
                if tp_sl_hit or (signal != 0 and signal != open_trade["signal"]):
                    pnl = (
                        (exit_price - open_trade["entry_price"]) if open_trade["direction"] == "buy" else
                        (open_trade["entry_price"] - exit_price)
                    )

                    # Record trade
                    self.trades.append({
                        "timestamp": open_trade["timestamp"],
                        "entry_price": open_trade["entry_price"],
                        "exit_price": exit_price,
                        "tp/sl": tp_sl_hit,
                        "direction": open_trade["direction"],
                        "pnl": pnl,
                    })

                    # Update last trade direction to block consecutive same-direction trades
                    self.last_trade_direction = open_trade["direction"]
                    open_trade = None  # Reset current trade

            # -------------------------------
            # Open a new trade if possible
            # -------------------------------
            if open_trade is None and signal in [1, -1]:
                direction = "buy" if signal == 1 else "sell"

                # Skip trade if last trade had same direction
                if self.last_trade_direction == direction:
                    continue

                entry_price = row["close"]
                open_trade = {
                    "timestamp": row["timestamp"],
                    "entry_price": entry_price,
                    "direction": direction,
                    "signal": signal,
                    "tp_price": entry_price * (1 + self.tp / 100) if signal == 1 else entry_price * (1 - self.tp / 100),
                    "sl_price": entry_price * (1 - self.sl / 100) if signal == 1 else entry_price * (1 + self.sl / 100),
                }

        logger.info(f"Backtesting completed. Total trades recorded: {len(self.trades)}")

    def get_results(self) -> pd.DataFrame:
        """
        Return the recorded trades as a pandas DataFrame.

        Columns:
            - timestamp: When the trade was opened.
            - entry_price: Entry price of the trade.
            - exit_price: Price at which the trade was closed.
            - tp/sl: Type of exit ('TP', 'SL', or None for signal flip exit).
            - direction: 'buy' or 'sell'.
            - pnl: Profit or loss for the trade.

        Returns:
            pd.DataFrame: DataFrame of all trades.
        """
        return pd.DataFrame(self.trades)
