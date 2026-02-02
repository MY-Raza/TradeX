import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    """
    Configuration class for the backtester.
    
    Attributes:
        starting_balance (float): Initial account balance.
        leverage (float): Trading leverage applied to positions.
        transaction_fee (float): Fee percentage per trade (e.g., 0.05 = 5%).
        slippage (float): Slippage percentage per trade.
        take_profit_pct (float): Take profit target as a percentage of buy price.
        stop_loss_pct (float): Stop loss target as a percentage of buy price.
        buy_after_minutes (int): Delay in minutes before entering a trade after a signal.
        min_balance_pct (float): Minimum allowable balance as a fraction of starting balance
                                 to continue trading.
    """
    starting_balance: float = 1000.0
    leverage: float = 1.0
    transaction_fee: float = 0.05
    slippage: float = 0.0
    take_profit_pct: float = 0.03
    stop_loss_pct: float = 0.01
    buy_after_minutes: int = 0
    min_balance_pct: float = 0.5


class Backtester:
    """
    Backtester class for simulating trades based on predicted market directions.
    
    Methods:
        run(price_df, signal_df):
            Runs the backtest using price and prediction dataframes.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initializes the backtester with a configuration object.
        
        Args:
            config (BacktestConfig): Configuration containing all backtest parameters.
        """
        self.config = config

    def run(self, price_df: pd.DataFrame, signal_df: pd.DataFrame):
        """
        Executes the backtesting algorithm.

        Args:
            price_df (pd.DataFrame): OHLCV price data with columns ['timestamp', 'open', 'high', 'low', ...].
            signal_df (pd.DataFrame): Predicted signals with columns ['timestamp', 'signals'].
            
        Returns:
            tuple: 
                - DataFrame containing all executed trades with columns
                  ['datetime', 'predicted_direction', 'action', 'buy_price', 'sell_price', 'balance', 'pnl', 'pnl_sum'].
                - Final balance after all trades.
                - Total cumulative PnL (%).
        """

        # -----------------------------
        # Copy and preprocess data
        # -----------------------------
        price_df = price_df.copy()
        signal_df = signal_df.copy()

        # Ensure timestamps are datetime objects
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
        signal_df["timestamp"] = pd.to_datetime(signal_df["timestamp"])

        # Sort data by timestamp
        price_df = price_df.sort_values("timestamp").reset_index(drop=True)
        signal_df = signal_df.sort_values("timestamp").reset_index(drop=True)

        # Convert DataFrames to numpy arrays for faster computations
        np_price = price_df[["timestamp", "open", "high", "low"]].to_numpy()
        np_signal = signal_df[["timestamp", "signals"]].to_numpy()

        ts_price = np_price[:, 0]   # Price timestamps
        ts_signal = np_signal[:, 0] # Signal timestamps

        # -----------------------------
        # Precompute intervals
        # -----------------------------
        # Each signal interval corresponds to a slice of the price array between consecutive signals
        intervals = []
        for i in range(len(ts_signal) - 1):
            start = np.searchsorted(ts_price, ts_signal[i])
            end = np.searchsorted(ts_price, ts_signal[i + 1])
            intervals.append((start, end))

        # -----------------------------
        # Initialize account parameters
        # -----------------------------
        balance = self.config.starting_balance
        breaking_balance = balance * self.config.min_balance_pct  # Stop if balance drops below this

        fee_pct = self.config.transaction_fee * self.config.leverage
        slippage = self.config.slippage

        # -----------------------------
        # Initialize trade state variables
        # -----------------------------
        in_position = False
        buy_price = 0
        sell_price = 0
        direction = 0
        pnl_sum = 0
        ledger = []

        prev_direction = np_signal[0][1]  # First signal

        # -----------------------------
        # Main backtesting loop
        # -----------------------------
        for i, (start_idx, end_idx) in enumerate(intervals):

            # Stop trading if balance falls below breaking threshold
            if balance < breaking_balance:
                break

            # Current predicted direction
            current_direction = np_signal[i][1]
            if current_direction == 0:
                # Use previous direction if signal is neutral (0)
                current_direction = prev_direction

            # Extract price data for this interval
            np_temp = np_price[start_idx:end_idx]
            if len(np_temp) <= 10:
                prev_direction = current_direction
                continue

            # Separate OHLC columns
            opens = np_temp[:, 1]
            highs = np_temp[:, 2]
            lows = np_temp[:, 3]
            times = np_temp[:, 0]

            # Decide which row to use for trade entry
            entry_idx = min(self.config.buy_after_minutes, len(opens) - 1)

            # -----------------------------
            # ENTER POSITION
            # -----------------------------
            if not in_position:
                buy_price = opens[entry_idx]
                direction = current_direction

                # Immediate PnL impact due to fee and slippage
                pnl = -fee_pct - slippage
                balance += balance * (pnl / 100)
                pnl_sum += pnl

                # Record trade in ledger
                ledger.append([times[entry_idx],
                               "long" if direction > 0 else "short",
                               "buy", buy_price, 0,
                               round(balance, 2), round(pnl, 2), round(pnl_sum, 2)])
                in_position = True

            # -----------------------------
            # CHECK TAKE PROFIT / STOP LOSS
            # -----------------------------
            if in_position:
                if direction > 0:
                    tp = buy_price * (1 + self.config.take_profit_pct)
                    sl = buy_price * (1 - self.config.stop_loss_pct)
                    tp_hits = np.where(highs >= tp)[0]
                    sl_hits = np.where(lows <= sl)[0]
                else:
                    tp = buy_price * (1 - self.config.take_profit_pct)
                    sl = buy_price * (1 + self.config.stop_loss_pct)
                    tp_hits = np.where(lows <= tp)[0]
                    sl_hits = np.where(highs >= sl)[0]

                # Determine which occurs first (TP or SL)
                hit_idx = None
                if len(tp_hits) and len(sl_hits):
                    hit_idx = min(tp_hits[0], sl_hits[0])
                elif len(tp_hits):
                    hit_idx = tp_hits[0]
                elif len(sl_hits):
                    hit_idx = sl_hits[0]

                if hit_idx is not None:
                    sell_price = highs[hit_idx] if direction > 0 else lows[hit_idx]
                    if direction < 0:
                        # Handle short positions
                        sell_price = lows[hit_idx] if hit_idx in tp_hits else highs[hit_idx]

                    # Compute PnL %
                    pnl = ((sell_price - buy_price) / buy_price * 100) if direction > 0 \
                        else ((buy_price - sell_price) / buy_price * 100)
                    pnl *= self.config.leverage
                    pnl -= fee_pct + slippage

                    balance += balance * (pnl / 100)
                    pnl_sum += pnl

                    # Record sell action (take profit or stop loss)
                    action = "sell - take_profit" if pnl > 0 else "sell - stop_loss"

                    ledger.append([times[hit_idx],
                                   "long" if direction > 0 else "short",
                                   action, buy_price, sell_price,
                                   round(balance, 2), round(pnl, 2), round(pnl_sum, 2)])

                    in_position = False

            # -----------------------------
            # EXIT ON DIRECTION CHANGE
            # -----------------------------
            if in_position and current_direction != prev_direction:
                sell_price = opens[entry_idx]

                # Compute PnL for direction-change exit
                pnl = ((sell_price - buy_price) / buy_price * 100) if prev_direction > 0 \
                    else ((buy_price - sell_price) / buy_price * 100)
                pnl *= self.config.leverage
                pnl -= fee_pct + slippage

                balance += balance * (pnl / 100)
                pnl_sum += pnl

                ledger.append([times[entry_idx],
                               "long" if prev_direction > 0 else "short",
                               "sell - direction change",
                               buy_price, sell_price,
                               round(balance, 2), round(pnl, 2), round(pnl_sum, 2)])

                in_position = False

            # Update previous direction
            prev_direction = current_direction

        # -----------------------------
        # Prepare final ledger DataFrame
        # -----------------------------
        columns = ["datetime", "predicted_direction", "action",
                   "buy_price", "sell_price", "balance", "pnl", "pnl_sum"]

        return pd.DataFrame(ledger, columns=columns), round(balance, 2), round(pnl_sum, 2)
