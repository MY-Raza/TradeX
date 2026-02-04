import pandas as pd
import numpy as np
from TradeX.utils.common.logs import get_logger

# Initialize module-level logger
logger = get_logger("backtest_class")


class HighPerfBacktest:
    """
    High-performance backtesting engine for direction-based trading strategies.

    This class:
    - Works with minute-level OHLCV price data
    - Consumes model-generated directional predictions (e.g., hourly)
    - Executes trades realistically (delayed entry, TP/SL, fees, slippage)
    - Avoids look-ahead bias by aligning predictions strictly forward in time
    - Tracks balance, PnL, and a full trade ledger
    """

    def __init__(
        self,
        df_price,
        df_predictions,
        starting_balance=1000,
        take_profit=1,
        stop_loss=1,
        buy_after_minutes=0,
        fee=0.05,
        leverage=1.0,
        slippage=0.0,
        max_delay_minutes=5
    ):
        """
        Initialize the backtest with price data, predictions, and trading parameters.

        Parameters
        ----------
        df_price : pd.DataFrame
            Minute-level OHLCV price data with columns:
            ['timestamp', 'open', 'high', 'low', 'close', ...]

        df_predictions : pd.DataFrame
            Prediction data with columns:
            ['timestamp', 'signals']
            signals: 1 (long), -1 (short), 0 (neutral)

        starting_balance : float
            Initial account balance

        take_profit : float
            Take-profit percentage (e.g., 1 = 1%)

        stop_loss : float
            Stop-loss percentage (e.g., 1 = 1%)

        buy_after_minutes : int
            Delay (in minutes) after prediction before entering a trade

        fee : float
            Trading fee percentage per trade

        leverage : float
            Leverage multiplier applied to PnL

        slippage : float
            Slippage percentage applied on entry and exit

        max_delay_minutes : int
            (Reserved) Maximum execution delay for realism
        """

        # ==========================
        # PRICE DATA PREPARATION
        # ==========================

        # Ensure timestamps are datetime objects
        df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])

        # Convert price DataFrame to NumPy array for performance
        self.np_price = df_price.to_numpy()

        # Cache column indices for fast access
        self.idx_time = df_price.columns.get_loc('timestamp')
        self.idx_open = df_price.columns.get_loc('open')
        self.idx_high = df_price.columns.get_loc('high')
        self.idx_low = df_price.columns.get_loc('low')
        self.idx_close = df_price.columns.get_loc('close')

        # Extract timestamps column separately (used for searchsorted)
        self.timestamps_price = self.np_price[:, self.idx_time]

        # ==========================
        # PREDICTION DATA PREPARATION
        # ==========================

        # Ensure prediction timestamps are datetime objects
        df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp'])

        # Convert predictions to NumPy for speed
        self.np_pred = df_predictions.to_numpy()

        # Cache prediction column indices
        self.idx_pred_time = df_predictions.columns.get_loc('timestamp')
        self.idx_pred_signal = df_predictions.columns.get_loc('signals')

        # Store prediction timestamps separately
        self.timestamps_pred = self.np_pred[:, self.idx_pred_time]

        # ==========================
        # PRECOMPUTE PRICE INTERVALS
        # ==========================

        """
        For each prediction timestamp:
        - Find the slice of price data that belongs to that prediction window
        - Interval = [prediction_time, next_prediction_time)
        """

        self.interval_indices = []

        for i in range(len(self.np_pred) - 1):
            # Current prediction timestamp
            pred_time = self.np_pred[i, self.idx_pred_time]

            # Find the first price candle AT or AFTER prediction time
            start_idx = np.searchsorted(
                self.np_price[:, self.idx_time],
                pred_time,
                side='left'
            )

            # Next prediction timestamp defines the end of interval
            end_pred_time = self.np_pred[i + 1, self.idx_pred_time]

            # Find where the next interval begins
            end_idx = np.searchsorted(
                self.np_price[:, self.idx_time],
                end_pred_time,
                side='left'
            )

            # Only keep valid intervals with actual data
            if end_idx > start_idx:
                self.interval_indices.append((start_idx, end_idx))
            else:
                # Mark invalid intervals explicitly
                self.interval_indices.append((-1, -1))

        # Count valid intervals for logging/debugging
        self.valid_intervals = sum(
            1 for s, e in self.interval_indices if s >= 0 and e > s
        )

        logger.info(
            f"Found {self.valid_intervals} valid intervals "
            f"out of {len(self.interval_indices)} predictions"
        )

        # ==========================
        # BACKTEST PARAMETERS
        # ==========================

        self.starting_balance = starting_balance
        self.balance = starting_balance

        # Stop backtest if balance drops below 50%
        self.breaking_balance = starting_balance * 0.5

        # Convert TP/SL percentages into decimal form
        self.take_profit = take_profit / 100
        self.stop_loss = stop_loss / 100

        # Entry delay in minutes
        self.buy_after_minutes = int(buy_after_minutes)

        # Fees scale with leverage (realistic futures behavior)
        self.fee = fee * leverage
        self.leverage = leverage
        self.slippage = slippage
        self.max_delay_minutes = max_delay_minutes

        # ==========================
        # TRADE STATE VARIABLES
        # ==========================

        self.in_position = False          # Whether a trade is currently open
        self.buy_price = 0.0              # Entry price
        self.sell_price = 0.0             # Exit price
        self.current_direction = 0        # 1 = long, -1 = short
        self.ledger = []                  # Trade history
        self.last_signal = 0              # Last non-zero signal used

    # ==========================
    # BUY LOGIC
    # ==========================
    def buy(self, np_interval, direction, timestamp=None):
        """
        Open a new position.

        Parameters
        ----------
        np_interval : np.ndarray
            Slice of price data for the prediction interval

        direction : int
            1 for long, -1 for short

        timestamp : optional
            Explicit timestamp for trade entry
        """

        # Safety check: do nothing if no data
        if len(np_interval) == 0:
            return

        # Choose entry candle after optional delay
        idx = min(self.buy_after_minutes, len(np_interval) - 1)

        # Set trade state
        self.buy_price = np_interval[idx, self.idx_open]
        self.sell_price = 0
        self.current_direction = direction
        self.in_position = True

        # Apply entry costs immediately
        pnl = -self.fee - self.slippage
        self.balance += self.balance * pnl / 100

        # Determine actual buy timestamp
        buy_time = timestamp if timestamp else np_interval[idx, self.idx_time]

        # Record the trade
        self.record_trade(buy_time, 'buy', pnl)

    # ==========================
    # SELL LOGIC
    # ==========================
    def sell(self, timestamp, price, reason):
        """
        Close the current position.

        Parameters
        ----------
        timestamp : datetime
            Time of exit

        price : float
            Exit price

        reason : str
            Reason for exit (TP, SL, direction change, end of test)
        """

        # Prevent accidental double-sell
        if not self.in_position:
            return

        self.sell_price = price

        # Direction-aware PnL calculation
        if self.current_direction > 0:  # long
            pnl = (self.sell_price - self.buy_price) / self.buy_price * 100
        else:  # short
            pnl = (self.buy_price - self.sell_price) / self.buy_price * 100

        # Apply leverage
        pnl *= self.leverage

        # Subtract exit costs
        pnl -= (self.fee + self.slippage)

        # Update balance
        self.balance += self.balance * pnl / 100

        # Reset position state
        self.in_position = False

        # Record the trade
        self.record_trade(timestamp, f'sell - {reason}', pnl)

    # ==========================
    # RECORD TRADE
    # ==========================
    def record_trade(self, timestamp, action, pnl=None):
        """
        Append a trade event to the ledger.
        """

        self.ledger.append({
            'datetime': timestamp,
            'predicted_direction': 'long' if self.current_direction > 0 else 'short',
            'action': action,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price if 'sell' in action else None,
            'balance': self.balance,
            'pnl': pnl
        })

    # ==========================
    # TAKE-PROFIT / STOP-LOSS
    # ==========================
    def check_tp_sl(self, np_interval):
        """
        Scan the price interval for the first TP or SL hit.
        """

        if not self.in_position:
            return

        highs = np_interval[:, self.idx_high]
        lows = np_interval[:, self.idx_low]
        timestamps = np_interval[:, self.idx_time]

        # Compute TP and SL price levels
        if self.current_direction > 0:  # long
            tp_price = self.buy_price * (1 + self.take_profit)
            sl_price = self.buy_price * (1 - self.stop_loss)
        else:  # short
            tp_price = self.buy_price * (1 - self.take_profit)
            sl_price = self.buy_price * (1 + self.stop_loss)

        # Find indices where TP or SL is hit
        tp_hits = np.where(
            highs >= tp_price if self.current_direction > 0 else lows <= tp_price
        )[0]

        sl_hits = np.where(
            lows <= sl_price if self.current_direction > 0 else highs >= sl_price
        )[0]

        # Determine which one happens first
        first_hit = None
        if len(tp_hits) > 0 and len(sl_hits) > 0:
            first_hit = tp_hits[0] if tp_hits[0] < sl_hits[0] else sl_hits[0]
        elif len(tp_hits) > 0:
            first_hit = tp_hits[0]
        elif len(sl_hits) > 0:
            first_hit = sl_hits[0]

        # Execute exit if TP or SL is triggered
        if first_hit is not None:
            price = tp_price if first_hit in tp_hits else sl_price
            reason = 'take_profit' if first_hit in tp_hits else 'stop_loss'
            self.sell(timestamps[first_hit], price, reason)

    # ==========================
    # RUN BACKTEST
    # ==========================
    def run(self):
        """
        Execute the backtest loop and return results.
        """

        # Find the first actionable signal
        first_non_zero_idx = None
        for i in range(len(self.np_pred)):
            if self.np_pred[i, self.idx_pred_signal] != 0:
                first_non_zero_idx = i
                break

        # Abort if no trades are possible
        if first_non_zero_idx is None:
            logger.info("No non-zero signals found!")
            return pd.DataFrame(self.ledger), self.balance, 0

        # Start one step earlier to capture the interval correctly
        start_idx = max(0, first_non_zero_idx - 1)

        for i in range(start_idx, len(self.np_pred) - 1):
            current_pred_signal = self.np_pred[i, self.idx_pred_signal]

            # Retrieve precomputed price interval
            start_price_idx, end_price_idx = self.interval_indices[i]
            if start_price_idx < 0 or end_price_idx <= start_price_idx:
                continue

            np_interval = self.np_price[start_price_idx:end_price_idx]

            # Skip intervals too short for delayed entry
            if len(np_interval) <= self.buy_after_minutes:
                continue

            # Carry forward last signal if current is neutral
            if current_pred_signal == 0:
                current_pred_signal = self.last_signal

            # Skip if still neutral
            if current_pred_signal == 0:
                continue

            # Entry price for direction change exits
            open_price = np_interval[0, self.idx_open]

            # Handle direction changes
            if self.in_position and current_pred_signal != self.last_signal:
                self.sell(np_interval[0, self.idx_time], open_price, 'direction_change')
                self.buy(np_interval, current_pred_signal)
            elif not self.in_position:
                self.buy(np_interval, current_pred_signal)

            # Update last signal
            self.last_signal = current_pred_signal

            # Check TP/SL within this interval
            self.check_tp_sl(np_interval)

            # Stop if balance collapses
            if self.balance < self.breaking_balance:
                logger.info("Breaking balance reached. Stopping backtest.")
                break

        # Force-close any open position at the end
        if self.in_position and len(self.np_price) > 0:
            self.sell(
                self.np_price[-1, self.idx_time],
                self.np_price[-1, self.idx_close],
                'end_of_backtest'
            )

        # Build final ledger DataFrame
        df_ledger = pd.DataFrame(self.ledger)
        if len(df_ledger) > 0:
            df_ledger['pnl_sum'] = df_ledger['pnl'].cumsum()
            df_ledger[['balance', 'pnl', 'pnl_sum']] = df_ledger[
                ['balance', 'pnl', 'pnl_sum']
            ].round(2)

        final_balance = round(self.balance, 2)
        total_pnl_percent = round(
            df_ledger['pnl_sum'].iloc[-1] if len(df_ledger) > 0 else 0, 2
        )

        return df_ledger, final_balance, total_pnl_percent
