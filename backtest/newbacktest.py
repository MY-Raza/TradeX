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
        max_delay_minutes=1
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
        df_price['datetime'] = pd.to_datetime(df_price['datetime'])

        # Convert price DataFrame to NumPy array for performance
        self.np_price = df_price.to_numpy()

        # Cache column indices for fast access
        self.idx_time = df_price.columns.get_loc('datetime')
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
        df_predictions['datetime'] = pd.to_datetime(df_predictions['datetime'])

        # Convert predictions to NumPy for speed
        self.np_pred = df_predictions.to_numpy()

        # Cache prediction column indices
        self.idx_pred_time = df_predictions.columns.get_loc('datetime')
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
        self.entry_candle_idx = None

    # ==========================
    # BUY LOGIC
    # ==========================
    def buy(self, np_interval, direction, timestamp=None):
        """
        Open a new position. No-op if already in a position.
        """

        # INVARIANT GUARD: never open a second position
        if self.in_position:
            logger.warning("buy() called while already in position — skipping.")
            return

        if len(np_interval) == 0:
            return

        # Choose entry candle after optional delay
        entry_idx = min(self.buy_after_minutes, len(np_interval) - 1)

        # Store entry candle index so TP/SL knows where to start scanning
        self.entry_candle_idx = entry_idx  # NEW: persisted for check_tp_sl

        self.buy_price = np_interval[entry_idx, self.idx_open]
        self.sell_price = 0
        self.current_direction = direction
        self.in_position = True

        # Apply entry costs immediately
        pnl = -self.fee - self.slippage
        self.balance += self.balance * pnl / 100

        buy_time = timestamp if timestamp else np_interval[entry_idx, self.idx_time]
        self.record_trade(buy_time, 'buy', pnl)


    # ==========================
    # SELL LOGIC
    # ==========================
    def sell(self, timestamp, price, reason):
        """
        Close the current position. No-op if not in a position.
        """

        # INVARIANT GUARD: already correct, but now also resets entry_candle_idx
        if not self.in_position:
            logger.warning(f"sell() called while not in position (reason={reason}) — skipping.")
            return

        self.sell_price = price

        if self.current_direction > 0:  # long
            pnl = (self.sell_price - self.buy_price) / self.buy_price * 100
        else:  # short
            pnl = (self.buy_price - self.sell_price) / self.buy_price * 100

        pnl *= self.leverage
        pnl -= (self.fee + self.slippage)

        self.balance += self.balance * pnl / 100
        self.in_position = False
        self.entry_candle_idx = None  # NEW: reset on close

        self.record_trade(timestamp, f'sell - {reason}', pnl)


    # ==========================
    # TAKE-PROFIT / STOP-LOSS
    # ==========================
    def check_tp_sl(self, np_interval):
        """
        Scan for the first TP or SL hit, starting strictly AFTER the entry candle.
        """

        if not self.in_position:
            return

        # INVARIANT: only evaluate candles after the entry candle
        # entry_candle_idx is set by buy(); default to 0 if somehow missing
        scan_start = (self.entry_candle_idx + 1) if self.entry_candle_idx is not None else 1

        # Nothing to scan if entry was on the last candle of the interval
        if scan_start >= len(np_interval):
            return

        # Slice to post-entry candles only
        np_scan = np_interval[scan_start:]

        highs = np_scan[:, self.idx_high]
        lows = np_scan[:, self.idx_low]
        timestamps = np_scan[:, self.idx_time]

        if self.current_direction > 0:  # long
            tp_price = self.buy_price * (1 + self.take_profit)
            sl_price = self.buy_price * (1 - self.stop_loss)
            tp_hits = np.where(highs >= tp_price)[0]
            sl_hits = np.where(lows <= sl_price)[0]
        else:  # short
            tp_price = self.buy_price * (1 - self.take_profit)
            sl_price = self.buy_price * (1 + self.stop_loss)
            tp_hits = np.where(lows <= tp_price)[0]
            sl_hits = np.where(highs >= sl_price)[0]

        first_tp = tp_hits[0] if len(tp_hits) > 0 else None
        first_sl = sl_hits[0] if len(sl_hits) > 0 else None

        if first_tp is not None and first_sl is not None:
            if first_tp <= first_sl:
                self.sell(timestamps[first_tp], tp_price, 'take_profit')
            else:
                self.sell(timestamps[first_sl], sl_price, 'stop_loss')
        elif first_tp is not None:
            self.sell(timestamps[first_tp], tp_price, 'take_profit')
        elif first_sl is not None:
            self.sell(timestamps[first_sl], sl_price, 'stop_loss')

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
            'sell_price': self.sell_price if 'sell' in action else 0,
            'balance': self.balance,
            'pnl': pnl
        })

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
            pred_time = self.np_pred[i, self.idx_pred_time]

            if self.in_position and current_pred_signal != self.current_direction:
                # Atomic flip: close at interval open, reopen immediately
                self.sell(pred_time, open_price, 'direction_change')
                self.buy(np_interval, current_pred_signal, timestamp=pred_time)
            elif not self.in_position:
                self.buy(np_interval, current_pred_signal, timestamp=pred_time)
            # else: in_position and same direction → hold, do nothing

            self.last_signal = current_pred_signal
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
