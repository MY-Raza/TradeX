import pandas as pd
import numpy as np
from TradeX.utils.common.logs import get_logger

logger = get_logger("backtest_class")

class HighPerfBacktest:
    """
    High-performance backtesting class for trading strategies based on predicted directions.
    Works with minute-level OHLCV data and model predictions.
    Handles realistic execution, TP/SL, direction changes, and avoids meaningless trades.
    """

    def __init__(self, df_price, df_predictions, starting_balance=1000,
                 take_profit=1, stop_loss=1, buy_after_minutes=0,
                 fee=0.05, leverage=1.0, slippage=0.0, max_delay_minutes=5):

        # --------------------------
        # Price data
        # --------------------------
        df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
        self.np_price = df_price.to_numpy()
        self.idx_time = df_price.columns.get_loc('timestamp')
        self.idx_open = df_price.columns.get_loc('open')
        self.idx_high = df_price.columns.get_loc('high')
        self.idx_low = df_price.columns.get_loc('low')
        self.idx_close = df_price.columns.get_loc('close')  # Added for closing positions
        self.timestamps_price = self.np_price[:, self.idx_time]

        # --------------------------
        # Predictions - ensure hourly alignment
        # --------------------------
        df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp'])
        self.np_pred = df_predictions.to_numpy()
        self.idx_pred_time = df_predictions.columns.get_loc('timestamp')
        self.idx_pred_signal = df_predictions.columns.get_loc('signals')
        self.timestamps_pred = self.np_pred[:, self.idx_pred_time]

        # --------------------------
        # Precompute interval indices with hourly alignment
        # --------------------------
        self.interval_indices = []
        
        for i in range(len(self.np_pred)-1):
            pred_time = self.np_pred[i, self.idx_pred_time]
            
            # Find the first price timestamp AT or AFTER the prediction time
            start_idx = np.searchsorted(
                self.np_price[:, self.idx_time], 
                pred_time, 
                side='left'
            )
            
            # Get the next prediction time for end of interval
            end_pred_time = self.np_pred[i+1, self.idx_pred_time]
            end_idx = np.searchsorted(
                self.np_price[:, self.idx_time], 
                end_pred_time, 
                side='left'
            )
            
            # Only add if we have some data
            if end_idx > start_idx:
                self.interval_indices.append((start_idx, end_idx))
            else:
                self.interval_indices.append((-1, -1))  # Mark as invalid

        # Store valid intervals count
        self.valid_intervals = sum(1 for s, e in self.interval_indices if s >= 0 and e > s)
        logger.info(f"Found {self.valid_intervals} valid intervals out of {len(self.interval_indices)} predictions")

        # --------------------------
        # Backtest parameters
        # --------------------------
        self.starting_balance = self.balance = starting_balance
        self.breaking_balance = starting_balance * 0.5
        self.take_profit = take_profit / 100
        self.stop_loss = stop_loss / 100
        self.buy_after_minutes = int(buy_after_minutes)
        self.fee = fee * leverage
        self.leverage = leverage
        self.slippage = slippage
        self.max_delay_minutes = max_delay_minutes

        # --------------------------
        # Trade tracking
        # --------------------------
        self.in_position = False
        self.buy_price = 0
        self.sell_price = 0
        self.current_direction = 0
        self.ledger = []
        self.last_signal = 0  # Track the last executed signal

    # --------------------------
    # Buy position
    # --------------------------
    def buy(self, np_interval, direction, timestamp=None):
        if len(np_interval) == 0:
            return
            
        # Use the first price in the interval (which should be at or very close to the hour)
        idx = min(self.buy_after_minutes, len(np_interval)-1)
        self.buy_price = np_interval[idx, self.idx_open]
        self.sell_price = 0
        self.current_direction = direction
        self.in_position = True
        
        # Apply entry fee/slippage
        pnl = -self.fee - self.slippage
        self.balance += self.balance * pnl/100
        buy_time = timestamp if timestamp else np_interval[idx, self.idx_time]
        
        self.record_trade(buy_time, 'buy', pnl)

    # --------------------------
    # Sell position
    # --------------------------
    def sell(self, timestamp, price, reason):
        if not self.in_position:
            return
        self.sell_price = price
        # PnL calculation based on direction
        pnl = ((self.sell_price - self.buy_price)/self.buy_price*100
               if self.current_direction > 0 else (self.buy_price - self.sell_price)/self.buy_price*100)
        pnl *= self.leverage
        pnl -= (self.fee + self.slippage)
        self.balance += self.balance * pnl/100
        self.in_position = False
        self.record_trade(timestamp, f'sell - {reason}', pnl)

    # --------------------------
    # Record trade
    # --------------------------
    def record_trade(self, timestamp, action, pnl=None):
        self.ledger.append({
            'datetime': timestamp,
            'predicted_direction': 'long' if self.current_direction > 0 else 'short',
            'action': action,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price if 'sell' in action else None,
            'balance': self.balance,
            'pnl': pnl
        })

    # --------------------------
    # TP / SL check within interval
    # --------------------------
    def check_tp_sl(self, np_interval):
        if not self.in_position:
            return
        highs = np_interval[:, self.idx_high]
        lows = np_interval[:, self.idx_low]
        timestamps = np_interval[:, self.idx_time]

        if self.current_direction > 0:  # long
            tp_price = self.buy_price * (1 + self.take_profit)
            sl_price = self.buy_price * (1 - self.stop_loss)
        else:  # short
            tp_price = self.buy_price * (1 - self.take_profit)
            sl_price = self.buy_price * (1 + self.stop_loss)

        # Find first hit of TP or SL
        tp_hits = np.where(highs >= tp_price if self.current_direction > 0 else lows <= tp_price)[0]
        sl_hits = np.where(lows <= sl_price if self.current_direction > 0 else highs >= sl_price)[0]

        first_hit = None
        if len(tp_hits) > 0 and len(sl_hits) > 0:
            first_hit = tp_hits[0] if tp_hits[0] < sl_hits[0] else sl_hits[0]
        elif len(tp_hits) > 0:
            first_hit = tp_hits[0]
        elif len(sl_hits) > 0:
            first_hit = sl_hits[0]

        if first_hit is not None:
            price = tp_price if first_hit in tp_hits else sl_price
            reason = 'take_profit' if first_hit in tp_hits else 'stop_loss'
            self.sell(timestamps[first_hit], price, reason)

    # --------------------------
    # Run backtest with improved logic
    # --------------------------
    def run(self):
        # Initialize with the first non-zero signal if exists
        first_non_zero_idx = None
        for i in range(len(self.np_pred)):
            if self.np_pred[i, self.idx_pred_signal] != 0:
                first_non_zero_idx = i
                break
        
        if first_non_zero_idx is None:
            logger.info("No non-zero signals found!")
            df_ledger = pd.DataFrame(self.ledger)
            return df_ledger, self.balance, 0

        # Start from the first non-zero signal
        start_idx = max(0, first_non_zero_idx - 1)  # Start one before to catch the interval
        
        for i in range(start_idx, len(self.np_pred)-1):
            current_pred_signal = self.np_pred[i, self.idx_pred_signal]
            pred_time = self.np_pred[i, self.idx_pred_time]
            
            # Skip if interval is invalid
            start_price_idx, end_price_idx = self.interval_indices[i]
            if start_price_idx < 0 or end_price_idx <= start_price_idx:
                continue

            np_interval = self.np_price[start_price_idx:end_price_idx]
            if len(np_interval) <= self.buy_after_minutes:
                continue
            
            # Handle zero signals: use previous non-zero signal
            if current_pred_signal == 0:
                current_pred_signal = self.last_signal if self.last_signal != 0 else 0
            
            # Skip if still zero (no valid signal)
            if current_pred_signal == 0:
                continue
            
            # Get the open price at the beginning of the interval
            open_price = np_interval[0, self.idx_open]
            
            # Check for direction change only if we're already in a position
            if self.in_position and current_pred_signal != self.last_signal:
                # Close the current position due to direction change
                self.sell(np_interval[0, self.idx_time], open_price, reason='direction_change')
                # Open new position with new direction
                self.buy(np_interval, current_pred_signal)
            elif not self.in_position:
                # No position open, open new position
                self.buy(np_interval, current_pred_signal)
            
            # Update last signal
            self.last_signal = current_pred_signal
            
            # Check for TP/SL in this interval
            self.check_tp_sl(np_interval)
            
            # Stop on huge loss
            if self.balance < self.breaking_balance:
                logger.info(f"Stopping backtest: balance {self.balance:.2f} below breaking point {self.breaking_balance:.2f}")
                break

        # Close any open position at the end
        if self.in_position and len(self.np_price) > 0:
            last_price = self.np_price[-1, self.idx_close]
            self.sell(self.np_price[-1, self.idx_time], last_price, reason='end_of_backtest')

        # Prepare ledger
        df_ledger = pd.DataFrame(self.ledger)
        if len(df_ledger) > 0:
            df_ledger['pnl_sum'] = df_ledger['pnl'].cumsum()
            df_ledger[['balance','pnl','pnl_sum']] = df_ledger[['balance','pnl','pnl_sum']].round(2)
        else:
            df_ledger = pd.DataFrame(columns=['datetime', 'predicted_direction', 'action', 'buy_price', 'sell_price', 'balance', 'pnl', 'pnl_sum'])

        final_balance = round(self.balance, 2)
        total_pnl_percent = round(df_ledger['pnl_sum'].iloc[-1] if len(df_ledger) > 0 else 0, 2)

        return df_ledger, final_balance, total_pnl_percent