import pandas as pd
import numpy as np

class HighPerfBacktest:
    """
    High-performance backtesting class for trading strategies based on predicted directions.
    Works with minute-level OHLCV data and model predictions.
    """

    def __init__(self, df_price, df_predictions, starting_balance=1000,
                 take_profit=1, stop_loss=1, fee=0.05, leverage=1.0, slippage=0.0,
                 buy_after_minutes=0):

        df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
        df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp'])

        self.np_price = df_price.to_numpy()
        self.idx_time = df_price.columns.get_loc('timestamp')
        self.idx_open = df_price.columns.get_loc('open')
        self.idx_high = df_price.columns.get_loc('high')
        self.idx_low = df_price.columns.get_loc('low')

        self.np_pred = df_predictions.to_numpy()
        self.idx_pred_time = df_predictions.columns.get_loc('timestamp')
        self.idx_pred_signal = df_predictions.columns.get_loc('signals')

        # Precompute interval indices for each prediction
        self.interval_indices = []
        for i in range(len(self.np_pred)-1):
            start_idx = np.searchsorted(self.np_price[:, self.idx_time], self.np_pred[i, self.idx_pred_time])
            end_idx = np.searchsorted(self.np_price[:, self.idx_time], self.np_pred[i+1, self.idx_pred_time])
            self.interval_indices.append((start_idx, end_idx))

        # Parameters
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.breaking_balance = starting_balance * 0.5
        self.take_profit = take_profit / 100
        self.stop_loss = stop_loss / 100
        self.fee = fee * leverage
        self.leverage = leverage
        self.slippage = slippage
        self.buy_after_minutes = buy_after_minutes

        # Trade tracking
        self.in_position = False
        self.buy_price = 0
        self.sell_price = 0
        self.current_direction = 0
        self.prev_direction = 0
        self.ledger = []

    # --------------------------
    # Buy position
    # --------------------------
    def buy(self, timestamp, price, direction):
        self.buy_price = price
        self.sell_price = 0
        self.current_direction = direction
        self.in_position = True
        # Entry fee/slippage
        pnl = -self.fee - self.slippage
        self.balance += self.balance * pnl / 100
        self.record_trade(timestamp, 'buy', pnl)

    # --------------------------
    # Sell position
    # --------------------------
    def sell(self, timestamp, price, reason):
        if not self.in_position:
            return
        self.sell_price = price
        pnl = ((self.sell_price - self.buy_price)/self.buy_price*100
               if self.prev_direction > 0 else (self.buy_price - self.sell_price)/self.buy_price*100)
        pnl *= self.leverage
        pnl -= (self.fee + self.slippage)
        self.balance += self.balance * pnl / 100
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

        if self.current_direction > 0:
            tp_price = self.buy_price * (1 + self.take_profit)
            sl_price = self.buy_price * (1 - self.stop_loss)
        else:
            tp_price = self.buy_price * (1 - self.take_profit)
            sl_price = self.buy_price * (1 + self.stop_loss)

        # First TP/SL hit in the interval
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
    # Run backtest
    # --------------------------
    def run(self):
        self.prev_direction = self.current_direction = self.np_pred[0, self.idx_pred_signal]

        for i in range(len(self.interval_indices)):
            start_idx, end_idx = self.interval_indices[i]
            np_interval = self.np_price[start_idx:end_idx]
            if len(np_interval) <= self.buy_after_minutes:
                continue

            # Get current prediction
            self.current_direction = self.np_pred[i, self.idx_pred_signal]
            if self.current_direction == 0:
                self.current_direction = self.prev_direction if self.prev_direction != 0 else 1

            # Only buy at the start of the interval
            open_price = np_interval[min(self.buy_after_minutes, len(np_interval)-1), self.idx_open]

            if not self.in_position:
                self.buy(np_interval[0, self.idx_time], open_price, self.current_direction)

            # Direction change only occurs at **new prediction interval**, not every minute
            if self.in_position and self.current_direction != self.prev_direction:
                sell_price = open_price
                self.sell(np_interval[0, self.idx_time], sell_price, reason='direction_change')
                self.buy(np_interval[0, self.idx_time], sell_price, self.current_direction)

            # TP/SL check
            self.check_tp_sl(np_interval)

            # Stop on huge loss
            if self.balance < self.breaking_balance:
                break

            self.prev_direction = self.current_direction

        df_ledger = pd.DataFrame(self.ledger)
        if len(df_ledger) > 0:
            df_ledger['pnl_sum'] = df_ledger['pnl'].cumsum()
            df_ledger[['balance','pnl','pnl_sum']] = df_ledger[['balance','pnl','pnl_sum']].round(2)
        else:
            df_ledger['pnl_sum'] = 0

        final_balance = round(self.balance, 2)
        total_pnl_percent = round(df_ledger['pnl_sum'].iloc[-1] if len(df_ledger) > 0 else 0, 2)

        return df_ledger, final_balance, total_pnl_percent
