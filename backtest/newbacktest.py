import pandas as pd
import numpy as np
from TradeX.utils.common.logs import get_logger

logger = get_logger("backtest_class")


class HighPerfBacktest:
    """
    High-performance backtesting engine for direction-based trading strategies.
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

        df_price['datetime'] = pd.to_datetime(df_price['datetime'])
        df_predictions['datetime'] = pd.to_datetime(df_predictions['datetime'])

        self.np_price = df_price.to_numpy()
        self.idx_time = df_price.columns.get_loc('datetime')
        self.idx_open = df_price.columns.get_loc('open')
        self.idx_high = df_price.columns.get_loc('high')
        self.idx_low = df_price.columns.get_loc('low')
        self.idx_close = df_price.columns.get_loc('close')
        self.timestamps_price = self.np_price[:, self.idx_time]

        self.np_pred = df_predictions.to_numpy()
        self.idx_pred_time = df_predictions.columns.get_loc('datetime')
        self.idx_pred_signal = df_predictions.columns.get_loc('signals')
        self.timestamps_pred = self.np_pred[:, self.idx_pred_time]

        # Precompute intervals
        self.interval_indices = []
        for i in range(len(self.np_pred) - 1):
            pred_time = self.np_pred[i, self.idx_pred_time]
            start_idx = np.searchsorted(self.timestamps_price, pred_time, side='left')

            end_pred_time = self.np_pred[i + 1, self.idx_pred_time]
            end_idx = np.searchsorted(self.timestamps_price, end_pred_time, side='left')

            if end_idx > start_idx:
                self.interval_indices.append((start_idx, end_idx))
            else:
                self.interval_indices.append((-1, -1))

        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.breaking_balance = starting_balance * 0.5
        self.take_profit = take_profit / 100
        self.stop_loss = stop_loss / 100
        self.buy_after_minutes = int(buy_after_minutes)
        self.fee = fee * leverage
        self.leverage = leverage
        self.slippage = slippage
        self.max_delay_minutes = max_delay_minutes

        # Trade state
        self.in_position = False
        self.buy_price = 0.0
        self.sell_price = 0.0
        self.current_direction = 0
        self.ledger = []
        self.last_signal = 0

    # =========================
    # Trade Execution Methods
    # =========================

    def buy(self, np_interval, direction, timestamp=None):
        if len(np_interval) == 0 or self.in_position:
            return

        idx = min(self.buy_after_minutes, len(np_interval) - 1)
        self.buy_price = np_interval[idx, self.idx_open]
        self.current_direction = direction
        self.in_position = True

        pnl = -self.fee - self.slippage
        self.balance += self.balance * pnl / 100

        buy_time = timestamp if timestamp else np_interval[idx, self.idx_time]
        self.record_trade(buy_time, 'buy', pnl)

    def sell(self, timestamp, price, reason):
        if not self.in_position:
            return

        self.sell_price = price

        if self.current_direction > 0:
            pnl = (self.sell_price - self.buy_price) / self.buy_price * 100
        else:
            pnl = (self.buy_price - self.sell_price) / self.buy_price * 100

        pnl *= self.leverage
        pnl -= (self.fee + self.slippage)

        self.balance += self.balance * pnl / 100
        self.in_position = False

        self.record_trade(timestamp, f'sell - {reason}', pnl)

    def record_trade(self, timestamp, action, pnl=None):
        self.ledger.append({
            'datetime': timestamp,
            'direction': 'long' if self.current_direction > 0 else 'short',
            'action': action,
            'buy_price': self.buy_price if 'buy' in action else 0.0,
            'sell_price': self.sell_price if 'sell' in action else 0.0,
            'balance': self.balance,
            'pnl': pnl
        })

    # =========================
    # TP / SL Logic
    # =========================

    def check_tp_sl(self, np_interval):
        if not self.in_position:
            return False

        highs = np_interval[:, self.idx_high]
        lows = np_interval[:, self.idx_low]
        timestamps = np_interval[:, self.idx_time]

        if self.current_direction > 0:
            tp_price = self.buy_price * (1 + self.take_profit)
            sl_price = self.buy_price * (1 - self.stop_loss)
            tp_hits = np.where(highs >= tp_price)[0]
            sl_hits = np.where(lows <= sl_price)[0]
        else:
            tp_price = self.buy_price * (1 - self.take_profit)
            sl_price = self.buy_price * (1 + self.stop_loss)
            tp_hits = np.where(lows <= tp_price)[0]
            sl_hits = np.where(highs >= sl_price)[0]

        first_hit = None
        reason = None

        if len(tp_hits) > 0 and len(sl_hits) > 0:
            if tp_hits[0] < sl_hits[0]:
                first_hit = tp_hits[0]
                reason = 'take_profit'
            else:
                first_hit = sl_hits[0]
                reason = 'stop_loss'
        elif len(tp_hits) > 0:
            first_hit = tp_hits[0]
            reason = 'take_profit'
        elif len(sl_hits) > 0:
            first_hit = sl_hits[0]
            reason = 'stop_loss'

        if first_hit is not None:
            price = tp_price if reason == 'take_profit' else sl_price
            self.sell(timestamps[first_hit], price, reason)
            return True

        return False

    # =========================
    # Backtest Runner
    # =========================

    def run(self):

        first_non_zero = None
        for i in range(len(self.np_pred)):
            if self.np_pred[i, self.idx_pred_signal] != 0:
                first_non_zero = i
                break

        if first_non_zero is None:
            return pd.DataFrame(self.ledger), self.balance, 0

        start_idx = max(0, first_non_zero - 1)

        for i in range(start_idx, len(self.np_pred) - 1):

            raw_signal = self.np_pred[i, self.idx_pred_signal]
            start_price, end_price = self.interval_indices[i]

            if start_price < 0 or end_price <= start_price:
                continue

            np_interval = self.np_price[start_price:end_price]

            if len(np_interval) <= self.buy_after_minutes:
                continue

            signal = raw_signal if raw_signal != 0 else self.last_signal
            if signal == 0:
                continue

            pred_time = self.np_pred[i, self.idx_pred_time]

            closed_this_interval = False

            # 1️⃣ First check TP/SL if in position
            if self.in_position:
                closed_this_interval = self.check_tp_sl(np_interval)

            # 2️⃣ If still in position after TP/SL → check direction change
            if self.in_position and self.current_direction != signal:
                open_price = np_interval[0, self.idx_open]
                self.sell(pred_time, open_price, 'direction_change')
                self.buy(np_interval, signal, timestamp=pred_time)

            # 3️⃣ If not in position → open only if NOT closed in same interval
            elif not self.in_position and not closed_this_interval:
                self.buy(np_interval, signal, timestamp=pred_time)

            self.last_signal = signal

            if self.balance < self.breaking_balance:
                logger.info("Breaking balance reached – stopping")
                break

        # Close open trade at end
        if self.in_position and len(self.np_price) > 0:
            self.sell(
                self.np_price[-1, self.idx_time],
                self.np_price[-1, self.idx_close],
                'end_of_backtest'
            )

        df_ledger = pd.DataFrame(self.ledger)

        if len(df_ledger) > 0:
            df_ledger['pnl_sum'] = df_ledger['pnl'].cumsum()
            df_ledger[['balance', 'pnl', 'pnl_sum']] = df_ledger[
                ['balance', 'pnl', 'pnl_sum']
            ].round(2)

        final_balance = round(self.balance, 2)
        total_pnl = round(df_ledger['pnl_sum'].iloc[-1] if len(df_ledger) > 0 else 0, 2)

        return df_ledger, final_balance, total_pnl