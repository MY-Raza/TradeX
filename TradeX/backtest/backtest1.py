# backtest1.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Backtest:
    """
    Optimized Backtesting class for trading strategies based on predicted directions.
    Works with minute-level OHLCV data and model predictions.
    Only 'buy' and 'sell' actions are recorded.
    """

    def __init__(self, df_price, df_predictions, starting_balance=1000,
                 take_profit=100, stop_loss=100, buy_after_minutes=0,
                 transaction_fee=0.05, leverage=1.0, slippage=0.0):

        # --------------------------
        # Prepare price data
        # --------------------------
        df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
        self.np_1m = df_price.to_numpy()
        self.index_1m_datetime = df_price.columns.get_loc("timestamp")
        self.index_1m_open = df_price.columns.get_loc("open")
        self.index_1m_high = df_price.columns.get_loc("high")
        self.index_1m_low = df_price.columns.get_loc("low")

        # Timestamps array for fast lookup
        self.timestamps_1m = self.np_1m[:, self.index_1m_datetime]

        # --------------------------
        # Prepare predictions data
        # --------------------------
        df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp'])
        self.np_model_predictions = df_predictions.to_numpy()
        self.index_pred_datetime = df_predictions.columns.get_loc("timestamp")
        self.index_pred_direction = df_predictions.columns.get_loc("signals")

        # Timestamps array for predictions
        self.timestamps_pred = self.np_model_predictions[:, self.index_pred_datetime]

        # Precompute 1-minute interval indices for each prediction
        self.interval_indices = []
        for i in range(len(self.timestamps_pred) - 1):
            start_idx = np.searchsorted(self.timestamps_1m, self.timestamps_pred[i])
            end_idx = np.searchsorted(self.timestamps_1m, self.timestamps_pred[i + 1])
            self.interval_indices.append((start_idx, end_idx))

        # --------------------------
        # Backtest parameters
        # --------------------------
        self.starting_balance = self.current_balance = starting_balance
        self.breaking_balance = self.current_balance * 0.5
        self.take_profit_percent = take_profit / 100
        self.stop_loss_percent = stop_loss / 100
        self.buy_after_minutes = int(buy_after_minutes)
        self.transaction_fee_percent = transaction_fee * leverage
        self.leverage = leverage
        self.slippage = slippage

        # --------------------------
        # Trade tracking
        # --------------------------
        self.in_position = False
        self.buy_price = 0
        self.sell_price = 0
        self.array_to_save = []
        self.header_names = [
            'datetime', 'predicted_direction', 'action', 'buy_price',
            'sell_price', 'balance', 'pnl'
        ]

    # --------------------------
    # Buy position
    # --------------------------
    def buy(self, np_temp):
        self.buy_price = np_temp[self.buy_after_minutes][self.index_1m_open]
        self.sell_price = 0
        pnl = -self.transaction_fee_percent - self.slippage
        self.current_balance += self.current_balance * (pnl / 100)
        self.in_position = True
        self.record_trade(np_temp[self.buy_after_minutes][self.index_1m_datetime], 'buy', pnl)

    # --------------------------
    # PnL when direction changes
    # --------------------------
    def pnl_direction_change(self, sell_datetime):
        if not self.in_position:
            return

        pnl = ((self.sell_price - self.buy_price) / self.buy_price * 100) if self.previous_pred_direction > 0 \
            else ((self.buy_price - self.sell_price) / self.buy_price * 100)
        pnl *= self.leverage
        pnl -= self.transaction_fee_percent + self.slippage

        self.current_balance += self.current_balance * (pnl / 100)
        self.in_position = False
        self.record_trade(sell_datetime, 'sell - direction change', pnl)

    # --------------------------
    # Find take profit or stop loss index
    # --------------------------
    def find_tp_sl_index(self, take_profit_amount, stop_loss_amount, np_temp_high, np_temp_low):
        if self.current_pred_direction > 0:
            high_hits = np.where(np_temp_high >= take_profit_amount)[0]
            low_hits = np.where(np_temp_low <= stop_loss_amount)[0]
        else:
            high_hits = np.where(np_temp_high >= stop_loss_amount)[0]
            low_hits = np.where(np_temp_low <= take_profit_amount)[0]

        if len(high_hits) == 0 and len(low_hits) == 0:
            return False, -1
        elif len(high_hits) > 0 and len(low_hits) == 0:
            self.sell_price = np_temp_high[high_hits[0]]
            return True, high_hits[0]
        elif len(high_hits) == 0 and len(low_hits) > 0:
            self.sell_price = np_temp_low[low_hits[0]]
            return True, low_hits[0]
        else:
            if high_hits[0] < low_hits[0]:
                self.sell_price = np_temp_high[high_hits[0]]
                return True, high_hits[0]
            else:
                self.sell_price = np_temp_low[low_hits[0]]
                return True, low_hits[0]

    # --------------------------
    # Check TP/SL conditions
    # --------------------------
    def check_tp_sl(self, np_temp, np_temp_high, np_temp_low):
        if not self.in_position:
            return

        if self.current_pred_direction > 0:  # long
            take_profit_amount = self.buy_price * (1 + self.take_profit_percent)
            stop_loss_amount = self.buy_price * (1 - self.stop_loss_percent)
        else:  # short
            take_profit_amount = self.buy_price * (1 - self.take_profit_percent)
            stop_loss_amount = self.buy_price * (1 + self.stop_loss_percent)

        tp_sl_hit, index = self.find_tp_sl_index(take_profit_amount, stop_loss_amount, np_temp_high, np_temp_low)
        if tp_sl_hit:
            pnl = ((self.sell_price - self.buy_price) / self.buy_price * 100) if self.previous_pred_direction > 0 \
                else ((self.buy_price - self.sell_price) / self.buy_price * 100)
            pnl *= self.leverage
            pnl -= self.transaction_fee_percent + self.slippage

            self.current_balance += self.current_balance * (pnl / 100)
            self.in_position = False

            action_type = ' - take_profit' if pnl > 0 else ' - stop_loss'
            self.record_trade(np_temp[index][self.index_1m_datetime], 'sell' + action_type, pnl)

    # --------------------------
    # Record trades
    # --------------------------
    def record_trade(self, datetime, action, pnl):
        # Only record buy or sell actions
        if 'buy' in action or 'sell' in action:
            self.array_to_save.append([
                datetime,
                'long' if self.current_pred_direction > 0 else 'short',
                action,
                self.buy_price,
                self.sell_price,
                self.current_balance,
                pnl
            ])

    # --------------------------
    # Get 1-minute interval data
    # --------------------------
    def get_interval_min_data(self, index):
        start_idx, end_idx = self.interval_indices[index]
        np_temp = self.np_1m[start_idx:end_idx]
        np_temp_high = np_temp[:, self.index_1m_high]
        np_temp_low = np_temp[:, self.index_1m_low]
        return np_temp, np_temp_high, np_temp_low

    # --------------------------
    # Run backtest
    # --------------------------
    def run(self):
        self.previous_pred_direction = self.current_pred_direction = self.np_model_predictions[0][self.index_pred_direction]
        break_on_huge_loss = False

        for i in range(len(self.np_model_predictions) - 1):
            self.current_pred_direction = self.np_model_predictions[i][self.index_pred_direction]

            # Handle 0 signals
            if self.current_pred_direction == 0:
                if self.previous_pred_direction == 0:
                    self.previous_pred_direction = self.current_pred_direction
                    continue
                self.current_pred_direction = self.previous_pred_direction

            np_temp, np_temp_high, np_temp_low = self.get_interval_min_data(i)
            if len(np_temp) <= 10:
                continue

            # Buy if not in position
            if not self.in_position:
                self.buy(np_temp)
                self.previous_pred_direction = self.current_pred_direction

            # Handle direction change
            if self.current_pred_direction != self.previous_pred_direction:
                if len(np_temp) >= 10:
                    self.sell_price = np_temp[self.buy_after_minutes][self.index_1m_open]
                    sell_datetime = np_temp[self.buy_after_minutes][self.index_1m_datetime]
                    self.pnl_direction_change(sell_datetime)

                    if not self.in_position:
                        self.buy(np_temp)

            # Check TP/SL
            self.check_tp_sl(np_temp, np_temp_high, np_temp_low)
            self.previous_pred_direction = self.current_pred_direction

            # Break on huge loss
            if self.current_balance < self.breaking_balance:
                break_on_huge_loss = True
                break

        # Prepare ledger
        df_ledger = pd.DataFrame(self.array_to_save, columns=self.header_names)
        df_ledger['pnl_sum'] = df_ledger['pnl'].cumsum()
        df_ledger[['balance', 'pnl', 'pnl_sum']] = df_ledger[['balance', 'pnl', 'pnl_sum']].round(2)

        pnl_percent = np.round(df_ledger["pnl_sum"].iloc[-1], 2) if len(df_ledger) > 1 else 0

        if break_on_huge_loss:
            return df_ledger, -1000, pnl_percent
        else:
            return df_ledger, round(self.current_balance, 2), round(pnl_percent, 2)
