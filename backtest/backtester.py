import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    starting_balance: float = 1000.0
    leverage: float = 1.0
    transaction_fee: float = 0.05
    slippage: float = 0.0
    take_profit_pct: float = 0.03  
    stop_loss_pct: float = 0.01   
    buy_after_minutes: int = 0
    min_balance_pct: float = 0.5

class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, price_df: pd.DataFrame, signal_df: pd.DataFrame):
        price_df = price_df.copy()
        signal_df = signal_df.copy()
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
        signal_df["timestamp"] = pd.to_datetime(signal_df["timestamp"])
        price_df = price_df.sort_values("timestamp").reset_index(drop=True)
        signal_df = signal_df.sort_values("timestamp").reset_index(drop=True)

        np_price = price_df[["timestamp", "open", "high", "low"]].to_numpy()
        np_signal = signal_df[["timestamp", "signals"]].to_numpy()

        ts_price = np_price[:, 0]
        ts_signal = np_signal[:, 0]

        # Precompute intervals
        intervals = []
        for i in range(len(ts_signal) - 1):
            start = np.searchsorted(ts_price, ts_signal[i])
            end = np.searchsorted(ts_price, ts_signal[i + 1])
            intervals.append((start, end))

        balance = self.config.starting_balance
        breaking_balance = balance * self.config.min_balance_pct
        fee_pct = self.config.transaction_fee * self.config.leverage
        slippage = self.config.slippage

        in_position = False
        buy_price = 0
        sell_price = 0
        direction = 0
        pnl_sum = 0
        ledger = []

        prev_direction = np_signal[0][1]

        for i, (start_idx, end_idx) in enumerate(intervals):
            if balance < breaking_balance:
                break

            current_direction = np_signal[i][1]
            if current_direction == 0:
                current_direction = prev_direction

            np_temp = np_price[start_idx:end_idx]
            if len(np_temp) <= 10:
                prev_direction = current_direction
                continue

            opens = np_temp[:, 1]
            highs = np_temp[:, 2]
            lows = np_temp[:, 3]
            times = np_temp[:, 0]

            entry_idx = min(self.config.buy_after_minutes, len(opens) - 1)

            # -------------------------
            # ENTER POSITION
            # -------------------------
            if not in_position:
                buy_price = opens[entry_idx]
                direction = current_direction

                pnl = -fee_pct - slippage
                balance += balance * (pnl / 100)
                pnl_sum += pnl

                ledger.append([times[entry_idx],
                               "long" if direction > 0 else "short",
                               "buy", buy_price, 0,
                               round(balance, 2), round(pnl, 2), round(pnl_sum, 2)])
                in_position = True

            # -------------------------
            # CHECK TP/SL (deterministic)
            # -------------------------
            if in_position:
                if direction > 0:
                    tp_price = buy_price * (1 + self.config.take_profit_pct)
                    sl_price = buy_price * (1 - self.config.stop_loss_pct)
                    tp_hits = np.where(highs >= tp_price)[0]
                    sl_hits = np.where(lows <= sl_price)[0]
                else:
                    tp_price = buy_price * (1 - self.config.take_profit_pct)
                    sl_price = buy_price * (1 + self.config.stop_loss_pct)
                    tp_hits = np.where(lows <= tp_price)[0]
                    sl_hits = np.where(highs >= sl_price)[0]

                hit_idx = None
                action_type = None

                # Determine which occurs first; TP has priority if both hit same candle
                if len(tp_hits) and len(sl_hits):
                    if tp_hits[0] <= sl_hits[0]:
                        hit_idx = tp_hits[0]
                        action_type = "take_profit"
                    else:
                        hit_idx = sl_hits[0]
                        action_type = "stop_loss"
                elif len(tp_hits):
                    hit_idx = tp_hits[0]
                    action_type = "take_profit"
                elif len(sl_hits):
                    hit_idx = sl_hits[0]
                    action_type = "stop_loss"

                if hit_idx is not None:
                    if direction > 0:
                        sell_price = highs[hit_idx] if action_type == "take_profit" else lows[hit_idx]
                    else:
                        sell_price = lows[hit_idx] if action_type == "take_profit" else highs[hit_idx]

                    pnl = ((sell_price - buy_price) / buy_price * 100) if direction > 0 \
                        else ((buy_price - sell_price) / buy_price * 100)
                    pnl *= self.config.leverage
                    pnl -= fee_pct + slippage

                    balance += balance * (pnl / 100)
                    pnl_sum += pnl

                    ledger.append([times[hit_idx],
                                   "long" if direction > 0 else "short",
                                   f"sell - {action_type}", buy_price, sell_price,
                                   round(balance, 2), round(pnl, 2), round(pnl_sum, 2)])
                    in_position = False

            # -------------------------
            # EXIT ON DIRECTION CHANGE
            # -------------------------
            if in_position and current_direction != prev_direction:
                sell_price = opens[entry_idx]
                pnl = ((sell_price - buy_price) / buy_price * 100) if prev_direction > 0 \
                    else ((buy_price - sell_price) / buy_price * 100)
                pnl *= self.config.leverage
                pnl -= fee_pct + slippage

                balance += balance * (pnl / 100)
                pnl_sum += pnl

                ledger.append([times[entry_idx],
                               "long" if prev_direction > 0 else "short",
                               "sell - direction change", buy_price, sell_price,
                               round(balance, 2), round(pnl, 2), round(pnl_sum, 2)])
                in_position = False

            prev_direction = current_direction

        columns = ["timestamp", "predicted_direction", "action",
                   "buy_price", "sell_price", "balance", "pnl", "pnl_sum"]

        return pd.DataFrame(ledger, columns=columns), round(balance, 2), round(pnl_sum, 2)
