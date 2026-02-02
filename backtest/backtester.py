import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


# =========================
# Configuration
# =========================
@dataclass
class BacktestConfig:
    starting_balance: float = 1000.0
    leverage: float = 1.0
    transaction_fee: float = 0.05
    slippage: float = 0.0
    take_profit_pct: float = 0.03
    stop_loss_pct: float = 0.01
    buy_after_minutes: int = 1
    min_balance_pct: float = 0.5


# =========================
# Backtester (interval-based)
# =========================
class Backtester:

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        self.balance = self.config.starting_balance
        self.breaking_balance = self.balance * self.config.min_balance_pct
        self.in_position = False
        self.buy_price = 0.0
        self.sell_price = 0.0
        self.current_direction = None
        self.trades = []

    # =========================
    # Helpers
    # =========================
    def _update_balance(self, pnl_pct: float):
        self.balance *= (1 + pnl_pct / 100)

    def _record(self, dt, direction, action, pnl):
        self.trades.append([
            dt,
            "long" if direction == 1 else "short",
            action,
            self.buy_price,
            self.sell_price,
            self.balance,
            pnl
        ])

    # =========================
    # TP / SL detection (EARLIEST)
    # =========================
    def _find_tp_sl(self, highs, lows, tp, sl, direction):
        if direction == 1:
            tp_hits = np.where(highs >= tp)[0]
            sl_hits = np.where(lows <= sl)[0]
        else:
            tp_hits = np.where(lows <= tp)[0]
            sl_hits = np.where(highs >= sl)[0]

        if len(tp_hits) == 0 and len(sl_hits) == 0:
            return None, None

        if len(tp_hits) and len(sl_hits):
            return ("tp", tp) if tp_hits[0] < sl_hits[0] else ("sl", sl)

        return ("tp", tp) if len(tp_hits) else ("sl", sl)

    # =========================
    # Run Backtest
    # =========================
    def run(self, price_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:

        self.reset()

        # ---- Prepare data
        price_df = price_df[["timestamp", "open", "high", "low"]].copy()
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

        price_np = price_df.to_numpy()
        ts_price = price_np[:, 0]

        pred_np = pred_df[["timestamp", "signals"]].to_numpy()
        ts_pred = pred_np[:, 0]
        signals = pred_np[:, 1]

        # ---- Build prediction intervals (CRITICAL FIX)
        intervals = []
        for i in range(len(ts_pred) - 1):
            s = np.searchsorted(ts_price, ts_pred[i])
            e = np.searchsorted(ts_price, ts_pred[i + 1])
            intervals.append((s, e, signals[i]))

        # ---- Main loop (INTERVAL BASED)
        for start, end, direction in intervals:

            if self.balance < self.breaking_balance:
                break

            if direction == 0:
                continue

            if end - start <= self.config.buy_after_minutes:
                continue

            np_temp = price_np[start:end]
            highs = np_temp[:, 2]
            lows = np_temp[:, 3]

            # ================= BUY =================
            if not self.in_position:
                entry_idx = start + self.config.buy_after_minutes
                self.buy_price = price_np[entry_idx][1]

                fee = (self.config.transaction_fee + self.config.slippage) * 100
                self._update_balance(-fee)

                self.current_direction = direction
                self.in_position = True

                self._record(
                    price_np[entry_idx][0],
                    direction,
                    "buy",
                    -fee
                )

            # ================= TP / SL =================
            tp = self.buy_price * (1 + self.config.take_profit_pct) if direction == 1 \
                else self.buy_price * (1 - self.config.take_profit_pct)

            sl = self.buy_price * (1 - self.config.stop_loss_pct) if direction == 1 \
                else self.buy_price * (1 + self.config.stop_loss_pct)

            hit_type, hit_price = self._find_tp_sl(highs, lows, tp, sl, direction)

            if hit_type:
                pnl = (
                    (hit_price - self.buy_price) / self.buy_price
                    if direction == 1
                    else (self.buy_price - hit_price) / self.buy_price
                ) * 100 * self.config.leverage

                pnl -= 2 * (self.config.transaction_fee + self.config.slippage) * 100
                self.sell_price = hit_price
                self._update_balance(pnl)

                self._record(
                    price_np[start][0],
                    direction,
                    f"sell - {'take_profit' if hit_type == 'tp' else 'stop_loss'}",
                    pnl
                )

                self.in_position = False
                continue

            # ================= DIRECTION CHANGE EXIT =================
            exit_price = price_np[start + self.config.buy_after_minutes][1]

            pnl = (
                (exit_price - self.buy_price) / self.buy_price
                if direction == 1
                else (self.buy_price - exit_price) / self.buy_price
            ) * 100 * self.config.leverage

            pnl -= 2 * (self.config.transaction_fee + self.config.slippage) * 100

            self.sell_price = exit_price
            self._update_balance(pnl)

            self._record(
                price_np[start][0],
                direction,
                "sell - direction change",
                pnl
            )

            self.in_position = False

        # ================= RESULTS =================
        df = pd.DataFrame(self.trades, columns=[
            "datetime", "predicted_direction", "action",
            "buy_price", "sell_price", "balance", "pnl"
        ])

        if df.empty:
            return df, self.balance, 0.0

        df["pnl_sum"] = df["pnl"].cumsum()
        df[["balance", "pnl", "pnl_sum"]] = df[["balance", "pnl", "pnl_sum"]].round(2)

        return df, round(self.balance, 2), round(df["pnl_sum"].iloc[-1], 2)
