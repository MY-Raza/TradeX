import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    starting_balance: float = 10000.0
    leverage: float = 1.0
    transaction_fee: float = 0.001
    slippage: float = 0.0005
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.01
    buy_after_minutes: int = 1
    min_balance_pct: float = 0.5


class Backtester:

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        self.balance = self.config.starting_balance
        self.trades = []
        self.current_position = None
        self.position_open = False

    # ==========================================================
    # 🔒 TRADE RECORD GUARD (YOUR RULES)
    # ==========================================================
    def _record_trade(self, trade: dict):
        direction = trade["predicted_direction"]
        action = trade["action"]

        allowed_actions = {"buy", "sell-tp", "sell-sl", "sell-direction-change"}

        if direction not in {"long", "short"}:
            return

        if action not in allowed_actions:
            return

        # buy/sell price exclusivity
        if trade["buy_price"] is not None:
            trade["sell_price"] = None

        if trade["sell_price"] is not None:
            trade["buy_price"] = None

        self.trades.append(trade)

    # ==========================================================
    def run(self, price_data: pd.DataFrame, prediction_data: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
        self.reset()

        price_data = self._prepare_price_data(price_data)
        signals = self._prepare_signals(prediction_data, price_data)

        timestamps = price_data["timestamp"].values
        opens = price_data["open"].values
        highs = price_data["high"].values
        lows = price_data["low"].values

        i = 0
        while i < len(signals):

            if self._check_risk_stop():
                break

            signal = signals[i]

            if not self.position_open:
                if signal != 0:
                    i = self._enter_position(i, signal, timestamps, opens)
            else:
                i = self._manage_position(i, signal, timestamps, opens, highs, lows)

            i += 1

        return self._build_results()

    # ==========================================================
    def _prepare_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[["timestamp", "open", "high", "low"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    def _prepare_signals(self, pred_df: pd.DataFrame, price_df: pd.DataFrame) -> np.ndarray:
        merged = price_df[["timestamp"]].merge(
            pred_df[["timestamp", "signals"]],
            on="timestamp",
            how="left"
        )

        signals = merged["signals"].ffill().fillna(0).values
        out = np.zeros(len(signals), dtype=int)

        last = 0
        for i, s in enumerate(signals):
            if s != 0:
                last = s
            out[i] = last

        return out

    def _check_risk_stop(self) -> bool:
        return self.balance < self.config.starting_balance * self.config.min_balance_pct

    # ==========================================================
    def _enter_position(self, idx, signal, timestamps, opens):
        entry_idx = min(idx + self.config.buy_after_minutes, len(opens) - 1)
        price = opens[entry_idx]
        price_adj = self._apply_entry_costs(price, signal)

        self.current_position = {
            "direction": signal,
            "buy_price": price,
            "buy_price_adj": price_adj,
            "tp": self._tp(price, signal),
            "sl": self._sl(price, signal)
        }

        self.position_open = True

        self._record_trade({
            "datetime": timestamps[entry_idx],
            "predicted_direction": "long" if signal == 1 else "short",
            "action": "buy",
            "buy_price": price,
            "sell_price": None,
            "balance": self.balance,
            "pnl": 0.0
        })

        return entry_idx

    # ==========================================================
    def _manage_position(self, idx, signal, timestamps, opens, highs, lows):
        pos = self.current_position

        tp_hit, sl_hit = self._check_tp_sl(highs[idx], lows[idx], pos)

        if tp_hit or sl_hit:
            exit_price = pos["tp"] if tp_hit else pos["sl"]
            action = "sell-tp" if tp_hit else "sell-sl"

            pnl = self._calculate_pnl(pos["buy_price_adj"], exit_price, pos["direction"])
            self._update_balance(pnl)

            self._record_trade({
                "datetime": timestamps[idx],
                "predicted_direction": "long" if pos["direction"] == 1 else "short",
                "action": action,
                "buy_price": None,
                "sell_price": exit_price,
                "balance": self.balance,
                "pnl": pnl
            })

            self.position_open = False
            self.current_position = None
            return idx

        if signal != 0 and signal != pos["direction"]:
            exit_idx = min(idx + self.config.buy_after_minutes, len(opens) - 1)
            exit_price = opens[exit_idx]

            pnl = self._calculate_pnl(pos["buy_price_adj"], exit_price, pos["direction"])
            self._update_balance(pnl)

            self._record_trade({
                "datetime": timestamps[exit_idx],
                "predicted_direction": "long" if pos["direction"] == 1 else "short",
                "action": "sell-direction-change",
                "buy_price": None,
                "sell_price": exit_price,
                "balance": self.balance,
                "pnl": pnl
            })

            self.position_open = False
            self.current_position = None
            return self._enter_position(exit_idx, signal, timestamps, opens)

        return idx

    # ==========================================================
    def _apply_entry_costs(self, price, direction):
        return price * (1 + self.config.transaction_fee) * (
            1 + self.config.slippage if direction == 1 else 1 - self.config.slippage
        )

    def _tp(self, price, direction):
        return price * (1 + self.config.take_profit_pct if direction == 1 else 1 - self.config.take_profit_pct)

    def _sl(self, price, direction):
        return price * (1 - self.config.stop_loss_pct if direction == 1 else 1 + self.config.stop_loss_pct)

    def _check_tp_sl(self, high, low, pos):
        if pos["direction"] == 1:
            return high >= pos["tp"], low <= pos["sl"]
        else:
            return low <= pos["tp"], high >= pos["sl"]

    def _calculate_pnl(self, entry, exit_price, direction):
        exit_adj = exit_price * (1 - self.config.transaction_fee)
        raw = (exit_adj - entry) / entry if direction == 1 else (entry - exit_adj) / entry
        return raw * self.config.leverage

    def _update_balance(self, pnl):
        self.balance *= (1 + pnl)

    # ==========================================================
    def _build_results(self):
        if not self.trades:
            return pd.DataFrame(), self.balance, 0.0

        df = pd.DataFrame(self.trades)
        df["pnl"] *= 100
        df["cumulative_pnl"] = (df["balance"] / self.config.starting_balance - 1) * 100

        return df, self.balance, df["cumulative_pnl"].iloc[-1]
