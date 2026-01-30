import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    starting_balance: float = 1000.0
    leverage: float = 1.0
    transaction_fee: float = 0.001
    slippage: float = 0.0
    take_profit_pct: float = 0.03
    stop_loss_pct: float = 0.01
    buy_after_minutes: int = 1
    min_balance_pct: float = 0.5


class Backtester:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        self.balance = self.config.starting_balance
        self.breaking_balance = self.config.starting_balance * self.config.min_balance_pct
        self.trades = []
        self.current_position = None
        self.position_open = False
        self.previous_direction = None
        self.header_names = [
            'datetime', 'predicted_direction', 'action', 'buy_price',
            'sell_price', 'balance', 'pnl', 'pnl_sum'
        ]

    def _record_trade(self, trade: dict):
        """Record all trades without strict alternation rules."""
        self.trades.append(trade)

    def _calculate_position_pnl(self, entry_price, exit_price, direction, is_exit=True):
        """Calculate PnL including entry/exit fees and slippage."""
        if direction == 1:  # Long
            price_change_pct = (exit_price - entry_price) / entry_price
        else:  # Short
            price_change_pct = (entry_price - exit_price) / entry_price
        
        # Apply leverage
        price_change_pct *= self.config.leverage
        
        # Deduct fees and slippage
        if is_exit:
            # Both entry and exit fees/slippage
            total_cost = 2 * self.config.transaction_fee + 2 * self.config.slippage
        else:
            # Only entry cost (for recording buy)
            total_cost = self.config.transaction_fee + self.config.slippage
        
        return (price_change_pct - total_cost) * 100  # Convert to percentage

    def _update_balance(self, pnl_percent):
        """Update balance based on PnL percentage."""
        self.balance *= (1 + pnl_percent / 100)

    def run(self, price_data: pd.DataFrame, prediction_data: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
        self.reset()
        
        # Prepare price data
        price_data = self._prepare_price_data(price_data)
        signals = self._prepare_signals(prediction_data, price_data)
        
        timestamps = price_data["timestamp"].values
        opens = price_data["open"].values
        highs = price_data["high"].values
        lows = price_data["low"].values
        
        break_on_huge_loss = False
        
        for i in range(len(signals) - 1):
            if self.balance < self.breaking_balance:
                break_on_huge_loss = True
                break
            
            current_signal = signals[i]
            
            # Handle 0 signals (use previous direction)
            if current_signal == 0:
                if self.previous_direction is None or self.previous_direction == 0:
                    self.previous_direction = current_signal
                    continue
                current_signal = self.previous_direction
            
            # Get current minute interval data
            start_idx = i
            end_idx = min(i + self.config.buy_after_minutes + 10, len(opens))
            np_temp = price_data.iloc[start_idx:end_idx][["timestamp", "open", "high", "low"]].values
            np_temp_high = np_temp[:, 2]
            np_temp_low = np_temp[:, 3]
            
            if len(np_temp) <= self.config.buy_after_minutes:
                continue
            
            # If not in position, try to buy
            if not self.position_open:
                self._enter_position(i, current_signal, timestamps, opens, np_temp)
                self.previous_direction = current_signal
            
            # If in position, manage it
            else:
                # Check for TP/SL first
                tp_sl_exit = self._check_tp_sl(i, highs, lows, np_temp_high, np_temp_low)
                if tp_sl_exit:
                    exit_type, exit_idx, exit_price = tp_sl_exit
                    self._exit_position(i, exit_type, exit_price, timestamps[exit_idx])
                    
                    # If direction changed and TP/SL hit, enter new position
                    if current_signal != self.current_position["direction"]:
                        self._enter_position(exit_idx, current_signal, timestamps, opens, np_temp)
                
                # Check for direction change
                elif current_signal != self.current_position["direction"]:
                    # Exit at current open price
                    exit_idx = min(i + self.config.buy_after_minutes, len(opens) - 1)
                    exit_price = opens[exit_idx]
                    self._exit_position(exit_idx, "direction-change", exit_price, timestamps[exit_idx])
                    
                    # Enter new position
                    self._enter_position(exit_idx, current_signal, timestamps, opens, np_temp)
                
                # Check TP/SL within current interval
                else:
                    self._check_interval_tp_sl(i, np_temp, np_temp_high, np_temp_low)
            
            self.previous_direction = current_signal
        
        return self._build_results(break_on_huge_loss)

    def _enter_position(self, idx, signal, timestamps, opens, np_temp):
        """Enter a new position."""
        entry_idx = min(idx + self.config.buy_after_minutes, len(opens) - 1)
        entry_price = opens[entry_idx]
        
        # Calculate entry PnL (negative due to fees/slippage)
        entry_pnl = -((self.config.transaction_fee + self.config.slippage) * 100)
        self._update_balance(entry_pnl)
        
        self.current_position = {
            "direction": signal,
            "buy_price": entry_price,
            "buy_price_adj": entry_price,
            "entry_idx": entry_idx,
            "tp": self._calculate_tp(entry_price, signal),
            "sl": self._calculate_sl(entry_price, signal),
            "entry_datetime": timestamps[entry_idx]
        }
        
        self.position_open = True
        
        self._record_trade({
            "datetime": timestamps[entry_idx],
            "predicted_direction": "long" if signal == 1 else "short",
            "action": "buy",
            "buy_price": entry_price,
            "sell_price": None,
            "balance": self.balance,
            "pnl": entry_pnl,
            "pnl_sum": 0
        })

    def _exit_position(self, idx, exit_type, exit_price, exit_timestamp):
        """Exit current position."""
        if not self.position_open:
            return
        
        pos = self.current_position
        pnl = self._calculate_position_pnl(
            pos["buy_price"], 
            exit_price, 
            pos["direction"],
            is_exit=True
        )
        
        self._update_balance(pnl)
        
        action_map = {
            "tp": "sell - take_profit",
            "sl": "sell - stop_loss",
            "direction-change": "sell - direction change"
        }
        
        self._record_trade({
            "datetime": exit_timestamp,
            "predicted_direction": "long" if pos["direction"] == 1 else "short",
            "action": action_map[exit_type],
            "buy_price": None,
            "sell_price": exit_price,
            "balance": self.balance,
            "pnl": pnl,
            "pnl_sum": 0  # Will be calculated in _build_results
        })
        
        self.position_open = False
        self.current_position = None

    def _check_tp_sl(self, idx, highs, lows, np_temp_high, np_temp_low):
        """Check if TP or SL was hit at current minute."""
        if not self.position_open:
            return None
        
        pos = self.current_position
        
        # Check current minute's high/low
        current_high = highs[idx]
        current_low = lows[idx]
        
        if pos["direction"] == 1:  # Long
            if current_high >= pos["tp"]:
                return ("tp", idx, pos["tp"])
            elif current_low <= pos["sl"]:
                return ("sl", idx, pos["sl"])
        else:  # Short
            if current_low <= pos["tp"]:
                return ("tp", idx, pos["tp"])
            elif current_high >= pos["sl"]:
                return ("sl", idx, pos["sl"])
        
        return None

    def _check_interval_tp_sl(self, idx, np_temp, np_temp_high, np_temp_low):
        """Check TP/SL within the current interval (for multi-minute positions)."""
        if not self.position_open:
            return
        
        pos = self.current_position
        
        if pos["direction"] == 1:  # Long
            # Check for TP hit
            tp_hits = np.where(np_temp_high >= pos["tp"])[0]
            # Check for SL hit
            sl_hits = np.where(np_temp_low <= pos["sl"])[0]
        else:  # Short
            # Check for TP hit
            tp_hits = np.where(np_temp_low <= pos["tp"])[0]
            # Check for SL hit
            sl_hits = np.where(np_temp_high >= pos["sl"])[0]
        
        # Determine which hit first
        if len(tp_hits) == 0 and len(sl_hits) == 0:
            return
        elif len(tp_hits) > 0 and len(sl_hits) == 0:
            exit_idx = tp_hits[0]
            exit_price = pos["tp"]
            exit_type = "tp"
        elif len(tp_hits) == 0 and len(sl_hits) > 0:
            exit_idx = sl_hits[0]
            exit_price = pos["sl"]
            exit_type = "sl"
        else:
            if tp_hits[0] < sl_hits[0]:
                exit_idx = tp_hits[0]
                exit_price = pos["tp"]
                exit_type = "tp"
            else:
                exit_idx = sl_hits[0]
                exit_price = pos["sl"]
                exit_type = "sl"
        
        # Calculate actual exit index
        actual_exit_idx = min(idx + exit_idx, len(np_temp) - 1)
        exit_timestamp = np_temp[actual_exit_idx][0]
        
        self._exit_position(actual_exit_idx, exit_type, exit_price, exit_timestamp)

    def _calculate_tp(self, price, direction):
        """Calculate take-profit price."""
        if direction == 1:  # Long
            return price * (1 + self.config.take_profit_pct)
        else:  # Short
            return price * (1 - self.config.take_profit_pct)

    def _calculate_sl(self, price, direction):
        """Calculate stop-loss price."""
        if direction == 1:  # Long
            return price * (1 - self.config.stop_loss_pct)
        else:  # Short
            return price * (1 + self.config.stop_loss_pct)

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

    def _build_results(self, break_on_huge_loss: bool = False):
        if not self.trades:
            return pd.DataFrame(), self.balance, 0.0
        
        df = pd.DataFrame(self.trades, columns=self.header_names)
        
        # Calculate cumulative PnL
        df["pnl_sum"] = df["pnl"].cumsum()
        
        # Round values
        df[["balance", "pnl", "pnl_sum"]] = df[["balance", "pnl", "pnl_sum"]].round(2)
        
        pnl_percent = df["pnl_sum"].iloc[-1] if len(df) > 0 else 0.0
        
        if break_on_huge_loss:
            return df, -1000, pnl_percent
        else:
            return df, round(self.balance, 2), round(pnl_percent, 2)