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
        
        i = 0
        while i < len(signals):
            if self.balance < self.breaking_balance:
                break
            
            current_signal = signals[i]
            
            # Handle 0 signals (use previous direction)
            if current_signal == 0:
                if self.previous_direction is None or self.previous_direction == 0:
                    self.previous_direction = current_signal
                    i += 1
                    continue
                current_signal = self.previous_direction
            
            # If not in position, try to enter
            if not self.position_open and current_signal != 0:
                entry_idx = min(i + self.config.buy_after_minutes, len(opens) - 1)
                if entry_idx >= len(opens):
                    i += 1
                    continue
                    
                entry_price = opens[entry_idx]
                entry_pnl = -((self.config.transaction_fee + self.config.slippage) * 100)
                self._update_balance(entry_pnl)
                
                self.current_position = {
                    "direction": current_signal,
                    "buy_price": entry_price,
                    "entry_idx": entry_idx,
                    "tp": self._calculate_tp(entry_price, current_signal),
                    "sl": self._calculate_sl(entry_price, current_signal),
                    "entry_datetime": timestamps[entry_idx]
                }
                
                self.position_open = True
                self.previous_direction = current_signal
                
                self._record_trade({
                    "datetime": timestamps[entry_idx],
                    "predicted_direction": "long" if current_signal == 1 else "short",
                    "action": "buy",
                    "buy_price": entry_price,
                    "sell_price": None,
                    "balance": self.balance,
                    "pnl": entry_pnl,
                    "pnl_sum": 0
                })
                
                i = entry_idx
                continue
            
            # If in position, manage it
            elif self.position_open:
                pos = self.current_position
                
                # Check for TP/SL at current minute
                tp_sl_hit = self._check_minute_tp_sl(i, highs[i], lows[i], pos)
                if tp_sl_hit:
                    exit_type, exit_price = tp_sl_hit
                    pnl = self._calculate_position_pnl(
                        pos["buy_price"], 
                        exit_price, 
                        pos["direction"],
                        is_exit=True
                    )
                    
                    self._update_balance(pnl)
                    
                    action_map = {
                        "tp": "sell - take_profit",
                        "sl": "sell - stop_loss"
                    }
                    
                    self._record_trade({
                        "datetime": timestamps[i],
                        "predicted_direction": "long" if pos["direction"] == 1 else "short",
                        "action": action_map[exit_type],
                        "buy_price": None,
                        "sell_price": exit_price,
                        "balance": self.balance,
                        "pnl": pnl,
                        "pnl_sum": 0
                    })
                    
                    self.position_open = False
                    self.current_position = None
                    
                    # If direction changed, enter new position next iteration
                    if current_signal != pos["direction"]:
                        i += 1
                        continue
                
                # Check for direction change
                elif current_signal != pos["direction"]:
                    # Exit at current open price
                    exit_price = opens[i]
                    pnl = self._calculate_position_pnl(
                        pos["buy_price"], 
                        exit_price, 
                        pos["direction"],
                        is_exit=True
                    )
                    
                    self._update_balance(pnl)
                    
                    self._record_trade({
                        "datetime": timestamps[i],
                        "predicted_direction": "long" if pos["direction"] == 1 else "short",
                        "action": "sell - direction change",
                        "buy_price": None,
                        "sell_price": exit_price,
                        "balance": self.balance,
                        "pnl": pnl,
                        "pnl_sum": 0
                    })
                    
                    self.position_open = False
                    self.current_position = None
                    
                    # Enter new position on next iteration
                    i += 1
                    continue
            
            i += 1
        
        return self._build_results()

    def _check_minute_tp_sl(self, idx, high, low, position):
        """Check if TP or SL was hit at current minute."""
        if position["direction"] == 1:  # Long
            if high >= position["tp"]:
                return ("tp", position["tp"])
            elif low <= position["sl"]:
                return ("sl", position["sl"])
        else:  # Short
            if low <= position["tp"]:
                return ("tp", position["tp"])
            elif high >= position["sl"]:
                return ("sl", position["sl"])
        return None

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

    def _build_results(self):
        if not self.trades:
            return pd.DataFrame(), self.balance, 0.0
        
        df = pd.DataFrame(self.trades)
        
        # Ensure all required columns exist
        for col in self.header_names:
            if col not in df.columns:
                df[col] = 0.0
        
        # Calculate cumulative PnL
        df["pnl_sum"] = df["pnl"].cumsum()
        
        # Round values
        df[["balance", "pnl", "pnl_sum"]] = df[["balance", "pnl", "pnl_sum"]].round(2)
        
        pnl_percent = df["pnl_sum"].iloc[-1] if len(df) > 0 else 0.0
        
        return df, round(self.balance, 2), round(pnl_percent, 2)