import pandas as pd


class Backtester:
    def __init__(
        self,
        price_df: pd.DataFrame,
        signal_df: pd.DataFrame,
        starting_balance: float = 1000,
        tp: float = 3,
        sl: float = 1,
        fee: float = 0.05,
        leverage: float = 1.0,
        slippage: float = 0.0,
    ):
        self.price_df = price_df.sort_values("timestamp").reset_index(drop=True)
        self.signal_df = signal_df.sort_values("timestamp").reset_index(drop=True)

        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.break_balance = starting_balance * 0.5

        self.tp = tp / 100
        self.sl = sl / 100
        self.fee = fee
        self.leverage = leverage
        self.slippage = slippage

        self.trades = []
        self.open_trade = None

        self.data = pd.merge_asof(
            self.price_df,
            self.signal_df[["timestamp", "signals"]],
            on="timestamp",
            direction="backward"
        ).fillna({"signals": 0})

    # -------------------------
    # Open Trade
    # -------------------------
    def open_position(self, row):
        direction = "long" if row.signals == 1 else "short"
        entry_price = row.close

        entry_action = "BUY" if direction == "long" else "SELL"

        self.open_trade = {
            "entry_time": row.timestamp,
            "entry_price": entry_price,
            "direction": direction,
            "tp_price": entry_price * (1 + self.tp) if direction == "long" else entry_price * (1 - self.tp),
            "sl_price": entry_price * (1 - self.sl) if direction == "long" else entry_price * (1 + self.sl),
            "entry_action": entry_action,
        }

        self.balance *= (1 - (self.fee + self.slippage) / 100)

    # -------------------------
    # Close Trade
    # -------------------------
    def close_position(self, row, exit_price, reason):
        entry_price = self.open_trade["entry_price"]
        direction = self.open_trade["direction"]

        pnl_pct = (
            (exit_price - entry_price) / entry_price * 100
            if direction == "long"
            else (entry_price - exit_price) / entry_price * 100
        )

        pnl_pct *= self.leverage
        pnl_pct -= (self.fee + self.slippage)

        self.balance *= (1 + pnl_pct / 100)

        # Determine exit action
        if direction == "long":
            exit_action = f"SELL ({reason})"
        else:
            exit_action = f"BUY ({reason})"

        self.trades.append({
            "entry_time": self.open_trade["entry_time"],
            "exit_time": row.timestamp,
            "direction": direction,
            "entry_action": self.open_trade["entry_action"],
            "exit_action": exit_action,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_%": round(pnl_pct, 2),
            "balance": round(self.balance, 2),
        })

        self.open_trade = None

    # -------------------------
    # Check TP / SL inside candle
    # -------------------------
    def check_tp_sl(self, row):
        if not self.open_trade:
            return False

        high, low = row.high, row.low
        tp_price = self.open_trade["tp_price"]
        sl_price = self.open_trade["sl_price"]
        direction = self.open_trade["direction"]

        if direction == "long":
            if high >= tp_price:
                self.close_position(row, tp_price, "TP")
                return True
            if low <= sl_price:
                self.close_position(row, sl_price, "SL")
                return True
        else:
            if low <= tp_price:
                self.close_position(row, tp_price, "TP")
                return True
            if high >= sl_price:
                self.close_position(row, sl_price, "SL")
                return True

        return False

    # -------------------------
    # Main Backtest Loop
    # -------------------------
    def run_backtest(self):
        for _, row in self.data.iterrows():
            signal = row.signals

            if self.open_trade:
                if self.check_tp_sl(row):
                    continue

                # Exit on signal flip
                if signal != 0 and (
                    (signal == 1 and self.open_trade["direction"] == "short") or
                    (signal == -1 and self.open_trade["direction"] == "long")
                ):
                    self.close_position(row, row.close, "Signal Flip")

            if not self.open_trade and signal in [1, -1]:
                self.open_position(row)

            if self.balance <= self.break_balance:
                print("⚠️ Account dropped below 50%. Stopping backtest.")
                break

    # -------------------------
    # Results
    # -------------------------
    def get_results(self):
        return pd.DataFrame(self.trades)

    def get_final_balance(self):
        return round(self.balance, 2)

    def get_total_return_pct(self):
        return round((self.balance - self.starting_balance) / self.starting_balance * 100, 2)
