import os
import csv
from datetime import datetime, timezone
from time import sleep
from binance.um_futures import UMFutures


class FuturesTrader:
    def __init__(self, api_key, api_secret, symbol):
        self.client = UMFutures(
            key=api_key,
            secret=api_secret,
            base_url="https://testnet.binancefuture.com"
        )
        self.symbol = symbol
        self.active_trade = None
        self.pnl_sum = 0
        self.csv_file = "trade_log.csv"

        # Create CSV file if it doesn't exist
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "datetime_utc",
                    "predicted_direction",
                    "action",
                    "buy_price",
                    "sell_price",
                    "fee",
                    "pnl",
                    "pnl_sum",
                    "current_balance"   # New column
                ])

    # =========================================
    # GET CURRENT BALANCE
    # =========================================
    def get_current_balance(self):
        try:
            balances = self.client.balance()
            for b in balances:
                if b["asset"] == "USDT":
                    return float(b["balance"])
        except Exception:
            return 0
        return 0

    # =========================================
    # OPEN TRADE
    # =========================================
    def open_trade(self, signal, quantity):
        if self.active_trade:
            return None, None

        side = "BUY" if signal == 1 else "SELL"
        direction = "LONG" if signal == 1 else "SHORT"

        order = self.client.new_order(
            symbol=self.symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )

        sleep(0.5)

        # Fetch executed price
        try:
            executed_price = float(order.get("avgPrice", 0))
            if executed_price == 0:
                trades = self.client.get_account_trades(symbol=self.symbol)
                last_trade = trades[-1]
                executed_price = float(last_trade["price"])
        except Exception:
            ticker = self.client.ticker_price(symbol=self.symbol)
            executed_price = float(ticker["price"])

        self.active_trade = {
            "direction": direction,
            "entry_price": executed_price,
            "quantity": quantity
        }

        self.log_trade(
            direction=direction,
            action=f"{side}-OPEN",
            buy_price=executed_price if side == "BUY" else "",
            sell_price=executed_price if side == "SELL" else "",
            fee=0,
            pnl=0
        )

        return executed_price, executed_price

    # =========================================
    # CLOSE TRADE
    # =========================================
    def close_trade(self, reason="Direction Change"):
        if not self.active_trade:
            return

        position_qty = float(self.active_trade["quantity"])
        direction = self.active_trade["direction"]
        side_to_close = "SELL" if direction == "LONG" else "BUY"

        order = self.client.new_order(
            symbol=self.symbol,
            side=side_to_close,
            type="MARKET",
            quantity=position_qty
        )

        sleep(0.5)

        # Get exit price
        try:
            exit_price = float(order.get("avgPrice", 0))
            if exit_price == 0:
                trades = self.client.get_account_trades(symbol=self.symbol)
                closing_trade = trades[-1]
                exit_price = float(closing_trade["price"])
                fee = float(closing_trade["commission"])
            else:
                fee = 0
        except Exception:
            ticker = self.client.ticker_price(symbol=self.symbol)
            exit_price = float(ticker["price"])
            fee = 0

        entry_price = self.active_trade["entry_price"]
        quantity = self.active_trade["quantity"]

        # Calculate PnL
        if direction == "LONG":
            pnl = (exit_price - entry_price) * quantity
            buy_price = entry_price
            sell_price = exit_price
        else:
            pnl = (entry_price - exit_price) * quantity
            buy_price = exit_price
            sell_price = entry_price

        pnl -= fee
        self.pnl_sum += pnl

        self.log_trade(
            direction=direction,
            action=f"{side_to_close}-{reason}",
            buy_price=round(buy_price, 6),
            sell_price=round(sell_price, 6),
            fee=round(fee, 6),
            pnl=round(pnl, 6)
        )

        self.active_trade = None

    # =========================================
    # TP / SL CHECK
    # =========================================
    def tp_sl_check(self, tp_percent=0.01, sl_percent=0.01):
        if not self.active_trade:
            return

        ticker = self.client.ticker_price(symbol=self.symbol)
        current_price = float(ticker["price"])

        entry_price = self.active_trade["entry_price"]
        direction = self.active_trade["direction"]

        if direction == "LONG":
            tp_price = entry_price * (1 + tp_percent / 100)
            sl_price = entry_price * (1 - sl_percent / 100)
            if current_price >= tp_price:
                self.close_trade("Take Profit Hit")
            elif current_price <= sl_price:
                self.close_trade("Stop Loss Hit")
        else:
            tp_price = entry_price * (1 - tp_percent / 100)
            sl_price = entry_price * (1 + sl_percent / 100)
            if current_price <= tp_price:
                self.close_trade("Take Profit Hit")
            elif current_price >= sl_price:
                self.close_trade("Stop Loss Hit")

    # =========================================
    # PROCESS SIGNAL
    # =========================================
    def process_signal(self, signal, quantity):
        self.tp_sl_check()

        if self.active_trade:
            current_direction = self.active_trade["direction"]
            if (current_direction == "LONG" and signal == -1) or \
               (current_direction == "SHORT" and signal == 1):
                self.close_trade("Direction Change")
                self.open_trade(signal, quantity)
        else:
            if signal in [1, -1]:
                self.open_trade(signal, quantity)

    # =========================================
    # LOG TO CSV
    # =========================================
    def log_trade(self, direction, action, buy_price, sell_price, fee, pnl):
        current_balance = self.get_current_balance()
        with open(self.csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                direction,
                action,
                buy_price,
                sell_price,
                fee,
                pnl,
                round(self.pnl_sum, 6),
                round(current_balance, 6)   # Add current balance
            ])
