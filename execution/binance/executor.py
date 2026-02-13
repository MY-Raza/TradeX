import os
import csv
from datetime import datetime, timezone
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

        # Create CSV file if not exists
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
                    "pnl_sum"
                ])

    # =========================================
    # OPEN TRADE
    # =========================================
    def open_trade(self, signal, quantity):

        if self.active_trade:
            return

        side = "BUY" if signal == 1 else "SELL"
        direction = "LONG" if signal == 1 else "SHORT"

        order = self.client.new_order(
            symbol=self.symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )

        executed_price = float(order["avgPrice"])

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

    # =========================================
    # CLOSE TRADE
    # =========================================
    def close_trade(self, reason="Direction Change"):

        if not self.active_trade:
            return

        positions = self.client.position_information(symbol=self.symbol)
        position_qty = float(positions[0]["positionAmt"])

        if position_qty == 0:
            return

        side_to_close = "SELL" if position_qty > 0 else "BUY"

        order = self.client.new_order(
            symbol=self.symbol,
            side=side_to_close,
            type="MARKET",
            quantity=abs(position_qty)
        )

        exit_price = float(order["avgPrice"])

        entry_price = self.active_trade["entry_price"]
        quantity = self.active_trade["quantity"]
        direction = self.active_trade["direction"]

        # Calculate PnL before fee
        if direction == "LONG":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        # ============================
        # GET REAL COMMISSION
        # ============================
        trades = self.client.get_account_trades(symbol=self.symbol)

        closing_trade = next(
            trade for trade in reversed(trades)
            if float(trade["qty"]) == abs(position_qty)
        )

        fee = float(closing_trade["commission"])

        pnl -= fee
        self.pnl_sum += pnl

        self.log_trade(
            direction=direction,
            action=f"{side_to_close}-{reason}",
            buy_price=entry_price if direction == "LONG" else exit_price,
            sell_price=exit_price if direction == "LONG" else entry_price,
            fee=round(fee, 6),
            pnl=round(pnl, 6)
        )

        self.active_trade = None

    # =========================================
    # TP / SL CHECK
    # =========================================
    def tp_sl_check(self, tp_percent=3, sl_percent=1):

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
    # MAIN SIGNAL PROCESSOR
    # =========================================
    def process_signal(self, signal, quantity):

        # First check TP/SL
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
    # CSV LOGGER
    # =========================================
    def log_trade(self, direction, action, buy_price, sell_price, fee, pnl):

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
                round(self.pnl_sum, 6)
            ])
