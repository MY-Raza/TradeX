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
                    "direction",
                    "action",
                    "buy_price",
                    "sell_price",
                    "fee_total",
                    "pnl",
                    "pnl_sum",
                    "order_id"
                ])

        self.tp_order_id = None
        self.sl_order_id = None

    # =========================================
    # WAIT FOR ORDER TO BE FILLED
    # =========================================
    def wait_for_fill(self, order_id, timeout=10):
        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < timeout:
            order = self.client.query_order(
                symbol=self.symbol,
                orderId=order_id
            )

            if order["status"] == "FILLED":
                return order

            sleep(0.2)

        raise RuntimeError(f"Order {order_id} not filled within timeout.")

    # =========================================
    # GET EXECUTION DETAILS
    # =========================================
    def get_execution_details(self, order_id):
        trades = self.client.get_account_trades(
            symbol=self.symbol,
            orderId=order_id
        )

        if not trades:
            raise RuntimeError(f"No trades found for order {order_id}")

        executed_qty = 0.0
        total_cost = 0.0
        total_fee = 0.0

        for t in trades:
            qty = float(t["qty"])
            price = float(t["price"])
            commission = float(t["commission"])

            executed_qty += qty
            total_cost += qty * price
            total_fee += commission

        avg_price = total_cost / executed_qty
        return avg_price, executed_qty, total_fee

    # =========================================
    # PLACE TP AND SL ORDERS ON BINANCE
    # =========================================
    def place_tp_sl_orders(self, direction, entry_price, quantity):
        # Cancel existing TP/SL orders if any
        for oid in [self.tp_order_id, self.sl_order_id]:
            if oid:
                try:
                    self.client.cancel_order(symbol=self.symbol, orderId=oid)
                except:
                    pass

        if direction == "LONG":
            tp_price = round(entry_price * 1.03, 2)
            sl_price = round(entry_price * 0.99, 2)
            self.tp_order_id = self.client.new_order(
                symbol=self.symbol,
                side="SELL",
                type="TAKE_PROFIT_MARKET",
                stopPrice=tp_price,
                closePosition=True
            )["orderId"]
            self.sl_order_id = self.client.new_order(
                symbol=self.symbol,
                side="SELL",
                type="STOP_MARKET",
                stopPrice=sl_price,
                closePosition=True
            )["orderId"]
        else:  # SHORT
            tp_price = round(entry_price * 0.97, 2)
            sl_price = round(entry_price * 1.01, 2)
            self.tp_order_id = self.client.new_order(
                symbol=self.symbol,
                side="BUY",
                type="TAKE_PROFIT_MARKET",
                stopPrice=tp_price,
                closePosition=True
            )["orderId"]
            self.sl_order_id = self.client.new_order(
                symbol=self.symbol,
                side="BUY",
                type="STOP_MARKET",
                stopPrice=sl_price,
                closePosition=True
            )["orderId"]

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
        order_id = order["orderId"]

        self.wait_for_fill(order_id)
        executed_price, executed_qty, entry_fee = self.get_execution_details(order_id)

        self.active_trade = {
            "direction": direction,
            "entry_price": executed_price,
            "quantity": executed_qty,
            "entry_fee": entry_fee
        }

        # Place TP and SL on Binance
        self.place_tp_sl_orders(direction, executed_price, executed_qty)

        self.log_trade(
            direction=direction,
            action=f"{side}-OPEN",
            buy_price=executed_price if side == "BUY" else "",
            sell_price=executed_price if side == "SELL" else "",
            fee=round(entry_fee, 6),
            pnl=0,
            order_id=order_id
        )

    # =========================================
    # CLOSE TRADE
    # =========================================
    def close_trade(self, reason="Manual Close"):
        if not self.active_trade:
            return

        direction = self.active_trade["direction"]
        quantity = self.active_trade["quantity"]
        side = "SELL" if direction == "LONG" else "BUY"

        order = self.client.new_order(
            symbol=self.symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )
        order_id = order["orderId"]

        self.wait_for_fill(order_id)
        exit_price, executed_qty, exit_fee = self.get_execution_details(order_id)

        entry_price = self.active_trade["entry_price"]
        entry_fee = self.active_trade["entry_fee"]
        total_fee = entry_fee + exit_fee

        if direction == "LONG":
            pnl = (exit_price - entry_price) * quantity - total_fee
            buy_price = entry_price
            sell_price = exit_price
        else:
            pnl = (entry_price - exit_price) * quantity - total_fee
            buy_price = exit_price
            sell_price = entry_price

        self.pnl_sum += pnl

        self.log_trade(
            direction=direction,
            action=f"{side}-{reason}",
            buy_price=round(buy_price, 6),
            sell_price=round(sell_price, 6),
            fee=round(total_fee, 6),
            pnl=round(pnl, 6),
            order_id=order_id
        )

        # Cancel TP/SL orders on close
        for oid in [self.tp_order_id, self.sl_order_id]:
            if oid:
                try:
                    self.client.cancel_order(symbol=self.symbol, orderId=oid)
                except:
                    pass
        self.tp_order_id = None
        self.sl_order_id = None
        self.active_trade = None

    # =========================================
    # PROCESS SIGNAL
    # =========================================
    def process_signal(self, signal, quantity):
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
    def log_trade(self, direction, action,
                  buy_price, sell_price,
                  fee, pnl, order_id):

        with open(self.csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                direction,
                action,
                buy_price,
                sell_price,
                fee,
                round(pnl, 6),
                round(self.pnl_sum, 6),
                order_id
            ])
