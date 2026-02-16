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
            base_url="https://testnet.binancefuture.com"  # change for live
        )

        self.symbol = symbol.upper()
        self.active_trade = None
        self.pnl_sum = 0
        self.csv_file = "trade_log.csv"
        self.tp_order_id = None
        self.sl_order_id = None

        # Get symbol precision
        self.price_precision, self.qty_precision = self.get_symbol_precision()

        # Create CSV if not exists
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

    # =========================================
    # GET SYMBOL PRECISION
    # =========================================
    def get_symbol_precision(self):
        info = self.client.exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == self.symbol:
                return s["pricePrecision"], s["quantityPrecision"]
        raise ValueError("Symbol not found")

    # =========================================
    # WAIT FOR ORDER FILL
    # =========================================
    def wait_for_fill(self, order_id, timeout=10):
        start = datetime.now()

        while (datetime.now() - start).seconds < timeout:
            order = self.client.query_order(
                symbol=self.symbol,
                orderId=order_id
            )
            if order["status"] == "FILLED":
                return order
            sleep(0.2)

        raise RuntimeError(f"Order {order_id} not filled")

    # =========================================
    # GET EXECUTION DETAILS
    # =========================================
    def get_execution_details(self, order_id):
        trades = self.client.get_account_trades(
            symbol=self.symbol,
            orderId=order_id
        )

        if not trades:
            raise RuntimeError(f"No trades for order {order_id}")

        qty_sum = 0
        cost_sum = 0
        fee_sum = 0

        for t in trades:
            qty = float(t["qty"])
            price = float(t["price"])
            commission = float(t["commission"])

            qty_sum += qty
            cost_sum += qty * price
            fee_sum += commission

        avg_price = cost_sum / qty_sum
        return avg_price, qty_sum, fee_sum

    # =========================================
    # PLACE TP & SL (PROPER VERSION)
    # =========================================
    def place_tp_sl_orders(self, direction, entry_price):

        # Cancel previous TP/SL
        for oid in [self.tp_order_id, self.sl_order_id]:
            if oid:
                try:
                    self.client.cancel_order(symbol=self.symbol, orderId=oid)
                except:
                    pass

        if direction == "LONG":
            tp_price = round(entry_price * 1.03, self.price_precision)
            sl_price = round(entry_price * 0.99, self.price_precision)
            side = "SELL"
        else:
            tp_price = round(entry_price * 0.97, self.price_precision)
            sl_price = round(entry_price * 1.01, self.price_precision)
            side = "BUY"

        tp = self.client.new_order(
            symbol=self.symbol,
            side=side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=tp_price,
            closePosition=True,
            reduceOnly=True,
            workingType="MARK_PRICE"
        )

        sl = self.client.new_order(
            symbol=self.symbol,
            side=side,
            type="STOP_MARKET",
            stopPrice=sl_price,
            closePosition=True,
            reduceOnly=True,
            workingType="MARK_PRICE"
        )

        self.tp_order_id = tp["orderId"]
        self.sl_order_id = sl["orderId"]

        print(f"TP placed at {tp_price}")
        print(f"SL placed at {sl_price}")

    # =========================================
    # CHECK IF POSITION CLOSED BY TP/SL
    # =========================================
    def check_position_closed(self):
        if not self.active_trade:
            return

        pos = self.client.get_position_risk(symbol=self.symbol)[0]
        position_amt = float(pos["positionAmt"])

        if position_amt == 0:
            print("Position closed by TP/SL")
            self.handle_auto_close()

    # =========================================
    # HANDLE AUTO CLOSE (TP OR SL HIT)
    # =========================================
    def handle_auto_close(self):
        direction = self.active_trade["direction"]
        entry_price = self.active_trade["entry_price"]
        quantity = self.active_trade["quantity"]
        entry_fee = self.active_trade["entry_fee"]

        # Get latest trades
        trades = self.client.get_account_trades(symbol=self.symbol)
        last_trade = trades[-1]

        exit_price = float(last_trade["price"])
        exit_fee = float(last_trade["commission"])

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
            direction,
            "AUTO-CLOSE",
            round(buy_price, self.price_precision),
            round(sell_price, self.price_precision),
            round(total_fee, 6),
            round(pnl, 6),
            "TP/SL"
        )

        self.active_trade = None
        self.tp_order_id = None
        self.sl_order_id = None

    # =========================================
    # OPEN TRADE
    # =========================================
    def open_trade(self, signal, quantity):

        if self.active_trade:
            return

        side = "BUY" if signal == 1 else "SELL"
        direction = "LONG" if signal == 1 else "SHORT"

        quantity = round(quantity, self.qty_precision)

        order = self.client.new_order(
            symbol=self.symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )

        order_id = order["orderId"]
        self.wait_for_fill(order_id)

        price, qty, fee = self.get_execution_details(order_id)

        self.active_trade = {
            "direction": direction,
            "entry_price": price,
            "quantity": qty,
            "entry_fee": fee
        }

        self.place_tp_sl_orders(direction, price)

        self.log_trade(
            direction,
            f"{side}-OPEN",
            price if side == "BUY" else "",
            price if side == "SELL" else "",
            round(fee, 6),
            0,
            order_id
        )

    # =========================================
    # PROCESS SIGNAL
    # =========================================
    def process_signal(self, signal, quantity):

        self.check_position_closed()

        if self.active_trade:
            current_direction = self.active_trade["direction"]
            if (current_direction == "LONG" and signal == -1) or \
               (current_direction == "SHORT" and signal == 1):
                print("Direction change detected")
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
