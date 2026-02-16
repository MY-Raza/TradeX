import os
import csv
from datetime import datetime, timezone
from time import sleep
from binance.client import Client

class FuturesTrader:
    """
    Binance Futures trader engine using the regular Client for USDT-margined futures.
    Handles market entries, TP/SL, PnL, trade reversal, and CSV logging.
    """

    def __init__(self, api_key, api_secret, symbol):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.active_trade = None
        self.pnl_sum = 0.0
        self.csv_file = "trade_log.csv"

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                csv.writer(f).writerow([
                    "datetime_utc", "direction", "action", "buy_price",
                    "sell_price", "fee_total", "pnl", "pnl_sum", "order_id"
                ])

    def _validate_trade(self):
        return isinstance(self.active_trade, dict)

    def wait_for_fill(self, order_id, timeout=10):
        start = datetime.now()
        while (datetime.now() - start).seconds < timeout:
            order = self.client.futures_get_order(symbol=self.symbol, orderId=order_id)
            if order["status"] == "FILLED":
                return order
            sleep(0.2)
        raise RuntimeError(f"Order {order_id} not filled")

    def get_execution_details(self, order_id):
        trades = self.client.futures_account_trades(symbol=self.symbol)
        qty = cost = fee = 0.0
        for t in trades:
            if t["orderId"] != order_id:
                continue
            q = float(t["qty"])
            p = float(t["price"])
            f = float(t["commission"])
            qty += q
            cost += q * p
            fee += f
        return cost / qty, qty, fee

    def get_exit_from_order(self, order):
        exit_price = float(order["avgPrice"])
        qty = float(order["executedQty"])
        fee = 0.0
        trades = self.client.futures_account_trades(symbol=self.symbol)
        for t in trades[-10:]:
            if float(t["qty"]) == qty:
                fee += float(t["commission"])
        return exit_price, qty, fee

    def place_tp_sl(self, entry_price, direction, tp_pct=0.1, sl_pct=0.1):
        if direction == "LONG":
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
            side = "SELL"
        else:
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)
            side = "BUY"

        tp = self.client.futures_create_order(
            symbol=self.symbol, side=side, type="TAKE_PROFIT_MARKET",
            stopPrice=round(tp_price, 2), closePosition=True, workingType="MARK_PRICE"
        )

        sl = self.client.futures_create_order(
            symbol=self.symbol, side=side, type="STOP_MARKET",
            stopPrice=round(sl_price, 2), closePosition=True, workingType="MARK_PRICE"
        )
        return tp["orderId"], sl["orderId"]

    def open_trade(self, signal, quantity):
        side = "BUY" if signal == 1 else "SELL"
        direction = "LONG" if signal == 1 else "SHORT"

        order = self.client.futures_create_order(
            symbol=self.symbol, side=side, type="MARKET", quantity=quantity
        )
        order_id = order["orderId"]
        self.wait_for_fill(order_id)

        entry_price, qty, entry_fee = self.get_execution_details(order_id)
        tp_id, sl_id = self.place_tp_sl(entry_price, direction)

        self.active_trade = {
            "direction": direction,
            "entry_price": entry_price,
            "quantity": qty,
            "entry_fee": entry_fee,
            "tp_id": tp_id,
            "sl_id": sl_id
        }

        self.log_trade(direction, f"{side}-OPEN",
                       entry_price if side == "BUY" else "",
                       entry_price if side == "SELL" else "",
                       entry_fee, 0.0, order_id)

    def close_active_trade_market(self, reason="DIRECTION CHANGE"):
        if not self._validate_trade():
            return

        try:
            self.client.futures_cancel_order(symbol=self.symbol, orderId=self.active_trade["tp_id"])
            self.client.futures_cancel_order(symbol=self.symbol, orderId=self.active_trade["sl_id"])
        except Exception:
            pass

        direction = self.active_trade["direction"]
        side = "SELL" if direction == "LONG" else "BUY"

        order = self.client.futures_create_order(
            symbol=self.symbol, side=side, type="MARKET", reduceOnly=True
        )
        self.wait_for_fill(order["orderId"])
        exit_price, qty, exit_fee = self.get_exit_from_order(order)

        entry = self.active_trade["entry_price"]
        entry_fee = self.active_trade["entry_fee"]
        total_fee = entry_fee + exit_fee

        if direction == "LONG":
            pnl = (exit_price - entry) * qty - total_fee
            buy, sell = entry, exit_price
        else:
            pnl = (entry - exit_price) * qty - total_fee
            buy, sell = exit_price, entry

        self.pnl_sum += pnl
        self.log_trade(direction, reason, buy, sell, total_fee, pnl, order["orderId"])
        self.active_trade = None

    def check_tp_sl_hit(self):
        if not self._validate_trade():
            return
        for reason, oid in [("TP-HIT", self.active_trade["tp_id"]),
                            ("SL-HIT", self.active_trade["sl_id"])]:
            order = self.client.futures_get_order(symbol=self.symbol, orderId=oid)
            if order["status"] == "FILLED":
                self.client.futures_cancel_order(symbol=self.symbol, orderId=self.active_trade["tp_id"])
                self.client.futures_cancel_order(symbol=self.symbol, orderId=self.active_trade["sl_id"])
                self.handle_exit(order, reason)
                return

    def handle_exit(self, order, reason):
        exit_price, qty, exit_fee = self.get_exit_from_order(order)
        entry = self.active_trade["entry_price"]
        entry_fee = self.active_trade["entry_fee"]
        direction = self.active_trade["direction"]
        total_fee = entry_fee + exit_fee

        if direction == "LONG":
            pnl = (exit_price - entry) * qty - total_fee
            buy, sell = entry, exit_price
        else:
            pnl = (entry - exit_price) * qty - total_fee
            buy, sell = exit_price, entry

        self.pnl_sum += pnl
        self.log_trade(direction, reason, buy, sell, total_fee, pnl, order["orderId"])
        self.active_trade = None

    def process_signal(self, signal, quantity):
        self.check_tp_sl_hit()

        if not self._validate_trade():
            if signal != 0:
                self.open_trade(signal, quantity)
            return

        current_dir = self.active_trade["direction"]
        if (signal == 1 and current_dir == "SHORT") or \
           (signal == -1 and current_dir == "LONG"):
            self.close_active_trade_market(reason="DIRECTION CHANGE")
            self.open_trade(signal, quantity)

    def log_trade(self, direction, action, buy, sell, fee, pnl, order_id):
        with open(self.csv_file, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                direction, action,
                round(buy, 6) if buy else "",
                round(sell, 6) if sell else "",
                round(fee, 6), round(pnl, 6),
                round(self.pnl_sum, 6), order_id
            ])
