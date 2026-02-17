import os
from datetime import datetime, timezone
from time import sleep
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd


class FuturesTrader:

    def __init__(self, api_key, api_secret, symbol):
        self.client = Client(api_key, api_secret, testnet=True)
        self.symbol = symbol
        self.active_trade = None
        self.pnl_sum = 0.0

        self.trade_log = pd.DataFrame(columns=[
            "datetime", "direction", "action",
            "buy_price", "sell_price",
            "fee_total", "pnl", "pnl_sum", "order_id"
        ])

        # Force ONE-WAY mode
        try:
            self.client.futures_change_position_mode(dualSidePosition=False)
        except BinanceAPIException as e:
            if e.code != -4059:
                raise

    # ---------------------------------------------------------
    # POSITION CHECK
    # ---------------------------------------------------------

    def has_open_position(self):
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            pos = next((p for p in positions if p["symbol"] == self.symbol), None)
            return pos and float(pos["positionAmt"]) != 0
        except Exception:
            return False

    def _validate_trade(self):
        return isinstance(self.active_trade, dict) and self.has_open_position()

    # ---------------------------------------------------------
    # ORDER HELPERS
    # ---------------------------------------------------------

    def wait_for_fill(self, order_id, timeout=15):
        start = datetime.now()
        while (datetime.now() - start).seconds < timeout:
            order = self.client.futures_get_order(
                symbol=self.symbol,
                orderId=order_id
            )
            if order["status"] == "FILLED":
                return order
            sleep(0.2)
        raise RuntimeError(f"Order {order_id} not filled")

    def get_execution_details(self, order_id):
        trades = self.client.futures_account_trades(symbol=self.symbol)
        qty = cost = fee = 0.0

        for t in trades:
            if t.get("orderId") == order_id:
                q = float(t["qty"])
                p = float(t["price"])
                f = float(t["commission"])
                qty += q
                cost += q * p
                fee += f

        if qty == 0:
            raise RuntimeError(f"No trades found for order {order_id}")

        return cost / qty, qty, fee

    # ---------------------------------------------------------
    # TP / SL
    # ---------------------------------------------------------

    def place_tp_sl(self, entry_price, direction, qty, tp_pct=3, sl_pct=1):

        if direction == "LONG":
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
            side = "SELL"
        else:
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)
            side = "BUY"

        tp = self.client.futures_create_order(
            symbol=self.symbol,
            side=side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=round(tp_price, 2),
            closePosition=True,
            workingType="MARK_PRICE"
        )

        sl = self.client.futures_create_order(
            symbol=self.symbol,
            side=side,
            type="STOP_MARKET",
            stopPrice=round(sl_price, 2),
            closePosition=True,
            workingType="MARK_PRICE"
        )

        # Support BOTH order types
        tp_id = tp.get("orderId") or tp.get("algoId")
        sl_id = sl.get("orderId") or sl.get("algoId")

        if not tp_id:
            raise RuntimeError(f"TP order failed: {tp}")

        if not sl_id:
            raise RuntimeError(f"SL order failed: {sl}")

        return tp_id, sl_id

    # ---------------------------------------------------------
    # AUTO CLOSE DETECTION
    # ---------------------------------------------------------

    def check_auto_closed_trade(self):

        if not isinstance(self.active_trade, dict):
            return

        if self.has_open_position():
            return  # still open

        entry_price = self.active_trade["entry_price"]
        direction = self.active_trade["direction"]
        entry_fee = self.active_trade["entry_fee"]
        qty_expected = self.active_trade["quantity"]

        trades = self.client.futures_account_trades(symbol=self.symbol)

    # Get last trades (most recent first)
        trades = sorted(trades, key=lambda x: x["time"], reverse=True)

        exit_qty = 0.0
        exit_cost = 0.0
        exit_fee = 0.0
        exit_order_id = None

        for t in trades:

            side = t["side"]  # BUY or SELL
            qty = float(t["qty"])

        # LONG closes with SELL
            if direction == "LONG" and side != "SELL":
                continue

        # SHORT closes with BUY
            if direction == "SHORT" and side != "BUY":
                continue

            price = float(t["price"])
            fee = float(t["commission"])

            exit_qty += qty
            exit_cost += qty * price
            exit_fee += fee
            exit_order_id = t["orderId"]

            if exit_qty >= qty_expected:
                break

        if exit_qty == 0:
            print("Warning: Position closed but no exit trades found")
            self.active_trade = None
            return

        exit_price = exit_cost / exit_qty
        total_fee = entry_fee + exit_fee

        if direction == "LONG":
            pnl = (exit_price - entry_price) * exit_qty - total_fee
            buy, sell = entry_price, exit_price
        else:
            pnl = (entry_price - exit_price) * exit_qty - total_fee
            buy, sell = exit_price, entry_price

        self.pnl_sum += pnl

    # Detect TP vs SL
        if direction == "LONG":
            if exit_price > entry_price:
                hit_type = "TP-HIT"
            else:
                hit_type = "SL-HIT"
        else:  # SHORT
            if exit_price < entry_price:
                hit_type = "TP-HIT"
            else:
                hit_type = "SL-HIT"

        self.log_trade({
        "direction": direction,
        "action": hit_type,
        "buy": buy,
        "sell": sell,
        "fee": total_fee,
        "pnl": pnl,
        "order_id": exit_order_id
        })

        self.active_trade = None


    # ---------------------------------------------------------
    # TRADE FLOW
    # ---------------------------------------------------------

    def open_trade(self, signal, quantity):

        side = "BUY" if signal == 1 else "SELL"
        direction = "LONG" if signal == 1 else "SHORT"

        order = self.client.futures_create_order(
            symbol=self.symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )

        order_id = order.get("orderId")
        if not order_id:
            raise RuntimeError(f"Market order failed: {order}")

        self.wait_for_fill(order_id)

        entry_price, qty, entry_fee = self.get_execution_details(order_id)

        tp_id, sl_id = self.place_tp_sl(entry_price, direction, qty)

        self.active_trade = {
            "direction": direction,
            "entry_price": entry_price,
            "quantity": qty,
            "entry_fee": entry_fee,
            "tp_id": tp_id,
            "sl_id": sl_id
        }

        self.log_trade({
            "direction": direction,
            "action": f"{side}-OPEN",
            "buy": entry_price if direction == "LONG" else None,
            "sell": entry_price if direction == "SHORT" else None,
            "fee": entry_fee,
            "pnl": 0.0,
            "order_id": order_id
        })

    def close_active_trade_market(self, reason="DIRECTION CHANGE"):

        if not self._validate_trade():
            self.active_trade = None
            return

        # Cancel TP/SL safely (handles normal + algo)
        for oid in [self.active_trade["tp_id"], self.active_trade["sl_id"]]:
            try:
                self.client.futures_cancel_order(
                    symbol=self.symbol,
                    orderId=oid
                )
            except Exception:
                try:
                    self.client.futures_cancel_algo_order(
                        symbol=self.symbol,
                        algoId=oid
                    )
                except Exception:
                    pass

        direction = self.active_trade["direction"]
        side = "SELL" if direction == "LONG" else "BUY"

        positions = self.client.futures_position_information(symbol=self.symbol)
        pos = next((p for p in positions if p["symbol"] == self.symbol), None)

        if not pos or float(pos["positionAmt"]) == 0:
            self.active_trade = None
            return

        qty_to_close = abs(float(pos["positionAmt"]))

        order = self.client.futures_create_order(
            symbol=self.symbol,
            side=side,
            type="MARKET",
            quantity=qty_to_close,
            reduceOnly=True
        )

        order_id = order.get("orderId")
        if not order_id:
            raise RuntimeError(f"Close order failed: {order}")

        self.wait_for_fill(order_id)

        exit_price, qty_filled, exit_fee = self.get_execution_details(order_id)

        entry = self.active_trade["entry_price"]
        total_fee = self.active_trade["entry_fee"] + exit_fee

        if direction == "LONG":
            pnl = (exit_price - entry) * qty_filled - total_fee
            buy, sell = entry, exit_price
        else:
            pnl = (entry - exit_price) * qty_filled - total_fee
            buy, sell = exit_price, entry

        self.pnl_sum += pnl

        self.log_trade({
            "direction": direction,
            "action": reason,
            "buy": buy,
            "sell": sell,
            "fee": total_fee,
            "pnl": pnl,
            "order_id": order_id
        })

        self.active_trade = None

    # ---------------------------------------------------------
    # SIGNAL HANDLER
    # ---------------------------------------------------------

    def process_signal(self, signal, quantity):

        self.check_auto_closed_trade()

        if not self._validate_trade():
            if signal != 0:
                self.open_trade(signal, quantity)
            return

        current_dir = self.active_trade["direction"]

        if (signal == 1 and current_dir == "SHORT") or \
           (signal == -1 and current_dir == "LONG"):

            self.close_active_trade_market("DIRECTION CHANGE")
            self.open_trade(signal, quantity)

    # ---------------------------------------------------------
    # LOGGING
    # ---------------------------------------------------------

    def log_trade(self, trade):

        buy_price = trade.get("buy")
        sell_price = trade.get("sell")

        row = {
        "datetime": datetime.now(timezone.utc),
        "direction": trade.get("direction"),
        "action": trade.get("action"),
        "buy_price": round(buy_price, 6) if buy_price is not None else 0.0,
        "sell_price": round(sell_price, 6) if sell_price is not None else 0.0,
        "fee_total": round(trade.get("fee", 0), 6),
        "pnl": round(trade.get("pnl", 0), 6),
        "pnl_sum": round(self.pnl_sum, 6),
        "order_id": trade.get("order_id")
        }

        self.trade_log = pd.concat(
            [self.trade_log, pd.DataFrame([row])],
            ignore_index=True
        )  


    def get_trade_log_df(self):
        return self.trade_log.copy()
