import os
from datetime import datetime, timezone
from time import sleep
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd


class FuturesTrader:
    """
    Binance USDT-M Futures trader with:
    - Market entries
    - TP/SL management
    - Direction reversal
    - Auto-recovery after TP/SL
    - Trade logging in pandas DataFrame
    """

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

    # -------------------- POSITION CHECK --------------------
    def has_open_position(self):
        """Check if there is any open position on Binance for this symbol."""
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            pos = next((p for p in positions if p["symbol"] == self.symbol), None)
            if not pos:
                return False
            return float(pos["positionAmt"]) != 0
        except Exception:
            return False

    def _validate_trade(self):
        """Check if active trade exists AND actual position is open."""
        return isinstance(self.active_trade, dict) and self.has_open_position()

    # -------------------- ORDER HELPERS --------------------
    def wait_for_fill(self, order_id, timeout=15):
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
            if t["orderId"] == order_id:
                q = float(t["qty"])
                p = float(t["price"])
                f = float(t["commission"])
                qty += q
                cost += q * p
                fee += f
        if qty == 0:
            raise RuntimeError(f"No trades found for order {order_id}")
        return cost / qty, qty, fee

    # -------------------- TP / SL --------------------
    def get_mark_price(self):
        return float(self.client.futures_mark_price(symbol=self.symbol)["markPrice"])

    def place_tp_sl(self, entry_price, direction, qty, tp_pct=3, sl_pct=1):
        """Place TP/SL with a small buffer to avoid instant trigger."""
        BUFFER = 0.001  # 0.1% safety

        if direction == "LONG":
            tp_price = max(entry_price * (1 + tp_pct / 100), self.get_mark_price() * (1 + BUFFER))
            sl_price = min(entry_price * (1 - sl_pct / 100), self.get_mark_price() * (1 - BUFFER))
            side = "SELL"
        else:
            tp_price = min(entry_price * (1 - tp_pct / 100), self.get_mark_price() * (1 - BUFFER))
            sl_price = max(entry_price * (1 + sl_pct / 100), self.get_mark_price() * (1 + BUFFER))
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

        return tp.get("algoId"), sl.get("algoId")
    
    def check_auto_closed_trade(self):
        """
    Check if the current active_trade has been closed automatically
    by TP or SL. If so, log it and reset active_trade.
    """
        if not self._validate_trade():
            return  # No active trade or already closed

        positions = self.client.futures_position_information(symbol=self.symbol)
        pos = next((p for p in positions if p["symbol"] == self.symbol), None)
        if pos and float(pos["positionAmt"]) == 0:
        # Position fully closed automatically
            entry = self.active_trade["entry_price"]
            total_fee = self.active_trade["entry_fee"]  # entry fee; exit fee already paid via TP/SL
        # Calculate PnL from last trades
        # Get last exit trade info
            trades = self.client.futures_account_trades(symbol=self.symbol)
            exit_trades = [t for t in trades if t["orderId"] in [self.active_trade["tp_id"], self.active_trade["sl_id"]]]
            exit_cost = exit_qty = exit_fee = 0.0
            for t in exit_trades:
                q = float(t["qty"])
                p = float(t["price"])
                f = float(t["commission"])
                exit_qty += q
                exit_cost += q * p
                exit_fee += f
            if exit_qty > 0:
                exit_price = exit_cost / exit_qty
                total_fee += exit_fee
                pnl = (exit_price - entry) * exit_qty - total_fee if self.active_trade["direction"] == "LONG" else (entry - exit_price) * exit_qty - total_fee
                buy, sell = (entry, exit_price) if self.active_trade["direction"] == "LONG" else (exit_price, entry)
            
                self.pnl_sum += pnl
            # Log the auto-closed trade
                self.log_trade({
                "direction": self.active_trade["direction"],
                "action": "TP-HIT" if exit_price > entry else "SL-HIT",
                "buy": buy,
                "sell": sell,
                "fee": total_fee,
                "pnl": pnl,
                "order_id": exit_trades[-1]["orderId"] if exit_trades else None
            })

        # Reset active trade
            self.active_trade = None


    # -------------------- TRADE FLOW --------------------
    def open_trade(self, signal, quantity):
        """Open a new trade based on signal (1=LONG, -1=SHORT)."""
        side = "BUY" if signal == 1 else "SELL"
        direction = "LONG" if signal == 1 else "SHORT"

        order = self.client.futures_create_order(
            symbol=self.symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )

        self.wait_for_fill(order["orderId"])
        entry_price, qty, entry_fee = self.get_execution_details(order["orderId"])
        tp_id, sl_id = self.place_tp_sl(entry_price, direction, qty)

        self.active_trade = {
            "direction": direction,
            "entry_price": entry_price,
            "quantity": qty,
            "entry_fee": entry_fee,
            "tp_id": tp_id,
            "sl_id": sl_id
        }

        # Log single trade
        self.log_trade({
            "direction": direction,
            "action": f"{side}-OPEN",
            "buy": entry_price if direction == "LONG" else None,
            "sell": entry_price if direction == "SHORT" else None,
            "fee": entry_fee,
            "pnl": 0.0,
            "order_id": order["orderId"]
        })

    def close_active_trade_market(self, reason="DIRECTION CHANGE"):
        """Close current trade safely, handling TP/SL or partial fills."""
        if not self._validate_trade():
            self.active_trade = None
            return

        # Cancel TP / SL safely
        for oid in [self.active_trade["tp_id"], self.active_trade["sl_id"]]:
            try:
                self.client.futures_cancel_order(symbol=self.symbol, orderId=oid)
            except Exception:
                pass

        direction = self.active_trade["direction"]
        side = "SELL" if direction == "LONG" else "BUY"
        qty = self.active_trade["quantity"]

        # Check actual open position
        positions = self.client.futures_position_information(symbol=self.symbol)
        pos = next((p for p in positions if p["symbol"] == self.symbol), None)
        if not pos or float(pos["positionAmt"]) == 0:
            self.active_trade = None
            return

        # Close the open position (partial/full)
        open_qty = abs(float(pos["positionAmt"]))
        qty_to_close = min(qty, open_qty)

        order = self.client.futures_create_order(
            symbol=self.symbol,
            side=side,
            type="MARKET",
            quantity=qty_to_close,
            reduceOnly=True
        )

        self.wait_for_fill(order["orderId"])
        exit_price, qty_filled, exit_fee = self.get_execution_details(order["orderId"])

        entry = self.active_trade["entry_price"]
        total_fee = self.active_trade["entry_fee"] + exit_fee
        pnl = (exit_price - entry) * qty_filled - total_fee if direction == "LONG" else (entry - exit_price) * qty_filled - total_fee
        buy, sell = (entry, exit_price) if direction == "LONG" else (exit_price, entry)

        self.pnl_sum += pnl

        # Log single trade
        self.log_trade({
            "direction": direction,
            "action": reason,
            "buy": buy,
            "sell": sell,
            "fee": total_fee,
            "pnl": pnl,
            "order_id": order["orderId"]
        })

        self.active_trade = None

    # -------------------- SIGNAL HANDLER --------------------
    def process_signal(self, signal, quantity):
        """
        Process incoming signal:
        - Open trade if no active trade or position
        - Reverse direction if opposite signal
        - Re-open if previous trade auto-closed
        """
        self.check_auto_closed_trade()
        if not self._validate_trade():
            if signal != 0:
                self.open_trade(signal, quantity)
            return

        current_dir = self.active_trade["direction"]

        # Reverse direction
        if (signal == 1 and current_dir == "SHORT") or (signal == -1 and current_dir == "LONG"):
            self.close_active_trade_market("DIRECTION CHANGE")
            self.open_trade(signal, quantity)

        # Re-open same direction if previous trade auto-closed
        elif not self.has_open_position() and signal != 0:
            self.open_trade(signal, quantity)

    # -------------------- LOGGING --------------------
    def log_trade(self, trades):
        """
        Log one or multiple trades into self.trade_log.

        trades: dict for single trade OR list of dicts for multiple trades.
        Each dict must contain keys: direction, action, buy, sell, fee, pnl, order_id
        """
        if isinstance(trades, dict):
            trades = [trades]  # wrap single trade into list

        new_rows = []
        for t in trades:
            new_rows.append({
                "datetime": datetime.now(timezone.utc),
                "direction": t.get("direction"),
                "action": t.get("action"),
                "buy_price": round(t.get("buy"), 6) if t.get("buy") is not None else None,
                "sell_price": round(t.get("sell"), 6) if t.get("sell") is not None else None,
                "fee_total": round(t.get("fee", 0), 6),
                "pnl": round(t.get("pnl", 0), 6),
                "pnl_sum": round(self.pnl_sum, 6),
                "order_id": t.get("order_id")
            })

        # Create DataFrame from all new rows at once
        new_df = pd.DataFrame(new_rows, columns=self.trade_log.columns)

        # Append safely
        self.trade_log = pd.concat([self.trade_log, new_df], ignore_index=True)

    def get_trade_log_df(self):
        """Return a copy of the current trade log."""
        return self.trade_log.copy()
