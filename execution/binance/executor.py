import os
import csv
from datetime import datetime, timezone
from time import sleep
from binance.um_futures import UMFutures


class FuturesTrader:
    """
    A simple Binance Futures trader bot that handles market entries, take-profit/stop-loss orders,
    trade reversal, PnL calculation, and CSV logging.

    Attributes:
        client (UMFutures): Binance Futures client instance.
        symbol (str): Trading symbol (e.g., "BTCUSDT").
        active_trade (dict or None): Stores the currently open trade details.
        pnl_sum (float): Cumulative PnL across all trades.
        csv_file (str): Path to CSV file where trades are logged.
    """

    def __init__(self, api_key, api_secret, symbol):
        """
        Initializes the trader instance with Binance API credentials and trading symbol.

        Args:
            api_key (str): Binance API key.
            api_secret (str): Binance API secret.
            symbol (str): Trading symbol, e.g., "BTCUSDT".
        """
        self.client = UMFutures(
            key=api_key,
            secret=api_secret,
            base_url="https://testnet.binancefuture.com"  # Using testnet for safety
        )

        self.symbol = symbol
        self.active_trade = None
        self.pnl_sum = 0.0
        self.csv_file = "trade_log.csv"

        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                csv.writer(f).writerow([
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

    # =========================
    # STATE GUARD
    # =========================
    def _validate_trade(self):
        """
        Checks if there is an active trade.

        Returns:
            bool: True if active_trade is a valid dict, False otherwise.
        """
        return isinstance(self.active_trade, dict)

    # =========================
    # WAIT FOR ORDER FILL
    # =========================
    def wait_for_fill(self, order_id, timeout=10):
        """
        Waits for a Binance order to be filled, polling every 0.2 seconds.

        Args:
            order_id (int): Binance order ID.
            timeout (int): Maximum seconds to wait for the order to fill.

        Returns:
            dict: Filled order data.

        Raises:
            RuntimeError: If the order is not filled within the timeout.
        """
        start = datetime.now()
        while (datetime.now() - start).seconds < timeout:
            order = self.client.query_order(symbol=self.symbol, orderId=order_id)
            if order["status"] == "FILLED":
                return order
            sleep(0.2)
        raise RuntimeError(f"Order {order_id} not filled")

    # =========================
    # ENTRY EXECUTION DETAILS
    # =========================
    def get_execution_details(self, order_id):
        """
        Fetches execution details for a filled market order.

        Args:
            order_id (int): Binance order ID.

        Returns:
            tuple: (average_price, total_quantity, total_fee)
        """
        trades = self.client.get_account_trades(symbol=self.symbol, orderId=order_id)

        qty = cost = fee = 0.0
        for t in trades:
            q = float(t["qty"])
            p = float(t["price"])
            f = float(t["commission"])
            qty += q
            cost += q * p
            fee += f

        return cost / qty, qty, fee  # avg_price, quantity, total fee

    # =========================
    # EXIT DETAILS (TP / SL)
    # =========================
    def get_exit_from_order(self, order):
        """
        Extracts exit price, quantity, and fee from a filled exit order (TP or SL).

        Args:
            order (dict): Filled Binance order object.

        Returns:
            tuple: (exit_price, executed_qty, total_fee)
        """
        exit_price = float(order["avgPrice"])
        qty = float(order["executedQty"])

        # Sum up recent trade commissions to approximate exit fee
        fee = 0.0
        recent_trades = self.client.get_account_trades(symbol=self.symbol, limit=10)
        for t in recent_trades:
            if float(t["qty"]) == qty:
                fee += float(t["commission"])

        return exit_price, qty, fee

    # =========================
    # PLACE TAKE-PROFIT AND STOP-LOSS
    # =========================
    def place_tp_sl(self, entry_price, direction, tp_pct=0.1, sl_pct=0.1):
        """
        Places TAKE_PROFIT_MARKET and STOP_MARKET orders based on entry price and direction.

        Args:
            entry_price (float): Price at which trade was entered.
            direction (str): "LONG" or "SHORT".
            tp_pct (float): Take-profit percentage.
            sl_pct (float): Stop-loss percentage.

        Returns:
            tuple: (tp_order_id, sl_order_id)
        """
        if direction == "LONG":
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
            side = "SELL"  # closing side for LONG
        else:
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)
            side = "BUY"   # closing side for SHORT

        tp = self.client.new_order(
            symbol=self.symbol,
            side=side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=round(tp_price, 2),
            closePosition=True,
            workingType="MARK_PRICE"
        )

        sl = self.client.new_order(
            symbol=self.symbol,
            side=side,
            type="STOP_MARKET",
            stopPrice=round(sl_price, 2),
            closePosition=True,
            workingType="MARK_PRICE"
        )

        return tp["orderId"], sl["orderId"]

    # =========================
    # OPEN NEW TRADE
    # =========================
    def open_trade(self, signal, quantity):
        """
        Opens a new market trade based on the signal.

        Args:
            signal (int): 1 for LONG, -1 for SHORT.
            quantity (float): Quantity to trade.
        """
        side = "BUY" if signal == 1 else "SELL"
        direction = "LONG" if signal == 1 else "SHORT"

        # Place market order
        order = self.client.new_order(
            symbol=self.symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )

        order_id = order["orderId"]
        self.wait_for_fill(order_id)

        # Get average entry price, quantity, and fee
        entry_price, qty, entry_fee = self.get_execution_details(order_id)

        # Place TP / SL orders
        tp_id, sl_id = self.place_tp_sl(entry_price, direction)

        # Store trade details
        self.active_trade = {
            "direction": direction,
            "entry_price": entry_price,
            "quantity": qty,
            "entry_fee": entry_fee,
            "tp_id": tp_id,
            "sl_id": sl_id
        }

        # Log trade opening
        self.log_trade(
            direction,
            f"{side}-OPEN",
            entry_price if side == "BUY" else "",
            entry_price if side == "SELL" else "",
            entry_fee,
            0.0,
            order_id
        )

    # =========================
    # CLOSE ACTIVE TRADE (MARKET)
    # =========================
    def close_active_trade_market(self, reason="REVERSE"):
        """
        Closes the currently active trade at market price. Can be triggered manually or during a reversal.

        Args:
            reason (str): Reason for closing the trade ("REVERSE", "MANUAL", etc.)
        """
        if not self._validate_trade():
            return

        direction = self.active_trade["direction"]
        side = "SELL" if direction == "LONG" else "BUY"

        # Close position at market
        order = self.client.new_order(
            symbol=self.symbol,
            side=side,
            type="MARKET",
            closePosition=True
        )

        self.wait_for_fill(order["orderId"])
        exit_price, qty, exit_fee = self.get_exit_from_order(order)

        entry = self.active_trade["entry_price"]
        entry_fee = self.active_trade["entry_fee"]
        total_fee = entry_fee + exit_fee

        # Calculate PnL and determine buy/sell for CSV logging
        if direction == "LONG":
            pnl = (exit_price - entry) * qty - total_fee
            buy, sell = entry, exit_price
        else:
            pnl = (entry - exit_price) * qty - total_fee
            buy, sell = exit_price, entry

        self.pnl_sum += pnl

        # Log trade closing
        self.log_trade(
            direction,
            reason,
            buy,
            sell,
            total_fee,
            pnl,
            order["orderId"]
        )

        self.active_trade = None

    # =========================
    # CHECK TP / SL ORDERS
    # =========================
    def check_tp_sl_hit(self):
        """
        Checks if the active trade's TP or SL orders have been filled and handles exit if hit.
        """
        if not self._validate_trade():
            return

        for reason, oid in [("TP", self.active_trade["tp_id"]),
                            ("SL", self.active_trade["sl_id"])]:
            order = self.client.query_order(symbol=self.symbol, orderId=oid)
            if order["status"] == "FILLED":
                self.handle_exit(order, reason)
                return

    # =========================
    # HANDLE EXIT
    # =========================
    def handle_exit(self, order, reason):
        """
        Handles exit logic for a filled TP or SL order.

        Args:
            order (dict): Filled Binance order.
            reason (str): Reason for exit ("TP", "SL", etc.)
        """
        exit_price, qty, exit_fee = self.get_exit_from_order(order)

        entry = self.active_trade["entry_price"]
        entry_fee = self.active_trade["entry_fee"]
        direction = self.active_trade["direction"]
        total_fee = entry_fee + exit_fee

        # PnL calculation based on trade direction
        if direction == "LONG":
            pnl = (exit_price - entry) * qty - total_fee
            buy, sell = entry, exit_price
        else:
            pnl = (entry - exit_price) * qty - total_fee
            buy, sell = exit_price, entry

        self.pnl_sum += pnl

        # Log the exit
        self.log_trade(
            direction,
            reason,
            buy,
            sell,
            total_fee,
            pnl,
            order["orderId"]
        )

        self.active_trade = None

    # =========================
    # PROCESS SIGNAL
    # =========================
    def process_signal(self, signal, quantity):
        """
        Main logic for processing new trading signals.

        Args:
            signal (int): 1 for LONG, -1 for SHORT.
            quantity (float): Trade quantity.
        """
        # First, check if TP/SL has been hit
        self.check_tp_sl_hit()

        # If no active trade, open a new trade
        if not self._validate_trade():
            self.open_trade(signal, quantity)
            return

        current_dir = self.active_trade["direction"]

        # If signal reverses current trade, close and open new trade
        if (signal == 1 and current_dir == "SHORT") or \
           (signal == -1 and current_dir == "LONG"):
            self.close_active_trade_market(reason="REVERSE")
            self.open_trade(signal, quantity)

    # =========================
    # LOG TRADE TO CSV
    # =========================
    def log_trade(self, direction, action, buy, sell, fee, pnl, order_id):
        """
        Logs trade details into the CSV file.

        Args:
            direction (str): "LONG" or "SHORT".
            action (str): Action description, e.g., "BUY-OPEN", "SELL-CLOSE".
            buy (float): Buy price.
            sell (float): Sell price.
            fee (float): Total fees.
            pnl (float): Profit or loss for this trade.
            order_id (int): Binance order ID.
        """
        with open(self.csv_file, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                direction,
                action,
                round(buy, 6) if buy else "",
                round(sell, 6) if sell else "",
                round(fee, 6),
                round(pnl, 6),
                round(self.pnl_sum, 6),
                order_id
            ])
