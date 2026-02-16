import os
import csv
from datetime import datetime, timezone
from time import sleep
from binance.um_futures import UMFutures

class FuturesTrader:
    """
    A simple Binance Futures trader execution engine that handles market entries, 
    take-profit/stop-loss orders, trade reversal, PnL calculation, and CSV logging.

    Attributes:
        client (UMFutures): The Binance Python SDK client for UM Futures.
        symbol (str): The trading pair (e.g., 'BTCUSDT').
        active_trade (dict|None): Stores current position details (entry, ids, fees).
        pnl_sum (float): Cumulative Profit and Loss for the session.
        csv_file (str): Filename for the persistent trade log.
    """

    def __init__(self, api_key, api_secret, symbol):
        """
        Initializes the Trader with API credentials and sets up the CSV log.

        Args:
            api_key (str): Binance API Key.
            api_secret (str): Binance API Secret.
            symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        """
        # Initialize the Binance UMFutures client using the Testnet URL
        self.client = UMFutures(
            key=api_key,
            secret=api_secret,
            base_url="https://testnet.binancefuture.com"
        )
        self.symbol = symbol
        self.active_trade = None  # Tracks state of the current open position
        self.pnl_sum = 0.0        # Session-based cumulative PnL
        self.csv_file = "trade_log.csv"

        # Check if log file exists; if not, create it and write the header row
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                csv.writer(f).writerow([
                    "datetime_utc", "direction", "action", "buy_price", 
                    "sell_price", "fee_total", "pnl", "pnl_sum", "order_id"
                ])

    def _validate_trade(self):
        """
        Checks if there is currently an active trade in the local state.

        Returns:
            bool: True if a trade dictionary exists, False otherwise.
        """
        return isinstance(self.active_trade, dict)

    def wait_for_fill(self, order_id, timeout=10):
        """
        Polls the Binance API until an order status changes to 'FILLED'.

        Args:
            order_id (int): The ID of the order to track.
            timeout (int): Maximum seconds to wait before raising an error.

        Returns:
            dict: The final order object from Binance.

        Raises:
            RuntimeError: If the order is not filled within the timeout period.
        """
        start = datetime.now()
        while (datetime.now() - start).seconds < timeout:
            # Fetch order status from Binance
            order = self.client.query_order(symbol=self.symbol, orderId=order_id)
            if order["status"] == "FILLED":
                return order
            sleep(0.2)  # Short sleep to avoid hitting rate limits
        raise RuntimeError(f"Order {order_id} not filled")

    def get_execution_details(self, order_id):
        """
        Retrieves actual filled price, quantity, and commission fees for an entry order.

        Args:
            order_id (int): The ID of the filled order.

        Returns:
            tuple: (average_price: float, total_qty: float, total_commission: float)
        """
        # Fetch individual trade fills for this specific order
        trades = self.client.get_account_trades(symbol=self.symbol, orderId=order_id)
        qty = cost = fee = 0.0
        for t in trades:
            q = float(t["qty"])
            p = float(t["price"])
            f = float(t["commission"])
            qty += q            # Sum up total quantity
            cost += q * p       # Calculate total cost for weighted average
            fee += f            # Sum up total commission paid
        return cost / qty, qty, fee

    def get_exit_from_order(self, order):
        """
        Extracts exit details (price and fees) from a filled closing order.

        Args:
            order (dict): The order object returned by Binance.

        Returns:
            tuple: (exit_price: float, quantity: float, total_fee: float)
        """
        exit_price = float(order["avgPrice"])    # Actual filled price
        qty = float(order["executedQty"])        # Actual filled quantity
        fee = 0.0
        # Fetch recent trades to find commission associated with this exit
        recent_trades = self.client.get_account_trades(symbol=self.symbol, limit=10)
        for t in recent_trades:
            # Match trades based on quantity (simplistic matching for this logic)
            if float(t["qty"]) == qty:
                fee += float(t["commission"])
        return exit_price, qty, fee

    def place_tp_sl(self, entry_price, direction, tp_pct=0.1, sl_pct=0.1):
        """
        Places Take-Profit and Stop-Loss Market orders on Binance.

        Args:
            entry_price (float): The price at which the position was opened.
            direction (str): "LONG" or "SHORT".
            tp_pct (float): Percentage gain for Take Profit.
            sl_pct (float): Percentage loss for Stop Loss.

        Returns:
            tuple: (tp_order_id: int, sl_order_id: int)
        """
        # Calculate trigger prices based on direction
        if direction == "LONG":
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
            side = "SELL"  # Closing a long requires a sell
        else:
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)
            side = "BUY"   # Closing a short requires a buy

        # Place Take Profit Market Order
        tp = self.client.new_order(
            symbol=self.symbol, side=side, type="TAKE_PROFIT_MARKET",
            stopPrice=round(tp_price, 2), closePosition=True, workingType="MARK_PRICE"
        )
        # Place Stop Loss Market Order
        sl = self.client.new_order(
            symbol=self.symbol, side=side, type="STOP_MARKET",
            stopPrice=round(sl_price, 2), closePosition=True, workingType="MARK_PRICE"
        )
        return tp["orderId"], sl["orderId"]

    def open_trade(self, signal, quantity):
        """
        Executes a market entry and sets up TP/SL protection.

        Args:
            signal (int): 1 for LONG, -1 for SHORT.
            quantity (float): Amount of the asset to trade.
        """
        side = "BUY" if signal == 1 else "SELL"
        direction = "LONG" if signal == 1 else "SHORT"

        # Execute Market Entry Order
        order = self.client.new_order(
            symbol=self.symbol, side=side, type="MARKET", quantity=quantity
        )
        order_id = order["orderId"]
        self.wait_for_fill(order_id)
        
        # Capture exact entry details from the fill
        entry_price, qty, entry_fee = self.get_execution_details(order_id)
        # Set up protection orders
        tp_id, sl_id = self.place_tp_sl(entry_price, direction)

        # Update local state
        self.active_trade = {
            "direction": direction, "entry_price": entry_price,
            "quantity": qty, "entry_fee": entry_fee,
            "tp_id": tp_id, "sl_id": sl_id
        }

        # Log the opening action
        self.log_trade(direction, f"{side}-OPEN", 
                       entry_price if side == "BUY" else "", 
                       entry_price if side == "SELL" else "", 
                       entry_fee, 0.0, order_id)

    def close_active_trade_market(self, reason="DIRECTION CHANGE"):
        """
        Closes the existing position immediately via Market order.

        Args:
            reason (str): Context for the exit (e.g., Signal Reversal).
        """
        if not self._validate_trade():
            return

        # Clean up pending TP/SL orders to prevent double-spending/errors
        try:
            self.client.cancel_order(symbol=self.symbol, orderId=self.active_trade["tp_id"])
            self.client.cancel_order(symbol=self.symbol, orderId=self.active_trade["sl_id"])
        except Exception:
            pass  # Already filled or canceled orders trigger errors, which we ignore

        direction = self.active_trade["direction"]
        side = "SELL" if direction == "LONG" else "BUY"

        # reduceOnly=True ensures we only close the position, never flip it accidentally
        order = self.client.new_order(
            symbol=self.symbol, side=side, type="MARKET", reduceOnly=True
        )
        self.wait_for_fill(order["orderId"])
        exit_price, qty, exit_fee = self.get_exit_from_order(order)

        # PnL logic: (Difference) * Quantity - (Entry Fee + Exit Fee)
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
        self.active_trade = None  # Reset state

    def check_tp_sl_hit(self):
        """
        Checks if the TP or SL orders have been filled on the exchange.
        """
        if not self._validate_trade():
            return
        # Iterate through TP and SL IDs to check their status
        for reason, oid in [("TP-HIT", self.active_trade["tp_id"]),
                            ("SL_HIT", self.active_trade["sl_id"])]:
            order = self.client.query_order(symbol=self.symbol, orderId=oid)
            if order["status"] == "FILLED":
                self.handle_exit(order, reason)
                return

    def handle_exit(self, order, reason):
        """
        Processes the logic for a trade that was closed by a TP/SL order.

        Args:
            order (dict): The filled TP or SL order object.
            reason (str): "TP-HIT" or "SL_HIT".
        """
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
        """
        Main entry point for processing trading signals (1, -1, or 0).

        Args:
            signal (int): 1 (Long), -1 (Short), 0 (Neutral).
            quantity (float): Position size for the trade.
        """
        # First, check if TP or SL were hit since the last loop
        self.check_tp_sl_hit()

        # If no trade is open, open one based on signal
        if not self._validate_trade():
            if signal != 0:
                self.open_trade(signal, quantity)
            return

        # If a trade is open, check if the signal has reversed
        current_dir = self.active_trade["direction"]
        if (signal == 1 and current_dir == "SHORT") or \
           (signal == -1 and current_dir == "LONG"):
            self.close_active_trade_market(reason="DIRECTION CHANGE")
            self.open_trade(signal, quantity)

    def log_trade(self, direction, action, buy, sell, fee, pnl, order_id):
        """
        Appends a trade record to the local CSV file.

        Args:
            direction (str): "LONG" or "SHORT".
            action (str): Description of what happened.
            buy (float|str): Buy price or empty string.
            sell (float|str): Sell price or empty string.
            fee (float): Commission paid.
            pnl (float): Net profit/loss for the trade.
            order_id (int): Binance order reference ID.
        """
        with open(self.csv_file, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                direction, action,
                round(buy, 6) if buy else "",
                round(sell, 6) if sell else "",
                round(fee, 6), round(pnl, 6),
                round(self.pnl_sum, 6), order_id
            ])