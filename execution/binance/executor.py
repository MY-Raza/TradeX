from binance.um_futures import UMFutures
from dotenv import load_dotenv
import os
import time

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '.env'))
load_dotenv(dotenv_path)

API_KEY = os.getenv("BINANCE_DEMO_API_KEY")
API_SECRET = os.getenv("BINANCE_DEMO_SECRET_KEY")

class BinanceTrader:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        base_url = "https://testnet.binancefuture.com" if testnet else None
        self.client = UMFutures(key=api_key, secret=api_secret, base_url=base_url)

    def open_trade(self, symbol: str, signal: int, quantity: float = 0.001, tp_percent: float = 3, sl_percent: float = 1):
        if signal == 0:
            print(f"{symbol} → Signal is HOLD. No trade.")
            return None, None

        positions = self.client.position_information(symbol=symbol)
        entry_qty = float(positions[0]["positionAmt"]) if positions else 0
        if entry_qty != 0:
            print(f"{symbol} → Already in position ({entry_qty}). Skipping open trade.")
            return None, None

        side = "BUY" if signal == 1 else "SELL"
        opposite_side = "SELL" if side == "BUY" else "BUY"

        order = self.client.new_order(symbol=symbol, side=side, type="MARKET", quantity=quantity)
        print(f"{symbol} → Opened {side} MARKET order: {order['orderId']}")

        executed_price = float(order["avgFillPrice"] if "avgFillPrice" in order else order["fills"][0]["price"])

        if signal == 1:  # LONG
            tp_price = round(executed_price * (1 + tp_percent / 100), 2)
            sl_price = round(executed_price * (1 - sl_percent / 100), 2)
        else:  # SHORT
            tp_price = round(executed_price * (1 - tp_percent / 100), 2)
            sl_price = round(executed_price * (1 + sl_percent / 100), 2)

        self.client.new_order(symbol=symbol, side=opposite_side, type="TAKE_PROFIT_MARKET", stopPrice=tp_price, closePosition=True)
        self.client.new_order(symbol=symbol, side=opposite_side, type="STOP_MARKET", stopPrice=sl_price, closePosition=True)

        print(f"{symbol} → TP set at {tp_price}, SL set at {sl_price}")
        return tp_price, sl_price

    def close_trade(self, symbol: str, prev_signal: int, new_signal: int):
        if new_signal == 0 or prev_signal == 0 or new_signal == prev_signal:
            print(f"{symbol} → No opposite signal detected. Skipping close trade.")
            return

        positions = self.client.position_information(symbol=symbol)
        if not positions:
            print(f"{symbol} → No position info found. Skipping close trade.")
            return

        pos = positions[0]
        entry_qty = float(pos["positionAmt"])
        if entry_qty == 0:
            print(f"{symbol} → No open position. Skipping close trade.")
            return

        side_to_close = "SELL" if entry_qty > 0 else "BUY"
        order = self.client.new_order(symbol=symbol, side=side_to_close, type="MARKET", quantity=abs(entry_qty))
        print(f"{symbol} → Closed position with MARKET order {order['orderId']} (Prev: {prev_signal}, New: {new_signal})")

    def tp_sl_check(self, symbol: str, tp_price: float, sl_price: float, quantity: float = 0.001, check_interval: int = 5):
        print(f"{symbol} → Starting TP/SL monitor. TP: {tp_price}, SL: {sl_price}")
        while True:
            ticker = self.client.mark_price(symbol=symbol)
            mark_price = float(ticker["markPrice"])

            positions = self.client.position_information(symbol=symbol)
            if not positions:
                print(f"{symbol} → No position info. Exiting monitor.")
                break

            pos = positions[0]
            entry_qty = float(pos["positionAmt"])
            if entry_qty == 0:
                print(f"{symbol} → Position closed already. Exiting monitor.")
                break

            if entry_qty > 0:  # LONG
                if mark_price >= tp_price:
                    print(f"{symbol} → LONG TP hit at {mark_price}. Closing position.")
                    self.client.new_order(symbol=symbol, side="SELL", type="MARKET", quantity=abs(entry_qty))
                    break
                elif mark_price <= sl_price:
                    print(f"{symbol} → LONG SL hit at {mark_price}. Closing position.")
                    self.client.new_order(symbol=symbol, side="SELL", type="MARKET", quantity=abs(entry_qty))
                    break
            elif entry_qty < 0:  # SHORT
                if mark_price <= tp_price:
                    print(f"{symbol} → SHORT TP hit at {mark_price}. Closing position.")
                    self.client.new_order(symbol=symbol, side="BUY", type="MARKET", quantity=abs(entry_qty))
                    break
                elif mark_price >= sl_price:
                    print(f"{symbol} → SHORT SL hit at {mark_price}. Closing position.")
                    self.client.new_order(symbol=symbol, side="BUY", type="MARKET", quantity=abs(entry_qty))
                    break

            sleep(check_interval)

    # ==========================
    # 4️⃣ Master executor
    # ==========================
    def execute_signal(self, symbol: str, prev_signal: int, new_signal: int, quantity: float = 0.001):
        """
        This function decides:
        1️⃣ Close trade if opposite
        2️⃣ Open trade if none
        3️⃣ Monitor TP/SL for current trade
        """
        # Step 1: Close if opposite
        self.close_trade(symbol, prev_signal, new_signal)

        # Step 2: Open new trade if no position
        tp, sl = self.open_trade(symbol, new_signal, quantity)

        # Step 3: Monitor TP/SL if trade exists
        if tp and sl:
            self.tp_sl_check(symbol, tp, sl, quantity)

