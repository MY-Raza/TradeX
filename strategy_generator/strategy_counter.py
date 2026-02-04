import json
import os
# ============================
# Strategy counter
# ============================
COUNTER_FILE = r"D:\trading\TradeX\strategy_generator\strategy_counter.json"

def _load_counters():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_counters(counters):
    with open(COUNTER_FILE, "w") as f:
        json.dump(counters, f)

def generate_strategy_id(flags: dict, timeframe="1h"):
    counters = _load_counters()
    counters[timeframe] = counters.get(timeframe, 0) + 1
    _save_counters(counters)
    return f"sig_{timeframe}_btc_{counters[timeframe]}"