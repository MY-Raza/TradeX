import json
import os

# ============================
# File path for storing strategy counters
# ============================
COUNTER_FILE = r"D:\trading\TradeX\strategy_generator\strategy_counter.json"


# ============================
# Load counters from JSON file
# ============================
def _load_counters():
    """
    Load strategy counters from a JSON file.

    Returns:
        dict: A dictionary of counters by timeframe, e.g.,
              {"1h": 5, "15m": 12}
              If the file does not exist, returns an empty dictionary.
    """
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r") as f:
            return json.load(f)
    # Return empty dict if file does not exist
    return {}


# ============================
# Save counters to JSON file
# ============================
def _save_counters(counters):
    """
    Save the strategy counters to a JSON file.

    Args:
        counters (dict): Dictionary of counters to save.
    """
    with open(COUNTER_FILE, "w") as f:
        json.dump(counters, f, indent=4)


# ============================
# Generate unique strategy ID
# ============================
def generate_strategy_id(flags: dict, timeframe="1h"):
    """
    Generate a unique strategy ID for a given timeframe.

    The function automatically increments the counter for the
    specified timeframe, saves it, and returns a string ID.

    Args:
        flags (dict): Dictionary of indicator flags (True/False).
                      Used for potential future use (not used currently).
        timeframe (str, optional): Strategy timeframe, e.g., "1h". Defaults to "1h".

    Returns:
        str: A unique strategy ID, e.g., "sig_1h_btc_7"
    """
    # Load existing counters
    counters = _load_counters()

    # Increment counter for this timeframe
    counters[timeframe] = counters.get(timeframe, 0) + 1

    # Save updated counters back to file
    _save_counters(counters)

    # Return unique strategy ID
    return f"sig_{timeframe}_btc_{counters[timeframe]}"
