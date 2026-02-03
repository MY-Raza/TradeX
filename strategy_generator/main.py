from TradeX.indicators.talib.signals import candlestick_signal, SIGNAL_FUNCTIONS
import os
import pandas as pd
from TradeX.utils.common.logs import get_logger
import numpy as np
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.backtest.backtester import Backtester, BacktestConfig
import datetime
import json
from TradeX.utils.db.utils import save_df_to_db

logger = get_logger("strategy_main")

ALL_INDICATORS = (
    # Overlap Studies
    "BBANDS", "DEMA", "EMA", "HT_TRENDLINE", "KAMA",
    "MA", "MAMA", "MIDPOINT", "MIDPRICE", "SAR",
    "SAREXT", "SMA", "T3", "TEMA", "TRIMA", "WMA",
    # Momentum Indicators
    "ADX", "ADXR", "APO", "AROON", "AROONOSC",
    "BOP", "CCI", "CMO", "DX", "MACD",
    "MACDEXT", "MACDFIX", "MFI", "MINUS_DI",
    "MINUS_DM", "MOM", "PLUS_DI", "PLUS_DM",
    "PPO", "ROC", "ROCP", "ROCR", "ROCR100",
    "RSI", "STOCH", "STOCHF", "STOCHRSI",
    "TRIX", "ULTOSC", "WILLR",
    # Volume Indicators
    "AD", "ADOSC", "OBV",
    # Volatility Indicators
    "ATR", "NATR", "TRANGE",
    # Price Transform Indicators
    "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE",
    # Cycle Indicators
    "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR",
    "HT_SINE", "HT_TRENDMODE",
    # Statistic Indicators
    "LINEARREG", "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT", "LINEARREG_SLOPE",
    "STDDEV", "TSF", "VAR",
    # Math Transform Indicators
    "ACOS", "ASIN", "ATAN", "CEIL", "COS", "COSH",
    "EXP", "FLOOR", "LN", "LOG10", "SIN", "SINH",
    "SQRT", "TAN", "TANH",
    # Candlestick Patterns
    "CDL2CROWS", "CDL3BLACKCROWS", "CDL3INSIDE",
    "CDL3LINESTRIKE", "CDL3OUTSIDE", "CDL3STARSINSOUTH",
    "CDL3WHITESOLDIERS", "CDLABANDONEDBABY",
    "CDLADVANCEBLOCK", "CDLBELTHOLD", "CDLBREAKAWAY",
    "CDLCLOSINGMARUBOZU", "CDLCONCEALBABYSWALL",
    "CDLCOUNTERATTACK", "CDLDARKCLOUDCOVER",
    "CDLDOJI", "CDLDOJISTAR", "CDLDRAGONFLYDOJI",
    "CDLENGULFING", "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR", "CDLGAPSIDESIDEWHITE",
    "CDLGRAVESTONEDOJI", "CDLHAMMER",
    "CDLHANGINGMAN", "CDLHARAMI",
    "CDLHARAMICROSS", "CDLHIGHWAVE",
    "CDLHIKKAKE", "CDLHIKKAKEMOD",
    "CDLHOMINGPIGEON", "CDLIDENTICAL3CROWS",
    "CDLINNECK", "CDLINVERTEDHAMMER",
    "CDLKICKING", "CDLKICKINGBYLENGTH",
    "CDLLADDERBOTTOM", "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE", "CDLMARUBOZU",
    "CDLMATCHINGLOW", "CDLMATHOLD",
    "CDLMORNINGDOJISTAR", "CDLMORNINGSTAR",
    "CDLONNECK", "CDLPIERCING",
    "CDLRICKSHAWMAN", "CDLRISEFALL3METHODS",
    "CDLSEPARATINGLINES", "CDLSHOOTINGSTAR",
    "CDLSHORTLINE", "CDLSPINNINGTOP",
    "CDLSTALLEDPATTERN", "CDLSTICKSANDWICH",
    "CDLTAKURI", "CDLTASUKIGAP",
    "CDLTHRUSTING", "CDLTRISTAR",
    "CDLUNIQUE3RIVER", "CDLUPSIDEGAP2CROWS",
    "CDLXSIDEGAP3METHODS"
)


def randomize_indicators(all_indicators):
    """Assign True/False randomly to each indicator."""
    flags_array = np.random.choice([True, False], size=len(all_indicators))
    return dict(zip(all_indicators, flags_array))


def run_active_signals_with_voting(flags, open_, high, low, close, volume, timestamps):
    """
    Executes all active indicator functions and aggregates them using majority voting.
    Returns a DataFrame with columns: timestamp, signals.
    """
    import inspect

    signals_dict = {}
    data = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}

    for name, active in flags.items():
        if not active:
            continue

        # Candlestick patterns
        if name.startswith("CDL"):
            sig, _ = candlestick_signal(open_, high, low, close, name)
            signals_dict[name] = sig
            continue

        # Normal indicators
        func = SIGNAL_FUNCTIONS.get(name)
        if func is None:
            logger.warning(f"⚠ No signal function found for {name}")
            continue

        sig_args = inspect.signature(func).parameters
        args_to_pass = [data[arg] for arg in sig_args if arg in data]

        try:
            sig = func(*args_to_pass)
            signals_dict[name] = sig
        except Exception as e:
            logger.warning(f"⚠ Error calling {name}: {e}")

    # ---------------------------
    # Majority Voting (NumPy implementation, avoids SciPy)
    # ---------------------------
    if signals_dict:
        all_signals = np.column_stack(list(signals_dict.values()))
        final_signal = np.zeros(all_signals.shape[0], dtype=int)

        # Assign the value that occurs most frequently per row
        for i in range(all_signals.shape[0]):
            vals, counts = np.unique(all_signals[i], return_counts=True)
            final_signal[i] = vals[np.argmax(counts)]
    else:
        final_signal = np.zeros(len(timestamps), dtype=int)

    # ---------------------------
    # Return DataFrame
    # ---------------------------
    df_signals = pd.DataFrame({
        "timestamp": timestamps,
        "signals": final_signal
    })
    return df_signals

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

# ============================
# Load and prepare data
# ============================
INPUT_CSV = r"D:\trading\TradeX\indicators\talib\btc_1m_data.csv"  
df_1m = pd.read_csv(INPUT_CSV)
if df_1m.empty:
    logger.error("No OHLCV data. Exiting.")
    exit()

df_1m["timestamp"] = pd.to_datetime(df_1m["datetime"])
df_1m = df_1m.drop(columns=["datetime"])
# Resample 1m -> 1h
df_1h = resample_ohlcv(df_1m, "1h")

# Prepare arrays
open_ = df_1h["open"].values
high = df_1h["high"].values
low = df_1h["low"].values
close = df_1h["close"].values
volume = df_1h["volume"].values

# Randomly activate indicators
flags = randomize_indicators(ALL_INDICATORS)

# Run active signals with voting
signals = run_active_signals_with_voting(
    flags,
    open_,
    high,
    low,
    close,
    volume,
    df_1h["timestamp"]
)

# Save signals to db

strategy_id = generate_strategy_id(flags, timeframe="1h")

save_df_to_db(
    df=signals,
    schema="strategy_signals",
    table_name=strategy_id,
    time_column="timestamp",
    is_timeseries=True
)


# ============================
# Backtesting
# ============================
config = BacktestConfig(
    starting_balance=1000,
    leverage=1,
    transaction_fee=0.05,   # percent
    slippage=0.02,          # percent
    take_profit_pct=0.03,   # 3%
    stop_loss_pct=0.01,     # 1%
    buy_after_minutes=1,
    min_balance_pct=0.5
)
bt = Backtester(config)
ledger_df, final_balance, total_pnl = bt.run(df_1m, signals)

logger.info("\n===== BACKTEST RESULTS =====")
logger.info(f"Final Balance: {final_balance}")
logger.info(f"Total PnL %: {total_pnl}")
logger.info(f"Number of Trades: {len(ledger_df)}")

# Save ledger
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
SIGNALS_FOLDER = os.path.join(BASE_DIR, "strategy_csv")  
os.makedirs(SIGNALS_FOLDER, exist_ok=True)

ledger_csv = f"{strategy_id}_ledger.csv"
output_csv = os.path.join(SIGNALS_FOLDER, ledger_csv)
ledger_df.to_csv(output_csv, index=False)

# ============================
# Save meta data to db including full flags
# ============================
strategy_df = pd.DataFrame([flags])
strategy_df.insert(0, "timeframe", "1h") 
strategy_df.insert(0, "strategy", strategy_id)
strategy_df.insert(0, "creation_time", datetime.datetime.utcnow())
strategy_df.columns = strategy_df.columns.str.lower()
print(strategy_df)

save_df_to_db(
    df=strategy_df,
    table_name="strategy_registry",
    schema="strategy_identifier",
    time_column="creation_time", 
    is_timeseries=False
)


