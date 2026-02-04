import os
import json
import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from TradeX.indicators.talib.signals import candlestick_signal, SIGNAL_FUNCTIONS
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.backtest.newbacktest import HighPerfBacktest
from TradeX.utils.db.utils import save_df_to_db

logger = get_logger("strategy_main")

# ============================
# All indicators
# ============================
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

# ============================
# Randomize indicators
# ============================
def randomize_indicators(all_indicators):
    flags_array = np.random.choice([True, False], size=len(all_indicators))
    return dict(zip(all_indicators, flags_array))

# ============================
# Compute signals in parallel
# ============================
def run_active_signals_with_voting(flags, open_, high, low, close, volume, timestamps):
    signals_dict = {}
    data = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}

    def compute_signal(name):
        try:
            if name.startswith("CDL"):
                sig, _ = candlestick_signal(open_, high, low, close, name)
                return name, sig.astype(np.int8)
            func = SIGNAL_FUNCTIONS.get(name)
            if func is None:
                return None, None
            args = [data[arg] for arg in func.__code__.co_varnames if arg in data]
            sig = func(*args)
            return name, sig.astype(np.int8)
        except Exception as e:
            logger.warning(f"Error calling {name}: {e}")
            return None, None

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(compute_signal, [n for n, a in flags.items() if a])

    for name, sig in results:
        if name is not None and sig is not None:
            signals_dict[name] = sig

    # Voting
    if signals_dict:
        all_signals = np.column_stack(list(signals_dict.values()))
        buy_votes = np.sum(all_signals == 1, axis=1)
        sell_votes = np.sum(all_signals == -1, axis=1)
        final_signal = np.where(buy_votes > sell_votes, 1,
                        np.where(sell_votes > buy_votes, -1, 0)).astype(np.int8)
    else:
        final_signal = np.zeros(len(timestamps), dtype=np.int8)

    return pd.DataFrame({"timestamp": timestamps, "signals": final_signal})

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

# ============================
# Load data
# ============================
INPUT_CSV = r"D:\trading\TradeX\indicators\talib\btc_1m_data.csv"
df_1m = pd.read_csv(INPUT_CSV)
if df_1m.empty:
    logger.error("No OHLCV data. Exiting.")
    exit()

df_1m["timestamp"] = pd.to_datetime(df_1m["datetime"])
df_1m = df_1m.drop(columns=["datetime"])
df_1h = resample_ohlcv(df_1m, "1h")

open_ = df_1h["open"].values
high = df_1h["high"].values
low = df_1h["low"].values
close = df_1h["close"].values
volume = df_1h["volume"].values

# ============================
# Run signals
# ============================
flags = randomize_indicators(ALL_INDICATORS)
signals = run_active_signals_with_voting(flags, open_, high, low, close, volume, df_1h["timestamp"])

# ============================
# Save signals to DB
# ============================
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
bt = HighPerfBacktest(
    df_price=df_1m,
    df_predictions=signals,
    starting_balance=1000,
    take_profit=3,
    stop_loss=1,
    fee=0.05,
    leverage=1,
    slippage=0
)
ledger, final_balance, total_pnl_percent = bt.run()

logger.info("\n===== BACKTEST RESULTS =====")
logger.info(f"Final Balance: {final_balance}")
logger.info(f"Total PnL %: {total_pnl_percent}")
logger.info(f"Number of Trades: {len(ledger)}")

# ============================
# Save ledger CSV
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FOLDER = os.path.join(BASE_DIR, "strategy_csv")
os.makedirs(SIGNALS_FOLDER, exist_ok=True)

ledger_csv = f"{strategy_id}_ledger.csv"
ledger.to_csv(os.path.join(SIGNALS_FOLDER, ledger_csv), index=False)

# ============================
# Save strategy metadata
# ============================
strategy_df = pd.DataFrame([flags])
strategy_df.insert(0, "timeframe", "1h")
strategy_df.insert(0, "symbol", "btc")
strategy_df.insert(0, "strategy", strategy_id)
strategy_df.columns = strategy_df.columns.str.lower()

save_df_to_db(
    df=strategy_df,
    table_name="strategy_registry",
    schema="strategy_identifier",
    time_column="strategy",
    is_timeseries=False
)
