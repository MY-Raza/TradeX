import random
from TradeX.indicators.talib.signals import candlestick_signal,SIGNAL_FUNCTIONS
import os
import pandas as pd
from TradeX.utils.common.logs import get_logger
import numpy as np
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.backtest.backtester import Backtester,BacktestConfig
import datetime
from scipy.stats import mode

logger = get_logger("strategy_main")
ALL_INDICATORS = (
    # -------------------------
    # Overlap Studies
    # -------------------------
    "BBANDS", "DEMA", "EMA", "HT_TRENDLINE", "KAMA",
    "MA", "MAMA", "MIDPOINT", "MIDPRICE", "SAR",
    "SAREXT", "SMA", "T3", "TEMA", "TRIMA", "WMA",
    # -------------------------
    # Momentum Indicators
    # -------------------------
    "ADX", "ADXR", "APO", "AROON", "AROONOSC",
    "BOP", "CCI", "CMO", "DX", "MACD",
    "MACDEXT", "MACDFIX", "MFI", "MINUS_DI",
    "MINUS_DM", "MOM", "PLUS_DI", "PLUS_DM",
    "PPO", "ROC", "ROCP", "ROCR", "ROCR100",
    "RSI", "STOCH", "STOCHF", "STOCHRSI",
    "TRIX", "ULTOSC", "WILLR",
    # -------------------------
    # Volume Indicators
    # -------------------------
    "AD", "ADOSC", "OBV",
    # -------------------------
    # Volatility Indicators
    # -------------------------
     "ATR", "NATR", "TRANGE",
    # -------------------------
    # Price Transform Indicators
    # -------------------------
    "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE",
    # -------------------------
    # Cycle Indicators
    # -------------------------
    "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR",
    "HT_SINE", "HT_TRENDMODE",
    # -------------------------
    # Statistic Indicators
    # -------------------------
    "LINEARREG", "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT", "LINEARREG_SLOPE",
    "STDDEV", "TSF", "VAR",
    # -------------------------
    # Math Transform Indicators
    # -------------------------
    "ACOS", "ASIN", "ATAN", "CEIL", "COS", "COSH",
    "EXP", "FLOOR", "LN", "LOG10", "SIN", "SINH",
    "SQRT", "TAN", "TANH",
    # ---------------------------
    # CANDLESTICK PATTERN 
    # --------------------------
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
    """
    Assigns True/False randomly to each indicator
    using random.choice.
    """
    flags_array = np.random.choice([True, False], size=len(all_indicators))
    indicator_flags = dict(zip(all_indicators, flags_array))
    return indicator_flags

def run_active_signals_with_voting(flags, open_, high, low, close, volume, timestamps):
    """
    Executes all active indicator functions and aggregates them using voting.
    Returns a DataFrame with columns: timestamp, signal
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
            print(f"⚠ No signal function found for {name}")
            continue

        sig_args = inspect.signature(func).parameters
        args_to_pass = [data[arg] for arg in sig_args if arg in data]

        try:
            sig = func(*args_to_pass)
            signals_dict[name] = sig
        except Exception as e:
            print(f"⚠ Error calling {name}: {e}")

    # ---------------------------
    # Voting / Aggregating signals
    # ---------------------------
    if signals_dict:
        all_signals = np.column_stack(list(signals_dict.values()))
        final_signal, _  = mode(all_signals,axis = 1)
        final_signal = final_signal.ravel().astype(int)
    else:
        final_signal = np.zeros(len(timestamps), dtype=int)

    # ---------------------------
    # Return DataFrame with timestamp and signal
    # ---------------------------
    df_signals = pd.DataFrame({
        "timestamp": timestamps,
        "signals": final_signal
    })

    return df_signals



INPUT_CSV = r"D:\trading\TradeX\indicators\talib\btc_1m_data.csv"  
df_1m = pd.read_csv(INPUT_CSV)
df_1m = pd.read_csv(INPUT_CSV)
if df_1m.empty:
    logger.error("No OHLCV data. Exiting.")
    exit()
df_1m["timestamp"] = pd.to_datetime(df_1m["datetime"])
# Optional: drop the old datetime column if you no longer need it
df_1m = df_1m.drop(columns=["datetime"])
# Resample 1m -> 1h
df_1h = resample_ohlcv(df_1m, "1h")

# ---------------------------
# Prepare arrays for automatic argument detection
# ---------------------------
open_ = df_1h["open"].values
high = df_1h["high"].values
low = df_1h["low"].values
close = df_1h["close"].values
volume = df_1h["volume"].values
ref = np.random.uniform(50, 200, len(df_1h)).astype(np.float64)
periods = np.random.randint(5, 30, len(df_1h)).astype(np.float64)


flags = randomize_indicators(ALL_INDICATORS)
for name, value in flags.items():
    if value:
        print("✔", name)
signals = run_active_signals_with_voting(
    flags,
    open_,
    high,
    low,
    close,
    volume,
    df_1h["timestamp"]
)
print(signals.head())
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
signals.to_csv(f"signals.csv", index=False) 
signals_df = pd.read_csv("signals.csv")
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
ledger_df, final_balance, total_pnl = bt.run(df_1m, signals_df)
print("\n===== BACKTEST RESULTS =====")
print("Final Balance:", final_balance)
print("Total PnL %:", total_pnl)
print("Number of Trades:", len(ledger_df))

    # Save trades
ledger_df.to_csv("ledger.csv", index=False)
print("\nTrade log saved to ledger.csv")
