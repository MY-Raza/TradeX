import os
import pandas as pd
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.backtest.newbacktest import HighPerfBacktest
from TradeX.utils.db.utils import save_df_to_db
from strategy_counter import generate_strategy_id
from signals_combiner import randomize_indicators, run_active_signals_with_voting

logger = get_logger("strategy_main")

# ============================
# List of all indicators
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
# Load 1-minute OHLCV data
# ============================
INPUT_CSV = r"D:\trading\TradeX\indicators\talib\btc_1m_data.csv"
df_1m = pd.read_csv(INPUT_CSV)

if df_1m.empty:
    logger.error("No OHLCV data found. Exiting script.")
    exit()

# Convert datetime column to pandas datetime
df_1m["timestamp"] = pd.to_datetime(df_1m["datetime"])
df_1m = df_1m.drop(columns=["datetime"])

# ============================
# Resample data to 1-hour OHLCV
# ============================
df_1h = resample_ohlcv(df_1m, "1h")

# Extract NumPy arrays for fast calculations
open_ = df_1h["open"].values
high = df_1h["high"].values
low = df_1h["low"].values
close = df_1h["close"].values
volume = df_1h["volume"].values

# ============================
# Randomly activate/deactivate indicators
# ============================
flags = randomize_indicators(ALL_INDICATORS)

# ============================
# Compute active signals with voting
# ============================
# Voting mechanism: each indicator outputs -1, 0, or 1
# The final signal per timestamp is determined by majority vote
signals = run_active_signals_with_voting(flags, open_, high, low, close, volume, df_1h["timestamp"])

# ============================
# Generate unique strategy ID
# ============================
strategy_id = generate_strategy_id(flags, timeframe="1h")

# ============================
# Save signals to database
# ============================
save_df_to_db(
    df=signals,
    schema="strategy_signals",
    table_name=strategy_id,
    time_column="timestamp",
    is_timeseries=True
)

# ============================
# Run backtest
# ============================
bt = HighPerfBacktest(
    df_price=df_1m,          # Original 1-minute price data
    df_predictions=signals,   # Signals computed from indicators
    starting_balance=1000,    # Initial capital
    take_profit=3,            # Take profit % per trade
    stop_loss=1,              # Stop loss % per trade
    fee=0.05,                 # Trading fee per trade
    leverage=1,               # Leverage multiplier
    slippage=0                # Slippage in price
)

# Execute backtest
ledger, final_balance, total_pnl_percent = bt.run()

# ============================
# Log backtest results
# ============================
logger.info("\n===== BACKTEST RESULTS =====")
logger.info(f"Final Balance: {final_balance}")
logger.info(f"Total PnL %: {total_pnl_percent}")
logger.info(f"Number of Trades: {len(ledger)}")

# ============================
# Save ledger as CSV
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_FOLDER = os.path.join(BASE_DIR, "strategy_csv")
os.makedirs(SIGNALS_FOLDER, exist_ok=True)

ledger_csv = f"{strategy_id}_ledger.csv"
ledger.to_csv(os.path.join(SIGNALS_FOLDER, ledger_csv), index=False)

# ============================
# Save strategy metadata
# ============================
strategy_df = pd.DataFrame([flags])  # Flags per indicator (True/False)
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
