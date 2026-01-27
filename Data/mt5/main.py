import MetaTrader5 as mt5
from datetime import datetime, timedelta
from TradeX.utils.common.config_loader import read_config
from TradeX.data.mt5.metatrader5_fetcher import MetaTrader5FutureFetcher
from dotenv import load_dotenv
import os
from TradeX.utils.data.data_cleaner import clean_df
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from TradeX.utils.db.utils import save_df_to_db, get_last_date, drop_schema,read_df_from_db
from TradeX.utils.common.logs import get_logger

# =========================================
# LOGGER INITIALIZATION
# =========================================
logger = get_logger("metatrader5_main")

# =========================================
# MT5 CONNECTION DETAILS FROM ENV
# =========================================
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path)

MT5_LOGIN = int(os.getenv("MT5_LOGIN"))      # MT5 login ID
MT5_PASSWORD = os.getenv("MT5_PASSWORD")     # MT5 password
MT5_SERVER = os.getenv("MT5_SERVER")         # MT5 server address

# =========================================
# INITIALIZE MT5 CONNECTION
# =========================================
if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    raise RuntimeError(f"❌ MT5 init failed: {mt5.last_error()}")
logger.info("✅ MT5 initialized successfully")

# =========================================
# LOAD CONFIGURATION FILE
# =========================================
config = read_config("config.yml")           # Reads config.yml for symbols and date ranges
raw_symbols = config["symbols"]              # List of symbols to fetch
start_date = config["start_date"]            # Start date for fetching data
end_date = config["end_date"]                # End date or "now" for current timestamp

# Convert start and end date strings to datetime objects
utc_from = datetime.fromisoformat(start_date)
utc_to = datetime.now() if end_date == "now" else datetime.fromisoformat(end_date)

# =========================================
# LOAD CONSTANTS
# =========================================
SCHEMA = EXCHANGE_SCHEMA_MAP["metatrader5"]  # Database schema for MT5 data

# =========================================
# CREATE MT5 FUTURES FETCHER INSTANCE
# =========================================
fetcher = MetaTrader5FutureFetcher(
    symbols=raw_symbols,
    utc_from=utc_from,
    utc_to=utc_to,
    timeframe=mt5.TIMEFRAME_M1       # 1-minute candlestick data
)

# =========================================
# FETCH AND SAVE DATA LOOP
# =========================================
for symbol in raw_symbols:
    # Check the last timestamp in DB to do incremental fetch
    last_ts = get_last_date(f"{symbol}_1m", schema=SCHEMA, time_column="timestamp")
    
    if last_ts:
        # Incremental fetch starts from the next minute after last record
        start_date = datetime.fromtimestamp(last_ts / 1000) + timedelta(minutes=1)
        logger.info(f"Incremental fetch for {symbol} from {start_date}")
    else:
        # If no data exists, fetch from the config start_date
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        logger.info(f"No existing data for {symbol}. Fetching from {start_date}")
    
    # Update fetcher start date
    fetcher.utc_from = start_date

    logger.info(f"Fetching data for {symbol}...")
    df = fetcher.fetch(symbol)  # Fetch raw data from MT5

    if df is not None and not df.empty:
        logger.info(f"Raw rows fetched for {symbol}: {len(df)}")

        # Clean and format the dataframe (fill missing timestamps, etc.)
        df = clean_df(df, "1m")

        # Save the cleaned dataframe to the database
        save_df_to_db(
            df=df,
            table_name=f"{symbol.lower()}_1m",
            schema=SCHEMA,
            time_column="timestamp",
            is_timeseries=True
        )

        logger.info(f"Data for {symbol} saved to DB successfully")
        logger.info(df.tail(5))
    else:
        logger.warning(f"⚠ No data returned for {symbol}")
# =========================================
# SHUTDOWN MT5 CONNECTION
# =========================================
mt5.shutdown()
logger.info("🔌 MT5 shutdown complete")
