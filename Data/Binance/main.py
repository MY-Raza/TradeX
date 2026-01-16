import os
import yaml
from dotenv import load_dotenv

from TradeX.utils.db.utils import (
    get_engine,
    create_schema,
    save_df_to_db,
    read_df_from_db,
    total_columns,
    total_rows,
    drop_schema
)
from TradeX.logs.logging import get_logger
from binance_fetcher import BinanceFuturesFetcher
from TradeX.utils.cleaning_utils import (
    clean_klines_df,
    fill_missing_timestamps,
    fill_missing_values,
    resample_ohlcv
)

logger = get_logger(__name__)

"""
main.py

End-to-end Binance Futures data ingestion pipeline:

1. Load environment variables and config
2. Initialize database and schema
3. Initialize Binance Futures fetcher (stateful)
4. Fetch raw OHLCV data
5. Clean and process data
6. Save to PostgreSQL / TimescaleDB
7. Verify insertion
"""

# -------------------------------------------------
# Load Environment Variables
# -------------------------------------------------
load_dotenv()

SCHEMA = os.getenv("DB_SCHEMA_BINANCE", "data_binance")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    raise RuntimeError("Binance API credentials not found in environment variables.")

logger.info("Environment variables loaded successfully.")

# -------------------------------------------------
# Load Config
# -------------------------------------------------
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

symbols = config.get("symbols", [])
start_date = config.get("start_date")
end_date = config.get("end_date", "now")
interval = config.get("interval", "1m")

if not symbols or not start_date:
    raise ValueError("Config file must contain at least 'symbols' and 'start_date'.")

logger.info(
    f"Config loaded | symbols={symbols} | start={start_date} | end={end_date}"
)

# -------------------------------------------------
# Initialize Database
# -------------------------------------------------
engine = get_engine()
if engine is None:
    raise RuntimeError("Database engine could not be initialized.")

create_schema(engine, schema=SCHEMA)
logger.info(f"Database schema '{SCHEMA}' is ready.")

# -------------------------------------------------
# Fetch, Clean & Store Data
# -------------------------------------------------
for symbol in symbols:
    logger.info(f"Starting pipeline for symbol: {symbol}")

    fetcher = BinanceFuturesFetcher(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_SECRET_KEY,
        symbol=f"{symbol.upper()}USDT",
        start_date=start_date,
        end_date=end_date,
        interval=interval
    )

    # ---------------------------
    # Fetch Raw Data
    # ---------------------------
    raw_df = fetcher.fetch_klines()

    if raw_df.empty:
        logger.warning(f"No data fetched for {symbol}. Skipping.")
        continue

    logger.info(f"Raw data fetched | rows={len(raw_df)}")

    # ---------------------------
    # Data Cleaning Pipeline
    # ---------------------------
    df = clean_klines_df(raw_df)
    logger.info("Basic OHLCV cleaning completed.")

    df = fill_missing_timestamps(df, interval=interval)
    logger.info("Missing timestamps filled.")

    df = fill_missing_values(df, method="ffill")
    logger.info("Missing values forward-filled.")

    # Optional resampling
    # df = resample_ohlcv(df, interval="5min")
    # logger.info("Data resampled to 5-minute candles.")

    # ---------------------------
    # Save to Database
    # ---------------------------
    table_name = f"{symbol.lower()}_{interval.replace('m', 'm')}"

    save_df_to_db(
        df=df,
        table_name=table_name,
        engine=engine,
        schema=SCHEMA,
        time_column="timestamp",
        is_timeseries=True
    )

    logger.info(f"Data saved to table '{SCHEMA}.{table_name}'")

    # ---------------------------
    # Verification
    # ---------------------------
    df_db = read_df_from_db(engine, table_name, schema=SCHEMA, limit=5)

    if not df_db.empty:
        logger.info(
            f"Verification successful | "
            f"{len(df_db)} rows read from '{table_name}'"
        )
    else:
        logger.warning(f"Verification failed for table '{table_name}'")

    col_count = total_columns(engine, table_name, schema=SCHEMA)
    row_count = total_rows(engine, table_name, schema=SCHEMA)

    logger.info(
        f"Table stats | table={table_name} | "
        f"columns={col_count} | rows={row_count}"
    )

# -------------------------------------------------
# Optional: Drop Schema (USE WITH CAUTION)
# -------------------------------------------------
# drop_schema(engine=engine, schema=SCHEMA)

logger.info("Binance Futures data ingestion pipeline completed successfully.")
