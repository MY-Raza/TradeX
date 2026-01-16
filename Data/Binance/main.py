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
    resample_ohlcv
)

logger = get_logger(__name__)

"""
main.py

End-to-end Binance Futures data ingestion pipeline.

Pipeline Steps:
1. Load environment variables (API keys, DB schema).
2. Load configuration from 'config.yml'.
3. Initialize database engine and schema.
4. Initialize Binance Futures fetcher for each symbol.
5. Fetch RAW OHLCV klines data.
6. Clean and process OHLCV data.
7. Save cleaned data to PostgreSQL / TimescaleDB.
8. Verify data insertion and log table statistics.

Notes:
- The fetcher returns RAW data; cleaning and preprocessing is handled separately.
- Optional resampling can be applied to higher timeframes.
- Schema drop is optional and should be used with caution.
"""

# -------------------------------------------------
# Load Environment Variables
# -------------------------------------------------
load_dotenv()

SCHEMA = os.getenv("DB_SCHEMA_BINANCE", "data_binance")  # Default schema
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    raise RuntimeError("Binance API credentials not found in environment variables.")

logger.info("Environment variables loaded successfully.")

# -------------------------------------------------
# Load Configuration
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
    f"Configuration loaded | symbols={symbols} | start={start_date} | end={end_date} | interval={interval}"
)

# -------------------------------------------------
# Initialize Database
# -------------------------------------------------
engine = get_engine()
if engine is None:
    raise RuntimeError("Database engine could not be initialized.")

# Create schema if it does not exist
create_schema(engine, schema=SCHEMA)
logger.info(f"Database schema '{SCHEMA}' is ready.")

# -------------------------------------------------
# Fetch, Clean & Store Data
# -------------------------------------------------
for symbol in symbols:
    logger.info(f"Starting data pipeline for symbol: {symbol}")

    # Initialize Binance fetcher
    fetcher = BinanceFuturesFetcher(
        symbol=f"{symbol.upper()}USDT",
        start_date=start_date,
        end_date=end_date,
        interval=interval
    )

    # ---------------------------
    # Fetch RAW OHLCV Data
    # ---------------------------
    raw_df = fetcher.fetch_klines()

    if raw_df.empty:
        logger.warning(f"No data fetched for {symbol}. Skipping to next symbol.")
        continue

    logger.info(f"RAW data fetched for {symbol} | rows={len(raw_df)}")

    # ---------------------------
    # Data Cleaning Pipeline
    # ---------------------------
    df = clean_klines_df(raw_df)
    logger.info("OHLCV cleaning completed.")

    # Optional: Resample to higher timeframe
    # df = resample_ohlcv(df, interval="5min")
    # logger.info("Data resampled to 5-minute candles.")

    if df.empty:
        logger.warning(f"Cleaned DataFrame empty for {symbol}. Skipping database save.")
        continue

    # ---------------------------
    # Save Data to Database
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

    logger.info(f"Cleaned data saved to table '{SCHEMA}.{table_name}'")

    # ---------------------------
    # Verification
    # ---------------------------
    df_db = read_df_from_db(engine, table_name, schema=SCHEMA, limit=5)

    if not df_db.empty:
        logger.info(
            f"Verification successful | {len(df_db)} rows read from '{table_name}'"
        )
    else:
        logger.warning(f"Verification failed for table '{table_name}'")

    # Log table statistics
    col_count = total_columns(engine, table_name, schema=SCHEMA)
    row_count = total_rows(engine, table_name, schema=SCHEMA)

    logger.info(
        f"Table stats | table={table_name} | columns={col_count} | rows={row_count}"
    )

# -------------------------------------------------
# Optional: Drop Schema (Use With Caution)
# -------------------------------------------------
# drop_schema(engine=engine, schema=SCHEMA)

logger.info("Binance Futures data ingestion pipeline completed successfully.")
