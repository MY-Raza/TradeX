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
)
from TradeX.logs.logging import get_logger
from bybit_fetcher import BybitFuturesFetcher
from TradeX.utils.cleaning_utils import (
    clean_klines_df,
    fill_missing_timestamps,
    fill_missing_values,
    resample_ohlcv,
)

logger = get_logger(__name__)

"""
main.py

End-to-end Bybit USDT Perpetual Futures data ingestion pipeline.

Pipeline steps:

1. Load environment variables (API keys, DB schema).
2. Load config.yml for symbols and date range.
3. Initialize database engine and schema.
4. Fetch RAW klines from Bybit using BybitFuturesFetcher.
5. Clean and process data using shared OHLCV utilities.
6. Save cleaned data to PostgreSQL / TimescaleDB.
7. Verify data insertion and log table statistics.

Note:
- The fetcher returns RAW data only; cleaning and processing is handled separately.
- Optional resampling can be applied if required.
"""

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

SCHEMA = os.getenv("DB_SCHEMA_BYBIT", "data_bybit")  # Default schema

# ---------------------------
# Load Configuration
# ---------------------------
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

symbols = config.get("symbols", [])
start_date_str = config.get("start_date")
end_date_str = config.get("end_date", "now")

if not symbols or not start_date_str:
    raise ValueError("Config must include 'symbols' and 'start_date'.")

logger.info(f"Configuration loaded | symbols={symbols}, start_date={start_date_str}, end_date={end_date_str}")

# ---------------------------
# Initialize Database
# ---------------------------
engine = get_engine()
if engine is None:
    raise RuntimeError("Database engine could not be initialized.")

# Create schema if it doesn't exist
create_schema(engine, schema=SCHEMA)
logger.info(f"Database schema ready: {SCHEMA}")

# ---------------------------
# Fetch, Clean & Store Data
# ---------------------------
for symbol in symbols:
    symbol = symbol.upper()
    logger.info(f"Processing symbol: {symbol}")

    # Initialize Bybit fetcher
    fetcher = BybitFuturesFetcher(
        symbol=f"{symbol}USDT",
        start_date=start_date_str,
        end_date=end_date_str,
        interval="1",  # 1-minute interval
    )

    # ---------------------------
    # Fetch RAW klines
    # ---------------------------
    raw_df = fetcher.fetch_klines()

    if raw_df.empty:
        logger.warning(f"No data fetched for {symbol}. Skipping.")
        continue

    logger.info(f"RAW data fetched for {symbol}: {len(raw_df)} rows")

    # ---------------------------
    # Cleaning & Processing Pipeline
    # ---------------------------
    df = clean_klines_df(raw_df)
    logger.info("Basic OHLCV cleaning completed.")

    df = fill_missing_timestamps(df, interval="1m")
    logger.info("Missing timestamps inserted.")

    df = fill_missing_values(df, method="ffill")
    logger.info("Missing values forward-filled.")

    # Optional: Resample to higher timeframe
    # df = resample_ohlcv(df, interval="5min")
    # logger.info("Resampling completed.")

    if df.empty:
        logger.warning(f"Cleaned DataFrame empty for {symbol}. Skipping DB save.")
        continue

    # ---------------------------
    # Save to Database
    # ---------------------------
    table_name = f"{symbol.lower()}_1m"

    save_df_to_db(
        df=df,
        table_name=table_name,
        engine=engine,
        schema=SCHEMA,
        time_column="timestamp",
        is_timeseries=True,
    )

    logger.info(f"Saved cleaned data to table: {SCHEMA}.{table_name}")

    # ---------------------------
    # Verification
    # ---------------------------
    df_db = read_df_from_db(engine, table_name, schema=SCHEMA, limit=5)

    if not df_db.empty:
        logger.info(f"Verification success | {len(df_db)} rows read from DB.")
    else:
        logger.warning("Verification failed: No rows read from DB.")

    # Log table statistics
    col_count = total_columns(engine, table_name, schema=SCHEMA)
    row_count = total_rows(engine, table_name, schema=SCHEMA)

    logger.info(f"Table stats | {table_name} → columns={col_count}, rows={row_count}")

logger.info("Bybit data ingestion pipeline completed successfully.")
