import os
from datetime import datetime
import yaml
from dotenv import load_dotenv

from TradeX.utils.db.utils import (
    get_engine, create_schema, save_df_to_db, read_df_from_db,
    total_columns, total_rows, drop_schema
)
from TradeX.logs.logging import get_logger
from bybit_fetcher import BybitFuturesFetcher
from TradeX.utils.cleaning_utils import (
    clean_klines_df, fill_missing_timestamps,
    fill_missing_values, resample_ohlcv
)

logger = get_logger(__name__)

"""
main_bybit.py

End-to-end Bybit USDT Perpetual Futures data ingestion pipeline:

1. Load environment variables and config.
2. Initialize database and schema.
3. Fetch raw klines from Bybit.
4. Clean and process data using multiple cleaning steps.
5. Save cleaned data to PostgreSQL/TimescaleDB.
6. Verify data insertion and log table stats.

Dependencies:
- Bybit API credentials (BYBIT_API_KEY, BYBIT_SECRET_KEY)
- DATABASE_URL environment variable
- config.yml containing symbols and date range
"""

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()
SCHEMA = os.getenv("DB_SCHEMA_BYBIT", "data_bybit")
logger.info("Environment variables loaded.")

# ---------------------------
# Load Config
# ---------------------------
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

symbols = config.get("symbols", [])
start_date_str = config.get("start_date")
end_date_str = config.get("end_date", "now")

# ---------------------------
# Initialize Database
# ---------------------------
engine = get_engine()
if engine is None:
    raise RuntimeError("Database engine could not be initialized.")
create_schema(engine, schema=SCHEMA)

# ---------------------------
# Initialize Bybit Fetcher
# ---------------------------
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_SECRET_KEY")

fetcher = BybitFuturesFetcher(
    api_key=API_KEY,
    api_secret=API_SECRET,
    testnet=False
)

# ---------------------------
# Fetch, Clean & Process Data
# ---------------------------
for symbol in symbols:
    raw_df = fetcher.fetch_klines(
        symbol=f"{symbol.upper()}USDT",
        start_date=start_date_str,
        end_date=end_date_str,
        interval="1"  # Bybit uses "1" instead of "1m"
    )

    if raw_df.empty:
        logger.warning(f"No data fetched for {symbol}. Skipping.")
        continue

    logger.info("Data fetching completed.")

    # ---------------------------
    # Cleaning Pipeline (Shared)
    # ---------------------------
    df = clean_klines_df(raw_df)
    logger.info("Basic data cleaning completed.")

    df = fill_missing_timestamps(df, interval="1m")
    logger.info("Missing timestamps inserted.")

    df = remove_outliers(df)
    logger.info("Outliers removed.")

    df = fill_missing_values(df, method="ffill")
    logger.info("Missing values forward-filled.")

    # Optional resampling
    # df = resample_ohlcv(df, interval="5min")

    # ---------------------------
    # Save cleaned DataFrame to DB
    # ---------------------------
    table_name = f"{symbol.lower()}_1m"
    save_df_to_db(
        df=df,
        table_name=table_name,
        engine=engine,
        schema=SCHEMA,
        time_column="timestamp",
        is_timeseries=True
    )

    # ---------------------------
    # Verification
    # ---------------------------
    df_db = read_df_from_db(engine, table_name, schema=SCHEMA, limit=5)
    if not df_db.empty:
        logger.info(f"Verification success | {len(df_db)} rows read from '{table_name}'.")
    else:
        logger.warning(f"No data found in database for '{symbol}'.")

    col_count = total_columns(engine, table_name, schema=SCHEMA)
    row_count = total_rows(engine, table_name, schema=SCHEMA)
    logger.info(f"Table '{table_name}' has {col_count} columns and {row_count} rows.")

logger.info("Bybit data ingestion pipeline completed successfully.")
