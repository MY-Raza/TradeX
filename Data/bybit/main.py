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
main_bybit.py

End-to-end Bybit USDT Perpetual Futures data ingestion pipeline:

1. Load environment variables and config.
2. Initialize database and schema.
3. Fetch RAW klines from Bybit.
4. Clean and process data (shared cleaning_utils).
5. Save cleaned data to PostgreSQL / TimescaleDB.
6. Verify data insertion.

Fetcher  -> RAW data only
Cleaner  -> Shared OHLCV cleaning
Main     -> Orchestration
"""

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

SCHEMA = os.getenv("DB_SCHEMA_BYBIT", "data_bybit")
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Bybit API credentials not found in environment variables.")

logger.info("Environment variables loaded.")

# ---------------------------
# Load Config
# ---------------------------
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

symbols = config.get("symbols", [])
start_date_str = config.get("start_date")
end_date_str = config.get("end_date", "now")

if not symbols or not start_date_str:
    raise ValueError("Config must include 'symbols' and 'start_date'.")

# ---------------------------
# Initialize Database
# ---------------------------
engine = get_engine()
if engine is None:
    raise RuntimeError("Database engine could not be initialized.")

create_schema(engine, schema=SCHEMA)

# ---------------------------
# Fetch, Clean & Store Data
# ---------------------------
for symbol in symbols:
    symbol = symbol.upper()
    logger.info(f"Processing symbol: {symbol}")

    fetcher = BybitFuturesFetcher(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol=f"{symbol}USDT",
        start_date=start_date_str,
        end_date=end_date_str,
        interval="1",  # 1-minute Bybit interval
        demo=False,
    )

    # ---------------------------
    # Fetch RAW data
    # ---------------------------
    raw_df = fetcher.fetch_klines()

    if raw_df.empty:
        logger.warning(f"No data fetched for {symbol}. Skipping.")
        continue

    logger.info(f"RAW data fetched: {len(raw_df)} rows")

    # ---------------------------
    # Cleaning Pipeline (Shared)
    # ---------------------------
    df = clean_klines_df(raw_df)
    logger.info("Basic OHLCV cleaning completed.")

    df = fill_missing_timestamps(df, interval="1m")
    logger.info("Missing timestamps inserted.")

    df = fill_missing_values(df, method="ffill")
    logger.info("Missing values forward-filled.")

    # Optional resampling
    # df = resample_ohlcv(df, interval="5min")

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

    logger.info(f"Saved data to table: {SCHEMA}.{table_name}")

    # ---------------------------
    # Verification
    # ---------------------------
    df_db = read_df_from_db(engine, table_name, schema=SCHEMA, limit=5)

    if not df_db.empty:
        logger.info(f"Verification success | {len(df_db)} rows read.")
    else:
        logger.warning("Verification failed: No rows read.")

    col_count = total_columns(engine, table_name, schema=SCHEMA)
    row_count = total_rows(engine, table_name, schema=SCHEMA)

    logger.info(
        f"Table stats | {table_name} → columns={col_count}, rows={row_count}"
    )

logger.info("Bybit data ingestion pipeline completed successfully.")
