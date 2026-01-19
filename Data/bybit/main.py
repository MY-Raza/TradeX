from TradeX.utils.db.utils import (
    create_schema,
    save_df_to_db,
)
from TradeX.utils.common.logs import get_logger
from bybit_fetcher import BybitFuturesFetcher
from TradeX.utils.data.data_cleaner import (
    clean_df,
)
from TradeX.utils.common.utils_common import load_config

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

SCHEMA = "data_bybit"

# ---------------------------
# Load Configuration
# ---------------------------
config = load_config("config.yml")
symbols = config["symbols"]
start_date = config["start_date"]
end_date = config["end_date"]

# ---------------------------
# Initialize Database
# ---------------------------

# Create schema if it doesn't exist
create_schema(schema=SCHEMA)
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
        start_date=start_date,
        end_date=end_date,
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
    df = clean_df(raw_df)
    logger.info("OHLCV cleaning completed.")

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
        schema=SCHEMA,
        time_column="timestamp",
        is_timeseries=True,
    )

    logger.info(f"Saved cleaned data to table: {SCHEMA}.{table_name}")

logger.info("Bybit data ingestion pipeline completed successfully.")
