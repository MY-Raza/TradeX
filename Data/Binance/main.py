from TradeX.utils.db.utils import (
    get_engine,
    create_schema,
    save_df_to_db,
)
from TradeX.utils.common.logs import get_logger
from binance_fetcher import BinanceFuturesFetcher
from TradeX.utils.data.data_cleaner import (
    clean_df,
)
from TradeX.utils.common.utils_common import load_config

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

SCHEMA = "data_binance"

# -------------------------------------------------
# Load Configuration
# -------------------------------------------------
config = load_config("config.yml")
symbols = config["symbols"]
start_date = config["start_date"]
end_date = config["end_date"]

# Create schema if it does not exist
create_schema(schema=SCHEMA)
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
        interval="1m"
    )

    # ---------------------------
    # Fetch RAW OHLCV Data
    # ---------------------------
    raw_df = fetcher.fetch_klines()

    if raw_df.empty:
        logger.warning(f"No data fetched for {symbol}. Skipping to next symbol.")
        continue

    logger.info(f"RAW data fetched for {symbol} | rows={len(raw_df)}")
    logger.info(f"Raw DF columns: {raw_df.columns.tolist()}")

    # ---------------------------
    # Data Cleaning Pipeline
    # ---------------------------
    df = clean_df(raw_df)
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
    table_name = f"{symbol.lower()}_1m"

    save_df_to_db(
        df=df,
        table_name=table_name,
        schema=SCHEMA,
        time_column="timestamp",
        is_timeseries=True
    )

    logger.info(f"Cleaned data saved to table '{SCHEMA}.{table_name}'")

logger.info("Binance Futures data ingestion pipeline completed successfully.")
