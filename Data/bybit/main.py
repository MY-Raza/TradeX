from TradeX.utils.db.utils import save_df_to_db
from TradeX.logs.logs import get_logger
from bybit_fetcher import BybitFuturesFetcher
from TradeX.utils.data.data_cleaner import clean_df
from TradeX.utils.common.config_loader import read_config
import os
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP

logger = get_logger("bybit_main")

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

SCHEMA = EXCHANGE_SCHEMA_MAP["bybit"]

# ---------------------------
# Load Configuration
# ---------------------------
config = read_config()
symbols = config["symbols"]
start_date = config["start_date"]
end_date = config["end_date"]

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
    # Fetch RAW DATA
    # ---------------------------
    raw_df = fetcher.fetch_data()
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

    # ---------------------------
    # Save to Database
    # ---------------------------

    save_df_to_db(
        df=df,
        table_name=symbol.lower(),
        schema=SCHEMA,
        time_column="timestamp",
        is_timeseries=True,
    )

    logger.info(f"Saved cleaned data to table: {SCHEMA}.{symbol.lower()}_1m")

logger.info("Bybit data ingestion pipeline completed successfully.")
