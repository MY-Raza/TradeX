# main.py (Kraken Futures)
from TradeX.utils.db.utils import save_df_to_db, get_last_date
from TradeX.utils.common.logs import get_logger
from TradeX.data.kraken.kraken_fetcher import KrakenFuturesFetcher
from TradeX.utils.data.data_cleaner import clean_df
from TradeX.utils.common.config_loader import read_config
from datetime import datetime, timezone
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP

logger = get_logger("kraken_main")

"""
main.py

End-to-end Kraken Futures data ingestion pipeline.

Pipeline steps:

1. Load environment variables (API keys, DB schema).
2. Load config.yml for symbols, tick type, and date range.
3. Initialize database engine and schema.
4. Fetch RAW OHLCV candles from Kraken using KrakenFuturesFetcher.
5. Clean and process data using shared OHLCV utilities.
6. Save cleaned data to PostgreSQL / TimescaleDB.
7. Verify data insertion and log table statistics.
"""

SCHEMA = EXCHANGE_SCHEMA_MAP["kraken"]
# ---------------------------
# Load Configuration
# ---------------------------
config = read_config()
symbols = config["symbols"]           
start_date_config = config["start_date"]
end_date_config = config["end_date"]

# ---------------------------
# Fetch, Clean & Store Data
# ---------------------------
for symbol in symbols:
    logger.info(f"Processing symbol: {symbol}")

    # Check last stored timestamp
    last_stored_date = get_last_date(
        table_name=f"{symbol}_1m",
        schema=SCHEMA,
        time_column="timestamp"
    )

    if last_stored_date:
        # Convert from timestamp to datetime
        last_stored_date_dt = datetime.fromtimestamp(
            last_stored_date, tz=timezone.utc
        )
        start_date = last_stored_date_dt.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Found existing data for {symbol}. Setting start_date={start_date}")
    else:
        start_date = start_date_config
        logger.info(f"No existing data found for {symbol}. Using start_date={start_date}")

    # Initialize Kraken futures fetcher
    fetcher = KrakenFuturesFetcher(
        symbol=f"PI_{symbol.upper()}USD",
        start_date=start_date,
        end_date=end_date_config,
        tick_type="trade",
        resolution="1m"
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

    # Optional: Resample to higher timeframe
    # df = resample_ohlcv(df, interval="5min")
    # logger.info(f"Resampling completed for {symbol}.")

    # ---------------------------
    # Save to Database
    # ---------------------------

    save_df_to_db(
        df=df,
        table_name=symbol,
        schema=SCHEMA,
        time_column="timestamp",
        is_timeseries=True
    )

    logger.info(f"Saved cleaned data to table: {SCHEMA}.{symbol.lower()}_1m")

logger.info("Kraken futures data ingestion pipeline completed successfully.")
