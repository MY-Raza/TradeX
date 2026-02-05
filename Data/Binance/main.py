from TradeX.utils.db.utils import save_df_to_db,get_last_date,ensure_unique_index
from TradeX.utils.common.logs import get_logger
from binance_fetcher import BinanceFuturesFetcher
from TradeX.utils.data.data_cleaner import clean_df
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from datetime import datetime, timezone
logger = get_logger("binance_main")

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

SCHEMA = EXCHANGE_SCHEMA_MAP["binance"]
# -------------------------------------------------
# Load Configuration
# -------------------------------------------------
config = read_config()
symbols = config["symbols"]
start_date = config["start_date"]
end_date = config["end_date"]

# -------------------------------------------------
# Fetch, Clean & Store Data
# -------------------------------------------------
for symbol in symbols:
    logger.info(f"Starting data pipeline for symbol: {symbol}")
    last_stored_date = get_last_date(table_name=f"{symbol}_1m", schema=SCHEMA, time_column="datetime")

    if last_stored_date:
          last_stored_date_dt = datetime.fromtimestamp(last_stored_date / 1000, tz=timezone.utc)
          start_date = last_stored_date_dt.strftime("%Y-%m-%d %H:%M:%S")
          logger.info(f"Found existing data for {symbol}. Setting start_date={start_date}")
    else:
        # Use default start date from config
        start_date = start_date
        logger.info(f"No existing data found for {symbol}. Using start_date={start_date}")

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
    raw_df = fetcher.fetch_data()

    if raw_df.empty:
        logger.warning(f"No data fetched for {symbol}. Skipping to next symbol.")
        continue

    logger.info(f"RAW data fetched for {symbol} | rows={len(raw_df)}")

    # ---------------------------
    # Data Cleaning Pipeline
    # ---------------------------
    df = clean_df(raw_df)

    # Optional: Resample to higher timeframe
    # df = resample_ohlcv(df, interval="5min")
    # logger.info("Data resampled to 5-minute candles.")

    # ---------------------------
    # Save Data to Database
    # ---------------------------

    save_df_to_db(
        df=df,
        table_name=f"{symbol.lower()}_1m",
        schema=SCHEMA,
        time_column="datetime",
        is_timeseries=True
    )

logger.info("Binance Futures data ingestion pipeline completed successfully.")
