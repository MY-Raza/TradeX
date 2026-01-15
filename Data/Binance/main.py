import os
from datetime import datetime
import yaml
from dotenv import load_dotenv

from TradeX.utils.db.utils import get_engine, create_schema, save_df_to_db, read_df_from_db, total_columns, total_rows, drop_schema
from TradeX.logs.logging import get_logger
from binance_fetcher import BinanceFuturesFetcher
from TradeX.utils.cleaning_utils import clean_klines_df

logger = get_logger(__name__)

"""
main.py

This script runs the end-to-end Binance Futures data ingestion pipeline:

1. Load environment variables and configuration.
2. Convert dates into timestamps.
3. Initialize the database and schema.
4. Fetch raw klines from Binance using BinanceFuturesFetcher.
5. Clean the data using cleaning_utils.
6. Save the cleaned data into PostgreSQL/TimescaleDB.
7. Verify data insertion and log table stats.

Dependencies:
- Binance API credentials (BINANCE_API_KEY, BINANCE_SECRET_KEY)
- DATABASE_URL environment variable
- config.yml containing symbols and date range
"""

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()  # Load variables from .env file
SCHEMA = os.getenv("DB_SCHEMA", "data_binance")  # Default schema if not in .env
logger.info("Environment variables loaded.")

# ---------------------------
# Load Config
# ---------------------------
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

symbols = config.get("symbols", [])          # List of trading symbols
start_date_str = config.get("start_date")   # Start date in YYYY-MM-DD
end_date_str = config.get("end_date", "now")  # End date or "now"



# ---------------------------
# Initialize Database
# ---------------------------
engine = get_engine()  # Create SQLAlchemy engine
if engine is None:
    raise RuntimeError("Database engine could not be initialized.")
create_schema(engine, schema=SCHEMA)  # Ensure schema exists

# ---------------------------
# Initialize Binance Fetcher
# ---------------------------
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")
fetcher = BinanceFuturesFetcher(api_key=API_KEY, api_secret=API_SECRET)

# ---------------------------
# Fetch, Clean & Save Data
# ---------------------------
for symbol in symbols:
    # Fetch raw Binance klines for the symbol
    raw_df = fetcher.fetch_klines(symbol=f"{symbol.upper()}USDT", start_date=start_date_str, end_date=end_date_str)
    if raw_df.empty:
        logger.warning(f"No data fetched for {symbol}. Skipping.")
        continue

    # Clean the raw data using cleaning_utils
    logger.info(f"Data Fetching Completed")
    df = clean_klines_df(raw_df)

    # Save cleaned DataFrame to DB
    logger.info(f"Data Cleaning Completed")
    table_name = f"{symbol.lower()}_1m"
    save_df_to_db(
        df=df,
        table_name=table_name,
        engine=engine,
        schema=SCHEMA,
        time_column="timestamp",
        is_timeseries=True  # Converts table to TimescaleDB hypertable
    )

    # ---------------------------
    # Verification
    # ---------------------------
    # Read back first 5 rows to ensure data saved
    df_db = read_df_from_db(engine, table_name, schema=SCHEMA, limit=5)
    if not df_db.empty:
        logger.info(f"Verification success | {len(df_db)} rows read from '{table_name}'.")
    else:
        logger.warning(f"No data found in database for '{symbol}'.")

    # Log table statistics
    col_count = total_columns(engine, table_name, schema=SCHEMA)
    row_count = total_rows(engine, table_name, schema=SCHEMA)
    logger.info(f"Table '{table_name}' has {col_count} columns and {row_count} rows.")

drop_schema(engine=engine,schema=SCHEMA)

logger.info("Data ingestion pipeline completed successfully.")
