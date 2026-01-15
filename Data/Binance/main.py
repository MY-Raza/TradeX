import os
from datetime import datetime

import yaml
from dotenv import load_dotenv

from TradeX.utils.db.utils import (
    get_engine,
    create_schema,
    read_df_from_db,
    total_columns,
    total_rows,
    drop_schema,
    drop_table
)
from TradeX.logs.logging import get_logger
from binance_fetcher import BinanceFuturesFetcher

logger = get_logger(__name__)

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()
logger.info("Environment variables loaded.")
SCHEMA = os.getenv("DB_SCHEMA", "data_binance")
# ---------------------------
# Load Configuration
# ---------------------------
try:
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    exchange_name = config.get("exchange_name", "binance")
    symbols = config.get("symbols", [])
    start_date_str = config.get("start_date")
    end_date_str = config.get("end_date", "now")

    logger.info("Configuration loaded successfully.")

except Exception:
    logger.exception("Failed to load configuration file.")
    raise

# ---------------------------
# Convert Dates to Timestamps (ms)
# ---------------------------
try:
    start_ts = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)

    if end_date_str == "now":
        end_ts = int(datetime.utcnow().timestamp() * 1000)
    else:
        end_ts = int(datetime.strptime(end_date_str, "%Y-%m-%d").timestamp() * 1000)

    logger.info(f"Date range resolved | start={start_date_str} | end={end_date_str}")

except Exception:
    logger.exception("Failed to parse date configuration.")
    raise

# ---------------------------
# Initialize Database
# ---------------------------
engine = get_engine()
if engine is None:
    logger.critical("Database engine initialization failed.")
    raise RuntimeError("Database engine could not be initialized.")

logger.info("Database engine initialized.")

# ---------------------------
# Resolve Schema (prompt user once)
# ---------------------------
logger.info(f"Using schema: '{SCHEMA}'")

# Ensure schema exists
create_schema(engine, schema=SCHEMA)
logger.info("Schema ensured.")

# ---------------------------
# Initialize Binance Fetcher
# ---------------------------
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")

if not API_KEY or not API_SECRET:
    logger.critical("Binance API credentials missing.")
    raise RuntimeError("BINANCE_API_KEY or BINANCE_SECRET_KEY not found.")

fetcher = BinanceFuturesFetcher(
    api_key=API_KEY,
    api_secret=API_SECRET,
    engine=engine,
    schema=SCHEMA   
)

logger.info("Binance Futures Fetcher initialized.")

# ---------------------------
# Fetch, Store & Verify Data
# ---------------------------
for symbol in symbols:
    try:
        symbol_pair = f"{symbol.upper()}USDT"
        logger.info(f"Starting fetch cycle for {symbol_pair}.")

        fetcher.fetch_and_save(
            symbol=symbol_pair,
            start_ts=start_ts,
            end_ts=end_ts,
            interval="1m"
        )

        table_name = f"{symbol.lower()}_1m"

        # ---------------------------
        # Read back data (verification)
        # ---------------------------
        df_db = read_df_from_db(
            engine=engine,
            table_name=table_name,
            schema=SCHEMA,   # pass schema
            limit=5
        )

        if not df_db.empty:
            logger.info(
                f"Verification success | {len(df_db)} rows read from '{table_name}'."
            )
        else:
            logger.warning(f"No data found in database for '{symbol_pair}'.")

        # ---------------------------
        # Column Count
        # ---------------------------
        col_count = total_columns(
            engine=engine,
            table_name=table_name,
            schema=SCHEMA   # pass schema
        )
        logger.info(f"Table '{table_name}' has {col_count} columns.")

        # ---------------------------
        # Row Count
        # ---------------------------
        row_count = total_rows(
            engine=engine,
            table_name=table_name,
            schema=SCHEMA  # pass schema
        )
        logger.info(f"Table '{table_name}' has {row_count} rows.")

    except Exception:
        logger.exception(f"Unexpected error during processing of symbol '{symbol}'.")

    
#for symbol in symbols:
 #   table_name = f"{symbol.lower()}_1m"
  #  drop_table(engine=engine,table_name=table_name,schema=SCHEMA)

drop_schema(engine=engine,schema=SCHEMA)

logger.info("Data ingestion pipeline completed successfully.")
