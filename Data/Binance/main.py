import os
from dotenv import load_dotenv
from datetime import datetime
import yaml
import pandas as pd

from TradeX.utils.db.db_utils import (
    get_engine,
    create_schema,
    read_df_from_db,
    total_columns,
    total_rows
)
from binance_fetcher import BinanceFuturesFetcher

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

# ---------------------------
# Load Configuration
# ---------------------------
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

exchange_name = config.get("exchange_name", "binance")
symbols = config.get("symbols", [])
start_date_str = config.get("start_date")
end_date_str = config.get("end_date", "now")

# ---------------------------
# Convert Dates to Timestamps (ms)
# ---------------------------
start_ts = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)

if end_date_str == "now":
    end_ts = int(datetime.utcnow().timestamp() * 1000)
else:
    end_ts = int(datetime.strptime(end_date_str, "%Y-%m-%d").timestamp() * 1000)

# ---------------------------
# Initialize Database
# ---------------------------
engine = get_engine()
if engine is None:
    raise RuntimeError("Database engine could not be initialized.")

# Ensure schema exists
create_schema(engine, schema="data_binance")

# ---------------------------
# Initialize Binance Fetcher
# ---------------------------
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET_KEY")
if not API_KEY or not API_SECRET:
    raise RuntimeError("API_KEY or API_SECRET_KEY not found in environment variables.")

fetcher = BinanceFuturesFetcher(API_KEY, API_SECRET, engine, schema="data_binance")

# ---------------------------
# Fetch and Save Data
# ---------------------------
for symbol in symbols:
    symbol_pair = symbol.upper() + "USDT"  # Append USDT for futures
    print(f"Fetching data for {symbol_pair}...")
    fetcher.fetch_and_save(symbol_pair, start_ts, end_ts, interval="1m")
    table_name = symbol + '_1m' 


    # ---------------------------
    # Read back data from DB for verification
    # ---------------------------
    df_db = read_df_from_db(engine, table_name=table_name.lower(), schema="data_binance", limit=5)
    if not df_db.empty:
        print(f"Preview of last 5 rows for '{symbol_pair}':")
        print(df_db.tail())
    else:
        print(f"No data found in DB for '{symbol_pair}'.")

    number_of_columns =  total_columns(engine, table_name=table_name.lower(), schema="data_binance")
    if number_of_columns:
        print(f"Total Columns are '{number_of_columns}'")
    else:
        print(f'Unexpected Error')
    
    number_of_rows = total_rows(engine, table_name=table_name.lower(), schema="data_binance")
    if number_of_rows:
        print(f"Total Rows are '{number_of_rows}'")
    else:
        print(f'Unexpected Error')
          

