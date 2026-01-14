import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import yaml

from TradeX.utils.db.db_utils import get_engine, create_schema  # functional DB utils
from TradeX.utils.db.db_utils import save_df_to_db
from binance_fetcher import BinanceFuturesFetcher

# Load environment variables
load_dotenv()

# Load configuration from config.yml
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

exchange_name = config.get("exchange_name", "binance")
symbols = config.get("symbols", [])
start_date_str = config.get("start_date")
end_date_str = config.get("end_date", "now")

# Convert dates to timestamps in milliseconds
start_ts = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)

if end_date_str == "now":
    end_ts = int(datetime.utcnow().timestamp() * 1000)
else:
    end_ts = int(datetime.strptime(end_date_str, "%Y-%m-%d").timestamp() * 1000)

# Initialize database engine and schema
engine = get_engine()
create_schema(engine, schema="data_binance")

# Initialize Binance fetcher
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET_KEY")
fetcher = BinanceFuturesFetcher(API_KEY, API_SECRET, engine, schema="data_binance")

# Fetch and save data for each symbol
for symbol in symbols:
    symbol_pair = symbol.upper() + "USDT"  # Append USDT to symbol for futures
    fetcher.fetch_and_save(symbol_pair, start_ts, end_ts, interval="1m")
