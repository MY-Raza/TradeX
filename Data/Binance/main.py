import yaml
from datetime import datetime, timedelta
from TradeX.Data.binance.binance_fetcher import BinanceFuturesFetcher

# Load configuration
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

symbols = config["symbols"]
start_date = config["start_date"]
end_date = config.get("end_date", "now")  # Can be "now" or a date string

# Initialize the fetcher
fetcher = BinanceFuturesFetcher()

# Convert end_date to timestamp in milliseconds
end_ts = int(datetime.utcnow().timestamp() * 1000) if end_date == "now" else int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

# Fetch previous 7 days from end_ts
start_ts = int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000)

# Fetch data for each symbol
for sym in symbols:
    symbol = sym.upper() + "USDT"
    output_file = f"data/futures/{symbol}_1m_last_7_days.csv"

    # Using the class-based method with interval as 1 minute
    fetcher.fetch_futures_data(
        symbol=symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        output_path=output_file,
        interval="1m"
    )
