import yaml
from datetime import datetime, timedelta
from TradeX.data.binance.binance_fetcher import BinanceFuturesFetcher

# Load config
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

symbols = config["symbols"]
start_date = config.get("start_date")
end_date = config.get("end_date", "now")

fetcher = BinanceFuturesFetcher()

end_ts = (
    int(datetime.utcnow().timestamp() * 1000)
    if end_date == "now"
    else int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
)

start_ts = (
    int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    if start_date
    else int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000)
)

for sym in symbols:
    symbol = sym.upper() + "USDT"
    print(f"\nProcessing {symbol}...")
    fetcher.fetch_and_save(symbol=symbol, start_ts=start_ts, end_ts=end_ts, interval="1m")
