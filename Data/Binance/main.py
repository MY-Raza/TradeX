import yaml
from datetime import datetime, timedelta
from BTC_Data_Fetcher import fetch_1m_data, fetch_1m_futures_data 

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

symbols = config["symbols"]
start_date = config["start_date"]
end_date = config["end_date"]

end_ts = int(datetime.utcnow().timestamp() * 1000)

if start_date == "last_7_days":
    start_ts = int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000)
else:
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)

# Fetch data
for sym in symbols:
    #symbol = sym.upper() + "USDT"
    #output_file = f"data/{symbol}_1m.csv"

    #fetch_1m_data(
    #    symbol=symbol,
    #    start_ts=start_ts,
    #    end_ts=end_ts,
    #    output_path=output_file
    #)
    symbol = sym.upper() + "USDT"
    output_file = f"data/futures/{symbol}_1m_last_7_days.csv"

    fetch_1m_futures_data(
        symbol=symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        output_path=output_file
    )
