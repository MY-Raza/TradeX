from TradeX.utils.common.config_loader import read_config
from TradeX.data.kraken.kraken_fetcher import KrakenFuturesFetcher  
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import clean_df
from TradeX.utils.db.utils import save_df_to_db,drop_schema,get_last_date
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from datetime import datetime, timezone

logger = get_logger("kraken_main")

SCHEMA = EXCHANGE_SCHEMA_MAP["kraken"]

def main():
    # -----------------------------
    # Load config
    # -----------------------------
    config = read_config("config.yml")
    exchange_name = config["exchange_name"]
    symbols = config["symbols"]
    start_date = config["start_date"]
    end_date = config["end_date"]

    # -----------------------------
    # Validate exchange
    # -----------------------------
    if exchange_name != "kraken":
        logger.error("This script only supports Kraken exchange")
        return

    # -----------------------------
    # Loop through symbols
    # -----------------------------
    for symbol in symbols:
        # Kraken uses PF_ prefix for futures, and uppercase
        kraken_symbol = f"PF_{symbol.upper()}USD"

        logger.info(f"Fetching data for {kraken_symbol} from {start_date} to {end_date}")
        last_stored_date = get_last_date(table_name=f"{symbol}_1m", schema=SCHEMA, time_column="timestamp")
        if last_stored_date:
            # Start from the last timestamp in the database
           last_stored_date_dt = datetime.fromtimestamp(last_stored_date / 1000, tz=timezone.utc)
           start_date = last_stored_date_dt.strftime("%Y-%m-%d %H:%M:%S")
           logger.info(f"Found existing data for {symbol}. Setting start_date={start_date}")

        else:
            start_date = start_date
            logger.info(f"No existing data found for {symbol}. Using start_date={start_date}")
            
        fetcher = KrakenFuturesFetcher(symbol=kraken_symbol, interval="1m")
        try:
            df = fetcher.fetch(start_date=start_date, end_date=end_date)
            df = clean_df(df)
            save_df_to_db(
                df=df,
                table_name=symbol.lower(),
                schema=SCHEMA,
                time_column="timestamp",
                is_timeseries= True)

        except Exception as e:
            logger.exception(f"Failed to fetch {kraken_symbol}: {e}")

if __name__ == "__main__":
    main()
