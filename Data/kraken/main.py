from kraken_fetcher import KrakenFuturesFetcher
from TradeX.utils.data.data_cleaner import clean_df
from TradeX.utils.db.utils import save_df_to_db,get_last_date,drop_schema
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from TradeX.utils.common.logs import get_logger
from datetime import datetime, timezone
logger = get_logger("kraken_main")
def timestamp_to_str(ms: int) -> str:
    """Convert epoch ms to 'YYYY-MM-DD HH:MM:SS' string for KrakenFuturesFetcher."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def main():
    # -------------------------------
    # 1. Load config
    # -------------------------------
    SCHEMA = EXCHANGE_SCHEMA_MAP["kraken"]
    config = read_config()
    symbols = config["symbols"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    
    # -------------------------------
    # 2. Loop through symbols and fetch, clean, save
    # -------------------------------
    for symbol in symbols:
        # Map symbol to Kraken futures pair (example: 'xbt' -> 'BTC/USD')
        kraken_symbol = symbol.upper()
        #read_df_from_db(table_name=f"{symbol}",schema=SCHEMA),
        #drop_schema(SCHEMA)
        if kraken_symbol == "XBT":
            kraken_symbol = "BTC/USD"
        else:
            kraken_symbol = f"{kraken_symbol}/USD"

        logger.info(f"\nFetching data for {kraken_symbol} from {start_date} to {end_date}...")
        
        last_ts = get_last_date(table_name=f"{symbol}_1m", schema=SCHEMA, time_column="timestamp")
        if last_ts:
            # Increment by 1ms to avoid duplicate
            start_date = timestamp_to_str(last_ts + 1)
            logger.info(f"Found existing data for {kraken_symbol}, starting from {start_date}")
        else:
            start_date = start_date
            logger.info(f"No existing data for {kraken_symbol}, using start date from config: {start_date}")

        # -------------------------------
        # Fetch raw data
        # -------------------------------
        fetcher = KrakenFuturesFetcher(
            symbol=kraken_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1m",
            limit=1000,
            sleep_seconds=0.5
        )

        raw_df = fetcher.fetch_data()
        logger.info(f"Raw data fetched | rows: {len(raw_df)}")

        if raw_df.empty:
            logger.info(f"No data for {kraken_symbol}, skipping.")
            continue

        # -------------------------------
        # Clean data
        # -------------------------------
        cleaned_df = clean_df(raw_df, interval="1m")
        logger.info(f"Cleaned data | rows: {len(cleaned_df)}")

        # -------------------------------
        # Save to DB
        # -------------------------------
        save_df_to_db(
            df=cleaned_df,
            table_name=f"{symbol}",
            schema=SCHEMA,
            time_column="timestamp",
            is_timeseries=True
        )
        logger.info(f"Data for {kraken_symbol} saved to DB.")

if __name__ == "__main__":
    main()
