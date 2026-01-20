from kraken_fetcher import KrakenFuturesFetcher
from TradeX.utils.data.data_cleaner import clean_df
from TradeX.utils.db.utils import save_df_to_db
from TradeX.utils.common.config_loader import read_config
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
from TradeX.utils.common.logs import get_logger
logger = get_logger("kraken_main")
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
        if kraken_symbol == "XBT":
            kraken_symbol = "BTC/USD"
        else:
            kraken_symbol = f"{kraken_symbol}/USD"

        logger.info(f"\nFetching data for {kraken_symbol} from {start_date} to {end_date}...")

        # -------------------------------
        # Fetch raw data
        # -------------------------------
        fetcher = KrakenFuturesFetcher(
            symbol=kraken_symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1m",
            limit=100,
            sleep_seconds=0.5
        )

        raw_df = fetcher.fetch_data()
        logger.info(f"Raw data fetched | rows: {len(raw_df)}")

        if raw_df.empty:
            print(f"No data for {kraken_symbol}, skipping.")
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
            schema="data_kraken",
            time_column="timestamp",
            is_timeseries=True
        )
        logger.info(f"Data for {kraken_symbol} saved to DB.")

if __name__ == "__main__":
    main()
