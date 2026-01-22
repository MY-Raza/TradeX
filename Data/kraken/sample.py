from TradeX.utils.common.config_loader import read_config
from TradeX.data.kraken.kraken_fetcher import KrakenFuturesFetcher  
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import clean_df
import os

logger = get_logger("kraken_main")

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

        fetcher = KrakenFuturesFetcher(symbol=kraken_symbol, interval="1m")
        try:
            df = fetcher.fetch(start_date=start_date, end_date=end_date)
            df = clean_df(df)
            # Save CSV
            filename = f"kraken_{kraken_symbol}_ohlcv_cleaned.csv"
            filepath = os.path.join(os.getcwd(), filename)
            fetcher.save_to_csv(df, filepath)

        except Exception as e:
            logger.exception(f"Failed to fetch {kraken_symbol}: {e}")

if __name__ == "__main__":
    main()
