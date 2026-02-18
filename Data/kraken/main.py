# kraken_main.py

from TradeX.utils.common.config_loader import read_config
from TradeX.data.kraken.kraken_fetcher import KrakenFuturesFetcher  
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import clean_df
from TradeX.utils.db.utils import save_df_to_db, drop_schema, get_last_date,read_df_from_db
from TradeX.utils.common.constants import EXCHANGE_SCHEMA_MAP
import pandas as pd
import os

# ---------------------------
# Initialize logger
# ---------------------------
logger = get_logger("kraken_main")

# Retrieve the schema for Kraken from constants
SCHEMA = EXCHANGE_SCHEMA_MAP["kraken"]


def main():
    """
    Main function to fetch Kraken Futures OHLCV data and save it to the database.

    Steps:
        1. Load configuration from 'config.yml'.
        2. Validate exchange is 'kraken'.
        3. Loop through each symbol in the config.
        4. Check the last stored timestamp in the database.
        5. Fetch data from Kraken Futures starting from last timestamp (or config start date).
        6. Clean the data using clean_df.
        7. Save the cleaned data into the database.
        8. Handle any exceptions during fetching or saving.
    """
    # -----------------------------
    # Load config
    # -----------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kraken_config_path = os.path.join(current_dir,"config.yml")
    config = read_config(kraken_config_path)
    exchange_name = config.get("exchange_name")
    symbols = config.get("symbols", [])
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    drop_schema(SCHEMA)
    # -----------------------------
    # Validate exchange
    # -----------------------------
    if exchange_name.lower() != "kraken":
        logger.error("This script only supports Kraken exchange")
        return
    # -----------------------------
    # Loop through all symbols
    # -----------------------------
    for symbol in symbols:
        # Kraken uses PF_ prefix for futures, and uppercase symbols
        kraken_symbol = f"PF_{symbol.upper()}USD"

        logger.info(f"Fetching data for {kraken_symbol} from {start_date} to {end_date}")

        # Check last stored timestamp in DB
        last_stored_date = get_last_date(
        table_name=f"{symbol.lower()}_1m",
        schema=SCHEMA,
        time_column="datetime"
    )

    if last_stored_date:
        # last_stored_date is already pd.Timestamp (UTC)
        start_date = (last_stored_date + pd.Timedelta(milliseconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Found existing data for {symbol}. Setting start_date={start_date}")
    else:
        # Use default start date from config
        start_date = start_date
        logger.info(f"No existing data found for {symbol}. Using start_date={start_date}")
            
        # Initialize Kraken fetcher for this symbol
        fetcher = KrakenFuturesFetcher(symbol=kraken_symbol, interval="1m")

        try:
            # Fetch OHLCV data
            df = fetcher.fetch_data(start_date=start_date, end_date=end_date)

            # Clean the data: remove duplicates, sort by timestamp, etc.
            df = clean_df(df)
            # Save to database
            save_df_to_db(
                df=df,
                table_name=f"{symbol.lower()}_1m",
                schema=SCHEMA,
                time_column="datetime",
                is_timeseries=True
            )
            logger.info(f"Successfully saved data for {kraken_symbol}. Rows: {len(df)}")

        except Exception as e:
            # Catch and log any exceptions
            logger.exception(f"Failed to fetch or save {kraken_symbol}: {e}")


# Entry point for script execution
if __name__ == "__main__":
    main()
