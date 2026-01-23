import MetaTrader5 as mt5
import pandas as pd
from datetime import timedelta, datetime
from TradeX.utils.common.logs import get_logger

logger = get_logger("metatrader5_fetcher")


class MetaTrader5FutureFetcher:
    """
    A class to fetch historical futures data from MetaTrader5 for given symbols.

    Attributes:
        raw_symbols (list): List of raw symbols from config.
        utc_from (datetime): Start datetime for fetching data.
        utc_to (datetime): End datetime for fetching data (max is now).
        timeframe (int): MT5 timeframe (default is mt5.TIMEFRAME_M1).
        symbols (list): List of resolved MT5 symbols with proper suffixes.
    """

    # Special suffixes for certain symbols that do not follow the default naming
    SPECIAL_SUFFIX = {
        "neth": "25"  # Example: neth -> NETH25
    }

    def __init__(self, symbols, utc_from, utc_to, timeframe=mt5.TIMEFRAME_M1):
        """
        Initialize the fetcher and enable symbols in MT5 terminal.

        Args:
            symbols (list): List of raw symbols.
            utc_from (datetime): Start datetime for fetching.
            utc_to (datetime): End datetime for fetching.
            timeframe (int): MT5 timeframe (default 1-minute).
        """
        self.raw_symbols = symbols
        self.utc_from = utc_from
        self.utc_to = min(utc_to, datetime.now())  # Avoid fetching future data
        self.timeframe = timeframe

        # Resolve raw symbols into MT5-compatible symbols
        self.symbols = [self.resolve_symbol(s) for s in self.raw_symbols]

        # Enable symbols in MT5 terminal
        for sym in self.symbols:
            info = mt5.symbol_info(sym)
            if info is None:
                logger.warning(f"⚠ Symbol not found: {sym}")
            elif mt5.symbol_select(sym, True):
                logger.info(f"✔ Enabled symbol: {sym}")
            else:
                logger.error(f"❌ Failed to enable symbol: {sym}")

    def resolve_symbol(self, symbol: str) -> str:
        """
        Convert a raw symbol to MT5 symbol with proper suffix.

        Args:
            symbol (str): Raw symbol from config.

        Returns:
            str: MT5-compatible symbol string.
        """
        symbol = symbol.lower()
        if symbol in self.SPECIAL_SUFFIX:
            # Use special suffix if defined
            return f"{symbol.upper()}{self.SPECIAL_SUFFIX[symbol]}"
        # Default: append USD as suffix
        return f"{symbol.upper()}USD"

    def fetch(self, symbol: str):
        """
        Fetch historical data for a single symbol in 1-day chunks.

        Args:
            symbol (str): Raw symbol string.

        Returns:
            pd.DataFrame or None: DataFrame with columns
                ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                or None if no data is available.
        """
        mt5_symbol = self.resolve_symbol(symbol)

        # Check if symbol exists on MT5 server
        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            logger.error(f"❌ Symbol {mt5_symbol} not found on MT5 server")
            return None

        # Enable symbol in MT5 terminal
        if not mt5.symbol_select(mt5_symbol, True):
            logger.error(f"❌ Failed to enable symbol {mt5_symbol}")
            return None

        dfs = []  # List to collect daily chunks
        current_from = self.utc_from

        # Fetch in 1-day increments to avoid huge requests
        while current_from < self.utc_to:
            current_to = min(current_from + timedelta(days=1), self.utc_to)

            try:
                # Fetch data from MT5
                rates = mt5.copy_rates_range(mt5_symbol, self.timeframe, current_from, current_to)
            except Exception as e:
                logger.error(f"❌ Error fetching {mt5_symbol} from {current_from} to {current_to}: {e}")
                current_from = current_to
                continue

            # Process fetched data
            if rates is not None and len(rates) > 0:
                df_chunk = pd.DataFrame(rates)
                # Convert MT5 time to Unix timestamp in milliseconds
                df_chunk["timestamp"] = df_chunk["time"].astype("int64") * 1000
                # Rename tick_volume -> volume
                df_chunk = df_chunk.rename(columns={"tick_volume": "volume"})
                # Keep only essential columns
                df_chunk = df_chunk[["timestamp", "open", "high", "low", "close", "volume"]]
                dfs.append(df_chunk)
            else:
                logger.warning(f"⚠ No data for {mt5_symbol} from {current_from} to {current_to}")

            current_from = current_to

        # Concatenate all daily chunks into a single DataFrame
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return None

    def fetch_all(self):
        """
        Fetch historical data for all configured symbols.

        Returns:
            dict: Dictionary mapping MT5 symbols to DataFrames {symbol: DataFrame}.
        """
        all_data = {}
        for s in self.raw_symbols:
            df = self.fetch(s)
            if df is not None:
                all_data[self.resolve_symbol(s)] = df
        return all_data
