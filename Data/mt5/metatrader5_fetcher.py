import MetaTrader5 as mt5
import pandas as pd
from datetime import timedelta, datetime

class MetaTrader5FutureFetcher:
    SPECIAL_SUFFIX = {
        "neth": "25"  # Example special suffix
    }

    def __init__(self, symbols, utc_from, utc_to, timeframe=mt5.TIMEFRAME_M1):
        """
        symbols: list of raw symbols from config
        utc_from, utc_to: datetime objects
        timeframe: MT5 timeframe (e.g., mt5.TIMEFRAME_M1)
        """
        self.raw_symbols = symbols
        self.utc_from = utc_from
        self.utc_to = min(utc_to, datetime.now())  # prevent future dates
        self.timeframe = timeframe

        # Resolve symbols
        self.symbols = [self.resolve_symbol(s) for s in self.raw_symbols]

        # Enable symbols
        for sym in self.symbols:
            info = mt5.symbol_info(sym)
            if info is None:
                print(f"⚠ Symbol not found: {sym}")
            elif mt5.symbol_select(sym, True):
                print(f"✔ Enabled symbol: {sym}")
            else:
                print(f"❌ Failed to enable symbol: {sym}")

    def resolve_symbol(self, symbol: str) -> str:
        """
        Convert raw symbol to MT5 symbol with proper suffix
        """
        symbol = symbol.lower()
        if symbol in self.SPECIAL_SUFFIX:
            return f"{symbol.upper()}{self.SPECIAL_SUFFIX[symbol]}"
        return f"{symbol.upper()}USD"

    def fetch(self, symbol: str):
        """
        Fetch historical data for a single symbol in 1-day chunks.
        Returns a DataFrame with columns:
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        mt5_symbol = self.resolve_symbol(symbol)

        # Check symbol info
        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            print(f"❌ Symbol {mt5_symbol} not found on MT5 server")
            return None

        # Enable symbol
        if not mt5.symbol_select(mt5_symbol, True):
            print(f"❌ Failed to enable symbol {mt5_symbol}")
            return None

        dfs = []
        current_from = self.utc_from

        while current_from < self.utc_to:
            current_to = min(current_from + timedelta(days=1), self.utc_to)

            try:
                rates = mt5.copy_rates_range(mt5_symbol, self.timeframe, current_from, current_to)
            except Exception as e:
                print(f"❌ Error fetching {mt5_symbol} from {current_from} to {current_to}: {e}")
                current_from = current_to
                continue

            if rates is not None and len(rates) > 0:
                df_chunk = pd.DataFrame(rates)
                # Convert time -> Unix timestamp in ms
                df_chunk["timestamp"] = df_chunk["time"].astype("int64") * 1000
                # Rename tick_volume -> volume
                df_chunk = df_chunk.rename(columns={"tick_volume": "volume"})
                df_chunk = df_chunk[["timestamp", "open", "high", "low", "close", "volume"]]
                dfs.append(df_chunk)
            else:
                print(f"⚠ No data for {mt5_symbol} from {current_from} to {current_to}")

            current_from = current_to

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return None

    def fetch_all(self):
        """
        Fetch historical data for all symbols.
        Returns a dictionary {symbol: DataFrame}.
        """
        all_data = {}
        for s in self.raw_symbols:
            df = self.fetch(s)
            if df is not None:
                all_data[self.resolve_symbol(s)] = df
        return all_data
