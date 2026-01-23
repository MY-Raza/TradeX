import MetaTrader5 as mt5
import pandas as pd

class MetaTrader5FutureFetcher:
    def __init__(self, symbols, utc_from, utc_to, timeframe):
        """
        symbols: list of MT5 symbol names (already resolved in main)
        utc_from, utc_to: datetime objects
        timeframe: MT5 timeframe (mt5.TIMEFRAME_M1 etc.)
        """
        self.symbols = symbols
        self.utc_from = utc_from
        self.utc_to = utc_to
        self.timeframe = timeframe

    def fetch(self, symbol):
        """
        Fetch historical data for a single symbol.
        Converts 'time' -> 'timestamp' in Unix ms
        Converts 'tick_volume' -> 'volume'
        Returns DataFrame
        """
        rates = mt5.copy_rates_range(symbol, self.timeframe, self.utc_from, self.utc_to)
        if rates is None or len(rates) == 0:
            print(f"❌ No data for {symbol}")
            return None

        df = pd.DataFrame(rates)

        # Convert time → timestamp (Unix ms)
        df["timestamp"] = (df["time"].astype("int64")) * 1000

        # Rename tick_volume → volume
        df = df.rename(columns={"tick_volume": "volume"})

        # Keep only relevant columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        return df
