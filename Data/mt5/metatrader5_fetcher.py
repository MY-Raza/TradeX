import MetaTrader5 as mt5
import pandas as pd

class MetaTrader5FutureFetcher:
    SPECIAL_SUFFIX = {
        "neth": "25"  # Example special suffix
    }

    def __init__(self, login, password, server, symbols, utc_from, utc_to, timeframe=mt5.TIMEFRAME_M1):
        """
        login, password, server: MT5 credentials
        symbols: list from config.yml (raw)
        utc_from, utc_to: datetime objects
        timeframe: MT5 timeframe
        """
        self.login = login
        self.password = password
        self.server = server
        self.raw_symbols = symbols
        self.utc_from = utc_from
        self.utc_to = utc_to
        self.timeframe = timeframe

        # Initialize MT5
        if not mt5.initialize(login=self.login, password=self.password, server=self.server):
            raise RuntimeError(f"❌ MT5 init failed: {mt5.last_error()}")
        print("✅ MT5 initialized inside class")

        # Resolve symbols
        self.symbols = [self.resolve_symbol(s) for s in self.raw_symbols]

        # Enable symbols
        for sym in self.symbols:
            if mt5.symbol_info(sym):
                mt5.symbol_select(sym, True)
                print(f"✔ Enabled symbol: {sym}")
            else:
                print(f"⚠ Symbol not found: {sym}")

    def resolve_symbol(self, symbol: str) -> str:
        """
        Converts config symbol to MT5 symbol.
        """
        symbol = symbol.lower()
        if symbol in self.SPECIAL_SUFFIX:
            return f"{symbol.upper()}{self.SPECIAL_SUFFIX[symbol]}"
        return f"{symbol.upper()}USD"

    def fetch(self, symbol: str):
        """
        Fetch historical data for a single symbol.
        Converts time -> timestamp in Unix ms
        Converts tick_volume -> volume
        """
        mt5_symbol = self.resolve_symbol(symbol)  # ensure consistent
        rates = mt5.copy_rates_range(mt5_symbol, self.timeframe, self.utc_from, self.utc_to)
        if rates is None or len(rates) == 0:
            print(f"❌ No data for {mt5_symbol}")
            return None

        df = pd.DataFrame(rates)

        # Convert time -> timestamp in Unix ms
        df["timestamp"] = df["time"].astype("int64") * 1000

        # Rename tick_volume -> volume
        df = df.rename(columns={"tick_volume": "volume"})

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def shutdown(self):
        mt5.shutdown()
        print("🔌 MT5 shutdown complete")
