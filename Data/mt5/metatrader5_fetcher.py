# metatrader5_fetcher.py

import MetaTrader5 as mt5
import pandas as pd
from datetime import timedelta, datetime, timezone
from TradeX.utils.common.logs import get_logger

logger = get_logger("metatrader5_fetcher")


class MetaTrader5FutureFetcher:
    """
    Fetch historical futures data from MetaTrader5.

    Returns DataFrames with columns:
        ['datetime', 'open', 'high', 'low', 'close', 'volume']
    where 'datetime' is UTC-aware.
    """

    SPECIAL_SUFFIX = {
        "neth": "25",  # Example: neth -> NETH25
    }

    def __init__(self, symbols, utc_from, utc_to, timeframe=mt5.TIMEFRAME_M1):
        self.raw_symbols = symbols
        self.utc_from = utc_from
        self.utc_to = min(utc_to, datetime.now(timezone.utc))  # Avoid future
        self.timeframe = timeframe

        # Resolve symbols
        self.symbols = [self.resolve_symbol(s) for s in self.raw_symbols]

        # Enable symbols
        for sym in self.symbols:
            info = mt5.symbol_info(sym)
            if info is None:
                logger.warning(f"⚠ Symbol not found: {sym}")
            elif mt5.symbol_select(sym, True):
                logger.info(f"✔ Enabled symbol: {sym}")
            else:
                logger.error(f"❌ Failed to enable symbol: {sym}")

    def resolve_symbol(self, symbol: str) -> str:
        symbol = symbol.lower()
        if symbol in self.SPECIAL_SUFFIX:
            return f"{symbol.upper()}{self.SPECIAL_SUFFIX[symbol]}"
        return f"{symbol.upper()}USD"

    def fetch(self, symbol: str) -> pd.DataFrame | None:
        mt5_symbol = self.resolve_symbol(symbol)

        # Ensure symbol exists
        if mt5.symbol_info(mt5_symbol) is None:
            logger.error(f"❌ Symbol {mt5_symbol} not found on MT5 server")
            return None
        if not mt5.symbol_select(mt5_symbol, True):
            logger.error(f"❌ Failed to enable symbol {mt5_symbol}")
            return None

        dfs = []
        current_from = self.utc_from

        while current_from < self.utc_to:
            current_to = min(current_from + timedelta(days=1), self.utc_to)

            try:
                rates = mt5.copy_rates_range(mt5_symbol, self.timeframe, current_from, current_to)
            except Exception as e:
                logger.error(f"❌ Error fetching {mt5_symbol} from {current_from} to {current_to}: {e}")
                current_from = current_to
                continue

            if rates is not None and len(rates) > 0:
                df_chunk = pd.DataFrame(rates)
                # Convert MT5 time to UTC datetime
                df_chunk["timestamp"] = pd.to_datetime(df_chunk["time"], unit="s", utc=True)
                # Rename tick_volume -> volume
                df_chunk = df_chunk.rename(columns={"tick_volume": "volume"})
                # Keep only required columns
                df_chunk = df_chunk[["timestamp", "open", "high", "low", "close", "volume"]]
                dfs.append(df_chunk)
            else:
                logger.warning(f"⚠ No data for {mt5_symbol} from {current_from} to {current_to}")

            current_from = current_to

        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
            # Ensure sorting by datetime
            df_all = df_all.sort_values("timestamp").reset_index(drop=True)
            return df_all
        else:
            return None

    def fetch_all(self) -> dict:
        """
        Fetch all configured symbols.

        Returns:
            dict: {resolved_symbol: DataFrame}
        """
        all_data = {}
        for s in self.raw_symbols:
            df = self.fetch(s)
            if df is not None and not df.empty:
                all_data[self.resolve_symbol(s)] = df
        return all_data
