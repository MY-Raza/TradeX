# kraken_futures_fetcher.py
import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from TradeX.utils.common.logs import get_logger

logger = get_logger("kraken_fetcher")

# ---------------------------
# Load Kraken API credentials from .env
# ---------------------------
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path)

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_PRIVATE_KEY = os.getenv("KRAKEN_SECRET_KEY")

if not KRAKEN_API_KEY or not KRAKEN_PRIVATE_KEY:
    logger.warning("Kraken API credentials not found. Public endpoints will still work.")


class KrakenFuturesFetcher:
    """
    Fetches OHLCV futures candlestick data from Kraken Futures API.

    Public endpoint: https://futures.kraken.com/api/charts/v1/{tick_type}/{symbol}/{resolution}

    Parameters:
        symbol: str       -> futures contract symbol, e.g., "PI_XBTUSD" (Perpetual BTC/USD)
        start_date: str   -> "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
        end_date: str     -> "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" or "now"
        tick_type: str    -> "trade", "mark", or "spot"
        resolution: str   -> candle interval: "1m", "5m", "1h", etc.
        max_loops: int    -> safety limit for pagination
    """

    BASE_URL = "https://futures.kraken.com/api/charts/v1/{tick_type}/{symbol}/{resolution}"

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        tick_type: str = "trade",
        resolution: str = "1m",
        max_loops: int = 20000,
    ):
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.tick_type = tick_type
        self.resolution = resolution
        self.max_loops = max_loops

        # Convert dates to epoch seconds
        self.start_ts = self._to_epoch(self.start_date)
        self.end_ts = (
            int(datetime.utcnow().timestamp())
            if self.end_date.lower() == "now"
            else self._to_epoch(self.end_date)
        )

        if self.start_ts >= self.end_ts:
            raise ValueError("start_date must be earlier than end_date")

        logger.info(
            f"KrakenFuturesFetcher initialized | {self.symbol} | {self.tick_type} | "
            f"{self.start_date} → {self.end_date}"
        )

    def _to_epoch(self, dt_str: str) -> int:
        """Convert datetime string to UNIX epoch seconds."""
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime.strptime(dt_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical OHLCV futures data and return as pandas DataFrame."""
        all_records = []
        current_from = self.start_ts
        loops = 0

        while current_from < self.end_ts:
            loops += 1
            if loops > self.max_loops:
                logger.error("Max loop count reached. Breaking fetch loop.")
                break

            url = self.BASE_URL.format(
                tick_type=self.tick_type,
                symbol=self.symbol,
                resolution=self.resolution,
            )
            params = {
                "from": current_from,
                "to": self.end_ts,
            }

            logger.info(f"Fetching {self.symbol} from {current_from} → {self.end_ts}")
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Request error: {e}")
                break

            data = resp.json()
            candles = data.get("candles", [])
            if not candles:
                logger.info("No more candles returned.")
                break

            # Convert to DataFrame
            df_part = pd.DataFrame(candles)
            df_part["time"] = pd.to_datetime(df_part["time"], unit="ms")
            all_records.append(df_part)

            # Advance 'from' for pagination
            last_time_ms = df_part["time"].astype("int64").max()
            current_from = int(last_time_ms // 10**9) + 1  # convert ns -> seconds

            time.sleep(0.3)  # rate limit buffer

            if current_from >= self.end_ts:
                break

        if not all_records:
            logger.warning("No data fetched.")
            return pd.DataFrame()

        df = pd.concat(all_records, ignore_index=True)
        logger.info(f"Fetched {len(df)} rows for {self.symbol}.")
        return df
