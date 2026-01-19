# bybit_fetcher.py
import os
import time
from datetime import datetime
import pandas as pd
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from TradeX.utils.common.logs import get_logger

logger = get_logger(__name__)

# ---------------------------
# Load API keys from environment
# ---------------------------
load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Bybit API credentials not found in environment variables.")

# ---------------------------
# Create global HTTP client
# ---------------------------
BYBIT_CLIENT = HTTP(
    api_key=API_KEY,
    api_secret=API_SECRET,
    demo=False,  # You can make this configurable if needed
)

logger.info("Bybit HTTP client initialized.")


class BybitFuturesFetcher:
    """
    Stateful Bybit USDT Perpetual Futures Kline Fetcher.

    Only fetches RAW klines; no cleaning, sorting, or resampling.
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        interval: str = "1",
        limit: int = 1000,
        max_loops: int = 20_000,
    ):
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.limit = limit
        self.max_loops = max_loops

        # Validate interval
        if interval == "1":
            self.interval_ms = 60_000
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Convert dates to timestamps
        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"BybitFuturesFetcher initialized | {self.symbol} | {self.start_date} → {self.end_date}"
        )

    def _convert_to_timestamp(self) -> tuple[int, int]:
        start_ts = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = (
            int(datetime.utcnow().timestamp() * 1000)
            if self.end_date.lower() == "now"
            else int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
        )

        if start_ts >= end_ts:
            raise ValueError("start_date must be earlier than end_date")

        return start_ts, end_ts

    def fetch_raw_data(self) -> pd.DataFrame:
        all_klines = []
        start_ts = self.start_ts
        last_start_ts = None
        loops = 0

        while start_ts < self.end_ts:
            loops += 1
            if loops > self.max_loops:
                logger.error("Max loop count reached. Breaking loop.")
                break
            if start_ts == last_start_ts:
                logger.error("Timestamp did not advance. Breaking loop.")
                break
            last_start_ts = start_ts

            logger.info(
                f"Fetching {self.symbol} from {datetime.utcfromtimestamp(start_ts / 1000)}"
            )

            response = BYBIT_CLIENT.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.interval,
                start=start_ts,
                limit=self.limit,
            )

            klines = response.get("result", {}).get("list", [])
            if not klines:
                logger.info("No more klines returned.")
                break

            klines.reverse()
            all_klines.extend(klines)

            start_ts = int(klines[-1][0]) + self.interval_ms
            time.sleep(0.25)

        if not all_klines:
            logger.warning("No klines fetched.")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_klines,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        logger.info(f"Fetched RAW klines: {len(df)} rows")
        return df
