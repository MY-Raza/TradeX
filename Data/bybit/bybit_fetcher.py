import time
from datetime import datetime
import pandas as pd
from pybit.unified_trading import HTTP
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)


class BybitFuturesFetcher:
    """
    Stateful Bybit USDT Perpetual Futures kline fetcher.
    Responsible ONLY for fetching raw data from Bybit.
    """

    INTERVAL_MAP = {
        "1": 60_000,  # 1 minute in milliseconds
    }

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        interval: str = "1",
        limit: int = 1000,
        max_loops: int = 20_000,
        demo: bool = False,
    ):
        self.client = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            demo=demo,
        )

        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}")

        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.limit = limit
        self.max_loops = max_loops
        self.interval_ms = self.INTERVAL_MAP[interval]

        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"BybitFuturesFetcher initialized | "
            f"{self.symbol} | {self.start_date} → {self.end_date}"
        )

    def _convert_to_timestamp(self) -> tuple[int, int]:
        """
        Convert YYYY-MM-DD dates to millisecond timestamps.
        """
        start_ts = int(
            datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000
        )

        end_ts = (
            int(datetime.utcnow().timestamp() * 1000)
            if self.end_date.lower() == "now"
            else int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
        )

        if start_ts >= end_ts:
            raise ValueError("start_date must be earlier than end_date")

        return start_ts, end_ts

    def fetch_klines(self) -> pd.DataFrame:
        """
        Fetch RAW klines from Bybit.
        No cleaning, casting, or sorting is performed here.
        """

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
                f"Fetching {self.symbol} from "
                f"{datetime.utcfromtimestamp(start_ts / 1000)}"
            )

            response = self.client.get_kline(
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

            # API returns newest → oldest
            klines.reverse()
            all_klines.extend(klines)

            start_ts = int(klines[-1][0]) + self.interval_ms
            time.sleep(0.25)  # polite rate limiting

        if not all_klines:
            logger.warning("No klines fetched.")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ],
        )

        logger.info(f"Fetched RAW klines: {len(df)} rows")
        return df
