import time
from datetime import datetime
import pandas as pd
from pybit.unified_trading import HTTP
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)


class BybitFuturesFetcher:
    """
    Stateful Bybit USDT Perpetual Futures Kline Fetcher.

    This class is responsible **ONLY** for fetching raw kline/candlestick data from Bybit.
    It does NOT perform any cleaning, numeric conversion, resampling, or sorting.

    Attributes:
        client (HTTP): Bybit API client.
        symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD) or "now".
        interval (str): Kline interval (currently only "1" supported for 1m).
        limit (int): Maximum number of klines per API call.
        max_loops (int): Safety limit to avoid infinite loops.
        interval_ms (int): Interval duration in milliseconds.
        start_ts (int): Start timestamp in milliseconds.
        end_ts (int): End timestamp in milliseconds.
    """

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
        """
        Initialize the BybitFuturesFetcher with API keys and fetch parameters.

        Args:
            api_key (str): Bybit API key.
            api_secret (str): Bybit API secret.
            symbol (str): Trading pair (e.g., "BTCUSDT").
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str, optional): End date (YYYY-MM-DD) or "now". Defaults to "now".
            interval (str, optional): Kline interval. Defaults to "1".
            limit (int, optional): Max klines per API call. Defaults to 1000.
            max_loops (int, optional): Safety max loop count. Defaults to 20,000.
            demo (bool, optional): Use demo/testnet environment. Defaults to False.
        """
        # Initialize Bybit HTTP client
        self.client = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            demo=demo,
        )

        # Validate and set interval in milliseconds (hardcoded)
        if interval == "1":
            self.interval_ms = 60_000  # 1 minute
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Store parameters
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.limit = limit
        self.max_loops = max_loops

        # Convert dates to timestamps in milliseconds
        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"BybitFuturesFetcher initialized | "
            f"{self.symbol} | {self.start_date} → {self.end_date}"
        )

    def _convert_to_timestamp(self) -> tuple[int, int]:
        """
        Convert start and end dates (YYYY-MM-DD) to millisecond timestamps.

        Returns:
            tuple[int, int]: start_ts, end_ts in milliseconds.

        Raises:
            ValueError: If start_date >= end_date
        """
        # Convert start date
        start_ts = int(
            datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000
        )

        # Convert end date or use current UTC time
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
        Fetch RAW klines from Bybit for the specified symbol and interval.

        Notes:
            - The API returns data from newest → oldest; this is reversed internally.
            - No cleaning, numeric conversion, or sorting is applied.
            - Rate limiting is respected with a short sleep between requests.
            - Stops if max_loops is reached or timestamps do not advance.

        Returns:
            pd.DataFrame: DataFrame containing raw OHLCV data with columns:
                - timestamp
                - open
                - high
                - low
                - close
                - volume
                - turnover
        """
        all_klines = []
        start_ts = self.start_ts
        last_start_ts = None
        loops = 0

        while start_ts < self.end_ts:
            loops += 1

            # Safety checks
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

            # Fetch a batch of klines from Bybit
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

            # API returns newest → oldest, reverse to chronological order
            klines.reverse()
            all_klines.extend(klines)

            # Update start timestamp for next batch
            start_ts = int(klines[-1][0]) + self.interval_ms

            # Polite rate limiting
            time.sleep(0.25)

        if not all_klines:
            logger.warning("No klines fetched.")
            return pd.DataFrame()

        # Convert to DataFrame
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
