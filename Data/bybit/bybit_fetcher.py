import time
from datetime import datetime
import pandas as pd
from pybit.unified_trading import HTTP
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)


class BybitFuturesFetcher:
    """
    Fetches raw Bybit USDT Perpetual Futures klines data.
    Safe forward-pagination implementation (no infinite loops).
    """

    INTERVAL_MAP = {
        "1": 60_000,  # 1 minute in ms
    }

    def __init__(self, api_key: str, api_secret: str, demo: bool = False):
        self.client = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            demo=demo
        )
        logger.info("BybitFuturesFetcher initialized.")

    def _convert_to_timestamp(self, start_date: str, end_date: str) -> tuple[int, int]:
        """
        Convert start and end dates (YYYY-MM-DD) to milliseconds timestamps.
        If end_date is 'now', use current UTC time.
        """
        try:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)

            end_ts = (
                int(datetime.utcnow().timestamp() * 1000)
                if end_date.lower() == "now"
                else int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            )

            if start_ts >= end_ts:
                raise ValueError("start_date must be earlier than end_date")

            logger.info(f"Timestamps resolved | start={start_date} | end={end_date}")
            return start_ts, end_ts

        except Exception:
            logger.exception("Failed to convert dates to timestamps.")
            raise

    def fetch_klines(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        interval: str = "1",
        limit: int = 1000,
        max_loops: int = 20_000
    ) -> pd.DataFrame:
        """
        Fetch Bybit Futures klines for a symbol between given dates.

        Returns:
            pd.DataFrame
        """

        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}")

        interval_ms = self.INTERVAL_MAP[interval]
        start_ts, end_ts = self._convert_to_timestamp(start_date, end_date)

        all_klines = []
        last_start_ts = None
        loops = 0

        while start_ts < end_ts:
            loops += 1

            if loops > max_loops:
                logger.error("Max loop count reached. Breaking to avoid infinite loop.")
                break

            if last_start_ts == start_ts:
                logger.error(
                    "Start timestamp did not advance. Breaking loop to avoid infinite loop."
                )
                break

            last_start_ts = start_ts

            logger.info(
                f"Fetching {symbol} from {datetime.utcfromtimestamp(start_ts / 1000)} | interval={interval}"
            )

            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                start=start_ts,
                limit=limit
            )

            klines = response.get("result", {}).get("list", [])

            if not klines:
                logger.info("No more klines returned.")
                break

            # Bybit returns newest → oldest
            klines.reverse()
            all_klines.extend(klines)

            # Advance strictly forward by interval
            start_ts = int(klines[-1][0]) + interval_ms

            time.sleep(0.25)  # polite rate limit

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
                "turnover"
            ]
        )

        # Clean & normalize
        df["timestamp"] = df["timestamp"].astype(int)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
        df[numeric_cols] = df[numeric_cols].astype(float)

        df.sort_values("timestamp", inplace=True)
        df.drop_duplicates(subset="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(
            f"Fetched {len(df)} klines | "
            f"{df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}"
        )

        return df
