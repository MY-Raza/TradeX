# kraken_fetcher.py

import time
from datetime import datetime
import pandas as pd
import ccxt
from TradeX.utils.common.logs import get_logger

logger = get_logger("kraken_fetcher")


class KrakenFuturesFetcher:
    """
    Fetch historical Kraken Futures OHLCV data in batches
    and verify it starts from the given start_date.
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str = "now",
        timeframe: str = "1m",
        limit: int = 1000,
        sleep_seconds: float = 0.5,
    ):
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.limit = limit
        self.sleep_seconds = sleep_seconds

        # Initialize Kraken Futures
        self.exchange = ccxt.kraken({
            "options": {"defaultType": "future"}
        })

        # Convert dates → epoch ms
        self.start_ts, self.end_ts = self._convert_to_timestamp()

        logger.info(
            f"KrakenFuturesFetcher initialized | "
            f"symbol={self.symbol} | timeframe={self.timeframe} | "
            f"start={self.start_date} | end={self.end_date}"
        )

    # --------------------------------------------------
    # Timestamp conversion
    # --------------------------------------------------
    def _convert_to_timestamp(self) -> tuple[int, int]:
        """Convert start_date and end_date to epoch milliseconds."""
        try:
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")

        start_ts = int(start_dt.timestamp() * 1000)

        if self.end_date.lower() == "now":
            end_ts = int(datetime.utcnow().timestamp() * 1000)
        else:
            try:
                end_dt = datetime.strptime(self.end_date, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")

            end_ts = int(end_dt.timestamp() * 1000)

        if start_ts >= end_ts:
            raise ValueError("start_date must be earlier than end_date")

        return start_ts, end_ts

    # --------------------------------------------------
    # DEBUG: Verify requested vs returned timestamps
    # --------------------------------------------------
    def _debug_since_vs_returned(self, since: int, ohlcv: list):
        if not ohlcv:
            logger.warning("DEBUG: Empty OHLCV response")
            return

        first_ts = ohlcv[0][0]
        diff_ms = first_ts - since
        diff_min = diff_ms / 60000

        logger.info(
            "DEBUG START CHECK | "
            f"requested_since={since} "
            f"({datetime.utcfromtimestamp(since / 1000)}) | "
            f"first_returned={first_ts} "
            f"({datetime.utcfromtimestamp(first_ts / 1000)}) | "
            f"diff={diff_ms} ms ({diff_min:.2f} min)"
        )

    # --------------------------------------------------
    # Fetch data
    # --------------------------------------------------
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data in batches until end_ts.
        Returns DataFrame with epoch ms timestamps.
        """
        all_ohlcv = []
        since = self.start_ts

        while since < self.end_ts:
            logger.info(
                f"Fetching {self.symbol} | "
                f"from {datetime.utcfromtimestamp(since / 1000)}"
            )

            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=self.limit,
                )

                # 🔍 DEBUG CHECK
                self._debug_since_vs_returned(since, ohlcv)

            except ccxt.NetworkError as e:
                logger.warning(
                    f"Network error: {e}, retrying in {self.sleep_seconds}s"
                )
                time.sleep(self.sleep_seconds)
                continue

            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                break

            if not ohlcv:
                logger.info("No more data returned from Kraken.")
                break

            all_ohlcv.extend(ohlcv)

            # Advance timestamp safely
            last_candle_ts = ohlcv[-1][0]

            if last_candle_ts == since:
                # Safety guard (Kraken edge case)
                logger.warning(
                    "Last candle timestamp equals 'since'. "
                    "Skipping forward 1 minute to avoid infinite loop."
                )
                since += 60_000
            else:
                since = last_candle_ts + 1

            time.sleep(self.sleep_seconds)

        if not all_ohlcv:
            logger.warning("No data fetched from Kraken.")
            return pd.DataFrame()

        # --------------------------------------------------
        # Build DataFrame
        # --------------------------------------------------
        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        # --------------------------------------------------
        # FINAL VALIDATION
        # --------------------------------------------------
        first_ts = int(df["timestamp"].min())
        last_ts = int(df["timestamp"].max())

        logger.info(
            "FINAL DATA RANGE | "
            f"start_ts={first_ts} "
            f"({datetime.utcfromtimestamp(first_ts / 1000)}) | "
            f"end_ts={last_ts} "
            f"({datetime.utcfromtimestamp(last_ts / 1000)}) | "
            f"expected_start>={datetime.utcfromtimestamp(self.start_ts / 1000)}"
        )

        if first_ts < self.start_ts:
            logger.error(
                "❌ DATA STARTS BEFORE REQUESTED start_date | "
                f"diff_ms={self.start_ts - first_ts}"
            )

        return df
