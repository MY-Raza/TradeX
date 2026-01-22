from TradeX.data.kraken.kraken_fetcher import KrakenFuturesFetcher
from TradeX.utils.data.data_cleaner import clean_df

fetcher = KrakenFuturesFetcher(symbol="PF_XBTUSD", interval="1m")
df = fetcher.fetch(start_date="2026-01-15", end_date="now")
df = clean_df(df)
fetcher.save_to_csv(df, "kraken_PF_XBTUSD_ohlcv.csv")