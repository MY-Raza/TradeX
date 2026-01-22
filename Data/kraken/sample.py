from TradeX.data.kraken.kraken_fetcher import KrakenFuturesFetcher

fetcher = KrakenFuturesFetcher(symbol="PF_XBTUSD", interval="1m")
df = fetcher.fetch(start_date="2024-01-01", end_date="now")
fetcher.save_to_csv(df, "kraken_PF_XBTUSD_ohlcv.csv")