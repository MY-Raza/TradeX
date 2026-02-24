import pandas as pd
from TradeX.backtest.backtest import BackTest
from TradeX.utils.db.utils import fetch_ohlcv_df
# Read CSV file
df = pd.read_csv("signals_audit.csv")

# Convert datetime column to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Create a new dataframe with only datetime and converted trade_signal
signal_df = df[['datetime', 'trade_signal']].copy()

# Convert trade_signal values
signal_df['trade_signal'] = signal_df['trade_signal'].map({
    'long': 1,
    'short': -1
})
signal_df = signal_df.rename(columns={'trade_signal': 'signals'})

df_1m = fetch_ohlcv_df(
            table_name=f"btc_1m",
            schema="data_binance",
            time_column="datetime",
            start_date='2023-01-01',
            end_date='2026-02-17'
)

bt = BackTest(
                    df_1m,
                    signal_df,
                    take_profit=3,
                    stop_loss=1
                )
ledger, final_balance, pnl = bt.run()
ledger.to_csv("test.csv",index=False)
print(f"Final Balance: {final_balance}")
print(f"Cummulative PnL: {pnl}")

