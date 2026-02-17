import pandas as pd


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    df.dropna(inplace=True)

    return df
