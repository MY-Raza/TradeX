import pandas as pd


def prepare_ml_data(df: pd.DataFrame):

    df = df.copy()

    drop_cols = ["datetime", "future_close"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = df.dropna()

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y
