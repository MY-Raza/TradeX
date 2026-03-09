from darts.models import NBEATSModel
from TradeX.ai.dl.utils import prepare_series, train_test_split
import pandas as pd
import numpy as np


def train(
    df: pd.DataFrame,
    target_col: str = "close",
    split_date: str = "2024-01-01",
    input_chunk_length: int = 48,
    output_chunk_length: int = 12,
    n_epochs: int = 50,
    batch_size: int = 32,
    **kwargs
):
    """
    Train an NBEATS model using Darts.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with datetime column
        target_col (str): Target column to forecast
        split_date (str): Date to split train/test sets
        input_chunk_length (int): Number of past steps for input
        output_chunk_length (int): Number of steps to forecast
        n_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        **kwargs: Extra arguments for NBEATSModel

    Returns:
        model: Trained NBEATS model
        preds: Predictions (Darts TimeSeries)
        test_index: Numeric index of test set
        df_test: Test DataFrame (empty, for interface consistency)
    """
    # Ensure datetime column is UTC-aware
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)

    # Prepare Darts TimeSeries
    series = prepare_series(df, target_col)

    # Split train/test
    train_series, test_series = train_test_split(series, split_date)

    # Initialize and train NBEATS
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=42,
        **kwargs
    )

    model.fit(train_series)

    # Make predictions
    preds = model.predict(len(test_series))

    # Return numeric index for backtesting
    test_index = np.arange(len(train_series), len(train_series) + len(test_series))
    df_test = pd.DataFrame(index=test_series.time_index)  # empty covariates

    return model, preds, test_index, df_test