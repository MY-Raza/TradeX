from darts.models import VARIMA
from darts import TimeSeries
import pandas as pd


def train(
    df,
    target_cols=["open","high","low","close","volume"],
    split_date="2024-01-01",
):

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    series = TimeSeries.from_dataframe(
        df,
        time_col="datetime",
        value_cols=target_cols
    )

    train_series, test_series = series.split_before(pd.Timestamp(split_date))

    model = VARIMA()

    model.fit(train_series)

    preds = model.predict(len(test_series))

    return model, preds, test_series.time_index, None