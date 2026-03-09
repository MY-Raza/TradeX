from darts.models import ARIMA
from TradeX.ai.dl.utils import prepare_series, train_test_split


def train(
    df,
    target_col="close",
    split_date="2024-01-01",
):

    series = prepare_series(df, target_col)
    train_series, test_series = train_test_split(series, split_date)

    model = ARIMA()

    model.fit(train_series)

    preds = model.predict(len(test_series))

    return model, preds, test_series.time_index, None