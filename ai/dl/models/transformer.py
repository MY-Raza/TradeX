from darts.models import TransformerModel
from TradeX.ai.dl.utils import prepare_series, train_test_split


def train(
    df,
    target_col="close",
    split_date="2024-01-01",
):

    series = prepare_series(df, target_col)
    train_series, test_series = train_test_split(series, split_date)

    model = TransformerModel(
        input_chunk_length=48,
        output_chunk_length=12,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        n_epochs=50,
        batch_size=32,
        random_state=42
    )

    model.fit(train_series)

    preds = model.predict(len(test_series))

    return model, preds, test_series.time_index, None