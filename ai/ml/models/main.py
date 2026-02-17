from TradeX.utils.db.utils import fetch_ohlcv_df
from TradeX.ai.ml.models.feature_engineering import generate_features
from TradeX.ai.ml.models.target import create_target
from TradeX.ai.ml.models.dataset import prepare_ml_data
from TradeX.ai.ml.models.model import train_model, save_model


def main():

    print("Fetching data from database...")

    df = fetch_ohlcv_df(
        table_name="btc_1h",       # change if needed
        schema="market_data",      # change if needed
        limit=5000
    )

    if df.empty:
        print("No data found.")
        return

    print("Generating indicators...")
    df = generate_features(df)

    print("Creating target...")
    df = create_target(df)

    print("Preparing dataset...")
    X, y = prepare_ml_data(df)

    print("Training model...")
    model = train_model(X, y)

    save_model(model)


if __name__ == "__main__":
    main()
