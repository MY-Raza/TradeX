from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def train(df, target_col="target", split_date="2024-01-01 00:00",**model_params):
    """
    Train RandomForestRegressor using a string-based date split.

    Args:
        df (pd.DataFrame): Input dataframe with features and target
        target_col (str): Name of the target column
        split_date (str): Date string to split train/test
                          All rows before this are train, after are test

    Returns:
        model: trained RandomForestRegressor
        X_train, X_test, y_train, y_test: split data (optional for evaluation)
    """

    # Ensure datetime column exists
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime")
    else:
        raise ValueError("DataFrame must have a 'datetime' column for string slicing.")

    # Split based on string date
    split_date = pd.to_datetime(split_date, utc=True)
    train_df = df[df["datetime"] < split_date]
    test_df = df[df["datetime"] >= split_date]

    X_train = train_df.drop(columns=[target_col, "datetime"])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col, "datetime"])
    y_test = test_df[target_col]

    # Train model
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    return model,preds, X_test.index
