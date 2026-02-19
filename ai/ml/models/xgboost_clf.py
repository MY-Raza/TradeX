from xgboost import XGBClassifier
import pandas as pd

def train(df, target_col="target", split_date="2024-01-01 00:00", **xgb_params):
    """
    Train XGBClassifier using string-based date split and dynamic hyperparameters.

    Args:
        df (pd.DataFrame): Input dataframe with features and target
        target_col (str): Name of the target column
        split_date (str): Date string to split train/test
        **xgb_params: XGBoost hyperparameters (n_estimators, max_depth, etc.)

    Returns:
        model: trained XGBClassifier
        preds: predictions on the test set
    """

    # Ensure datetime column exists
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime")
    else:
        raise ValueError("DataFrame must have a 'datetime' column for string slicing.")

    # Split data
    split_date = pd.to_datetime(split_date, utc=True)
    train_df = df[df["datetime"] < split_date]
    test_df = df[df["datetime"] >= split_date]

    X_train = train_df.drop(columns=[target_col, "datetime"])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col, "datetime"])
    y_test = test_df[target_col]

    # Train model with dynamic params
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return model, preds, X_test.index
