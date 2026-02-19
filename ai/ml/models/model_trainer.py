import os
import pickle

# Import model modules
from TradeX.ai.ml.models import randomforest_clf, xgboost_clf
from TradeX.ai.ml.models import randomforest_reg, xgboost_reg
from TradeX.utils.common.config_loader import get_logger
import numpy as np  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
import pandas as pd # pyright: ignore[reportMissingModuleSource]

logger = get_logger("model_trainer")

# Mapping model names to train functions
CLASSIFIERS = {
    "random_forest": randomforest_clf.train,
    "xgboost": xgboost_clf.train,
}

REGRESSORS = {
    "random_forest": randomforest_reg.train,
    "xgboost": xgboost_reg.train,
}


def train_model(model_type: str, model_name: str, df, target_col="target",
                split_date="2024-01-01 00:00", **kwargs):
    """
    Train a model with string-based train/test split and optional hyperparameters.
    """
    if model_type == "classifier":
        trainer = CLASSIFIERS.get(model_name)
    elif model_type == "regressor":
        trainer = REGRESSORS.get(model_name)
    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    if trainer is None:
        raise ValueError(f"Unknown model name: {model_name}")

    # Call trainer with kwargs (XGBoost params)
    model, preds = trainer(df, target_col=target_col, split_date=split_date, **kwargs)
    return model, preds




def save_model(model, feature_columns, symbol, model_name, folder="saved_models"):
    """
    Save the trained model along with its feature columns.

    Args:
        model: Trained model object
        feature_columns (list): List of feature column names
        symbol (str): Symbol name (e.g., BTC)
        model_name (str): Model name (e.g., random_forest_classifier)
        folder (str): Folder path to save the model
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{symbol}_{model_name}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": feature_columns
        }, f)

    logger.info(f"Saved model {model_name} for {symbol} at {file_path}")

def prepare_predictions(df, preds, model_type, threshold=0.0):
    """
    Convert raw model predictions into backtest-ready DataFrame.
    """
    df_test = df[df["datetime"] >= "2024-01-01 00:00"].copy()

    if model_type == "classifier":
        signals = preds

    elif model_type == "regressor":
        signals = np.where(preds > threshold, 1,
                  np.where(preds < -threshold, -1, 0))

    df_predictions = pd.DataFrame({
        "datetime": df_test["datetime"].values,
        "signals": signals
    })

    return df_predictions
