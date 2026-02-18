import os
import pickle

# Import model modules
from TradeX.ai.ml.models import randomforest_clf, xgboost_clf
from TradeX.ai.ml.models import randomforest_reg, xgboost_reg
from TradeX.utils.common.config_loader import get_logger

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


def train_model(model_type: str, model_name: str, df, target_col="target", split_date="2024-01-01 00:00"):
    """
    Train a model using the corresponding trainer with string-based date split.

    Args:
        model_type (str): 'classifier' or 'regressor'
        model_name (str): 'random_forest' or 'xgboost'
        df (pd.DataFrame): DataFrame containing features, target, and datetime column
        target_col (str): Target column name
        split_date (str): Date string to split train/test

    Returns:
        model: trained model
        preds: predictions on the test set
    """

    if model_type == "classifier":
        trainer = CLASSIFIERS.get(model_name)
    elif model_type == "regressor":
        trainer = REGRESSORS.get(model_name)
    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    if trainer is None:
        raise ValueError(f"Unknown model name: {model_name}")

    # Call the trainer with string-based splitting
    model, preds = trainer(df, target_col=target_col, split_date=split_date)
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
