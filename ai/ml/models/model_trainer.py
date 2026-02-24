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


def train_model(model_type: str,
                model_name: str,
                df,
                target_col="target",
                split_date="2024-01-01 00:00",
                use_optuna=False,
                n_trials=50,
                **kwargs):

    if model_type == "classifier":
        trainer = CLASSIFIERS.get(model_name)
    elif model_type == "regressor":
        trainer = REGRESSORS.get(model_name)
    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    if trainer is None:
        raise ValueError(f"Unknown model name: {model_name}")

    # =====================================
    # OPTUNA SECTION
    # =====================================
    if use_optuna:

        import optuna
        from sklearn.metrics import accuracy_score

        def objective(trial):

            # Hyperparameter search space for RandomForest
            if model_name == "random_forest":

                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "max_depth": trial.suggest_int("max_depth", 3, 40),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
                }

            elif model_name == "xgboost":

                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                }

            else:
                params = {}

            # Train with suggested params
            model, preds, test_index = trainer(
                df,
                target_col=target_col,
                split_date=split_date,
                **params
            )

            # Get true labels
            y_test = df.loc[test_index, target_col].values

            if model_type == "classifier":
                score = accuracy_score(y_test, preds)
            else:
                # For regression we minimize MSE
                from sklearn.metrics import mean_squared_error
                score = -mean_squared_error(y_test, preds)

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best parameters found: {study.best_params}")

        kwargs.update(study.best_params)

    # =====================================
    # FINAL TRAINING
    # =====================================
    model, preds, test_index = trainer(
        df,
        target_col=target_col,
        split_date=split_date,
        **kwargs
    )

    return model, preds, test_index

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

import pandas as pd
import numpy as np

def prepare_predictions(df, preds, test_index, model_type, threshold=None, k=0.5):
    """
    Prepare a predictions DataFrame for backtesting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original price DataFrame with 'datetime' column.
        
    preds : np.ndarray
        Model predictions (classifier or regressor outputs)
        
    test_index : array-like
        Indices of the test set in df
        
    model_type : str
        'classifier' or 'regressor'
        
    threshold : float or None
        If None and model_type='regressor', automatically computed as k*std(preds)
        
    k : float
        Multiplier for standard deviation when auto thresholding (default 0.5)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'signals'] with -1, 0, 1 signals
    """
    
    # Ensure datetime is UTC-aware
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    if model_type == "classifier":
        # Use predictions directly
        signals = preds

    elif model_type == "regressor":
        # Auto-compute threshold if not provided
        if threshold is None:
            threshold = k * np.std(preds)
        
        # Convert continuous predictions into discrete signals
        signals = np.where(preds > threshold, 1,
                  np.where(preds < -threshold, -1, 0))

    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    # Build prediction DataFrame
    df_predictions = pd.DataFrame({
        "datetime": df.loc[test_index, "datetime"].values,
        "signals": signals
    })

    return df_predictions




