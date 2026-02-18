import os
import pickle

# Import model modules
from TradeX.ai.ml.models import randomforest_clf, xgboost_clf
from TradeX.ai.ml.models import randomforest_reg, xgboost_reg


CLASSIFIERS = {
    "random_forest": randomforest_clf.train,
    "xgboost": xgboost_clf.train,
}

REGRESSORS = {
    "random_forest": randomforest_reg.train,
    "xgboost": xgboost_reg.train,
}


def train_model(model_type: str, model_name: str, X, y):
    """
    model_type: 'classifier' or 'regressor'
    model_name: 'random_forest' or 'xgboost'
    """

    if model_type == "classifier":
        trainer = CLASSIFIERS.get(model_name)
    elif model_type == "regressor":
        trainer = REGRESSORS.get(model_name)
    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    if trainer is None:
        raise ValueError(f"Unknown model name: {model_name}")

    return trainer(X, y)


def save_model(model, feature_columns, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump({
            "model": model,
            "features": feature_columns
        }, f)
