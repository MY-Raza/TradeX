from __future__ import annotations

import os
import pickle

from TradeX.ai.dl.models import gru, lstm, tcn, tft
from TradeX.utils.common.logs import get_logger

logger = get_logger("dl_model_trainer")

# ──────────────────────────────────────────────────────────────────────────────
# Dispatch tables
# ──────────────────────────────────────────────────────────────────────────────

CLASSIFIERS: dict[str, callable] = {
    "gru":  gru.train,
    "lstm": lstm.train,
    "tcn":  tcn.train,
    "tft":  tft.train,
}

REGRESSORS: dict[str, callable] = {
    "gru":  gru.train,
    "lstm": lstm.train,
    "tcn":  tcn.train,
    "tft":  tft.train,
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model_type: str,
    model_name: str,
    df,
    df_1m,
    target_col: str = "target",
    n_trails: int = 0,
    split_date: str = "2024-01-01 00:00",
):
    """
    Train a DL model — identical call signature to the ML ``train_model``.

    Args:
        model_type  : ``'classifier'`` or ``'regressor'``.
        model_name  : One of ``'gru'``, ``'lstm'``, ``'tcn'``, ``'tft'``.
        df          : Feature DataFrame (output of ``data_pipeline``).
        df_1m       : 1-minute OHLCV DataFrame for backtesting.
        target_col  : Name of the target column in ``df``.
        n_trails    : Optuna trial budget (note: intentional typo kept for
                      API compatibility with ``main.py``).
        split_date  : ISO date string for train/test temporal boundary.

    Returns:
        ``(model, preds, test_index, X_test)`` — same as ML ``train_model``.

    Raises:
        ValueError : If ``model_type`` or ``model_name`` is unrecognised.
    """
    if model_type == "classifier":
        trainer = CLASSIFIERS.get(model_name)
    elif model_type == "regressor":
        trainer = REGRESSORS.get(model_name)
    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    if trainer is None:
        raise ValueError(
            f"Unknown DL model name: '{model_name}'. "
            f"Available: {list(CLASSIFIERS.keys())}"
        )

    model, preds, test_index, X_test = trainer(
        df,
        df_1m,
        target_col=target_col,
        split_date=split_date,
        n_trials=n_trails,
        model_type=model_type,
    )
    return model, preds, test_index, X_test


def save_model(
    model,
    feature_columns: list[str],
    symbol: str,
    model_name: str,
    folder: str = "saved_models",
) -> None:
    """
    Persist a trained DL model alongside its feature column list.

    Saves a dict ``{'model': model, 'features': feature_columns}`` as a
    ``pickle`` file under ``{folder}/{symbol}_{model_name}.pkl``.

    Args:
        model           : Fitted ``BaseDLModel`` (or subclass) instance.
        feature_columns : List of feature names the model was trained on.
        symbol          : Asset symbol (e.g. ``'btc'``).
        model_name      : Descriptive name (e.g. ``'gru_classifier_20250101_120000'``).
        folder          : Directory to save into (created if absent).
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{symbol}_{model_name}.pkl")

    with open(file_path, "wb") as fh:
        pickle.dump({"model": model, "features": feature_columns}, fh)

    logger.info(f"Saved DL model '{model_name}' for {symbol} at {file_path}")