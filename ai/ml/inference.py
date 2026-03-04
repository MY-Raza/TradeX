import os
import numpy as np
import pandas as pd
from datetime import datetime

from TradeX.utils.db.utils import fetch_ohlcv_df
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.utils.common.logs import get_logger
from TradeX.ai.ml.main import generate_features
from TradeX.backtest.backtest import BackTest

import joblib
import re

logger = get_logger("inference")


# ==========================================================
# FEATURE MAP (Same as training)
# ==========================================================

FEATURE_MAP = {
    "BB_UPPER": "BBANDS",
    "BB_MIDDLE": "BBANDS",
    "BB_LOWER": "BBANDS",
}


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(model_name: str, model_dir: str = "saved_models"):
    """
    Load model .pkl file.

    Returns:
        model
        feature_list
        symbols
    """
    model_path = os.path.join(model_dir, f"{model_name}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    saved_obj = joblib.load(model_path)

    model = saved_obj["model"]
    feature_list = saved_obj["features"]
    symbols = saved_obj.get("symbols", [])

    logger.info(f"Loaded model: {model_name}")
    logger.info(f"Features used: {len(feature_list)}")

    return model, feature_list, symbols


# ==========================================================
# GENERATE ONLY REQUIRED FEATURES
# ==========================================================

def generate_required_features(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """
    Generate only required indicators based on saved feature list.
    """

    base_indicators = set()

    for f in feature_list:
        if f in FEATURE_MAP:
            base_indicators.add(FEATURE_MAP[f])
        elif f.startswith("HT_"):
            base_indicators.add(f)
        else:
            m = re.match(r"([A-Z]+)", f)
            if m:
                base_indicators.add(m.group(1))

    df_generated = generate_features(df, list(base_indicators))

    missing_cols = set(feature_list) - set(df_generated.columns)
    if missing_cols:
        logger.warning(f"Missing features during inference: {missing_cols}")

    df_final = pd.concat(
        [df, df_generated[[c for c in feature_list if c in df_generated.columns]]],
        axis=1,
    )

    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    return df_final


# ==========================================================
# MAIN INFERENCE FUNCTION
# ==========================================================

def run_inference(
    model_name: str,
    start_date: str,
    end_date: str,
    table_name: str = "btc_1m",
    schema: str = "data_binance",
    timehorizon: str = "1h",
    classifier_threshold_high: float = 0.55,
    classifier_threshold_low: float = 0.45,
    run_backtest: bool = False,
    take_profit: float = 3,
    stop_loss: float = 1,
    k=0.5
):
    """
    Full inference pipeline.
    """

    # ------------------------------------------------------
    # 1️⃣ Load model
    # ------------------------------------------------------
    model, feature_list, symbols = load_model(model_name)

    # Detect model type from name
    model_type = "classifier" if "classifier" in model_name else "regressor"

    # ------------------------------------------------------
    # 2️⃣ Fetch new data
    # ------------------------------------------------------
    df_1m = fetch_ohlcv_df(
        table_name=table_name,
        schema=schema,
        time_column="datetime",
        start_date=start_date,
        end_date=end_date,
    )

    df_tf = resample_ohlcv(df_1m, timehorizon)

    # ------------------------------------------------------
    # 3️⃣ Generate required features
    # ------------------------------------------------------
    df_features = generate_required_features(df_tf, feature_list)

    # Keep only feature columns
    X = df_features[feature_list].dropna()

    if X.empty:
        raise ValueError("No valid rows after feature generation.")

    # Ensure correct column order
    X = X[feature_list]

    # ------------------------------------------------------
    # 4️⃣ Predict
    # ------------------------------------------------------
    if model_type == "classifier":
        preds_proba = model.predict_proba(X)[:, 1]

        signals = np.where(
            preds_proba > classifier_threshold_high,
            1,
            np.where(preds_proba < classifier_threshold_low, -1, 0),
        )

    else:  # regressor
        preds = model.predict(X)
        threshold = k * np.std(preds)
        signals = np.where(preds > threshold, 1,
                  np.where(preds < -threshold, -1, 0))

    # ------------------------------------------------------
    # Create prediction dataframe
    # ------------------------------------------------------
    df_predictions = pd.DataFrame({
        "datetime": df_features.loc[X.index, "datetime"],
        "signal": signals,
    })

    df_predictions["datetime"] = pd.to_datetime(df_predictions["datetime"], utc=True)

    logger.info(f"Generated {len(df_predictions)} predictions.")

    return df_predictions