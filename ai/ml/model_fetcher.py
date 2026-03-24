import numpy as np
import os
from TradeX.utils.db.utils import get_best_model, get_important_features,fetch_ohlcv_df
from TradeX.utils.common.logs import get_logger
from TradeX.utils.common.config_loader import read_config
import json
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.ai.ml.main import generate_features,create_classification_target,create_regression_target
import pandas as pd
import re
from TradeX.ai.ml.models.model_trainer import train_model,save_model
from TradeX.ai.ml.utils import prepare_predictions
from TradeX.backtest.backtest import BackTest
from datetime import datetime

logger = get_logger("model_fetcher")


FEATURE_MAP = {
    # Bollinger Bands
    "BB_UPPER": "BBANDS",
    "BB_MIDDLE": "BBANDS",
    "BB_LOWER": "BBANDS",
    # Hilbert Transform — Phasor (inphase=_0, quadrature=_1)
    "HT_PHASOR_0": "HT_PHASOR",
    "HT_PHASOR_1": "HT_PHASOR",
    # Hilbert Transform — Sine (sine=_0, leadsine=_1)
    "HT_SINE_0": "HT_SINE",
    "HT_SINE_1": "HT_SINE",
    # Linear Regression variants (suffixed with window, e.g. _14)
    "LINEARREG_SLOPE_14": "LINEARREG_SLOPE",
    "LINEARREG_ANGLE_14": "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT_14": "LINEARREG_INTERCEPT",
    # T3 (suffixed with window, e.g. _14)
    "T3_14": "T3",
    # Directional Movement — with period suffix
    "PLUS_DI_14": "PLUS_DI",
    "MINUS_DI_14": "MINUS_DI",
    # Directional Movement — no period suffix (stored as PLUS_DM / MINUS_DM)
    "PLUS_DM": "PLUS_DM",
    "MINUS_DM": "MINUS_DM",
}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
def generate_best_features(df: pd.DataFrame, best_features: list[str]) -> pd.DataFrame:
    """
    Generate only the technical indicators needed for best_features (with suffixes).

    Args:
        df (pd.DataFrame): DataFrame containing ['open','high','low','close','volume']
        best_features (list[str]): e.g. ["STOCH_K", "DX_14", "MACD_2", "ADOSC"]

    Returns:
        pd.DataFrame: df with only the best_features added
    """
    import json

    if isinstance(best_features, str):
        best_features = json.loads(best_features)

    # ----------------------------------------
    # Step 1: Determine base indicators to generate
    # ----------------------------------------
    base_indicators = set()
    for f in best_features:
        # Check explicit mapping first (handles suffixed variants like HT_SINE_0,
        # LINEARREG_ANGLE_14, PLUS_DI_14, MINUS_DM, T3_14, etc.)
        if f in FEATURE_MAP:
            base_indicators.add(FEATURE_MAP[f])
        # Candlestick patterns — keep as-is
        elif f.startswith("CDL"):
            base_indicators.add(f)
        # Hilbert Transform cycle indicators — strip trailing _0 / _1 index suffix
        # so "HT_SINE_0" -> "HT_SINE", "HT_PHASOR_1" -> "HT_PHASOR"
        elif f.startswith("HT_"):
            base = re.sub(r"_\d+$", "", f)
            base_indicators.add(base)
        else:
            # General case: strip trailing numeric period suffix (_14, _20, etc.)
            # then extract uppercase word(s) joined by underscores.
            # STOCH_K -> STOCH, MACD_2 -> MACD, DX_14 -> DX
            # PLUS_DI_14 -> PLUS_DI (via FEATURE_MAP above, but fallback works too)
            stripped = re.sub(r"_\d+$", "", f)
            m = re.match(r"([A-Z]+(?:_[A-Z]+)*)", stripped)
            if m:
                base_indicators.add(m.group(1))

    # ----------------------------------------
    # Step 2: Generate all base indicators
    # ----------------------------------------
    df_generated = generate_features(df, list(base_indicators))

    # ----------------------------------------
    # Step 3: Keep only columns that exactly match best_features
    # ----------------------------------------
    missing_cols = set(best_features) - set(df_generated.columns)
    if missing_cols:
        logger.warning(f"Some best_features were not generated: {missing_cols}")

    # Keep original df columns + only the best_features
    final_cols = [col for col in best_features if col in df_generated.columns]
    df_final = pd.concat([df[df.columns.tolist()], df_generated[final_cols]], axis=1)
    # After concatenation, remove duplicates
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    return df_final

# ==================================================
# MAIN EXECUTION
# ==================================================

best_model = get_best_model()

if best_model:
    logger.info(f"The best model: {best_model}")
else:
    logger.info("No valid model found")
    best_model = None

important_features = None

if best_model:
    important_features = get_important_features(best_model)

current_dir = os.path.dirname(os.path.abspath(__file__))
ml_config_path = os.path.join(current_dir, "config.yml")
config = read_config(ml_config_path)
start_date = config.get("start_date")
end_date = config.get("end_date")
split_date = config.get("split_date")

symbols = ["btc"]
timehorizon = config.get("timehorizon", "1h")
classifiers_config = config.get("classifiers", {})
regressors_config = config.get("regressors", {})

df_1m = fetch_ohlcv_df(
    table_name="btc_1m",
    schema="data_binance",
    time_column="datetime",
    start_date=start_date,
    end_date=end_date
)

df_1h = resample_ohlcv(
    df_1m,
    timehorizon
)

df_gf = generate_best_features(df_1h,important_features)

for clf_name, is_active in classifiers_config.items():
    if not is_active:
        continue

    logger.info(f"Training classifier: {clf_name} for {symbols}")
    try:
        df_clf = create_classification_target(df_gf)
        df_clf = df_clf.drop(columns=["open", "high", "low"], errors="ignore")
        model, preds, test_index, X_test = train_model(
                    model_type="classifier",
                    model_name=clf_name,
                    df=df_clf,
                    target_col="target",
                    split_date=split_date,
                    n_trails=10,
                    df_1m=df_1m
                )
        try:
            sample_preds = model.predict(X_test.head(5))
            logger.info(f"[Dry-run] {clf_name} predictions on first 5 test rows:\n{sample_preds}")
            preds_df = pd.DataFrame({
                 "prediction": sample_preds
              })
            preds_df.to_csv(f"{clf_name}_classifier_sample_preds.csv", index=False)
        except Exception as e:
            logger.error(f"[Dry-run] Failed for {clf_name}: {e}")
        df_predictions = prepare_predictions(df_clf,preds,test_index,model_type="classifier")
        df_predictions['datetime'] = pd.to_datetime(df_predictions['datetime'], utc=True)
        bt = BackTest(
                    df_1m,
                    df_predictions,
                    take_profit=3,
                    stop_loss=1
                )
        ledger, final_balance, pnl = bt.run()
        logger.info(f"Final Ledger for Classifier: {ledger.head()}")
        logger.info(f"Final Balance for Classifier: {final_balance}")
        logger.info(f"Final PnL for Classifier: {pnl}")
        save_model(
             model,
             X_test.columns.tolist(),
             symbols[0],
             f"{clf_name}_classifier_{timestamp}"
        )
    except Exception as e:
                logger.error(f"Classifier {clf_name} failed for {symbols}: {e}")

for reg_name, is_active in regressors_config.items():
     if not is_active:
          continue
     logger.info(f"Training Regressor: {reg_name} for {symbols}")
     try:
        df_reg = create_regression_target(df_gf)
        df_reg = df_reg.drop(columns=["open", "high", "low"], errors="ignore")
        model, preds, test_index, X_test = train_model(
                    model_type="regressor",
                    model_name=reg_name,
                    df=df_reg,
                    target_col="target",
                    split_date=split_date,
                    n_trails=10,
                    df_1m=df_1m
                )
        try:
            sample_preds = model.predict(X_test.head(5))
            logger.info(f"[Dry-run] {reg_name} predictions on first 5 test rows:\n{sample_preds}")
            preds_df = pd.DataFrame({
                 "prediction": sample_preds
              })
            preds_df.to_csv(f"{reg_name}_regressor_sample_preds.csv", index=False)
        except Exception as e:
            logger.error(f"[Dry-run] Failed for {clf_name}: {e}")
        df_predictions = prepare_predictions(df_reg,preds,test_index,model_type="regressor")
        df_predictions['datetime'] = pd.to_datetime(df_predictions['datetime'], utc=True)
        bt = BackTest(
                    df_1m,
                    df_predictions,
                    take_profit=3,
                    stop_loss=1
                )
        ledger, final_balance, pnl = bt.run()
        logger.info(f"Final Ledger for Regressor: {ledger.head()}")
        logger.info(f"Final Balance for Regressor: {final_balance}")
        logger.info(f"Final PnL for Regressor: {pnl}")
        save_model(
             model,
             X_test.columns.tolist(),
             symbols[0],
             f"{reg_name}_regressor_{timestamp}"
        )

     except Exception as e:
                logger.error(f"Regressor {reg_name} failed for {symbols}: {e}")