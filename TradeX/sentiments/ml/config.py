# ============================================================
# DATABASE / SCHEMA
# ============================================================
DB_SCHEMA_FEATURES   = "reddit"          # where ml_features lives
DB_SCHEMA_OUTPUT     = "reddit"          # where we write predictions / results
DB_SCHEMA_PRICE      = "data_binance"    # where OHLCV data lives

FEATURES_TABLE       = "ml_features"
PRICE_TABLE          = "btc_1m"
PRICE_TIME_COLUMN    = "datetime"

# Output tables
TABLE_ML_PREDICTIONS    = "ml_predictions"
TABLE_BACKTEST_RESULTS  = "backtest_results"
TABLE_BACKTEST_SUMMARY  = "backtest_summary"
LAG_RANGE            = range(1, 6)

# ============================================================
# FEATURE LIST
# Mirrors ALL_FEATURES from feature_pipeline.py
# Keep in sync if the pipeline adds/removes features.
# ============================================================
SENTIMENT_FEATURES = [
    "sentiment_combined",
    "sentiment_volume_total",
    "sentiment_disagreement",
    "post_momentum",
    "comment_momentum",
    *[f"post_lag_{i}"    for i in LAG_RANGE],
    *[f"comment_lag_{i}" for i in LAG_RANGE],
]

MARKET_FEATURES = [
    "returns",
    "volatility",
    "volume_change",
    "price_momentum",
]

ALPHA_FEATURES = [
    "divergence",
    "sentiment_spike",
    "fear_greed_index",
    "sentiment_price_interaction",
]


ALL_FEATURES = SENTIMENT_FEATURES + MARKET_FEATURES + ALPHA_FEATURES

TARGET_CLASS_COL  = "target"         # binary 0/1
TARGET_RETURN_COL = "target_return"  # continuous next-bar return
DATETIME_COL      = "datetime"

# ============================================================
# TRAIN / VAL / TEST SPLIT  (time-based, no shuffle)
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO  (implicit)

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================
CLASSIFIER_PARAMS = {
    "n_estimators":  200,
    "max_depth":     10,
    "min_samples_split": 10,
    "min_samples_leaf":   5,
    "max_features": "sqrt",
    "class_weight": "balanced",   # handles class imbalance
    "n_jobs":        -1,
    "random_state":  42,
}

REGRESSOR_PARAMS = {
    "n_estimators":  200,
    "max_depth":     8,
    "min_samples_split": 10,
    "min_samples_leaf":   5,
    "max_features": "sqrt",
    "n_jobs":        -1,
    "random_state":  42,
}

# ============================================================
# SIGNAL GENERATION
# ============================================================
# Signals: 1 = long, -1 = short, 0 = neutral
# A LONG signal requires:   class_pred == 1  AND  reg_pred >  SIGNAL_THRESHOLD
# A SHORT signal requires:  class_pred == 0  AND  reg_pred < -SIGNAL_THRESHOLD
# Otherwise: neutral (0)
SIGNAL_THRESHOLD = 0.0002      # 10× lower — matches actual RF prediction scale
MIN_CLASS_PROBABILITY = 0.55    # 0.0 = disable

# ============================================================
# BACKTEST PARAMETERS
# ============================================================
BACKTEST_PARAMS = {
    "starting_balance":  1_000,
    "take_profit":       3,    # %
    "stop_loss":         1.0,    # %
    "buy_after_minutes": 0,      # enter on open of signal bar
    "fee":               0.05,   # % per side (Binance maker ~0.02, taker 0.05)
    "leverage":          1.0,
    "slippage":          0.01,   # %
    "max_delay_minutes": 1,
}