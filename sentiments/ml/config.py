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

# ============================================================
# FEATURE LIST
# Mirrors ALL_FEATURES from feature_pipeline.py
# Keep in sync if the pipeline adds/removes features.
# ============================================================
SENTIMENT_FEATURES = [
    # Post-based
    "post_sentiment_mean", "post_sentiment_std", "post_sentiment_combined",
    "post_sentiment_ema", "post_sentiment_spike", "post_sentiment_divergence",
    # Lag features — posts
    "post_sentiment_combined_lag1", "post_sentiment_combined_lag2",
    "post_sentiment_combined_lag3", "post_sentiment_combined_lag4",
    "post_sentiment_combined_lag5",
    # Comment-based
    "comment_sentiment_mean", "comment_sentiment_std", "comment_sentiment_combined",
    "comment_sentiment_ema", "comment_sentiment_spike", "comment_sentiment_divergence",
    # Lag features — comments
    "comment_sentiment_combined_lag1", "comment_sentiment_combined_lag2",
    "comment_sentiment_combined_lag3", "comment_sentiment_combined_lag4",
    "comment_sentiment_combined_lag5",
]

MARKET_FEATURES = [
    "returns",
    "log_returns",
    "volatility",
    "volume_change",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "atr",
    "price_momentum",
]

ALPHA_FEATURES = [
    "sentiment_momentum",
    "sentiment_vol_interaction",
    "cross_source_divergence",
    "sentiment_market_alignment",
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
SIGNAL_THRESHOLD = 0.002   # ~0.2% predicted move — tune freely

# Minimum classification probability to act (avoids low-confidence trades)
MIN_CLASS_PROBABILITY = 0.55   # 0.0 = disable

# ============================================================
# BACKTEST PARAMETERS
# ============================================================
BACKTEST_PARAMS = {
    "starting_balance":  1_000,
    "take_profit":       1.5,    # %
    "stop_loss":         1.0,    # %
    "buy_after_minutes": 0,      # enter on open of signal bar
    "fee":               0.05,   # % per side (Binance maker ~0.02, taker 0.05)
    "leverage":          1.0,
    "slippage":          0.01,   # %
    "max_delay_minutes": 1,
}