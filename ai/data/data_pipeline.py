from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from TradeX.utils.db.utils import fetch_ohlcv_df
from TradeX.utils.common.logs import get_logger
from TradeX.utils.data.data_cleaner import resample_ohlcv
from TradeX.indicators.talib.indicators import call_indicator
logger = get_logger("data_pipeline")

# ---------------------------------------------------------------------------
# Columns that should NEVER be differenced
# (identifiers, targets, and already-stationary bounded indicators)
# ---------------------------------------------------------------------------
_SKIP_STATIONARITY = frozenset(
    {
        "datetime",
        "target",
        "future_close",
        # Bounded oscillators — always stationary by construction
        "RSI_14", "MFI_14", "CCI_14", "WILLR_14",
        "STOCH_K", "STOCH_D", "STOCHF_K", "STOCHF_D",
        "STOCHRSI_K", "STOCHRSI_D",
        "AROON_UP", "AROON_DOWN", "AROONOSC",
        "ADX_14", "ADXR_14", "DX_14",
        "BOP",
        # Candlestick patterns — integer {-100, 0, 100}
    }
)


# ===========================================================================
# 1. DATA ACQUISITION
# ===========================================================================

def fetch_raw_data(
    symbol: str,
    schema: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV data from the database.

    Args:
        symbol     : e.g. ``'btc'``
        schema     : Database schema name, e.g. ``'data_binance'``
        start_date : ISO date string or None (fetch all)
        end_date   : ISO date string or None (fetch all)

    Returns:
        Raw 1-minute OHLCV DataFrame.

    Raises:
        ValueError: If the returned DataFrame is empty.
    """
    table_name = f"{symbol}_1m"
    logger.info(f"Fetching data: schema={schema}, table={table_name}, "
                f"start={start_date}, end={end_date}")

    df = fetch_ohlcv_df(
        table_name=table_name,
        schema=schema,
        time_column="datetime",
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for symbol='{symbol}' "
            f"(table={schema}.{table_name})."
        )

    logger.info(f"Fetched {len(df):,} rows for {symbol}.")
    return df


# ===========================================================================
# 2. RESAMPLING
# ===========================================================================

def resample_data(df_1m: pd.DataFrame, timehorizon: str) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV to ``timehorizon`` (e.g. ``'1h'``, ``'4h'``).

    Args:
        df_1m        : 1-minute OHLCV DataFrame.
        timehorizon  : Target timeframe string accepted by ``resample_ohlcv``.

    Returns:
        Resampled OHLCV DataFrame.
    """
    logger.info(f"Resampling to {timehorizon}…")
    df_resampled = resample_ohlcv(df_1m, timehorizon)
    logger.info(f"Resampled shape: {df_resampled.shape}")
    return df_resampled


# ===========================================================================
# 3. FEATURE ENGINEERING
# ===========================================================================

def build_features(
    df: pd.DataFrame,
    active_indicators: list[str],
) -> pd.DataFrame:
    """
    Compute technical indicator columns for ``df``.

    Args:
        df                : OHLCV DataFrame (resampled).
        active_indicators : List of indicator names read from config.

    Returns:
        DataFrame with indicator columns appended.
    """
    logger.info(f"Generating {len(active_indicators)} indicator(s)…")
    df_features = generate_features(df, active_indicators)
    logger.info(f"Feature shape after generation: {df_features.shape}")
    return df_features


# ===========================================================================
# 4. STATIONARITY
# ===========================================================================

def check_stationarity(
    df: pd.DataFrame,
    significance: float = 0.05,
) -> dict[str, bool]:
    """
    Run the Augmented Dickey-Fuller test on every eligible numeric column.

    A column is considered **stationary** if the ADF p-value is strictly
    below ``significance`` (default 5 %).

    Columns in ``_SKIP_STATIONARITY`` and columns whose names start with
    ``'CDL'`` (candlestick patterns) are excluded from testing.

    Args:
        df           : Feature DataFrame (must include ``'datetime'``).
        significance : ADF p-value threshold (default 0.05).

    Returns:
        Dict mapping column name → ``True`` (stationary) / ``False`` (not).
    """
    results: dict[str, bool] = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in _SKIP_STATIONARITY or col.startswith("CDL"):
            continue

        series = df[col].dropna()

        if len(series) < 20:
            logger.warning(
                f"[ADF] '{col}' has only {len(series)} non-NaN values — "
                "skipping stationarity test (treating as stationary)."
            )
            results[col] = True
            continue

        try:
            adf_stat, p_value, _, _, _, _ = adfuller(series, autolag="AIC")
            is_stationary = bool(p_value < significance)
            results[col] = is_stationary

            if not is_stationary:
                logger.info(
                    f"[ADF] '{col}' — NON-STATIONARY  "
                    f"(p={p_value:.4f}, stat={adf_stat:.4f})"
                )
        except Exception as exc:
            logger.warning(f"[ADF] '{col}' — test failed ({exc}); treating as stationary.")
            results[col] = True

    n_non = sum(1 for v in results.values() if not v)
    logger.info(
        f"[ADF] Tested {len(results)} column(s): "
        f"{len(results) - n_non} stationary, {n_non} non-stationary."
    )
    return results


def make_stationary(
    df: pd.DataFrame,
    stationarity_map: dict[str, bool],
    max_diffs: int = 2,
    significance: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Difference non-stationary columns until they pass the ADF test.

    Only columns flagged as non-stationary in ``stationarity_map`` are
    touched.  Each column is differenced up to ``max_diffs`` times.  If a
    column still fails after ``max_diffs`` rounds it is left as-is and a
    warning is logged.

    Rows that become NaN after differencing are dropped at the end.

    Args:
        df               : Feature DataFrame (pre-stationarity).
        stationarity_map : Output of :func:`check_stationarity`.
        max_diffs        : Maximum number of differences to apply (default 2).
        significance     : ADF p-value threshold (default 0.05).

    Returns:
        Tuple of:
            - Transformed DataFrame (NaN rows dropped, index reset).
            - Dict mapping column name → number of diffs applied (0 if none).
    """
    df = df.copy()
    diff_counts: dict[str, int] = {}

    non_stationary_cols = [col for col, ok in stationarity_map.items() if not ok]

    if not non_stationary_cols:
        logger.info("[make_stationary] All columns are already stationary — nothing to do.")
        return df, diff_counts

    logger.info(
        f"[make_stationary] Differencing {len(non_stationary_cols)} "
        f"non-stationary column(s)…"
    )

    for col in non_stationary_cols:
        if col not in df.columns:
            continue

        n_diffs = 0
        series = df[col].copy()

        for d in range(1, max_diffs + 1):
            series = series.diff()
            n_diffs = d

            clean = series.dropna()
            if len(clean) < 20:
                break

            try:
                _, p_value, _, _, _, _ = adfuller(clean, autolag="AIC")
                if p_value < significance:
                    logger.info(
                        f"[make_stationary] '{col}' became stationary "
                        f"after {d} diff(s)  (p={p_value:.4f})."
                    )
                    break
            except Exception:
                break

        else:
            logger.warning(
                f"[make_stationary] '{col}' still non-stationary after "
                f"{max_diffs} diff(s) — keeping as-is."
            )

        df[col] = series
        diff_counts[col] = n_diffs

    # Drop rows that became NaN due to differencing and reset index
    df = df.dropna().reset_index(drop=True)
    logger.info(f"[make_stationary] Shape after differencing & dropna: {df.shape}")

    return df, diff_counts


# ===========================================================================
# 5. COMBINED FEATURE + STATIONARITY PIPELINE
# ===========================================================================

def prepare_features(
    df: pd.DataFrame,
    active_indicators: list[str],
    adf_significance: float = 0.05,
    max_diffs: int = 2,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Full feature engineering + stationarity-fixing pipeline.

    Steps
    -----
    1. Generate technical indicator columns via :func:`build_features`.
    2. Run ADF tests via :func:`check_stationarity`.
    3. Difference non-stationary columns via :func:`make_stationary`.

    Args:
        df                : Resampled OHLCV DataFrame.
        active_indicators : Indicator names from config.
        adf_significance  : ADF p-value threshold (default 0.05).
        max_diffs         : Maximum differencing rounds (default 2).

    Returns:
        Tuple of:
            - Feature DataFrame (stationary, NaN rows dropped).
            - Diff-count dict (see :func:`make_stationary`).
    """
    # Step 1 — generate features
    df_features = build_features(df, active_indicators)

    # Step 2 — ADF stationarity test
    logger.info("Running ADF stationarity tests…")
    stationarity_map = check_stationarity(df_features, significance=adf_significance)

    # Step 3 — fix non-stationary columns
    df_stationary, diff_counts = make_stationary(
        df_features,
        stationarity_map,
        max_diffs=max_diffs,
        significance=adf_significance,
    )

    if diff_counts:
        logger.info(
            f"Differencing summary: "
            + ", ".join(f"{c}={d}d" for c, d in diff_counts.items())
        )
    else:
        logger.info("No differencing was necessary.")

    return df_stationary, diff_counts


# ===========================================================================
# 6. TARGET CREATION
# ===========================================================================
def create_classification_target(df: pd.DataFrame, window: int = 15, threshold: float = 0.001) -> pd.DataFrame:
    """
    Create classification targets for trading:
    1 = significant upward move
    0 = neutral
    -1 = significant downward move
    """
    future_close = df["close"].shift(-window)

    future_return = (future_close - df["close"]) / df["close"]

    df["target"] = np.where(
        future_return > threshold, 1,
        np.where(future_return < -threshold, -1, 0)
    )

    df = df.dropna().reset_index(drop=True)
    return df


def create_regression_target(
    df: pd.DataFrame,
    window: int = 15
) -> pd.DataFrame:
    """
    Creates a regression target as the actual future max price over a rolling window.
    
    df: DataFrame with 'close' column
    window: how many future periods to look ahead
    """
    # Compute future max over rolling window, shift to align with current row
    future_max = df["close"].rolling(window=window).max().shift(-window + 1)

    # Target is just the future max price
    df["target"] = future_max

    df = df.dropna().reset_index(drop=True)
    return df
    

def build_classification_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a classification target column to ``df``.

    Wraps :func:`create_classification_target` and drops ``open``, ``high``,
    ``low`` columns (not needed for tree-based classifiers).

    Args:
        df : Stationary feature DataFrame (output of :func:`prepare_features`).

    Returns:
        DataFrame with ``'target'`` column (values: -1, 0, 1).
    """
    df_clf = create_classification_target(df)
    df_clf = df_clf.drop(columns=["open", "high", "low"], errors="ignore")
    logger.info(
        f"Classification target created — shape: {df_clf.shape}, "
        f"class counts:\n{df_clf['target'].value_counts().to_dict()}"
    )
    return df_clf


def build_regression_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a regression target column to ``df``.

    Wraps :func:`create_regression_target` and drops ``open``, ``high``,
    ``low`` columns.

    Args:
        df : Stationary feature DataFrame (output of :func:`prepare_features`).

    Returns:
        DataFrame with continuous ``'target'`` column.
    """
    df_reg = create_regression_target(df)
    df_reg = df_reg.drop(columns=["open", "high", "low"], errors="ignore")
    logger.info(f"Regression target created — shape: {df_reg.shape}")
    return df_reg


# ===========================================================================
# 7. SPLIT DATE RESOLVER
# ===========================================================================

def resolve_split_date(df: pd.DataFrame, config: dict) -> str:
    """
    Determine the train/test split date from config.

    Priority
    --------
    1. ``train_ratio`` (float 0–1) — computes the split from the data itself
       so the boundary always falls at exactly that percentage of rows.
    2. ``split_date`` (str ``'YYYY-MM-DD'``) — hard calendar boundary, used
       only when ``train_ratio`` is null / absent.

    Args:
        df     : Feature DataFrame (must have a ``'datetime'`` column or a
                 DatetimeIndex).
        config : Loaded config dict.

    Returns:
        ISO date string ``'YYYY-MM-DD HH:MM:SS'`` ready to pass to
        ``train_model``.

    Raises:
        ValueError : If neither ``train_ratio`` nor ``split_date`` is present.
        ValueError : If ``train_ratio`` is not in the open interval (0, 1).
    """
    train_ratio = config.get("train_ratio")

    if train_ratio is not None:
        train_ratio = float(train_ratio)
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(
                f"train_ratio must be between 0 and 1 (exclusive), got {train_ratio}"
            )

        if "datetime" in df.columns:
            dt_series = pd.to_datetime(df["datetime"], utc=True).sort_values()
        elif isinstance(df.index, pd.DatetimeIndex):
            dt_series = df.index.sort_values().to_series()
        else:
            raise ValueError(
                "resolve_split_date: DataFrame must have a 'datetime' column "
                "or a DatetimeIndex to use train_ratio."
            )

        n_train    = int(len(dt_series) * train_ratio)
        split_ts   = dt_series.iloc[n_train]
        split_date = split_ts.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(
            f"train_ratio={train_ratio} → split at row {n_train}/{len(dt_series)} "
            f"→ split_date='{split_date}' "
            f"({train_ratio * 100:.0f}% train / {(1 - train_ratio) * 100:.0f}% test)"
        )
        return split_date

    # Fallback: hard calendar split_date from config
    split_date = config.get("split_date")
    if split_date is None:
        raise ValueError(
            "Config must contain either 'train_ratio' or 'split_date'."
        )
    logger.info(f"Using fixed split_date='{split_date}' from config.")
    return str(split_date)

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

_SINGLE_SERIES = frozenset({
    "RSI", "EMA", "SMA", "WMA", "DEMA", "TEMA", "TRIMA",
    "KAMA", "T3", "MOM", "ROC", "ROCP", "ROCR", "ROCR100",
    "LINEARREG", "LINEARREG_SLOPE", "LINEARREG_ANGLE",
    "LINEARREG_INTERCEPT", "STDDEV", "VAR", "TSF",
    "MA", "CMO",
})
_HLC_SERIES       = frozenset({"ATR", "NATR", "ADX", "ADXR", "DX", "CCI",
                                "PLUS_DI", "MINUS_DI", "PLUS_DM", "MINUS_DM", "WILLR"})
_MACD_SERIES      = frozenset({"MACD", "MACDEXT", "PPO", "APO", "TRIX"})
_BBAND_SERIES     = frozenset({"BBANDS"})
_STOCH_SERIES     = frozenset({"STOCH", "STOCHF", "STOCHRSI"})
_VOLUME_SERIES    = frozenset({"OBV", "AD", "ADOSC", "MFI"})
_AROON_SERIES     = frozenset({"AROON", "AROONOSC"})
_SAR_SERIES       = frozenset({"SAR", "SAREXT"})
_PRICE_TRANSFORMS = frozenset({"AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"})

def _arr(x) -> np.ndarray:
    """Return *x* as a flat float64 numpy array."""
    return np.asarray(x, dtype=np.float64).ravel()

def generate_features(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """
    Compute technical indicators and append them as new columns.

    All indicator arrays are collected into a single dict and concatenated
    once at the end to avoid repeated DataFrame copies.
    """
    n = len(df)

    close  = df["close"].to_numpy(dtype=np.float64)
    high   = df["high"].to_numpy(dtype=np.float64)
    low    = df["low"].to_numpy(dtype=np.float64)
    open_  = df["open"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    indicator_set   = set(indicators)
    cycle_series    = frozenset(i for i in indicator_set if i.startswith("HT_"))
    candle_patterns = frozenset(i for i in indicator_set if i.startswith("CDL"))

    new_cols: dict[str, np.ndarray] = {}

    def _store(key: str, arr):
        if key in new_cols:
            logger.warning(f"Indicator key collision: '{key}' will be overwritten.")
        new_cols[key] = _arr(arr)

    for ind in indicators:
        try:
            if ind in _SINGLE_SERIES:
                values, window = call_indicator(ind, close, timeperiod=14)
                _store(f"{ind}_{window}", values)

            elif ind == "MAMA":
                (mama_raw, fama_raw), _ = call_indicator(
                    "MAMA", close, fastlimit=0.5, slowlimit=0.05
                )
                mama = _arr(mama_raw)
                fama = _arr(fama_raw)
                if len(mama) != n:
                    pad = n - len(mama)
                    if pad > 0:
                        mama = np.concatenate([np.full(pad, np.nan), mama])
                        fama = np.concatenate([np.full(pad, np.nan), fama])
                    else:
                        mama = mama[-n:]
                        fama = fama[-n:]
                _store("MAMA", mama)
                _store("FAMA", fama)

            elif ind == "MIDPOINT":
                _store("MIDPOINT_14", call_indicator("MIDPOINT", close, timeperiod=14)[0])

            elif ind == "MIDPRICE":
                _store("MIDPRICE_14", call_indicator("MIDPRICE", high, low, timeperiod=14)[0])

            elif ind == "BOP":
                _store("BOP", call_indicator("BOP", open=open_, high=high, low=low, close=close)[0])

            elif ind == "TRANGE":
                _store("TRANGE", call_indicator("TRANGE", high=high, low=low, close=close)[0])

            elif ind in _HLC_SERIES:
                if ind in {"MINUS_DM", "PLUS_DM"}:
                    _store(ind, call_indicator(ind, high=high, low=low, timeperiod=14)[0])
                else:
                    values, window = call_indicator(ind, high=high, low=low, close=close, timeperiod=14)
                    _store(f"{ind}_{window}", values)

            elif ind in _MACD_SERIES:
                result, _ = call_indicator(ind, close)
                if isinstance(result, (tuple, list)):
                    for i, arr in enumerate(result):
                        _store(f"{ind}_{i}", arr)
                else:
                    _store(f"{ind}_0", result)

            elif ind in _BBAND_SERIES:
                (upper, mid, lower), _ = call_indicator("BBANDS", close, timeperiod=20)
                _store("BB_UPPER",  upper)
                _store("BB_MIDDLE", mid)
                _store("BB_LOWER",  lower)

            elif ind in _STOCH_SERIES:
                if ind == "STOCHRSI":
                    (slowk, slowd), _ = call_indicator(ind, close)
                else:
                    (slowk, slowd), _ = call_indicator(ind, high=high, low=low, close=close)
                _store(f"{ind}_K", slowk)
                _store(f"{ind}_D", slowd)

            elif ind in _VOLUME_SERIES:
                if ind == "OBV":
                    _store(ind, call_indicator(ind, close, volume)[0])
                elif ind == "AD":
                    _store(ind, call_indicator(ind, high=high, low=low, close=close, volume=volume)[0])
                elif ind == "ADOSC":
                    _store(ind, call_indicator("ADOSC", high=high, low=low, close=close, volume=volume)[0])
                elif ind == "MFI":
                    _store(f"{ind}_14", call_indicator("MFI", high=high, low=low, close=close, volume=volume, timeperiod=14)[0])

            elif ind in _AROON_SERIES:
                if ind == "AROON":
                    (aroon_up, aroon_down), _ = call_indicator(ind, high=high, low=low, timeperiod=14)
                    _store("AROON_UP",   aroon_up)
                    _store("AROON_DOWN", aroon_down)
                else:
                    _store("AROONOSC", call_indicator(ind, high=high, low=low, timeperiod=14)[0])

            elif ind in _SAR_SERIES:
                _store(ind, call_indicator(ind, high=high, low=low)[0])

            elif ind == "AVGPRICE":
                _store(ind, call_indicator(ind, open=open_, high=high, low=low, close=close)[0])
            elif ind == "MEDPRICE":
                _store(ind, call_indicator(ind, high=high, low=low)[0])
            elif ind in {"TYPPRICE", "WCLPRICE"}:
                _store(ind, call_indicator(ind, high=high, low=low, close=close)[0])

            elif ind in cycle_series:
                result, _ = call_indicator(ind, close)
                if isinstance(result, (tuple, list)):
                    for i, arr in enumerate(result):
                        _store(f"{ind}_{i}", arr)
                else:
                    _store(ind, result)

            elif ind in candle_patterns:
                _store(ind, call_indicator(ind, open=open_, high=high, low=low, close=close)[0])

            else:
                logger.warning(f"Unsupported indicator: {ind}")

        except Exception as exc:
            logger.error(f"Indicator '{ind}' failed: {exc}")

    if new_cols:
        safe = {
            k: v for k, v in (
                (k, np.asarray(v, dtype=np.float64).ravel()) for k, v in new_cols.items()
            )
            if v.shape == (n,)
        }
        skipped = set(new_cols) - set(safe)
        for k in skipped:
            logger.warning(f"Skipping column '{k}': length mismatch.")
        if safe:
            df = pd.concat([df, pd.DataFrame(safe, index=df.index)], axis=1)

    return df