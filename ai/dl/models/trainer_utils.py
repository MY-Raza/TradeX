from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "normalise_datetime",
    "ensure_log_return",
    "rolling_train_test_split",
    "check_min_rows",
    "concat_dedup_sort",
    "build_pl_trainer_kwargs",
    "make_test_artifacts",
    # helpers (exported for testing)
    "_log_diff",
    "_signed_log_diff",
]

# ---------------------------------------------------------------------------
# 1. Datetime normalisation
# ---------------------------------------------------------------------------

def normalise_datetime(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """
    Ensure the DataFrame is indexed by a tz-naive UTC DatetimeIndex.

    Handles two cases:
      a) A 'datetime' column exists  → parse, convert, set as index.
      b) The index is already a DatetimeIndex  → strip tz if present.

    Args:
        df   : Input DataFrame. Must have either a 'datetime' column or a
               DatetimeIndex.
        copy : If True (default) work on a copy so the caller's frame is
               not mutated.

    Returns:
        DataFrame with a tz-naive DatetimeIndex named 'datetime'.

    Raises:
        ValueError : If neither a 'datetime' column nor a DatetimeIndex
                     can be found.
    """
    if copy:
         

        if "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"])
            if dt.dt.tz is None:
                dt = dt.dt.tz_localize("UTC")
            df["datetime"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)
            df = df.set_index("datetime")

        elif isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            df.index = idx

        else:
            raise ValueError(
                "normalise_datetime: DataFrame must have a 'datetime' column "
                "or a DatetimeIndex."
            )

    df.index.name = "datetime"
    return df


# ---------------------------------------------------------------------------
# 2. Log-return column derivation
# ---------------------------------------------------------------------------

def _signed_log_diff(series: pd.Series) -> pd.Series:
    """
    Signed log-difference for series that may contain non-positive values.

    Formula: sign(x) * log(|x| + 1), then .diff()

    This preserves the sign of the original values and is defined for all
    real numbers (including zero and negatives), unlike plain log().

    Args:
        series : Numeric pandas Series.

    Returns:
        First-differenced signed-log series (first element is NaN).
    """
    return series.apply(lambda x: np.sign(x) * np.log(abs(x) + 1)).diff()


def _log_diff(series: pd.Series) -> pd.Series:
    """
    Standard log-difference: log(x).diff().

    Args:
        series : Numeric pandas Series with strictly positive values.

    Returns:
        First-differenced log series (first element is NaN).
    """
    return np.log(series).diff()


def ensure_log_return(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    suffix: str = "_lr",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Compute log-difference returns for every numeric column in *df* and
    append them as ``<col><suffix>`` columns.

    For columns that are **strictly positive** everywhere, the standard
    log-difference is used::

        lr = log(x[t]) - log(x[t-1])

    For columns that contain **zero or negative values**, the signed
    log-difference is used instead (defined for all real numbers)::

        lr = sign(x[t]) * log(|x[t]| + 1)  -  sign(x[t-1]) * log(|x[t-1]| + 1)

    Special back-compat behaviour
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * If ``columns`` is not given (default), **all numeric columns** are
      processed — *except* those whose name already ends with ``suffix``
      (avoids double-differencing previously computed return columns).
    * A column named ``'close'`` always produces a ``'log_return'`` alias in
      addition to its normal ``'close_lr'`` entry so that downstream code that
      reads ``df['log_return']`` keeps working without modification.
    * If ``'log_return'`` is already present **and** ``columns`` is None, the
      function returns immediately (full back-compat shortcut).

    Args:
        df      : Input DataFrame.  Modified in-place when *inplace* is True.
        columns : Explicit list of source columns to process.  Defaults to all
                  numeric columns that don't already end with *suffix*.
        suffix  : Suffix appended to each source column name.  Default ``'_lr'``.
        inplace : If True (default) add columns to *df* directly.  If False,
                  return a copy.

    Returns:
        The (possibly modified) DataFrame with new ``<col><suffix>`` columns.

    Raises:
        ValueError : If an explicitly requested column is missing from *df*.
    """
    # --- Back-compat shortcut -------------------------------------------
    if columns is None and "log_return" in df.columns:
        return df

    if not inplace:
        df = df.copy()

    # --- Resolve target columns -----------------------------------------
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Skip columns that are already log-return / suffix columns,
        # and exclude any column named 'datetime' (it's a time index, not a feature).
        _DATETIME_NAMES = {"datetime", "date", "time", "timestamp"}
        columns = [
            c for c in numeric_cols
            if not c.endswith(suffix) and c.lower() not in _DATETIME_NAMES
        ]
    else:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"ensure_log_return: columns not found in DataFrame: {missing}"
            )

    if not columns:
        raise ValueError(
            "ensure_log_return: no numeric source columns found to process."
        )

    # --- Compute per-column log-differences -----------------------------
    for col in columns:
        out_col = f"{col}{suffix}"
        if out_col in df.columns:
            # Already computed — skip to avoid overwriting.
            continue

        series = df[col]

        # Use signed-log-diff when any value is <= 0; plain log-diff otherwise.
        if (series <= 0).any():
            df[out_col] = _signed_log_diff(series)
        else:
            df[out_col] = _log_diff(series)

    # --- Back-compat alias: 'close' → 'log_return' ----------------------
    if "close" in columns and "log_return" not in df.columns:
        df["log_return"] = df[f"close{suffix}"]

    return df


# ---------------------------------------------------------------------------
# 3. Rolling train / test split
# ---------------------------------------------------------------------------

def rolling_train_test_split(
    df_target: pd.DataFrame,
    split_date: str,
    rolling_rows: Optional[int],
    logger: Optional[logging.Logger] = None,
    label: str = "model",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a sorted, single-column target DataFrame at split_date and
    optionally cap the training portion to rolling_rows most-recent rows.

    Args:
        df_target    : Sorted, tz-naive DatetimeIndex DataFrame (one column).
        split_date   : ISO date string, e.g. '2024-01-01'.
        rolling_rows : Maximum number of training rows to keep.  Pass 0
                       or None to disable the cap (may be slow on long series).
        logger       : Optional logger for the rolling-window info message.
        label        : Short model name used in log messages.

    Returns:
        (df_train, df_test)

    Raises:
        ValueError : If either split produces an empty frame.
    """
    split_ts = pd.Timestamp(split_date)
    if split_ts.tz is not None:
        split_ts = split_ts.tz_convert("UTC").tz_localize(None)

    df_train = df_target[df_target.index < split_ts]
    df_test  = df_target[df_target.index >= split_ts]

    if df_train.empty:
        raise ValueError(
            f"{label}: no training rows before split_date '{split_date}'."
        )
    if df_test.empty:
        raise ValueError(
            f"{label}: no test rows on/after split_date '{split_date}'."
        )

    if rolling_rows and len(df_train) > rolling_rows:
        n_total   = len(df_train)
        n_dropped = n_total - rolling_rows
        df_train  = df_train.iloc[-rolling_rows:]
        msg = (
            f"{label} rolling window: using last {rolling_rows} of "
            f"{n_total} training rows (dropped {n_dropped} older rows)."
        )
        if logger is not None:
            logger.info(msg)
        else:
            logging.getLogger(label).info(msg)

    return df_train, df_test


# ---------------------------------------------------------------------------
# 4. Minimum-row guard
# ---------------------------------------------------------------------------

def check_min_rows(
    df_train: pd.DataFrame,
    min_rows: int,
    context: str = "",
) -> None:
    """
    Raise ValueError when the training DataFrame has fewer rows than required.

    Args:
        df_train : Training DataFrame after any rolling-window slice.
        min_rows : Minimum acceptable number of rows.
        context  : Human-readable description inserted into the error message
                   (e.g. "ARIMA(5,1,0)" or "NBEATS input+output=25").

    Raises:
        ValueError : If len(df_train) < min_rows.
    """
    if len(df_train) < min_rows:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}not enough training data — need at least {min_rows} "
            f"rows, got {len(df_train)}."
        )


# ---------------------------------------------------------------------------
# 5. Concat + dedup + sort
# ---------------------------------------------------------------------------

def concat_dedup_sort(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concatenate train and test frames, remove duplicate index entries
    (keeping the last occurrence), and sort by index.

    This is required before building a Darts TimeSeries from the combined
    window when the split boundary may produce a shared timestamp.

    Args:
        df_train : Training portion of the target DataFrame.
        df_test  : Test portion of the target DataFrame.

    Returns:
        A single sorted, de-duplicated DataFrame.
    """
    combined = pd.concat([df_train, df_test])
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


# ---------------------------------------------------------------------------
# 6. PyTorch-Lightning trainer kwargs builder
# ---------------------------------------------------------------------------

def build_pl_trainer_kwargs(
    base_kwargs: Optional[dict] = None,
    use_early_stopping: bool = True,
    monitor: str = "train_loss",
    patience: int = 3,
    min_delta: float = 1e-4,
) -> dict:
    """
    Build the pl_trainer_kwargs dict consumed by Darts DL models.

    Sets sensible CPU defaults and optionally appends an EarlyStopping
    callback.  The caller's dict is never mutated — a copy is made first.

    Args:
        base_kwargs         : Existing pl_trainer_kwargs from the caller
                              (e.g. popped from **kwargs).  Defaults are
                              applied only when the key is absent.
        use_early_stopping  : Whether to append an EarlyStopping callback.
        monitor             : Metric to monitor (default 'train_loss').
        patience            : EarlyStopping patience (default 3).
        min_delta           : Minimum improvement to reset patience counter.

    Returns:
        A new dict ready to pass as pl_trainer_kwargs=... to the Darts model.
    """
    kwargs = (base_kwargs or {})  

    kwargs.setdefault("accelerator",          "cpu")
    kwargs.setdefault("enable_progress_bar",  False)
    kwargs.setdefault("enable_model_summary", False)
    kwargs.setdefault("log_every_n_steps",    10)

    if use_early_stopping:
        try:
            from pytorch_lightning.callbacks import EarlyStopping
            callbacks = kwargs.pop("callbacks", [])
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    patience=patience,
                    min_delta=min_delta,
                    mode="min",
                )
            )
            kwargs["callbacks"] = callbacks
        except ImportError:
            warnings.warn(
                "build_pl_trainer_kwargs: pytorch_lightning not available; "
                "EarlyStopping not added.",
                RuntimeWarning,
                stacklevel=2,
            )

    return kwargs


# ---------------------------------------------------------------------------
# 7. Test artifacts builder
# ---------------------------------------------------------------------------

def make_test_artifacts(
    n_train: int,
    test_series,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build the (test_index, df_test) tuple returned by every trainer.

    Args:
        n_train     : Number of training rows in the windowed dataset.
                      test_index starts immediately after this offset.
        test_series : Darts TimeSeries for the test period.

    Returns:
        test_index  : 1-D integer NumPy array of shape (n_test,).
        df_test     : Empty pd.DataFrame indexed by test_series.time_index.
    """
    n_test     = len(test_series)
    test_index = np.arange(n_train, n_train + n_test)
    df_test    = pd.DataFrame(index=test_series.time_index)
    return test_index, df_test