from __future__ import annotations

import numpy as np
import pandas as pd

from TradeX.utils.common.logs import get_logger
from TradeX.sentiments.ml.model import ModelBundle
from TradeX.sentiments.ml.data.preprocessing import PreparedData
from TradeX.sentiments.ml.config import SIGNAL_THRESHOLD, MIN_CLASS_PROBABILITY, DATETIME_COL

logger = get_logger("signals")


# =========================================================
# PUBLIC API
# =========================================================

def generate_signals(
    bundle: ModelBundle,
    data:   PreparedData,
    threshold:        float | None = None,
    min_class_proba:  float | None = None,
) -> pd.DataFrame:
    """
    Combine classification and regression predictions into trading signals.

    Parameters
    ----------
    bundle : ModelBundle
        Output of models.evaluate_models().
    data : PreparedData
        Output of preprocessing.prepare_features() — provides dt_test timestamps.
    threshold : float, optional
        Override SIGNAL_THRESHOLD from config.
    min_class_proba : float, optional
        Override MIN_CLASS_PROBABILITY from config.

    Returns
    -------
    pd.DataFrame
        Columns: [datetime, signals]
        datetime — UTC-aware, hourly, matching the test split rows.
        signals  — int8 in {-1, 0, 1}.
    """
    thr   = threshold       if threshold       is not None else SIGNAL_THRESHOLD
    mcp   = min_class_proba if min_class_proba is not None else MIN_CLASS_PROBABILITY

    class_preds = bundle.class_preds_test          # shape (n,)  int {0,1}
    reg_preds   = bundle.reg_preds_test             # shape (n,)  float
    class_proba = bundle.class_proba_test           # shape (n,2) float

    if thr == SIGNAL_THRESHOLD:   # only if not manually overridden
        thr = float(np.percentile(np.abs(reg_preds), 70))
        logger.info(f"  Auto-calibrated threshold from reg_preds: {thr:.6f}")

    n = len(class_preds)
    signals = np.zeros(n, dtype=np.int8)

    # ── LONG condition ────────────────────────────────────────
    long_mask = (
        (class_preds == 1) &
        (reg_preds   >  thr) &
        (class_proba[:, 1] >= mcp)
    )

    # ── SHORT condition ───────────────────────────────────────
    short_mask = (
        (class_preds == 0) &
        (reg_preds   < -thr) &
        (class_proba[:, 0] >= mcp)
    )

    signals[long_mask]  =  1
    signals[short_mask] = -1

    # ── Logging ───────────────────────────────────────────────
    n_long    = int(long_mask.sum())
    n_short   = int(short_mask.sum())
    n_neutral = int(n - n_long - n_short)

    logger.info(
        f"Signal generation complete (n={n}) → "
        f"LONG: {n_long} ({n_long/n:.1%}) | "
        f"SHORT: {n_short} ({n_short/n:.1%}) | "
        f"NEUTRAL: {n_neutral} ({n_neutral/n:.1%})"
    )
    logger.info(
        f"  Config — threshold: {thr}  |  min_class_proba: {mcp}"
    )

    if n_long == 0 and n_short == 0:
        logger.warning(
            "All signals are neutral! "
            "Consider lowering SIGNAL_THRESHOLD or MIN_CLASS_PROBABILITY."
        )

    # ── Build output DataFrame ────────────────────────────────
    df_signals = pd.DataFrame({
        DATETIME_COL: data.dt_test.values,
        "signals":    signals,
    })

    # Ensure datetime is UTC-aware
    df_signals[DATETIME_COL] = pd.to_datetime(df_signals[DATETIME_COL], utc=True)

    # Sort ascending (should already be, but be safe)
    df_signals = df_signals.sort_values(DATETIME_COL).reset_index(drop=True)

    return df_signals