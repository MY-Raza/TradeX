from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
)

from TradeX.utils.common.logs import get_logger
from TradeX.sentiments.ml.data.preprocessing import PreparedData
from TradeX.sentiments.ml.config import CLASSIFIER_PARAMS, REGRESSOR_PARAMS

logger = get_logger("models")


# =========================================================
# RESULT CONTAINER
# =========================================================

@dataclass
class ModelBundle:
    """Carries trained models and all prediction artefacts."""

    classifier:  RandomForestClassifier
    regressor:   RandomForestRegressor

    # --- Test-set predictions (used for signal generation) ---
    class_preds_test:  np.ndarray          # shape (n_test,)  int   {0, 1}
    class_proba_test:  np.ndarray          # shape (n_test, 2) float
    reg_preds_test:    np.ndarray          # shape (n_test,)  float

    # --- Val-set predictions (used for hyper-param evaluation) ---
    class_preds_val:   np.ndarray
    class_proba_val:   np.ndarray
    reg_preds_val:     np.ndarray

    # --- Feature importances ---
    clf_importances: pd.Series = field(default_factory=pd.Series)
    reg_importances: pd.Series = field(default_factory=pd.Series)


# =========================================================
# TRAINING
# =========================================================

def train_classification_model(
    data: PreparedData,
) -> RandomForestClassifier:
    """
    Fit a RandomForestClassifier on the training split.

    Parameters
    ----------
    data : PreparedData
        Output of preprocessing.prepare_features().

    Returns
    -------
    Fitted RandomForestClassifier.
    """
    logger.info("Training RandomForestClassifier …")
    logger.info(f"  Params: {CLASSIFIER_PARAMS}")

    clf = RandomForestClassifier(**CLASSIFIER_PARAMS)
    clf.fit(data.X_train, data.y_class_train)

    logger.info("  Classifier training complete.")
    return clf


def train_regression_model(
    data: PreparedData,
) -> RandomForestRegressor:
    """
    Fit a RandomForestRegressor on the training split.

    Parameters
    ----------
    data : PreparedData
        Output of preprocessing.prepare_features().

    Returns
    -------
    Fitted RandomForestRegressor.
    """
    logger.info("Training RandomForestRegressor …")
    logger.info(f"  Params: {REGRESSOR_PARAMS}")

    reg = RandomForestRegressor(**REGRESSOR_PARAMS)
    reg.fit(data.X_train, data.y_reg_train)

    logger.info("  Regressor training complete.")
    return reg


# =========================================================
# EVALUATION
# =========================================================

def evaluate_models(
    clf: RandomForestClassifier,
    reg: RandomForestRegressor,
    data: PreparedData,
) -> ModelBundle:
    """
    Generate predictions on val and test sets, compute metrics, log results.

    Parameters
    ----------
    clf : RandomForestClassifier
    reg : RandomForestRegressor
    data : PreparedData

    Returns
    -------
    ModelBundle
        All predictions + feature importances packed together.
    """
    logger.info("=" * 64)
    logger.info("EVALUATION")
    logger.info("=" * 64)

    # ── Classification ─────────────────────────────────────
    class_preds_val  = clf.predict(data.X_val)
    class_proba_val  = clf.predict_proba(data.X_val)
    class_preds_test = clf.predict(data.X_test)
    class_proba_test = clf.predict_proba(data.X_test)

    val_acc  = accuracy_score(data.y_class_val,  class_preds_val)
    test_acc = accuracy_score(data.y_class_test, class_preds_test)

    logger.info(f"  Classifier — Val Accuracy : {val_acc:.4f}")
    logger.info(f"  Classifier — Test Accuracy: {test_acc:.4f}")
    logger.info(
        "  Val classification report:\n"
        + classification_report(data.y_class_val, class_preds_val, digits=4)
    )

    # ── Regression ─────────────────────────────────────────
    reg_preds_val  = reg.predict(data.X_val)
    reg_preds_test = reg.predict(data.X_test)

    val_rmse  = np.sqrt(mean_squared_error(data.y_reg_val,  reg_preds_val))
    test_rmse = np.sqrt(mean_squared_error(data.y_reg_test, reg_preds_test))

    logger.info(f"  Regressor — Val  RMSE: {val_rmse:.6f}")
    logger.info(f"  Regressor — Test RMSE: {test_rmse:.6f}")

    # ── Feature Importances ────────────────────────────────
    clf_importances = pd.Series(
        clf.feature_importances_, index=data.feature_cols
    ).sort_values(ascending=False)

    reg_importances = pd.Series(
        reg.feature_importances_, index=data.feature_cols
    ).sort_values(ascending=False)

    logger.info("  Top 10 Classifier feature importances:")
    for feat, imp in clf_importances.head(10).items():
        logger.info(f"    {feat:<50} {imp:.5f}")

    logger.info("  Top 10 Regressor feature importances:")
    for feat, imp in reg_importances.head(10).items():
        logger.info(f"    {feat:<50} {imp:.5f}")

    logger.info("=" * 64)

    return ModelBundle(
        classifier=clf,
        regressor=reg,
        class_preds_test=class_preds_test,
        class_proba_test=class_proba_test,
        reg_preds_test=reg_preds_test,
        class_preds_val=class_preds_val,
        class_proba_val=class_proba_val,
        reg_preds_val=reg_preds_val,
        clf_importances=clf_importances,
        reg_importances=reg_importances,
    )