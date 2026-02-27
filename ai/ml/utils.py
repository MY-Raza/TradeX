import pandas as pd
import numpy as np
from TradeX.backtest.backtest import BackTest


def prepare_predictions(df, preds, test_index, model_type, threshold=None, k=0.5):
    """
    Prepare a predictions DataFrame for backtesting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original price DataFrame with 'datetime' column.
        
    preds : np.ndarray
        Model predictions (classifier or regressor outputs)
        
    test_index : array-like
        Indices of the test set in df
        
    model_type : str
        'classifier' or 'regressor'
        
    threshold : float or None
        If None and model_type='regressor', automatically computed as k*std(preds)
        
    k : float
        Multiplier for standard deviation when auto thresholding (default 0.5)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'signals'] with -1, 0, 1 signals
    """
    
    # Ensure datetime is UTC-aware
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    if model_type == "classifier":
        # Convert probability into 3 trading zones
        upper = 0.55
        lower = 0.45

        signals = np.where(preds > upper, 1,
              np.where(preds < lower, -1, 0))

        print("Unique signals:", np.unique(signals))

    elif model_type == "regressor":
        # Auto-compute threshold if not provided
        if threshold is None:
            threshold = k * np.std(preds)
        # Convert continuous predictions into discrete signals
        signals = np.where(preds > threshold, 1,
                  np.where(preds < -threshold, -1, 0))
        print("Unique signals:", np.unique(signals))
        

    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    # Build prediction DataFrame
    df_predictions = pd.DataFrame({
        "datetime": df.loc[test_index, "datetime"].values,
        "signals": signals
    })

    return df_predictions

def pnl_permutation_importance(
    model,
    X_test,
    df,
    df_1m,
    base_pnl,
    model_type="classifier",
    k=0.5,
    threshold=None,
    n_repeats=3
):

    results = []


    for col in X_test.columns:
        pnl_scores = []

        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)

            # ----------------------------
            # Generate predictions
            # ----------------------------
            if model_type == "classifier":
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_perm)[:, 1]
                else:
                    preds = model.predict(X_perm)
            else:
                preds = model.predict(X_perm)

            # ----------------------------
            # Convert → trades
            # ----------------------------
            df_preds = prepare_predictions(
                df,
                preds,
                X_perm.index,
                model_type=model_type,
                threshold=threshold,
                k=k
            )

            df_preds["datetime"] = pd.to_datetime(
                df_preds["datetime"], utc=True
            )

            bt = BackTest(
                df_1m,
                df_preds,
                take_profit=3,
                stop_loss=1
            )
            _, _, pnl = bt.run()

            pnl_scores.append(pnl)

        pnl_drop = base_pnl - np.mean(pnl_scores)

        results.append({
            "feature": col,
            "pnl_drop": pnl_drop
        })

    return pd.DataFrame(results).sort_values("pnl_drop", ascending=False)

def extract_important_features(pnl_importance_wide: pd.DataFrame, model_name: str):
    # Drop pnl column
    feature_row = pnl_importance_wide.drop(columns=["pnl"], errors="ignore").iloc[0]

    # Keep only features with value > 0
    important_features = feature_row[feature_row > 0].index.tolist()

    return pd.DataFrame({
        "model_name": [model_name],
        "important_features": [important_features]
    })