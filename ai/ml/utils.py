import pandas as pd
import numpy as np

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
        # Use predictions directly
        signals = preds

    elif model_type == "regressor":
        # Auto-compute threshold if not provided
        if threshold is None:
            threshold = k * np.std(preds)
        
        # Convert continuous predictions into discrete signals
        signals = np.where(preds > threshold, 1,
                  np.where(preds < -threshold, -1, 0))

    else:
        raise ValueError("model_type must be 'classifier' or 'regressor'")

    # Build prediction DataFrame
    df_predictions = pd.DataFrame({
        "datetime": df.loc[test_index, "datetime"].values,
        "signals": signals
    })

    return df_predictions