"""
indicators.py

This module provides wrapper functions for all TA-Lib indicators.
Each indicator is dynamically exposed with proper documentation.

Author: Yasir Raza
"""

import talib
import numpy as np
import inspect


def get_all_indicators():
    """
    Returns a dictionary of all available TA-Lib indicators.

    Returns
    -------
    dict
        {indicator_name: function_reference}
    """
    return talib.get_functions()


def call_indicator(name: str, **kwargs):
    """
    Call any TA-Lib indicator dynamically by name.

    Parameters
    ----------
    name : str
        Name of the TA-Lib indicator (e.g. 'RSI', 'MACD')
    **kwargs
        Inputs required by the indicator (e.g. close, high, low)

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Indicator output
    """
    name = name.upper()

    if not hasattr(talib, name):
        raise ValueError(f"Indicator '{name}' not found in TA-Lib.")

    indicator_func = getattr(talib, name)
    return indicator_func(**kwargs)


def indicator_help(name: str):
    """
    Print full documentation and required parameters of an indicator.

    Parameters
    ----------
    name : str
        Indicator name
    """
    name = name.upper()
    func = getattr(talib, name)

    print("=" * 60)
    print(f"Indicator: {name}")
    print("=" * 60)
    print(inspect.getdoc(func))
