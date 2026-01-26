"""
indicators.py

Dynamic TA-Lib indicator wrapper with automatic argument mapping.
"""

import talib
import inspect

# Standard OHLCV ordering used by TA-Lib
INPUT_ORDER = ["open", "high", "low", "close", "volume"]


def call_indicator(name: str, **kwargs):
    """
    Call any TA-Lib indicator using human-friendly keyword arguments.

    Parameters
    ----------
    name : str
        Indicator name (e.g. RSI, MACD, EMA)
    **kwargs
        Indicator inputs (close, high, low, etc.)

    Returns
    -------
    numpy.ndarray or tuple
        Indicator output
    """
    name = name.upper()

    if not hasattr(talib, name):
        raise ValueError(f"Indicator '{name}' not found in TA-Lib")

    func = getattr(talib, name)
    spec = inspect.getfullargspec(func)

    positional_args = []
    keyword_args = {}

    for arg in spec.args:
        if arg in INPUT_ORDER and arg in kwargs:
            positional_args.append(kwargs[arg])
        elif arg in kwargs:
            keyword_args[arg] = kwargs[arg]

    return func(*positional_args, **keyword_args)


def indicator_help(name: str):
    """
    Print indicator documentation.
    """
    name = name.upper()
    func = getattr(talib, name)
    print(inspect.getdoc(func))
