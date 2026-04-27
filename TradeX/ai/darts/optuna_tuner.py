from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger("optuna_tuner")


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def _arima_space(trial) -> dict[str, Any]:
    return {
        "p":                trial.suggest_int("p", 1, 10),
        "d":                0,          # log_return is I(0) — never difference
        "q":                trial.suggest_int("q", 0, 3),
        # ARIMA pred std ~1e-5; search around that scale
        "signal_threshold": trial.suggest_float("signal_threshold", 1e-6, 5e-5, log=True),
        # Seasonal component: try with / without 24h cycle
        "seasonal_order": (
            trial.suggest_int("P", 0, 2),
            0,
            trial.suggest_int("Q_s", 0, 1),
            24,
        ),
    }


def _varima_space(trial) -> dict[str, Any]:
    return {
        "p": trial.suggest_int("p", 1, 6),   # FIX: was 1–4; raised to 6
        "d": 0,                               # log-returns are I(0)
        "q": 0,                               # VARMA(q>0) is non-identifiable
        # FIX: raised rolling_rows range to match new 8640-row default.
        # Allowing very short windows (1000) produced near-zero close_lr std
        # which meant the signal_threshold silenced almost everything.
        "rolling_rows": trial.suggest_int("rolling_rows", 4_000, 10_000, step=500),
        # FIX: lowered search range — VARIMA close_lr std on a 12-month BTC
        # window is ~3–8e-6, so the previous lower bound of 1e-6 was too tight
        # and the upper bound of 5e-5 was too loose (kills all signals).
        "signal_threshold": trial.suggest_float("signal_threshold", 5e-7, 1e-5, log=True),
    }


def _nbeats_space(trial) -> dict[str, Any]:
    input_chunk = trial.suggest_int("input_chunk_length", 24, 120, step=24)
    output_chunk = trial.suggest_int("output_chunk_length", 1, max(1, input_chunk // 4), step=1)
    return {
        "input_chunk_length":  input_chunk,
        "output_chunk_length": output_chunk,
        "n_epochs":            trial.suggest_int("n_epochs", 10, 60, step=10),
        "batch_size":          trial.suggest_categorical("batch_size", [32, 64, 128]),
        "num_blocks":          trial.suggest_int("num_blocks", 1, 4),
        "num_layers":          trial.suggest_int("num_layers", 1, 4),
        "layer_widths":        trial.suggest_categorical("layer_widths", [64, 128, 256]),
        "rolling_rows":        trial.suggest_int("rolling_rows", 1000, 6000, step=500),
        # NBEATS pred std ~1e-4; search 1e-5 to 5e-4
        "signal_threshold":    trial.suggest_float("signal_threshold", 1e-5, 5e-4, log=True),
    }


def _transformer_space(trial) -> dict[str, Any]:
    d_model_choices = [32, 64, 128]
    d_model = trial.suggest_categorical("d_model", d_model_choices)
    valid_nheads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
    nhead = trial.suggest_categorical("nhead", valid_nheads)
    input_chunk = trial.suggest_int("input_chunk_length", 24, 96, step=24)
    output_chunk = trial.suggest_int("output_chunk_length", 1, max(1, input_chunk // 4), step=1)
    return {
        "input_chunk_length":  input_chunk,
        "output_chunk_length": output_chunk,
        "d_model":             d_model,
        "nhead":               nhead,
        "num_encoder_layers":  trial.suggest_int("num_encoder_layers", 1, 4),
        "num_decoder_layers":  trial.suggest_int("num_decoder_layers", 1, 4),
        "dim_feedforward":     trial.suggest_categorical("dim_feedforward", [64, 128, 256]),
        "dropout":             trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        "n_epochs":            trial.suggest_int("n_epochs", 10, 50, step=10),
        "batch_size":          trial.suggest_categorical("batch_size", [32, 64]),
        "rolling_rows":        trial.suggest_int("rolling_rows", 1000, 4000, step=500),
        "signal_threshold":    trial.suggest_float("signal_threshold", 1e-4, 1e-3, log=True),
    }


_SPACE_BUILDERS = {
    "arima":       _arima_space,
    "varima":      _varima_space,
    "nbeats":      _nbeats_space,
    "transformer": _transformer_space,
}


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------

def _make_objective(
    model_name: str,
    df: pd.DataFrame,
    df_1m: pd.DataFrame,
    split_date: str,
    take_profit: float,
    stop_loss: float,
    high_performance: bool,
):
    """Return a callable ``objective(trial) -> float`` for Optuna."""

    from TradeX.ai.dl.models.model_trainer import train_model
    from TradeX.ai.ml.utils import prepare_predictions
    from TradeX.backtest.backtest import BackTest

    space_builder = _SPACE_BUILDERS[model_name]

    def objective(trial) -> float:
        params = space_builder(trial)

        try:
            # Disable recursive inline Optuna inside each trial to avoid
            # infinite nesting (trial → train_model → inline_optuna → trial…).
            model, preds, test_index, df_test = train_model(
                model_type="dl",
                model_name=model_name,
                df=df.copy(),
                split_date=split_date,
                high_performance=high_performance,
                model_params=params,
                run_inline_optuna=False,   # ← prevents recursion
            )

            sig_thresh = getattr(model, "signal_threshold", params.get("signal_threshold", 3e-4))

            df_predictions = prepare_predictions(
                df, preds, test_index,
                model_type="dl_darts",
                threshold=sig_thresh,
            )

            if "datetime" in df_predictions.columns:
                dt_col = pd.to_datetime(df_predictions["datetime"])
                if dt_col.dt.tz is None:
                    dt_col = dt_col.dt.tz_localize("UTC")
                df_predictions["datetime"] = dt_col

            bt = BackTest(df_1m, df_predictions, take_profit=take_profit, stop_loss=stop_loss)
            ledger, final_balance, pnl = bt.run()

            trial.set_user_attr("final_balance", float(final_balance))
            trial.set_user_attr("n_trades",      int(len(ledger)))

            logger.info(
                f"[{model_name}] Trial {trial.number}: pnl={pnl:.2f}, "
                f"params={params}"
            )
            return float(pnl)

        except Exception as exc:
            logger.warning(f"[{model_name}] Trial {trial.number} failed: {exc}")
            return -1e6

    return objective


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune_model(
    model_name: str,
    df: pd.DataFrame,
    df_1m: pd.DataFrame,
    split_date: str,
    n_trials: int = 50,
    timeout: float | None = None,
    take_profit: float = 2.0,
    stop_loss: float = 1.0,
    high_performance: bool = True,
    storage: str | None = None,
    study_name: str | None = None,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter search for *model_name* and return the best params.

    Args:
        model_name        : One of 'arima', 'varima', 'nbeats', 'transformer'.
        df                : Feature-engineered OHLCV DataFrame (df_gf from main).
        df_1m             : 1-minute OHLCV DataFrame for backtesting.
        split_date        : Train / test boundary (same as used in training).
        n_trials          : Number of Optuna trials to run.
        timeout           : Wall-clock time limit in seconds (None = no limit).
        take_profit       : BackTest take-profit multiplier.
        stop_loss         : BackTest stop-loss multiplier.
        high_performance  : If False, each trial uses half-resources.
        storage           : Optuna storage URL.  None = in-memory.
        study_name        : Optuna study name. Defaults to 'tradex_{model_name}'.
        show_progress_bar : Show tqdm progress bar.

    Returns:
        dict of best hyperparameters, ready to pass as ``model_params=`` to
        ``train_model()``.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "optuna is required for hyperparameter tuning. "
            "Install it with: pip install optuna"
        )

    if model_name not in _SPACE_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_SPACE_BUILDERS)}"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    _study_name = study_name or f"tradex_{model_name}"

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0,
    )

    study = optuna.create_study(
        study_name=_study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = _make_objective(
        model_name=model_name,
        df=df,
        df_1m=df_1m,
        split_date=split_date,
        take_profit=take_profit,
        stop_loss=stop_loss,
        high_performance=high_performance,
    )

    logger.info(
        f"Starting Optuna search for '{model_name}': "
        f"n_trials={n_trials}, timeout={timeout}s, "
        f"high_performance={high_performance}"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        gc_after_trial=True,
    )

    best = study.best_trial
    logger.info(
        f"[{model_name}] Best trial #{best.number}: "
        f"pnl={best.value:.2f}, params={best.params}"
    )

    print(f"\n{'='*60}")
    print(f"  Optuna results for: {model_name}")
    print(f"{'='*60}")
    print(f"  Best PnL      : {best.value:.4f}")
    print(f"  Best params   :")
    for k, v in best.params.items():
        print(f"    {k:30s}: {v}")
    print(f"  Trials total  : {len(study.trials)}")
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    print(f"  Trials ok     : {len(completed)}")
    print(f"{'='*60}\n")

    full_params = _SPACE_BUILDERS[model_name](
        _ParamOverrideTrial(best.params)
    )
    return full_params


def tune_all_models(
    df: pd.DataFrame,
    df_1m: pd.DataFrame,
    split_date: str,
    models: list[str] | None = None,
    n_trials: int = 50,
    timeout_per_model: float | None = None,
    take_profit: float = 2.0,
    stop_loss: float = 1.0,
    high_performance: bool = True,
    storage: str | None = "sqlite:///optuna_tradex.db",
) -> dict[str, dict[str, Any]]:
    """
    Run ``tune_model`` for each model in *models* and return a dict of
    ``{model_name: best_params}``.
    """
    if models is None:
        models = ["arima", "varima", "nbeats", "transformer"]

    results: dict[str, dict] = {}
    for model_name in models:
        logger.info(f"Tuning {model_name} …")
        try:
            results[model_name] = tune_model(
                model_name=model_name,
                df=df,
                df_1m=df_1m,
                split_date=split_date,
                n_trials=n_trials,
                timeout=timeout_per_model,
                take_profit=take_profit,
                stop_loss=stop_loss,
                high_performance=high_performance,
                storage=storage,
            )
        except Exception as exc:
            logger.error(f"Tuning failed for {model_name}: {exc}")
            results[model_name] = {}

    return results


# ---------------------------------------------------------------------------
# Internal helper: replay best.params through the space builder
# ---------------------------------------------------------------------------

class _ParamOverrideTrial:
    """
    Minimal stub that replays a fixed params dict through a space-builder
    function so we can reconstruct the full params dict (including hard-coded
    constants like d=0) from a completed Optuna trial.
    """
    def __init__(self, params: dict):
        self._params = params

    def suggest_int(self, name, low, high, step=1):
        return self._params.get(name, low)

    def suggest_float(self, name, low, high, log=False, step=None):
        return self._params.get(name, low)

    def suggest_categorical(self, name, choices):
        return self._params.get(name, choices[0])