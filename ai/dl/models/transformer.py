from __future__ import annotations

import os
import numpy as np
import pandas as pd

from TradeX.ai.dl.utils import prepare_series, train_test_split
from TradeX.ai.dl.models.trainer_utils import normalise_datetime,ensure_log_return,rolling_train_test_split,check_min_rows,concat_dedup_sort,build_pl_trainer_kwargs,make_test_artifacts
from TradeX.utils.common.logs import get_logger

logger = get_logger("transformer")

_DEFAULT_ROLLING_ROWS = 4_320   # ~6 months at 1h


def _configure_cpu(n_cores: int = 4) -> None:
    """
    Set all thread-count env-vars BEFORE torch is imported.
    On Windows, OMP/MKL read these at import time.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_cores)
    try:
        import torch
        torch.set_num_threads(n_cores)
        torch.set_num_interop_threads(1)
        logger.info(
            f"Transformer CPU: {n_cores} intraop threads | "
            f"MKL={torch.backends.mkl.is_available()}"
        )
    except Exception as e:
        logger.warning(f"Could not configure PyTorch threads: {e}")


def _try_quantize(model):
    """Apply dynamic INT8 quantization to the trained model's Linear layers."""
    try:
        import torch
        nn_model = getattr(model, "model", None)
        if nn_model is None:
            return model
        quantized = torch.quantization.quantize_dynamic(
            nn_model, {torch.nn.Linear}, dtype=torch.qint8,
        )
        model.model = quantized
        logger.info("Dynamic INT8 quantization applied to inference model.")
    except Exception as e:
        logger.warning(f"Quantization skipped: {e}")
    return model


def _try_compile(model):
    """torch.compile() fuses ops into a single kernel graph (PyTorch >= 2.0)."""
    try:
        import torch
        if not hasattr(torch, "compile"):
            return model
        nn_model = getattr(model, "model", None)
        if nn_model is None:
            return model
        model.model = torch.compile(nn_model, backend="inductor", mode="reduce-overhead")
        logger.info("torch.compile() (inductor, reduce-overhead) applied.")
    except Exception as e:
        logger.warning(f"torch.compile skipped: {e}")
    return model


def _bf16_supported() -> bool:
    """Return True only when bfloat16 is actually usable on this CPU."""
    try:
        import torch
        t = torch.tensor([1.0], dtype=torch.float32)
        _ = t.to(torch.bfloat16)
        a = torch.ones(4, 4, dtype=torch.bfloat16)
        _ = a @ a
        return True
    except Exception:
        return False


def train(
    df: pd.DataFrame,
    target_col: str = "log_return",
    split_date: str = "2024-01-01",
    input_chunk_length: int = 72,      # SIGNAL-3: was 24; 3 days of 1h context
    output_chunk_length: int = 4,      # SIGNAL-2: was 1; predict 4h forward
    d_model: int = 64,                 # SIGNAL-5: was 32
    nhead: int = 4,                    # updated: must divide d_model (64/4=16 ✓)
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dim_feedforward: int | None = 256, # SIGNAL-5: was 64
    dropout: float = 0.1,              # SIGNAL-6: was 0.0; regularise
    n_epochs: int = 40,                # SIGNAL-4: was 15
    batch_size: int = 64,
    random_state: int = 42,
    rolling_rows: int = 2_000,
    n_cores: int = 4,
    use_quantization: bool = True,
    use_compile: bool = False,
    use_bf16_fit: bool = True,
    signal_threshold: float = 3e-4,   # SIGNAL-1: dead-band on log_return
    lookback: int | None = None,
    epochs: int | None = None,
    **kwargs,
) -> tuple:
    """
    Optimized Transformer training for CPU.
    Returns (model, preds, test_index, df_test).
    """
    # --- CPU setup --------------------------------------------------------
    _configure_cpu(n_cores)

    # --- Aliases ----------------------------------------------------------
    if lookback is not None:
        input_chunk_length = lookback
    if epochs is not None:
        if "n_epochs" in kwargs:
            logger.warning("Both 'epochs' alias and 'n_epochs' in kwargs; alias wins.")
        n_epochs = epochs

    kwargs.pop("random_state", None)

    # --- Architecture guards ----------------------------------------------
    if d_model % nhead != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
    if num_encoder_layers < 1 or num_decoder_layers < 1:
        raise ValueError("num_encoder_layers and num_decoder_layers must be >= 1")
    if output_chunk_length >= input_chunk_length:
        raise ValueError("output_chunk_length must be < input_chunk_length")

    if dim_feedforward is None:
        dim_feedforward = 4 * d_model
    if dim_feedforward > 4 * d_model:
        logger.warning(
            f"dim_feedforward={dim_feedforward} is large relative to d_model={d_model}."
        )

    # --- 1. Datetime normalisation + log_return ---------------------------
    df = normalise_datetime(df)
    if target_col == "log_return":
        df = ensure_log_return(df)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found.")

    df_target = df[[target_col]].dropna().astype({target_col: "float32"}).sort_index()

    # --- 2. Rolling train / test split ------------------------------------
    df_train, df_test_raw = rolling_train_test_split(
        df_target,
        split_date=split_date,
        rolling_rows=rolling_rows,
        logger=logger,
        label="Transformer",
    )

    # --- 3. Minimum-row guard ---------------------------------------------
    check_min_rows(
        df_train,
        min_rows=input_chunk_length + output_chunk_length,
        context="Transformer training window",
    )

    # --- 4. Build Darts series --------------------------------------------
    df_windowed = concat_dedup_sort(df_train, df_test_raw)
    series = prepare_series(df_windowed.reset_index(), target_col)
    train_series, test_series = train_test_split(series, split_date)

    # --- 5. Build pl_trainer_kwargs ---------------------------------------
    pl_trainer_kwargs = build_pl_trainer_kwargs(
        base_kwargs=kwargs.pop("pl_trainer_kwargs", {}),
        use_early_stopping=True,
        monitor="train_loss",
        patience=5,                    # SIGNAL-4: was 2; allow fuller convergence
    )
    pl_trainer_kwargs.setdefault("devices", 1)
    pl_trainer_kwargs.setdefault("gradient_clip_val", 1.0)
    pl_trainer_kwargs.setdefault(
        "precision",
        "bf16-mixed" if use_bf16_fit and _bf16_supported() else "32-true",
    )

    dataloader_kwargs = kwargs.pop("dataloader_kwargs", {})  
    dataloader_kwargs["num_workers"] = 0
    dataloader_kwargs.setdefault("pin_memory", False)

    # --- 6. Fit -----------------------------------------------------------
    from darts.models import TransformerModel
    import inspect

    model_kwargs = dict(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=random_state,
        pl_trainer_kwargs=pl_trainer_kwargs,
        **kwargs,
    )
    try:
        if "dataloader_kwargs" in inspect.signature(TransformerModel.__init__).parameters:
            model_kwargs["dataloader_kwargs"] = dataloader_kwargs
    except Exception:
        pass

    model = TransformerModel(**model_kwargs)
    model.fit(train_series)

    # SIGNAL-1: attach threshold so downstream prepare_predictions can apply it.
    model.signal_threshold = signal_threshold
    logger.info(f"Transformer signal_threshold set to {signal_threshold:.2e}")

    # --- 7. Post-training boosts ------------------------------------------
    if use_compile:
        model = _try_compile(model)
    if use_quantization:
        model = _try_quantize(model)

    # --- 8. Predict -------------------------------------------------------
    preds = model.predict(len(test_series))

    # --- 9. Return artifacts ----------------------------------------------
    test_index, df_test = make_test_artifacts(len(df_train), test_series)
    return model, preds, test_index, df_test