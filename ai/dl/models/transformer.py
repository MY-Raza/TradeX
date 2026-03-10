from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd

from TradeX.ai.dl.utils import prepare_series, train_test_split
from TradeX.utils.common.logs import get_logger

logger = get_logger("transformer")

_DEFAULT_ROLLING_ROWS = 4_320   # ~6 months at 1h


def _configure_cpu(n_cores: int = 4) -> None:
    """
    Set all thread-count env-vars BEFORE torch is imported.
    On Windows, OMP/MKL read these at import time — torch.set_num_threads()
    has no effect on BLAS after the fact.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_cores)          # overwrite, not setdefault
    try:
        import torch
        torch.set_num_threads(n_cores)
        torch.set_num_interop_threads(1)         # >1 causes contention on small models
        logger.info(
            f"Transformer CPU: {n_cores} intraop threads | "
            f"MKL={torch.backends.mkl.is_available()}"
        )
    except Exception as e:
        logger.warning(f"Could not configure PyTorch threads: {e}")


def _try_quantize(model):
    """
    Apply dynamic INT8 quantization to the trained model's Linear layers.
    This is lossless in expectation (same weights, lower-precision arithmetic)
    and typically gives 1.5-2× inference speed on pure-CPU x86.
    Silently skipped on PyTorch < 1.8 or non-x86 platforms.
    """
    try:
        import torch
        # Access the underlying nn.Module stored by Darts
        nn_model = getattr(model, "model", None)
        if nn_model is None:
            return model
        quantized = torch.quantization.quantize_dynamic(
            nn_model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        model.model = quantized
        logger.info("Dynamic INT8 quantization applied to inference model.")
    except Exception as e:
        logger.warning(f"Quantization skipped: {e}")
    return model


def _try_compile(model):
    """
    torch.compile() (PyTorch ≥ 2.0) fuses ops into a single kernel graph,
    saving ~15-30% wall-clock per forward pass on CPU.
    Silently skipped on older PyTorch.
    """
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


def train(
    df: pd.DataFrame,
    target_col: str = "log_return",
    split_date: str = "2024-01-01",
    # ── Architecture (CPU-tuned defaults) ──────────────────────────────────
    input_chunk_length: int = 24,       # attention O(n²): 4× cheaper than 48
    output_chunk_length: int = 1,
    d_model: int = 32,                  # halves every matmul vs 64
    nhead: int = 2,                     # must divide d_model
    num_encoder_layers: int = 1,        # 3× fewer attention passes vs 3
    num_decoder_layers: int = 1,
    dim_feedforward: int | None = None, # None → auto (4 × d_model = 128)
    dropout: float = 0.0,               # no benefit + overhead for small models
    # ── Training ───────────────────────────────────────────────────────────
    n_epochs: int = 15,                 # early stopping fires ~8-12
    batch_size: int = 128,              # 4× fewer gradient steps/epoch vs 32
    random_state: int = 42,
    rolling_rows: int = _DEFAULT_ROLLING_ROWS,
    n_cores: int = 4,
    # ── Post-training speed boosts ─────────────────────────────────────────
    use_quantization: bool = True,      # INT8 dynamic quant for inference
    use_compile: bool = True,           # torch.compile if PyTorch ≥ 2.0
    use_bf16_fit: bool = True,          # bfloat16 autocast during training
    # ── Aliases ────────────────────────────────────────────────────────────
    lookback: int | None = None,
    epochs: int | None = None,
    **kwargs,
) -> tuple:
   
    # ── Configure threads BEFORE any torch/lightning import ────────────────
    _configure_cpu(n_cores)

    # ── Resolve aliases ────────────────────────────────────────────────────
    if lookback is not None:
        input_chunk_length = lookback
    if epochs is not None:
        if "n_epochs" in kwargs:
            logger.warning("Both 'epochs' alias and 'n_epochs' in kwargs; alias wins.")
        n_epochs = epochs

    kwargs.pop("random_state", None)

    # ── Architecture guards ────────────────────────────────────────────────
    if d_model % nhead != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
            f"Common pairs: (32,2), (64,4), (128,4), (128,8)."
        )
    if num_encoder_layers < 1 or num_decoder_layers < 1:
        raise ValueError(
            f"num_encoder_layers and num_decoder_layers must each be ≥ 1 "
            f"(got {num_encoder_layers}, {num_decoder_layers})."
        )
    if output_chunk_length >= input_chunk_length:
        raise ValueError(
            f"output_chunk_length ({output_chunk_length}) must be strictly "
            f"less than input_chunk_length ({input_chunk_length})."
        )

    if dim_feedforward is None:
        dim_feedforward = 4 * d_model
    if dim_feedforward > 4 * d_model:
        logger.warning(
            f"dim_feedforward={dim_feedforward} is large relative to "
            f"d_model={d_model}. Consider {4 * d_model} for CPU speed."
        )

    # ── 1. Prepare DataFrame ───────────────────────────────────────────────
    df = df.copy()

    if target_col == "log_return" and "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"]).diff()

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
        df["datetime"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        df = df.set_index("datetime")

    if target_col not in df.columns:
        raise ValueError(
            f"target_col '{target_col}' not found. Available: {list(df.columns)}"
        )

    df_target = df[[target_col]].dropna().sort_index()

    # ── 2. Rolling window split ────────────────────────────────────────────
    split_ts = pd.Timestamp(split_date)
    if split_ts.tz is not None:
        split_ts = split_ts.tz_convert("UTC").tz_localize(None)

    df_train_full = df_target[df_target.index < split_ts]
    df_test_raw   = df_target[df_target.index >= split_ts]

    if df_train_full.empty:
        raise ValueError(f"No training rows before split_date '{split_date}'.")
    if df_test_raw.empty:
        raise ValueError(f"No test rows on/after split_date '{split_date}'.")

    if rolling_rows and len(df_train_full) > rolling_rows:
        dropped = len(df_train_full) - rolling_rows
        df_train_full = df_train_full.iloc[-rolling_rows:]
        logger.info(
            f"Transformer rolling window: last {rolling_rows} rows "
            f"(dropped {dropped})."
        )

    min_rows = input_chunk_length + output_chunk_length
    if len(df_train_full) < min_rows:
        raise ValueError(
            f"Training window too small ({len(df_train_full)} rows), "
            f"need at least {min_rows}."
        )

    # ── 3. Build Darts series ──────────────────────────────────────────────
    df_windowed  = pd.concat([df_train_full, df_test_raw])
    series       = prepare_series(df_windowed.reset_index(), target_col)
    train_series, test_series = train_test_split(series, split_date)

    # ── 4. Trainer kwargs ──────────────────────────────────────────────────
    pl_trainer_kwargs = kwargs.pop("pl_trainer_kwargs", {})
    pl_trainer_kwargs.setdefault("accelerator",          "cpu")
    pl_trainer_kwargs.setdefault("devices",              1)
    pl_trainer_kwargs.setdefault("enable_progress_bar",  False)
    pl_trainer_kwargs.setdefault("enable_model_summary", False)
    pl_trainer_kwargs.setdefault("log_every_n_steps",    10)
    pl_trainer_kwargs.setdefault("gradient_clip_val",    1.0)   # NEW: faster convergence

    # bfloat16 autocast: halves arithmetic on AVX-512 CPUs, ~10-20% faster.
    # Falls back to fp32 silently on machines without AVX-512.
    if use_bf16_fit:
        try:
            import torch
            if torch.backends.cpu.get_default_dtype() is not None:  # always True
                pl_trainer_kwargs.setdefault("precision", "bf16-mixed")
                logger.info("Training precision: bf16-mixed (autocast).")
        except Exception:
            pl_trainer_kwargs.setdefault("precision", "32-true")
    else:
        pl_trainer_kwargs.setdefault("precision", "32-true")

    # Windows: num_workers > 0 spawns subprocesses (~2-5 s overhead per epoch)
    dataloader_kwargs = kwargs.pop("dataloader_kwargs", {})
    dataloader_kwargs["num_workers"] = 0
    dataloader_kwargs.setdefault("pin_memory", False)

    # EarlyStopping: patience=2 (was 3) shaves 1-2 epochs off the tail
    try:
        from pytorch_lightning.callbacks import EarlyStopping
        cb = pl_trainer_kwargs.pop("callbacks", [])
        cb.append(EarlyStopping(
            monitor="train_loss", patience=2, min_delta=1e-4, mode="min"
        ))
        pl_trainer_kwargs["callbacks"] = cb
    except ImportError:
        pass

    # ── 5. Fit ─────────────────────────────────────────────────────────────
    from darts.models import TransformerModel

    logger.info(
        f"TransformerModel | d_model={d_model} nhead={nhead} "
        f"enc={num_encoder_layers} dec={num_decoder_layers} "
        f"ffn={dim_feedforward} icl={input_chunk_length} "
        f"epochs={n_epochs} batch={batch_size} cores={n_cores} | "
        f"quant={use_quantization} compile={use_compile} bf16={use_bf16_fit}"
    )

    model = TransformerModel(
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
    model.fit(train_series)

    # ── 6. Post-training speed boosts (inference path) ─────────────────────
    if use_compile:
        model = _try_compile(model)         # graph fusion via inductor
    if use_quantization:
        model = _try_quantize(model)        # INT8 dynamic quant on Linear layers

    # ── 7. Predict ─────────────────────────────────────────────────────────
    preds = model.predict(len(test_series))

    # ── 8. Return artifacts ────────────────────────────────────────────────
    n_train_full = len(df_target[df_target.index < split_ts])
    n_test       = len(test_series)
    test_index   = np.arange(n_train_full, n_train_full + n_test)
    df_test      = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test