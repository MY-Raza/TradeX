from __future__ import annotations

import os
import numpy as np
import pandas as pd

from TradeX.ai.dl.utils import prepare_series, train_test_split
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
    """torch.compile() fuses ops into a single kernel graph (PyTorch ≥ 2.0)."""
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
    """
    BUG-5 FIX: return True only when bfloat16 is actually usable on this CPU.
    Uses torch.Tensor.is_floating_point + bfloat16 cast as a reliable probe.
    """
    try:
        import torch
        t = torch.tensor([1.0], dtype=torch.float32)
        _ = t.to(torch.bfloat16)
        # Further check: bfloat16 matmul must not raise
        a = torch.ones(4, 4, dtype=torch.bfloat16)
        _ = a @ a
        return True
    except Exception:
        return False


def train(
    df: pd.DataFrame,
    target_col: str = "log_return",
    split_date: str = "2024-01-01",
    input_chunk_length: int = 24,   # shorter input window
    output_chunk_length: int = 1,
    d_model: int = 32,              # smaller model
    nhead: int = 2,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dim_feedforward: int | None = 64,  # smaller feedforward
    dropout: float = 0.0,
    n_epochs: int = 15,             # fewer epochs
    batch_size: int = 64,           # larger batch
    random_state: int = 42,
    rolling_rows: int = 2_000,      # smaller rolling window
    n_cores: int = 4,
    use_quantization: bool = True,
    use_compile: bool = True,
    use_bf16_fit: bool = True,
    lookback: int | None = None,
    epochs: int | None = None,
    **kwargs,
) -> tuple:
    """
    Optimized Transformer training for CPU:
    - Smaller model, shorter input window, fewer epochs.
    - Uses bf16 + torch.compile for speed.
    - Returns (model, preds, test_index, df_test).
    """
    # --- CPU setup
    _configure_cpu(n_cores)

    # --- Aliases
    if lookback is not None:
        input_chunk_length = lookback
    if epochs is not None:
        if "n_epochs" in kwargs:
            logger.warning("Both 'epochs' alias and 'n_epochs' in kwargs; alias wins.")
        n_epochs = epochs

    kwargs.pop("random_state", None)

    # --- Architecture guards
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

    # --- Prepare DataFrame
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
        raise ValueError(f"target_col '{target_col}' not found.")

    df_target = df[[target_col]].dropna().astype({target_col: "float32"}).sort_index()

    # --- Rolling window split
    split_ts = pd.Timestamp(split_date)
    if split_ts.tz is not None:
        split_ts = split_ts.tz_convert("UTC").tz_localize(None)

    df_train_full = df_target[df_target.index < split_ts]
    df_test_raw = df_target[df_target.index >= split_ts]

    if rolling_rows and len(df_train_full) > rolling_rows:
        df_train_full = df_train_full.iloc[-rolling_rows:]
        logger.info(f"Using last {rolling_rows} rows for training.")

    min_rows = input_chunk_length + output_chunk_length
    if len(df_train_full) < min_rows:
        raise ValueError(f"Training window too small: {len(df_train_full)} rows.")

    # --- Build Darts series
    df_windowed = (
        pd.concat([df_train_full, df_test_raw])
        .loc[~pd.concat([df_train_full, df_test_raw]).index.duplicated(keep="last")]
        .sort_index()
    )
    series = prepare_series(df_windowed.reset_index(), target_col)
    train_series, test_series = train_test_split(series, split_date)

    # --- Trainer kwargs
    pl_trainer_kwargs = kwargs.pop("pl_trainer_kwargs", {}).copy()
    pl_trainer_kwargs.setdefault("accelerator", "cpu")
    pl_trainer_kwargs.setdefault("devices", 1)
    pl_trainer_kwargs.setdefault("enable_progress_bar", False)
    pl_trainer_kwargs.setdefault("enable_model_summary", False)
    pl_trainer_kwargs.setdefault("log_every_n_steps", 10)
    pl_trainer_kwargs.setdefault("gradient_clip_val", 1.0)
    pl_trainer_kwargs.setdefault("precision", "bf16-mixed" if use_bf16_fit and _bf16_supported() else "32-true")

    try:
        from pytorch_lightning.callbacks import EarlyStopping
        cb = pl_trainer_kwargs.pop("callbacks", [])
        cb.append(EarlyStopping(monitor="train_loss", patience=2, min_delta=1e-4, mode="min"))
        pl_trainer_kwargs["callbacks"] = cb
    except Exception as e:
        logger.warning(f"EarlyStopping not added: {e}")

    dataloader_kwargs = kwargs.pop("dataloader_kwargs", {}).copy()
    dataloader_kwargs["num_workers"] = 0
    dataloader_kwargs.setdefault("pin_memory", False)

    # --- Fit
    from darts.models import TransformerModel

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
        import inspect
        if "dataloader_kwargs" in inspect.signature(TransformerModel.__init__).parameters:
            model_kwargs["dataloader_kwargs"] = dataloader_kwargs
    except Exception:
        pass

    model = TransformerModel(**model_kwargs)
    model.fit(train_series)

    # --- Post-training boosts
    if use_compile:
        model = _try_compile(model)
    if use_quantization:
        model = _try_quantize(model)

    # --- Predict
    preds = model.predict(len(test_series))

    # --- Return
    n_train_windowed = len(df_train_full)
    n_test = len(test_series)
    test_index = np.arange(n_train_windowed, n_train_windowed + n_test)
    df_test = pd.DataFrame(index=test_series.time_index)

    return model, preds, test_index, df_test