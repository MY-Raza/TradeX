# utils.py

import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from TradeX.utils.common.logs import get_logger
from dotenv import load_dotenv
from types import SimpleNamespace
import json
import numpy as np

# ---------------------------
# Initialize logger
# ---------------------------
logger = get_logger("utils")

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(dotenv_path)

# ---------------------------
# Globals
# ---------------------------
_ENGINE = None
USER_SCHEMA: str | None = None


# =====================================================
# Database Engine
# =====================================================
def get_engine(db_url: str | None = None):
    global _ENGINE

    if _ENGINE:
        return _ENGINE

    db_url = db_url or os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not provided")

    _ENGINE = create_engine(db_url, pool_pre_ping=True)
    logger.info("Database engine initialized")
    return _ENGINE


# =====================================================
# Schema Utilities
# =====================================================
def ensure_schema(schema: str | None) -> str:
    global USER_SCHEMA

    if schema:
        USER_SCHEMA = schema
        return schema

    if USER_SCHEMA:
        return USER_SCHEMA

    schema = input("Enter schema name: ").strip()
    if not schema:
        raise ValueError("Schema cannot be empty")

    USER_SCHEMA = schema
    return schema


def create_schema(schema: str | None = None):
    engine = get_engine()
    schema = ensure_schema(schema)

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

    logger.info(f"Schema '{schema}' ready")


def drop_schema(schema: str | None = None):
    engine = get_engine()
    schema = ensure_schema(schema)

    confirm = input(f"⚠️ DROP schema '{schema}' and ALL objects? Type 'yes' to continue: ").lower()
    if confirm != "yes":
        logger.warning("Schema drop cancelled")
        return

    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))

    logger.warning(f"Schema '{schema}' dropped")


# =====================================================
# Table Repair Helpers
# =====================================================
def ensure_unique_index(table_name: str, schema: str, time_column: str):
    engine = get_engine()
    index_name = f"{table_name}_{time_column}_uidx"
    desc_index = f"{table_name}_{time_column}_desc_idx"

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
            ON {schema}.{table_name} ({time_column});
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS {desc_index}
            ON {schema}.{table_name} ({time_column} DESC);
        """))

    logger.info(f"Indexes ensured: {index_name}, {desc_index}")


def ensure_hypertable(table, schema, time_column):
    full_table_name = f'"{schema}"."{table}"'
    engine = get_engine()

    try:
        with engine.begin() as conn:
            # Ensure TimescaleDB extension is installed
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))

            # Create hypertable if not exists
            query = text(f"""
                SELECT create_hypertable(
                   '{full_table_name}'::regclass,
                   '{time_column}'::name,
                   migrate_data => :migrate,
                   if_not_exists => :if_not_exists
                );
            """)
            params = {"migrate": True, "if_not_exists": True}
            conn.execute(query, params)
            logger.info(f"Hypertable created/verified for {full_table_name}")

    except Exception as e:
        logger.warning(f"Hypertable check for {full_table_name} failed: {e}")


# =====================================================
# Core DB Operations
# =====================================================
def get_last_date(table_name: str, schema: str, time_column: str) -> pd.Timestamp | None:
    """
    Get the last datetime stored in the table.
    Returns pd.Timestamp in UTC.
    """
    engine = get_engine()
    inspector = inspect(engine)
    if not inspector.has_table(table_name, schema=schema):
        return None

    query = f"""
        SELECT {time_column} AT TIME ZONE 'UTC' AS last_dt
        FROM {schema}.{table_name}
        ORDER BY {time_column} DESC
        LIMIT 1
    """
    with engine.begin() as conn:
        result = conn.execute(text(query)).scalar()
        if result:
            return pd.to_datetime(result, utc=True)
        return None

def save_df_to_db(
    df: pd.DataFrame,
    table_name: str,
    schema: str | None = None,
    time_column: str | None = "datetime",
    is_timeseries: bool = True,
    enforce_unique_time: bool = True,  
    use_on_conflict: bool = True
):
    """
    Save a DataFrame to the database.
    - Automatically handles datetime columns for time-series.
    - Dynamically adds missing columns for strategies.strategy_registry.
    - Converts tuple/list columns to JSON for JSONB storage.
    """
    if df.empty:
        logger.warning("Empty DataFrame, skipping insert")
        return

    engine = get_engine()
    schema = ensure_schema(schema)
    create_schema(schema)
    table = table_name

    # --------------------------------------------------
    # Handle time column safely (NO forced UTC conversion)
    # --------------------------------------------------
    if time_column is None:
        if "datetime" in df.columns:
            time_column = "datetime"
            logger.info("time_column not provided, using 'datetime' by default")
        else:
            time_column = None
            logger.info("No time_column provided and 'datetime' missing")

    if time_column:
        if time_column not in df.columns:
            raise ValueError(f"time_column '{time_column}' not found in DataFrame")

        df[time_column] = pd.to_datetime(df[time_column])

        # Localize ONLY if naive
        if df[time_column].dt.tz is None:
            df[time_column] = df[time_column].dt.tz_localize("UTC")

    # --------------------------------------------------
    # Dynamically add missing columns (strategy_registry)
    # --------------------------------------------------
    if schema == "strategies" and table == "strategy_registry":
        df.head(0).to_sql(table, engine, schema=schema, if_exists="append", index=False)

        inspector = inspect(engine)
        existing_cols = [col["name"] for col in inspector.get_columns(table, schema=schema)]

        for col in df.columns:
            if col not in existing_cols:
                sample_value = df[col].iloc[0]

                if isinstance(sample_value, (tuple, list)):
                    col_type = "JSONB"
                    df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
                elif pd.api.types.is_integer_dtype(df[col]):
                    col_type = "BIGINT"
                elif pd.api.types.is_float_dtype(df[col]):
                    col_type = "DOUBLE PRECISION"
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    col_type = "TIMESTAMP WITH TIME ZONE"
                else:
                    col_type = "TEXT"

                alter_sql = text(
                    f'ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS "{col}" {col_type}'
                )
                with engine.begin() as conn:
                    conn.execute(alter_sql)

                logger.info(f"Added missing column '{col}' as {col_type}")

    # --------------------------------------------------
    # Convert tuple/list columns to JSON (safety)
    # --------------------------------------------------
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (tuple, list))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)

    # --------------------------------------------------
    # Create table if missing
    # --------------------------------------------------
    df.head(0).to_sql(table, engine, schema=schema, if_exists="append", index=False)

    # --------------------------------------------------
    # Time-series handling
    # --------------------------------------------------
    if time_column and enforce_unique_time:
        ensure_unique_index(table, schema, time_column)

        if is_timeseries:
            ensure_hypertable(table, schema, time_column)

        # Incremental ingestion (timezone-safe)
        last_dt = get_last_date(table, schema, time_column)

        if last_dt is not None:
            last_dt = pd.to_datetime(last_dt)

            if last_dt.tzinfo is None:
                last_dt = last_dt.tz_localize("UTC")

            df = df[df[time_column] > last_dt]

    if df.empty:
        logger.info("No new rows to insert")
        return

    # --------------------------------------------------
    # Insert safely
    # --------------------------------------------------
    cols = ",".join([f'"{c}"' for c in df.columns])
    placeholders = ",".join([f":{c}" for c in df.columns])
    conflict_clause = ""
    if time_column and enforce_unique_time and use_on_conflict:
        conflict_clause = f"ON CONFLICT ({time_column}) DO NOTHING" if time_column else ""

    insert_sql = text(f"""
        INSERT INTO {schema}.{table} ({cols})
        VALUES ({placeholders})
        {conflict_clause}
    """)

    with engine.begin() as conn:
        conn.execute(insert_sql, df.to_dict(orient="records"))

    logger.info(f"Inserted {len(df)} rows into {schema}.{table}")


# =====================================================
# Read Helpers
# =====================================================
def read_df_from_db(table_name: str, schema: str | None = None, limit: int | None = None) -> pd.DataFrame:
    engine = get_engine()
    schema = ensure_schema(schema)
    table = table_name

    query = f"SELECT * FROM {schema}.{table}"
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql_query(query, engine)
    if not df.empty:
        logger.info(f"Table Found")
    else:
        logger.info(f"No data found in {schema}.{table}")

    return df


# =====================================================
# Drop Helpers
# =====================================================
def drop_table(table_name: str, schema: str | None = None):
    engine = get_engine()
    schema = ensure_schema(schema)
    table = f"{table_name}_1h"

    confirm = input(f"Drop table {schema}.{table}? (yes/no): ").lower()
    if confirm != "yes":
        logger.warning("Table drop cancelled")
        return

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{table} CASCADE"))

    logger.warning(f"Table {schema}.{table} dropped")


# =====================================================
# OHLCV Fetch Helper (returns datetime column)
# =====================================================
def fetch_ohlcv_df(
    table_name: str,
    schema: str,
    time_column: str = "datetime",
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from DB filtered by datetime range.

    Args:
        table_name (str): Table name
        schema (str): DB schema
        time_column (str): Datetime column name
        start_date (str | None): e.g. "2024-01-01"
        end_date (str | None): e.g. "2024-12-31"

    Returns:
        pd.DataFrame
    """

    df = read_df_from_db(table_name, schema)

    if df.empty:
        return df

    # Ensure datetime type (UTC safe)
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column], utc=True)

    df = df.sort_values(time_column)

    # -------------------------
    # Apply Date Filtering
    # -------------------------
    if start_date:
        start_date = pd.to_datetime(start_date, utc=True)
        df = df[df[time_column] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date, utc=True)
        df = df[df[time_column] <= end_date]

    if limit:
        df = df.tail(limit)
    return df.reset_index(drop=True)

#======================================
# Strategy Fetcher Operation From DB
#======================================
def get_profitable_strategies(
    symbol: str,    
    timehorizon: str,
    min_pnl: float = 100,
    top_n: int = 1,
    best: str = "highest" 
):
    """
    Fetch the top N profitable strategies for a given timeframe where pnl_sum > min_pnl.
    Keeps only columns where value is:
      - True (boolean)
      - OR non-zero number (int/float)
      - OR non-null and not False

    Args:
        timehorizon (str): e.g. "1h", "15m"
        min_pnl (float): minimum cumulative PnL
        top_n (int): number of top strategies to return based on pnl_sum
        best (str): "highest" for top PnL, "lowest" for lowest PnL

    Returns:
        List[SimpleNamespace]
    """
    engine = get_engine()

    query = text("""
        SELECT *
        FROM strategies.strategy_registry
        WHERE pnl_sum > :min_pnl
          AND timehorizon = :timehorizon
          AND symbol = :symbol
    """)

    with engine.begin() as conn:
        result = conn.execute(
            query,
            {"min_pnl": min_pnl, "timehorizon": timehorizon, "symbol": symbol}
        )
        rows = result.fetchall()
        columns = result.keys()

    # Convert rows to dicts
    row_dicts = [dict(zip(columns, row)) for row in rows]

    # Sort by pnl_sum
    reverse_sort = True if best == "highest" else False
    row_dicts.sort(key=lambda x: x.get("pnl_sum", 0), reverse=reverse_sort)

    # Pick top N
    row_dicts = row_dicts[:top_n]

    strategies = []

    for row_dict in row_dicts:
        filtered_dict = {}
        for k, v in row_dict.items():
            if v is True:
                filtered_dict[k] = v
            elif isinstance(v, (int, float)) and v != 0:
                filtered_dict[k] = v
            elif v not in (None, False):
                filtered_dict[k] = v

        strategy_obj = SimpleNamespace(**filtered_dict)
        strategies.append(strategy_obj)

    logger.info(
        f"Total profitable strategies returned for timeframe {timehorizon}: {len(strategies)}"
    )

    return strategies

def get_best_model(
    schema: str = "model_stats",
    table_name: str = "ml_results"
) -> str | None:
    """
    Returns the best model_name based on:
        score = pnl * sharpe_ratio / abs(max_drawdown)

    Reads data from DB and calculates score dynamically.
    """

    # ----------------------------------------
    # Load data from DB
    # ----------------------------------------
    df = read_df_from_db(table_name=table_name, schema=schema)

    if df.empty:
        logger.warning("No model stats found in database.")
        return None

    required_cols = ["model_name", "pnl", "sharpe_ratio", "max_drawdown"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ----------------------------------------
    # Clean data
    # ----------------------------------------
     

    # Avoid division by zero
    df = df[df["max_drawdown"] != 0]

    if df.empty:
        logger.warning("All models have zero max_drawdown. Cannot compute score.")
        return None

    # ----------------------------------------
    # Compute Score
    # ----------------------------------------
    df["score"] = (
        df["pnl"]
        * df["sharpe_ratio"]
        / df["max_drawdown"].abs()
    )

    # Remove infinite / NaN scores
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])

    if df.empty:
        logger.warning("No valid models after score computation.")
        return None

    # ----------------------------------------
    # Get Best Model
    # ----------------------------------------
    best_row = df.sort_values("score", ascending=False).iloc[0]

    best_model_name = best_row["model_name"]

    logger.info(f"Best model selected: {best_model_name}")
    logger.info(f"Score: {best_row['score']}")

    return best_model_name

def get_important_features(
    model_name: str,
    schema: str = "ml_features",
    table_name: str = "best_features",
):
    """
    Fetch important_features for a given model_name
    from ml_features.best_features table.
    """

    if not model_name:
        logger.warning("No model_name provided.")
        return None

    # ----------------------------------------
    # Load table
    # ----------------------------------------
    df = read_df_from_db(table_name=table_name, schema=schema)

    if df.empty:
        logger.warning("best_features table is empty.")
        return None

    required_cols = ["model_name", "important_features"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ----------------------------------------
    # Filter for model
    # ----------------------------------------
    row = df[df["model_name"] == model_name]

    if row.empty:
        logger.warning(f"No features found for model: {model_name}")
        return None

    important_features = row.iloc[0]["important_features"]

    logger.info(f"Fetched important features for model: {model_name}")

    return important_features





