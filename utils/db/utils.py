# utils.py

import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from TradeX.utils.common.logs import get_logger
from dotenv import load_dotenv

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
    time_column: str = "datetime",
    is_timeseries: bool = True
):
    """
    Save a DataFrame to the database.
    Stores datetime column directly as TIMESTAMP WITH TIME ZONE.
    """
    if df.empty:
        logger.warning("Empty DataFrame, skipping insert")
        return

    engine = get_engine()
    schema = ensure_schema(schema)
    create_schema(schema)
    table = table_name

    # Ensure datetime column dtype
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column], utc=True)

    # Deduplicate
    df = df.drop_duplicates(subset=[time_column])

    # Create table if missing
    df.head(0).to_sql(table, engine, schema=schema, if_exists="append", index=False)

    # Ensure indexes & hypertable
    ensure_unique_index(table, schema, time_column)
    if is_timeseries:
        ensure_hypertable(table, schema, time_column)

    # Filter already ingested rows (incremental ingestion)
    last_dt = get_last_date(table, schema, time_column)
    if last_dt:
        df = df[df[time_column] > last_dt]

    if df.empty:
        logger.info("No new rows to insert")
        return

    # Insert safely
    cols = ",".join(df.columns)
    placeholders = ",".join([f":{c}" for c in df.columns])
    insert_sql = text(f"""
        INSERT INTO {schema}.{table} ({cols})
        VALUES ({placeholders})
        ON CONFLICT ({time_column}) DO NOTHING
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
        logger.info(f"\nLast 5 rows from {schema}.{table}:\n{df.tail(5)}")
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
    limit: int | None = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data from DB with datetime column directly.
    """
    df = read_df_from_db(table_name, schema, limit)
    if df.empty:
        return df

    # Ensure datetime dtype
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column], utc=True)

    df = df.sort_values(time_column)
    return df
