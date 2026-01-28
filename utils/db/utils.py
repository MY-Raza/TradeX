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
    """
    Create and return a singleton SQLAlchemy engine.

    Args:
        db_url (str | None): Optional database URL; defaults to DATABASE_URL in .env.

    Returns:
        sqlalchemy.Engine: Engine object for DB operations.
    """
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
    """
    Ensure a schema is defined. If not, prompt the user.

    Args:
        schema (str | None): Optional schema name.

    Returns:
        str: Valid schema name.
    """
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
    """
    Create a database schema if it does not exist.

    Args:
        schema (str | None): Optional schema name.
    """
    engine = get_engine()
    schema = ensure_schema(schema)

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

    logger.info(f"Schema '{schema}' ready")


def drop_schema(schema: str | None = None):
    """
    Drop a database schema after user confirmation.

    Args:
        schema (str | None): Optional schema name.
    """
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
    """
    Ensure a UNIQUE index exists on a table for the time column.
    Handles older tables or hypertables safely.

    Args:
        table_name (str): Table name.
        schema (str): Schema name.
        time_column (str): Timestamp column name.
    """
    engine = get_engine()
    index_name = f"{table_name}_{time_column}_uidx"
    desc_index = f"{table_name}_{time_column}_desc_idx"

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
            ON {schema}.{table_name} ({time_column});
        """))

        # For fast "last row" queries
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS {desc_index}
            ON {schema}.{table_name} ({time_column} DESC);
        """))

    logger.info(f"Indexes ensured: {index_name}, {desc_index}")


def ensure_hypertable(table_name: str, schema: str, time_column: str):
    """
    Convert a table to a TimescaleDB hypertable if not already.

    Args:
        table_name (str): Table name.
        schema (str): Schema name.
        time_column (str): Timestamp column name.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"""
            SELECT create_hypertable(
                '{schema}.{table_name}',
                '{time_column}',
                migrate_data => TRUE,
                if_not_exists => TRUE
            );
        """))


# =====================================================
# Core DB Operations
# =====================================================
def get_last_date(table_name: str, schema: str, time_column: str) -> int | None:
    """
    Get the last timestamp in a table.

    Args:
        table_name (str): Table name.
        schema (str): Schema name.
        time_column (str): Timestamp column name.

    Returns:
        int | None: Last timestamp (ms) or None if table doesn't exist.
    """
    engine = get_engine()
    inspector = inspect(engine)
    if not inspector.has_table(table_name, schema=schema):
        return None

    query = f"""
        SELECT {time_column}
        FROM {schema}.{table_name}
        ORDER BY {time_column} DESC
        LIMIT 1
    """
    with engine.begin() as conn:
        return conn.execute(text(query)).scalar()


def save_df_to_db(
    df: pd.DataFrame,
    table_name: str,
    schema: str | None = None,
    time_column: str = "timestamp",
    is_timeseries: bool = True
):
    """
    Save a DataFrame to the database safely.

    Steps:
        1. Deduplicate batch.
        2. Create table if missing.
        3. Ensure unique index.
        4. Convert to hypertable if needed.
        5. Filter already ingested rows.
        6. Insert remaining rows safely.

    Args:
        df (pd.DataFrame): DataFrame to insert.
        table_name (str): Table name.
        schema (str | None): Schema name.
        time_column (str): Timestamp column name.
        is_timeseries (bool): Convert table to Timescale hypertable if True.
    """
    if df.empty:
        logger.warning("Empty DataFrame, skipping insert")
        return

    engine = get_engine()
    schema = ensure_schema(schema)
    create_schema(schema)

    table = table_name

    # 1. Deduplicate
    df = df.drop_duplicates(subset=[time_column])

    # 2. Create table if missing
    df.head(0).to_sql(table, engine, schema=schema, if_exists="append", index=False)

    # 3. Self-heal
    ensure_unique_index(table, schema, time_column)
    if is_timeseries:
        ensure_hypertable(table, schema, time_column)

    # 4. Filter already ingested rows
    last_ts = get_last_date(table, schema, time_column)
    if last_ts:
        df = df[df[time_column] > last_ts]

    if df.empty:
        logger.info("No new rows to insert")
        return

    # 5. Safe insert with ON CONFLICT DO NOTHING
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
    """
    Read data from a table and log last 5 rows.

    Args:
        table_name (str): Table name.
        schema (str | None): Schema name.
        limit (int | None): Limit number of rows to fetch.

    Returns:
        pd.DataFrame: Retrieved rows.
    """
    engine = get_engine()
    schema = ensure_schema(schema)
    table = table_name

    query = f"SELECT * FROM {schema}.{table}"
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql_query(query, engine)

    if not df.empty:
        logger.info(f"\nFirst 5 rows from {schema}.{table}:\n")
        logger.info(df.tail(5))
    else:
        logger.info(f"No data found in {schema}.{table}")

    return df


# =====================================================
# Drop Helpers
# =====================================================
def drop_table(table_name: str, schema: str | None = None):
    """
    Drop a table after user confirmation.

    Args:
        table_name (str): Table name.
        schema (str | None): Schema name.
    """
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

def fetch_ohlcv_df(
    table_name: str,
    schema: str,
    time_column: str = "timestamp",
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data with UNIX timestamp column (int64).
    """
    df = read_df_from_db(table_name, schema, limit)

    if df.empty:
        return df

    # Ensure correct dtype
    df[time_column] = df[time_column].astype("int64")

    # Sort but DO NOT set index
    df = df.sort_values(time_column)

    return df