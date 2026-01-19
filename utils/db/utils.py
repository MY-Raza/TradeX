import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect 
from TradeX.utils.common.logs import get_logger
from datetime import datetime


logger = get_logger("utils")

# ---------------------------
# Global variable to store the schema
# ---------------------------
USER_SCHEMA: str | None = None

# ---------------------------
# Schema Utilities
# ---------------------------
def ensure_schema(schema: str | None) -> str:
    """
    Resolve the database schema to use.

    If a schema is explicitly provided, it is used. 
    Otherwise, the user is prompted to enter one. 
    The resolved schema is stored globally to avoid repeated prompts.

    Args:
        schema (str | None): Optional schema name.

    Returns:
        str: Valid schema name.

    Raises:
        ValueError: If no schema is provided or entered.
    """
    global USER_SCHEMA
    if schema:  # If schema provided, use it
        USER_SCHEMA = schema
        return schema
    if USER_SCHEMA:  # Return previously stored schema
        return USER_SCHEMA

    # Prompt user for schema
    user_schema = input("Enter schema name: ").strip()
    if not user_schema:
        raise ValueError("Schema cannot be empty.")
    
    USER_SCHEMA = user_schema
    return USER_SCHEMA

# ---------------------------
# Engine Utilities
# ---------------------------
_ENGINE = None


def get_engine(db_url: str | None = None):
    """
    Return a singleton SQLAlchemy engine.
    Engine is created only once and reused everywhere.
    """
    global _ENGINE

    if _ENGINE is not None:
        return _ENGINE

    try:
        db_url = db_url or os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not provided.")

        _ENGINE = create_engine(db_url, pool_pre_ping=True)
        logger.info("Database engine created successfully.")
        return _ENGINE

    except Exception:
        logger.exception("Failed to create engine.")
        return None

# ---------------------------
# Schema Management
# ---------------------------
def create_schema(schema: str | None = None):
    """
    Create a schema if it does not exist.

    Args:
        engine: SQLAlchemy engine.
        schema (str | None): Optional schema name.
    """
    engine = get_engine()
    try:
        schema = ensure_schema(schema)
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
        logger.info(f"Schema '{schema}' is ready.")
    except Exception:
        logger.exception("Failed to create schema.")

def drop_schema(schema: str | None = None):
    """
    Drop a schema after user confirmation.

    Args:
        engine: SQLAlchemy engine.
        schema (str | None): Optional schema name.
    """
    engine = get_engine()
    try:
        schema = ensure_schema(schema)
        confirm = input(f"Are you sure to drop '{schema}'? (yes/no): ").lower()
        if confirm != "yes":
            logger.warning("Schema drop cancelled.")
            return
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE;"))
        logger.warning(f"Schema '{schema}' dropped.")
    except Exception:
        logger.exception("Failed to drop schema.")

# ---------------------------
# DataFrame Storage Utilities
# ---------------------------
def save_df_to_db(
    df: pd.DataFrame,
    table_name: str,
    schema: str | None = None,
    time_column: str | None = None,
    is_timeseries: bool = False
):
    """
    Save a pandas DataFrame to the database.

    Args:
        df (pd.DataFrame): DataFrame to save.
        table_name (str): Target table name.
        engine: SQLAlchemy engine.
        schema (str | None): Optional schema name.
        time_column (str | None): Column to use as time for timeseries.
        is_timeseries (bool): If True, create a TimescaleDB hypertable.
    """
    if df.empty:
        logger.warning("DataFrame empty. Nothing to insert.")
        return
    engine = get_engine()
    try:
        schema = ensure_schema(schema)
        create_schema(schema=schema)
        # Insert DataFrame into database
        df.to_sql(table_name + "_1m", engine, schema=schema, if_exists="append", index=False, method="multi")
        logger.info(f"Inserted {len(df)} rows into '{schema}.{table_name}_1m'.")


        # Convert table to hypertable if required
        if is_timeseries and time_column:
            with engine.begin() as conn:
                conn.execute(text(f"""
                    SELECT create_hypertable('{schema}.{table_name+"_1m"} ', '{time_column}', migrate_data => TRUE, if_not_exists => TRUE);
                """))
            logger.info(f"Hypertable ensured for '{schema}.{table_name}_1m' on column '{time_column}'.")
    except Exception:
        logger.exception("Failed to save DataFrame to database.")

def read_df_from_db(table_name: str, schema: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """
    Read data from a table into a pandas DataFrame.

    Args:
        engine: SQLAlchemy engine.
        table_name (str): Table to read from.
        schema (str | None): Optional schema name.
        limit (int | None): Optional number of rows to read.

    Returns:
        pd.DataFrame: DataFrame containing table data.
    """
    engine = get_engine()
    try:
        schema = ensure_schema(schema)
        query = f"SELECT * FROM {schema}.{table_name}_1m"
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql_query(query, engine)
        logger.info(f"Read {len(df)} rows from '{schema}.{table_name}_1m'.")
        return df
    except Exception:
        logger.exception("Failed to read table.")
        return pd.DataFrame()

# ---------------------------
# Table Information Utilities
# ---------------------------

def drop_table(table_name : str, schema: str | None = None):
    """
    Drop a table after user confirmation.

    Args:
        engine: SQLAlchemy engine.
        table_name (str): Name of the table to drop.
        schema (str | None): Optional schema name.
    """
    table_name = table_name +"_1m"
    engine = get_engine()
    try:
        # Resolve schema name
        schema = ensure_schema(schema)
        full_name = f"{schema}.{table_name}"

        # Ask user for confirmation before dropping
        confirm = input(f"Are you sure to drop table '{full_name}'? (yes/no): ").lower()
        if confirm != "yes":
            logger.warning("Table drop cancelled by user.")
            return

        # Drop the table
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {full_name} CASCADE;"))

        logger.warning(f"Table '{full_name}' dropped.")

    except Exception:
        logger.exception("Failed to drop table.")

def get_last_date(table_name: str, schema: str | None = None, time_column: str = "timestamp") -> datetime | None:
    """
    Fetch the latest timestamp from a table and convert it to a datetime object.

    Args:
        engine: SQLAlchemy engine.
        table_name (str): Name of the table to query.
        schema (str | None): Optional schema name. If None, resolves via `ensure_schema`.
        time_column (str, optional): Column storing the timestamp in milliseconds. Defaults to "timestamp".

    Returns:
        datetime | None: Latest timestamp as a datetime object, or None if table is empty or error occurs.

    Notes:
        - Assumes timestamps are stored in **milliseconds** since epoch.
        - Converts the timestamp to a timezone-naive UTC datetime.
    """
    engine = get_engine()
    try:
        inspector = inspect(engine)
        full_table_name = f"{table_name}_1m"
        if not inspector.has_table(full_table_name, schema=schema):
          logger.info(f"Table '{schema}.{full_table_name}' does not exist. Will start from config start_date.")
          return None
        schema = ensure_schema(schema)
        query = f"SELECT MAX({time_column}) as last_ts FROM {schema}.{table_name}_1m"
        with engine.begin() as conn:
            result = conn.execute(text(query)).scalar()

        # Convert milliseconds timestamp to datetime
        last_dt = datetime.utcfromtimestamp(result / 1000)
        return last_dt

    except Exception:
        logger.exception(f"Failed to fetch last timestamp from '{table_name}_1m'.")
        return None 