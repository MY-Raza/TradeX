import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from TradeX.logs.logging import get_logger

logger = get_logger(__name__)

# ---------------------------
# Global variable to store the schema
# ---------------------------
USER_SCHEMA: str | None = None

# ---------------------------
# Schema Utilities
# ---------------------------
def resolve_schema(schema: str | None) -> str:
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
def get_engine(db_url: str | None = None):
    """
    Create a SQLAlchemy engine to connect to the database.

    Args:
        db_url (str | None): Optional database URL. 
                             If not provided, read from 'DATABASE_URL' environment variable.

    Returns:
        Engine | None: SQLAlchemy engine object or None if creation fails.
    """
    try:
        db_url = db_url or os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not provided.")
        engine = create_engine(db_url)
        logger.info("Database engine created successfully.")
        return engine
    except Exception:
        logger.exception("Failed to create engine.")
        return None

# ---------------------------
# Schema Management
# ---------------------------
def create_schema(engine, schema: str | None = None):
    """
    Create a schema if it does not exist.

    Args:
        engine: SQLAlchemy engine.
        schema (str | None): Optional schema name.
    """
    try:
        schema = resolve_schema(schema)
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
        logger.info(f"Schema '{schema}' is ready.")
    except Exception:
        logger.exception("Failed to create schema.")

def drop_schema(engine, schema: str | None = None):
    """
    Drop a schema after user confirmation.

    Args:
        engine: SQLAlchemy engine.
        schema (str | None): Optional schema name.
    """
    try:
        schema = resolve_schema(schema)
        confirm = input(f"Drop schema '{schema}'? (yes/no): ").lower()
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
    engine,
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
    try:
        schema = resolve_schema(schema)
        # Insert DataFrame into database
        df.to_sql(table_name, engine, schema=schema, if_exists="append", index=False, method="multi")
        logger.info(f"Inserted {len(df)} rows into '{schema}.{table_name}'.")

        # Convert table to hypertable if required
        if is_timeseries and time_column:
            with engine.begin() as conn:
                conn.execute(text(f"""
                    SELECT create_hypertable('{schema}.{table_name}', '{time_column}', migrate_data => TRUE, if_not_exists => TRUE);
                """))
            logger.info(f"Hypertable ensured for '{schema}.{table_name}' on column '{time_column}'.")
    except Exception:
        logger.exception("Failed to save DataFrame to database.")

def read_df_from_db(engine, table_name: str, schema: str | None = None, limit: int | None = None) -> pd.DataFrame:
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
    try:
        schema = resolve_schema(schema)
        query = f"SELECT * FROM {schema}.{table_name}"
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql_query(query, engine)
        logger.info(f"Read {len(df)} rows from '{schema}.{table_name}'.")
        return df
    except Exception:
        logger.exception("Failed to read table.")
        return pd.DataFrame()

# ---------------------------
# Table Information Utilities
# ---------------------------
def total_rows(engine, table_name: str, schema: str | None = None) -> int:
    """
    Get total number of rows in a table.

    Args:
        engine: SQLAlchemy engine.
        table_name (str): Table name.
        schema (str | None): Optional schema name.

    Returns:
        int: Total row count, 0 if error occurs.
    """
    try:
        schema = resolve_schema(schema)
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{table_name}"))
            return result.scalar()
    except Exception:
        logger.exception("Failed to count rows.")
        return 0

def total_columns(engine, table_name: str, schema: str | None = None) -> int:
    """
    Get total number of columns in a table.

    Args:
        engine: SQLAlchemy engine.
        table_name (str): Table name.
        schema (str | None): Optional schema name.

    Returns:
        int: Total number of columns, 0 if error occurs.
    """
    try:
        schema = resolve_schema(schema)
        inspector = inspect(engine)
        return len(inspector.get_columns(table_name, schema=schema))
    except Exception:
        logger.exception("Failed to count columns.")
        return 0

def drop_table(engine, table_name: str, schema: str | None = None):
    """
    Drop a table after user confirmation.

    Args:
        engine: SQLAlchemy engine.
        table_name (str): Name of the table to drop.
        schema (str | None): Optional schema name.
    """
    try:
        # Resolve schema name
        schema = resolve_schema(schema)
        full_name = f"{schema}.{table_name}"

        # Ask user for confirmation before dropping
        confirm = input(f"⚠️ Drop table '{full_name}'? (yes/no): ").lower()
        if confirm != "yes":
            logger.warning("Table drop cancelled by user.")
            return

        # Drop the table
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {full_name} CASCADE;"))

        logger.warning(f"Table '{full_name}' dropped.")

    except Exception:
        logger.exception("Failed to drop table.")
