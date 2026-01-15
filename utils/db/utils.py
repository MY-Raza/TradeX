import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

from logging import get_logger

logger = get_logger(__name__)


# ---------------------------
# Engine Initialization
# ---------------------------
def get_engine(db_url: str | None = None):
    """
    Create and return a SQLAlchemy engine.

    Args:
        db_url (str | None): Database connection URL. If None, reads from DATABASE_URL env.

    Returns:
        Engine | None: SQLAlchemy engine or None if creation fails.
    """
    try:
        db_url = db_url or os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not provided.")

        engine = create_engine(db_url)
        logger.info("Database engine created successfully.")
        return engine

    except Exception as e:
        logger.exception("Failed to create database engine.")
        return None


# ---------------------------
# Schema Management
# ---------------------------
def create_schema(engine, schema: str | None = None):
    """
    Create a schema if it does not exist.

    Args:
        engine: SQLAlchemy engine
        schema (str | None): Schema name. If None, defaults to PostgreSQL default schema.
    """
    if not schema:
        logger.info("Schema not provided. Skipping schema creation.")
        return

    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
        logger.info(f"Schema '{schema}' is ready.")

    except SQLAlchemyError:
        logger.exception(f"Failed to create schema '{schema}'.")


def drop_schema(engine, schema: str):
    """
    Drop a schema after user confirmation.

    Args:
        engine: SQLAlchemy engine
        schema (str): Schema name
    """
    confirm = input(f"⚠️ Drop schema '{schema}'? This is irreversible (yes/no): ").lower()
    if confirm != "yes":
        logger.warning("Schema drop cancelled by user.")
        return

    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE;"))
        logger.warning(f"Schema '{schema}' dropped.")

    except SQLAlchemyError:
        logger.exception(f"Failed to drop schema '{schema}'.")


def ensure_schema_exists(engine, schema: str | None = None):
    """
    Ensure a schema exists. Create it if missing.

    Args:
        engine: SQLAlchemy engine
        schema (str | None): Schema name
    """
    if not schema:
        logger.info("No schema specified. Using default schema.")
        return

    try:
        inspector = inspect(engine)
        if schema not in inspector.get_schema_names():
            create_schema(engine, schema)
        else:
            logger.info(f"Schema '{schema}' already exists.")

    except Exception:
        logger.exception("Failed to ensure schema exists.")


# ---------------------------
# Table Management
# ---------------------------
def drop_table(engine, table_name: str, schema: str | None = None):
    """
    Drop a table after user confirmation.

    Args:
        engine: SQLAlchemy engine
        table_name (str): Table name
        schema (str | None): Schema name
    """
    full_name = f"{schema}.{table_name}" if schema else table_name
    confirm = input(f"⚠️ Drop table '{full_name}'? (yes/no): ").lower()

    if confirm != "yes":
        logger.warning("Table drop cancelled by user.")
        return

    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {full_name} CASCADE;"))
        logger.warning(f"Table '{full_name}' dropped.")

    except SQLAlchemyError:
        logger.exception(f"Failed to drop table '{full_name}'.")


# ---------------------------
# DataFrame Operations
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
    Save a pandas DataFrame to PostgreSQL.

    Args:
        df (pd.DataFrame): Data to insert
        table_name (str): Target table
        engine: SQLAlchemy engine
        schema (str | None): Schema name
        time_column (str | None): Time column for TimescaleDB
        is_timeseries (bool): Convert to hypertable if True
    """
    if df.empty:
        logger.warning("DataFrame is empty. Nothing to insert.")
        return

    try:
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="append",
            index=False,
            method="multi"
        )

        logger.info(f"Inserted {len(df)} rows into '{table_name}'.")

        if is_timeseries:
            if not time_column:
                raise ValueError("time_column is required for timeseries tables.")

            full_name = f"{schema}.{table_name}" if schema else table_name

            with engine.begin() as conn:
                conn.execute(text(f"""
                    SELECT create_hypertable(
                        '{full_name}',
                        '{time_column}',
                        migrate_data => TRUE,
                        if_not_exists => TRUE
                    );
                """))

            logger.info(f"Hypertable ensured for '{full_name}'.")

    except Exception:
        logger.exception("Failed to save DataFrame to database.")


def read_df_from_db(engine, table_name: str, schema: str | None = None, limit: int | None = None):
    """
    Read a table into a pandas DataFrame.

    Args:
        engine: SQLAlchemy engine
        table_name (str): Table name
        schema (str | None): Schema name
        limit (int | None): Row limit

    Returns:
        pd.DataFrame
    """
    try:
        full_name = f"{schema}.{table_name}" if schema else table_name
        query = f"SELECT * FROM {full_name}"
        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, engine)
        logger.info(f"Read {len(df)} rows from '{full_name}'.")
        return df

    except Exception:
        logger.exception("Failed to read table.")
        return pd.DataFrame()


# ---------------------------
# Database Info Utilities
# ---------------------------
def total_rows(engine, table_name: str, schema: str | None = None):
    """
    Get total number of rows in a table.

    Args:
        engine: SQLAlchemy engine
        table_name (str): Table name
        schema (str | None): Schema name

    Returns:
        int
    """
    try:
        full_name = f"{schema}.{table_name}" if schema else table_name
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {full_name}"))
            count = result.scalar()

        logger.info(f"'{full_name}' contains {count} rows.")
        return count

    except Exception:
        logger.exception("Failed to count rows.")
        return 0


def get_table_last_state(
    engine,
    table_name: str,
    schema: str | None = None,
    order_by_column: str = "timestamp"
) -> pd.DataFrame:
    """
    Retrieve the latest row from a table ordered by a specific column.

    Args:
        engine: SQLAlchemy engine instance
        table_name (str): Table name
        schema (str | None): Schema name (defaults to PostgreSQL default schema)
        order_by_column (str): Column used for ordering

    Returns:
        pd.DataFrame: DataFrame containing the latest row or empty DataFrame
    """
    try:
        full_name = f"{schema}.{table_name}" if schema else table_name

        query = f"""
        SELECT *
        FROM {full_name}
        ORDER BY {order_by_column} DESC
        LIMIT 1
        """

        df = pd.read_sql_query(query, engine)
        logger.info(f"Retrieved last row from '{full_name}'.")
        return df

    except SQLAlchemyError:
        logger.exception(f"SQLAlchemy error while fetching last row from '{table_name}'.")
    except Exception:
        logger.exception("Unexpected error while fetching last table state.")

    return pd.DataFrame()


def total_columns(
    engine,
    table_name: str,
    schema: str | None = None
) -> int:
    """
    Get the total number of columns in a table.

    Args:
        engine: SQLAlchemy engine instance
        table_name (str): Table name
        schema (str | None): Schema name (defaults to PostgreSQL default schema)

    Returns:
        int: Number of columns in the table
    """
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name, schema=schema)

        full_name = f"{schema}.{table_name}" if schema else table_name
        logger.info(f"Table '{full_name}' has {len(columns)} columns.")

        return len(columns)

    except SQLAlchemyError:
        logger.exception(f"SQLAlchemy error while inspecting table '{table_name}'.")
    except Exception:
        logger.exception("Unexpected error while counting table columns.")

    return 0

