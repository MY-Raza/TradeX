import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

from TradeX.logs.logging import get_logger

logger = get_logger(__name__)


# ---------------------------
# Helper Utilities
# ---------------------------
def resolve_schema(schema: str | None) -> str:
    """
    Resolve schema value. Prompt user if schema is not provided.

    Args:
        schema (str | None): Schema name

    Returns:
        str: Valid schema name

    Raises:
        ValueError: If schema is empty
    """
    if schema:
        return schema

    user_schema = input("🔎 Please enter schema name: ").strip()
    if not user_schema:
        raise ValueError("Schema name cannot be empty.")

    logger.info(f"Schema provided by user: '{user_schema}'")
    return user_schema


# ---------------------------
# Engine Initialization
# ---------------------------
def get_engine(db_url: str | None = None):
    """
    Create and return a SQLAlchemy engine.

    Args:
        db_url (str | None): Database connection URL. If None, reads from DATABASE_URL.

    Returns:
        Engine | None
    """
    try:
        db_url = db_url or os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not provided.")

        engine = create_engine(db_url)
        logger.info("Database engine created successfully.")
        return engine

    except Exception:
        logger.exception("Failed to create database engine.")
        return None


# ---------------------------
# Schema Management
# ---------------------------
def create_schema(engine, schema: str | None = None):
    """
    Create a schema if it does not exist.
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
    """
    try:
        schema = resolve_schema(schema)

        confirm = input(
            f"⚠️ Drop schema '{schema}'? This is irreversible (yes/no): "
        ).lower()

        if confirm != "yes":
            logger.warning("Schema drop cancelled by user.")
            return

        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE;"))

        logger.warning(f"Schema '{schema}' dropped.")

    except Exception:
        logger.exception("Failed to drop schema.")


def ensure_schema_exists(engine, schema: str | None = None):
    """
    Ensure a schema exists, create it if missing.
    """
    try:
        schema = resolve_schema(schema)
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
    """
    try:
        schema = resolve_schema(schema)
        full_name = f"{schema}.{table_name}"

        confirm = input(f"⚠️ Drop table '{full_name}'? (yes/no): ").lower()
        if confirm != "yes":
            logger.warning("Table drop cancelled by user.")
            return

        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {full_name} CASCADE;"))

        logger.warning(f"Table '{full_name}' dropped.")

    except Exception:
        logger.exception("Failed to drop table.")


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
    """
    if df.empty:
        logger.warning("DataFrame is empty. Nothing to insert.")
        return

    try:
        schema = resolve_schema(schema)

        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="append",
            index=False,
            method="multi"
        )

        logger.info(f"Inserted {len(df)} rows into '{schema}.{table_name}'.")

        if is_timeseries:
            if not time_column:
                raise ValueError("time_column is required for timeseries tables.")

            with engine.begin() as conn:
                conn.execute(text(f"""
                    SELECT create_hypertable(
                        '{schema}.{table_name}',
                        '{time_column}',
                        migrate_data => TRUE,
                        if_not_exists => TRUE
                    );
                """))

            logger.info(
                f"Hypertable ensured for '{schema}.{table_name}' "
                f"on column '{time_column}'."
            )

    except Exception:
        logger.exception("Failed to save DataFrame to database.")


def read_df_from_db(
    engine,
    table_name: str,
    schema: str | None = None,
    limit: int | None = None
) -> pd.DataFrame:
    """
    Read a table into a pandas DataFrame.
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
# Database Info Utilities
# ---------------------------
def total_rows(engine, table_name: str, schema: str | None = None) -> int:
    """
    Get total number of rows in a table.
    """
    try:
        schema = resolve_schema(schema)

        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM {schema}.{table_name}")
            )
            count = result.scalar()

        logger.info(f"'{schema}.{table_name}' contains {count} rows.")
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
    """
    try:
        schema = resolve_schema(schema)

        query = f"""
        SELECT *
        FROM {schema}.{table_name}
        ORDER BY {order_by_column} DESC
        LIMIT 1
        """

        df = pd.read_sql_query(query, engine)
        logger.info(f"Retrieved last row from '{schema}.{table_name}'.")
        return df

    except Exception:
        logger.exception("Failed to fetch last table state.")
        return pd.DataFrame()


def total_columns(engine, table_name: str, schema: str | None = None) -> int:
    """
    Get the total number of columns in a table.
    """
    try:
        schema = resolve_schema(schema)
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name, schema=schema)

        logger.info(
            f"Table '{schema}.{table_name}' has {len(columns)} columns."
        )
        return len(columns)

    except Exception:
        logger.exception("Failed to count table columns.")
        return 0
