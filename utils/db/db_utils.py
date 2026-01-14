import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

# ---------------------------
# Engine Initialization
# ---------------------------
def get_engine(db_url=None):
    """
    Create and return a SQLAlchemy engine.
    """
    try:
        db_url = db_url or os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("Database URL not provided.")
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        print(f"[get_engine] Error: {e}")
        return None

# ---------------------------
# Schema Management
# ---------------------------
def create_schema(engine, schema="data_binance"):
    """
    Create schema if it doesn't exist.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
        print(f"[create_schema] Schema '{schema}' is ready.")
    except SQLAlchemyError as e:
        print(f"[create_schema] SQLAlchemyError: {e}")
    except Exception as e:
        print(f"[create_schema] Error: {e}")

def drop_schema(engine, schema="data_binance"):
    """
    Drop a schema if it exists.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE;"))
        print(f"[drop_schema] Schema '{schema}' dropped.")
    except SQLAlchemyError as e:
        print(f"[drop_schema] SQLAlchemyError: {e}")
    except Exception as e:
        print(f"[drop_schema] Error: {e}")

def ensure_schema_exists(engine, schema="data_binance"):
    """
    Ensure the schema exists, create if missing.
    """
    try:
        inspector = inspect(engine)
        if schema not in inspector.get_schema_names():
            create_schema(engine, schema)
        else:
            print(f"[ensure_schema_exists] Schema '{schema}' already exists.")
    except Exception as e:
        print(f"[ensure_schema_exists] Error: {e}")

# ---------------------------
# Table Management
# ---------------------------
def drop_table(engine, table_name, schema="data_binance"):
    """
    Drop a table if it exists.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{table_name} CASCADE;"))
        print(f"[drop_table] Table '{schema}.{table_name}' dropped.")
    except SQLAlchemyError as e:
        print(f"[drop_table] SQLAlchemyError: {e}")
    except Exception as e:
        print(f"[drop_table] Error: {e}")

# ---------------------------
# DataFrame Operations
# ---------------------------
def save_df_to_db(df: pd.DataFrame, table_name: str, engine, schema="data_binance"):
    """
    Save a pandas DataFrame to PostgreSQL under the given schema.
    """
    if df.empty:
        print("[save_df_to_db] No data to insert.")
        return

    try:
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="append",
            index=False,
            method='multi'
        )
        print(f"[save_df_to_db] Inserted {len(df)} rows into table '{schema}.{table_name}'.")
    except SQLAlchemyError as e:
        print(f"[save_df_to_db] SQLAlchemyError: {e}")
    except Exception as e:
        print(f"[save_df_to_db] Error: {e}")

def read_df_from_db(engine, table_name, schema="data_binance", limit=None):
    """
    Read a DataFrame from PostgreSQL table.
    """
    try:
        query = f"SELECT * FROM {schema}.{table_name}"
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql_query(query, engine)
        print(f"[read_df_from_db] Read {len(df)} rows from table '{schema}.{table_name}'.")
        return df
    except SQLAlchemyError as e:
        print(f"[read_df_from_db] SQLAlchemyError: {e}")
    except Exception as e:
        print(f"[read_df_from_db] Error: {e}")
    return pd.DataFrame()

# ---------------------------
# Database Info Utilities
# ---------------------------
def get_last_state(engine, table_name, schema="data_binance", order_by_column="timestamp"):
    """
    Get the last row of a table ordered by a specific column.
    """
    try:
        query = f"""
        SELECT * FROM {schema}.{table_name} 
        ORDER BY {order_by_column} DESC 
        LIMIT 1
        """
        df = pd.read_sql_query(query, engine)
        print(f"[get_last_state] Retrieved last row from '{schema}.{table_name}'.")
        return df
    except SQLAlchemyError as e:
        print(f"[get_last_state] SQLAlchemyError: {e}")
    except Exception as e:
        print(f"[get_last_state] Error: {e}")
    return pd.DataFrame()

def total_columns(engine, table_name, schema="data_binance"):
    """
    Get total number of columns in a table.
    """
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name, schema=schema)
        print(f"[total_columns] Table '{schema}.{table_name}' has {len(columns)} columns.")
        return len(columns)
    except SQLAlchemyError as e:
        print(f"[total_columns] SQLAlchemyError: {e}")
    except Exception as e:
        print(f"[total_columns] Error: {e}")
    return 0

def total_rows(engine, table_name, schema="data_binance"):
    """
    Get total number of rows in a table.
    """
    try:
        query = text(f"SELECT COUNT(*) FROM {schema}.{table_name}")
        
        with engine.connect() as conn:
            result = conn.execute(query)
            row_count = result.scalar()

        print(f"[total_rows] Table '{schema}.{table_name}' has {row_count} rows.")
        return row_count

    except SQLAlchemyError as e:
        print(f"[total_rows] SQLAlchemyError: {e}")
    except Exception as e:
        print(f"[total_rows] Error: {e}")

    return 0
