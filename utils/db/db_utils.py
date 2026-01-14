from sqlalchemy import create_engine, text
import pandas as pd
import os

# Initialize engine globally (you can also pass db_url to functions if needed)
def get_engine(db_url=None):
    """
    Create and return a SQLAlchemy engine.
    """
    db_url = db_url or os.getenv("DATABASE_URL")
    return create_engine(db_url)

def create_schema(engine, schema="data_binance"):
    """
    Create schema if it doesn't exist.
    """
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
        conn.commit()
    print(f"Schema '{schema}' is ready.")

def save_df_to_db(df: pd.DataFrame, table_name: str, engine, schema="data_binance"):
    """
    Save a pandas DataFrame to PostgreSQL under the given schema.
    """
    if df.empty:
        print("No data to insert.")
        return

    df.to_sql(
        table_name,
        engine,
        schema=schema,
        if_exists="append",
        index=False,
        method='multi'
    )
    print(f"Inserted {len(df)} rows into table '{schema}.{table_name}'.")
