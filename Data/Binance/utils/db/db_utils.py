from sqlalchemy import create_engine, text
import pandas as pd
import os

class DBUtils:
    def __init__(self, db_url=None, schema="data_binance"):
        """
        Initialize SQLAlchemy engine and schema.
        """
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.engine = create_engine(self.db_url)
        self.schema = schema
        self.create_schema()

    def create_schema(self):
        """
        Create schema if it doesn't exist.
        """
        with self.engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema};"))
            conn.commit()

    def save_dataframe(self, df: pd.DataFrame, table_name: str):
        """
        Save a pandas DataFrame to PostgreSQL under the given schema.
        """
        if df.empty:
            print("No data to insert.")
            return

        df.to_sql(
            table_name,
            self.engine,
            schema=self.schema,
            if_exists="append",
            index=False,
            method='multi'
        )
        print(f"Inserted {len(df)} rows into table '{self.schema}.{table_name}'.")
