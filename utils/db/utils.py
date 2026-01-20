import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from TradeX.utils.common.logs import get_logger

logger = get_logger("utils")

# =====================================================
# Globals
# =====================================================
_ENGINE = None
USER_SCHEMA: str | None = None


# =====================================================
# Engine
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

    confirm = input(
        f"⚠️ DROP schema '{schema}' and ALL objects? Type 'yes' to continue: "
    ).lower()

    if confirm != "yes":
        logger.warning("Schema drop cancelled")
        return

    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))

    logger.warning(f"Schema '{schema}' dropped")


# =====================================================
# Table Repair Helpers (CRITICAL)
# =====================================================
def ensure_unique_index(
    table_name: str,
    schema: str,
    time_column: str
):
    """
    Ensures a UNIQUE INDEX exists.
    Works even for OLD tables and hypertables.
    """
    engine = get_engine()
    index_name = f"{table_name}_{time_column}_uidx"

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
            ON {schema}.{table_name} ({time_column});
        """))

    logger.info(f"Unique index ensured: {index_name}")


def ensure_hypertable(
    table_name: str,
    schema: str,
    time_column: str
):
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
def get_last_date(
    table_name: str,
    schema: str,
    time_column: str
) -> int | None:
    engine = get_engine()
    inspector = inspect(engine)

    if not inspector.has_table(table_name, schema=schema):
        return None

    query = f"SELECT MAX({time_column}) FROM {schema}.{table_name}"
    with engine.begin() as conn:
        return conn.execute(text(query)).scalar()


def save_df_to_db(
    df: pd.DataFrame,
    table_name: str,
    schema: str | None = None,
    time_column: str = "timestamp",
    is_timeseries: bool = True
):
    if df.empty:
        logger.warning("Empty DataFrame, skipping insert")
        return

    engine = get_engine()
    schema = ensure_schema(schema)
    create_schema(schema)

    table = f"{table_name}_1m"

    # -------------------------------------------------
    # 1. Deduplicate incoming batch
    # -------------------------------------------------
    df = df.drop_duplicates(subset=[time_column])

    # -------------------------------------------------
    # 2. Create table if missing
    # -------------------------------------------------
    df.head(0).to_sql(
        table,
        engine,
        schema=schema,
        if_exists="append",
        index=False
    )

    # -------------------------------------------------
    # 3. SELF-HEAL table (this fixes Binance issue)
    # -------------------------------------------------
    ensure_unique_index(table, schema, time_column)

    if is_timeseries:
        ensure_hypertable(table, schema, time_column)

    # -------------------------------------------------
    # 4. Filter already ingested rows
    # -------------------------------------------------
    last_ts = get_last_date(table, schema, time_column)
    if last_ts:
        df = df[df[time_column] > last_ts]

    if df.empty:
        logger.info("No new rows to insert")
        return

    # -------------------------------------------------
    # 5. Safe insert (NOW guaranteed to work)
    # -------------------------------------------------
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
def read_df_from_db(
    table_name: str,
    schema: str | None = None,
    limit: int | None = None
) -> pd.DataFrame:
    engine = get_engine()
    schema = ensure_schema(schema)
    table = f"{table_name}_1m"

    query = f"SELECT * FROM {schema}.{table}"
    if limit:
        query += f" LIMIT {limit}"

    return pd.read_sql_query(query, engine)


# =====================================================
# Drop Helpers
# =====================================================
def drop_table(table_name: str, schema: str | None = None):
    engine = get_engine()
    schema = ensure_schema(schema)
    table = f"{table_name}_1m"

    confirm = input(f"Drop table {schema}.{table}? (yes/no): ").lower()
    if confirm != "yes":
        logger.warning("Table drop cancelled")
        return

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{table} CASCADE"))

    logger.warning(f"Table {schema}.{table} dropped")
