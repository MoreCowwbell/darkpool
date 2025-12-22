from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


def get_connection(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS finra_otc_volume_raw (
            symbol TEXT,
            week_start_date DATE,
            off_exchange_volume DOUBLE,
            trade_count DOUBLE,
            source TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS equity_trades_raw (
            symbol TEXT,
            timestamp TIMESTAMP,
            price DOUBLE,
            size DOUBLE,
            bid DOUBLE,
            ask DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS equity_lit_directional_flow (
            symbol TEXT,
            date DATE,
            lit_buy_volume DOUBLE,
            lit_sell_volume DOUBLE,
            lit_buy_ratio DOUBLE,
            classification_method TEXT,
            lit_coverage_pct DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS darkpool_estimated_flow (
            symbol TEXT,
            date DATE,
            finra_off_exchange_volume DOUBLE,
            estimated_dark_buy_volume DOUBLE,
            estimated_dark_sell_volume DOUBLE,
            applied_lit_buy_ratio DOUBLE,
            inference_version TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS darkpool_daily_summary (
            date DATE,
            symbol TEXT,
            estimated_bought DOUBLE,
            estimated_sold DOUBLE,
            buy_ratio DOUBLE,
            total_off_exchange_volume DOUBLE,
            finra_period_type TEXT
        )
        """
    )


def upsert_dataframe(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    df: pd.DataFrame,
    key_columns: Iterable[str],
) -> None:
    if df.empty:
        return
    conn.register("df_view", df)
    keys = list(key_columns)
    if keys:
        join_clause = " AND ".join([f"{table_name}.{col} = df_view.{col}" for col in keys])
        conn.execute(f"DELETE FROM {table_name} USING df_view WHERE {join_clause}")
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM df_view")
    conn.unregister("df_view")
