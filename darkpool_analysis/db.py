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
        CREATE TABLE IF NOT EXISTS finra_otc_weekly_raw (
            symbol TEXT,
            week_start_date DATE,
            off_exchange_volume DOUBLE,
            trade_count DOUBLE,
            tier_identifier TEXT,
            tier_description TEXT,
            issue_name TEXT,
            market_participant_name TEXT,
            mpid TEXT,
            last_update_date DATE,
            source_file TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS finra_short_daily_raw (
            symbol TEXT,
            trade_date DATE,
            short_volume DOUBLE,
            short_exempt_volume DOUBLE,
            total_volume DOUBLE,
            market TEXT,
            source TEXT,
            source_file TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS polygon_equity_trades_raw (
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
        CREATE TABLE IF NOT EXISTS polygon_daily_agg_raw (
            symbol TEXT,
            trade_date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            vwap DOUBLE,
            volume DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lit_direction_daily (
            symbol TEXT,
            date DATE,
            lit_buy_volume DOUBLE,
            lit_sell_volume DOUBLE,
            lit_buy_ratio DOUBLE,
            log_buy_sell DOUBLE,
            classification_method TEXT,
            lit_coverage_pct DOUBLE,
            inference_version TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_metrics (
            date DATE,
            symbol TEXT,
            log_buy_sell DOUBLE,
            short_volume DOUBLE,
            short_exempt_volume DOUBLE,
            short_total_volume DOUBLE,
            short_ratio DOUBLE,
            short_ratio_z DOUBLE,
            short_ratio_denominator_type TEXT,
            short_ratio_denominator_value DOUBLE,
            short_ratio_source TEXT,
            close DOUBLE,
            vwap DOUBLE,
            high DOUBLE,
            low DOUBLE,
            volume DOUBLE,
            return_1d DOUBLE,
            return_z DOUBLE,
            range_pct DOUBLE,
            otc_off_exchange_volume DOUBLE,
            otc_week_used DATE,
            data_quality TEXT,
            has_otc BOOLEAN,
            has_short BOOLEAN,
            has_lit BOOLEAN,
            has_price BOOLEAN,
            pressure_context_label TEXT,
            inference_version TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_constituent_short_agg_daily (
            index_symbol TEXT,
            trade_date DATE,
            total_short_volume DOUBLE,
            total_denominator_volume DOUBLE,
            denominator_type TEXT,
            agg_short_ratio DOUBLE,
            agg_short_ratio_z DOUBLE,
            coverage_count INTEGER,
            expected_constituent_count INTEGER,
            coverage_pct DOUBLE,
            index_price_symbol TEXT,
            index_price_source TEXT,
            index_price_return DOUBLE,
            index_return_z DOUBLE,
            interpretation_label TEXT,
            data_quality TEXT,
            inference_version TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS composite_signal (
            symbol TEXT,
            date DATE,
            directional_score DOUBLE,
            pressure_score DOUBLE,
            participation_score DOUBLE,
            composite_score DOUBLE,
            final_label TEXT,
            data_quality TEXT,
            inference_version TEXT
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
    columns = list(df.columns)
    column_list = ", ".join(columns)
    conn.execute(
        f"INSERT INTO {table_name} ({column_list}) SELECT {column_list} FROM df_view"
    )
    conn.unregister("df_view")
