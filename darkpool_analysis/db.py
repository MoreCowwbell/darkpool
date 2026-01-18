from __future__ import annotations

from datetime import date, datetime
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
            short_sell_volume DOUBLE,
            short_ratio DOUBLE,
            short_ratio_z DOUBLE,
            short_buy_sell_ratio DOUBLE,
            short_buy_sell_ratio_z DOUBLE,
            combined_ratio DOUBLE,
            vw_flow DOUBLE,
            vw_flow_z DOUBLE,
            finra_buy_volume DOUBLE,
            finra_buy_volume_z DOUBLE,
            short_ratio_denominator_type TEXT,
            short_ratio_denominator_value DOUBLE,
            short_ratio_source TEXT,
            lit_buy_volume DOUBLE,
            lit_sell_volume DOUBLE,
            lit_total_volume DOUBLE,
            lit_buy_ratio DOUBLE,
            lit_buy_ratio_z DOUBLE,
            lit_flow_imbalance DOUBLE,
            lit_flow_imbalance_z DOUBLE,
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
            otc_weekly_buy_ratio DOUBLE,
            otc_buy_volume DOUBLE,
            otc_sell_volume DOUBLE,
            otc_buy_ratio_z DOUBLE,
            otc_status TEXT,
            weekly_total_volume DOUBLE,
            otc_participation_rate DOUBLE,
            otc_participation_z DOUBLE,
            otc_participation_delta DOUBLE,
            accumulation_score DOUBLE,
            accumulation_score_display DOUBLE,
            confidence DOUBLE,
            data_quality TEXT,
            has_otc BOOLEAN,
            has_short BOOLEAN,
            has_lit BOOLEAN,
            has_price BOOLEAN,
            pressure_context_label TEXT,
            inference_version TEXT,
            accumulation_short_z_source TEXT
        )
        """
    )
    daily_metrics_columns = [
        "short_sell_volume DOUBLE",
        "short_buy_sell_ratio DOUBLE",
        "short_buy_sell_ratio_z DOUBLE",
        "combined_ratio DOUBLE",
        "vw_flow DOUBLE",
        "vw_flow_z DOUBLE",
        "finra_buy_volume DOUBLE",
        "finra_buy_volume_z DOUBLE",
        "lit_buy_volume DOUBLE",
        "lit_sell_volume DOUBLE",
        "lit_total_volume DOUBLE",
        "lit_buy_ratio DOUBLE",
        "lit_buy_ratio_z DOUBLE",
        "lit_flow_imbalance DOUBLE",
        "lit_flow_imbalance_z DOUBLE",
        "otc_weekly_buy_ratio DOUBLE",
        "otc_buy_volume DOUBLE",
        "otc_sell_volume DOUBLE",
        "otc_buy_ratio_z DOUBLE",
        "otc_status TEXT",
        "weekly_total_volume DOUBLE",
        "otc_participation_rate DOUBLE",
        "otc_participation_z DOUBLE",
        "otc_participation_delta DOUBLE",
        "accumulation_score DOUBLE",
        "accumulation_score_display DOUBLE",
        "confidence DOUBLE",
        "accumulation_short_z_source TEXT",
    ]
    for column_def in daily_metrics_columns:
        conn.execute(f"ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS {column_def}")
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
    # Polygon ingestion state for caching (avoid re-fetching)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS polygon_ingestion_state (
            symbol TEXT,
            trade_date DATE,
            data_source TEXT,
            fetch_timestamp TIMESTAMP,
            record_count INTEGER,
            status TEXT,
            PRIMARY KEY (symbol, trade_date, data_source)
        )
        """
    )
    # Add data_source column to polygon_equity_trades_raw if missing
    conn.execute(
        "ALTER TABLE polygon_equity_trades_raw ADD COLUMN IF NOT EXISTS data_source TEXT DEFAULT 'tick'"
    )
    # Add fetch_timestamp column to polygon_daily_agg_raw if missing
    conn.execute(
        "ALTER TABLE polygon_daily_agg_raw ADD COLUMN IF NOT EXISTS fetch_timestamp TIMESTAMP"
    )
    # Options premium tables for ATM premium panel
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS options_premium_daily (
            symbol TEXT,
            trade_date DATE,
            expiration_date DATE,
            expiration_type TEXT,
            strike DOUBLE,
            option_type TEXT,
            premium DOUBLE,
            volume DOUBLE,
            close_price DOUBLE,
            fetch_timestamp TIMESTAMP,
            PRIMARY KEY (symbol, trade_date, expiration_date, strike, option_type)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS options_premium_summary (
            symbol TEXT,
            trade_date DATE,
            expiration_type TEXT,
            total_call_premium DOUBLE,
            total_put_premium DOUBLE,
            net_premium DOUBLE,
            log_ratio DOUBLE,
            strikes_count INTEGER,
            atm_strike DOUBLE,
            otm_call_premium DOUBLE,
            itm_call_premium DOUBLE,
            otm_put_premium DOUBLE,
            itm_put_premium DOUBLE,
            directional_score DOUBLE,
            fetch_timestamp TIMESTAMP,
            PRIMARY KEY (symbol, trade_date, expiration_type)
        )
        """
    )
    # Migration: add ITM/OTM columns if they don't exist
    options_premium_columns = [
        "otm_call_premium DOUBLE",
        "itm_call_premium DOUBLE",
        "otm_put_premium DOUBLE",
        "itm_put_premium DOUBLE",
        "directional_score DOUBLE",
    ]
    for column_def in options_premium_columns:
        conn.execute(f"ALTER TABLE options_premium_summary ADD COLUMN IF NOT EXISTS {column_def}")

    # =========================================================================
    # Backtest tables (bt_ prefix) for accumulation signal optimization
    # =========================================================================
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bt_signal_events (
            event_id INTEGER PRIMARY KEY,
            date DATE NOT NULL,
            symbol VARCHAR NOT NULL,
            sector_etf VARCHAR,
            score_variant VARCHAR NOT NULL,
            score_value DOUBLE,
            buy_threshold INTEGER,
            sell_threshold INTEGER,
            entry_price DOUBLE,
            signal_close DOUBLE,
            trend_state VARCHAR,
            sma_20_slope DOUBLE,
            price_vs_sma_20_pct DOUBLE,
            rsi_14 DOUBLE,
            opt_put_call_ratio DOUBLE,
            opt_iv_percentile DOUBLE,
            opt_put_call_ratio_5d_chg DOUBLE,
            regime VARCHAR,
            spy_vs_sma_50 DOUBLE,
            vix_level DOUBLE,
            fwd_return_1d DOUBLE,
            fwd_return_3d DOUBLE,
            fwd_return_5d DOUBLE,
            fwd_return_10d DOUBLE,
            fwd_return_20d DOUBLE,
            days_to_first_dist INTEGER,
            days_to_2x_consec_dist INTEGER,
            return_at_first_dist DOUBLE,
            return_at_2x_consec_dist DOUBLE,
            max_drawdown_before_dist DOUBLE,
            max_gain_before_dist DOUBLE,
            hit_at_dist BOOLEAN,
            hit_5d BOOLEAN,
            hit_10d BOOLEAN,
            parent_etf_aligned BOOLEAN,
            signal_quality VARCHAR,
            failure_mode VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (date, symbol, score_variant, buy_threshold, sell_threshold)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bt_backtest_runs (
            run_id INTEGER PRIMARY KEY,
            run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            score_variant VARCHAR NOT NULL,
            short_z_source VARCHAR,
            weight_short DOUBLE,
            weight_lit DOUBLE,
            weight_price DOUBLE,
            z_window INTEGER,
            buy_threshold INTEGER,
            sell_threshold INTEGER,
            exit_strategy VARCHAR,
            ticker_universe VARCHAR,
            regime_filter VARCHAR,
            date_start DATE,
            date_end DATE,
            total_signals INTEGER,
            total_trades INTEGER,
            hit_rate DOUBLE,
            avg_return DOUBLE,
            median_return DOUBLE,
            win_loss_ratio DOUBLE,
            sharpe_ratio DOUBLE,
            max_drawdown DOUBLE,
            avg_hold_days DOUBLE,
            signals_per_month DOUBLE,
            false_positive_rate DOUBLE,
            high_quality_hit_rate DOUBLE,
            UNIQUE (score_variant, z_window, buy_threshold, sell_threshold,
                    exit_strategy, ticker_universe, regime_filter, date_start, date_end)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bt_frequency_baseline (
            id INTEGER PRIMARY KEY,
            symbol VARCHAR NOT NULL,
            score_variant VARCHAR NOT NULL,
            date_start DATE NOT NULL,
            date_end DATE NOT NULL,
            accum_signals_count INTEGER,
            dist_signals_count INTEGER,
            avg_gap_accum_to_dist DOUBLE,
            median_gap_accum_to_dist DOUBLE,
            avg_gap_dist_to_accum DOUBLE,
            signals_per_month DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (symbol, score_variant, date_start, date_end)
        )
        """
    )
    # Backtest table indexes
    bt_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_bt_signal_events_date ON bt_signal_events(date)",
        "CREATE INDEX IF NOT EXISTS idx_bt_signal_events_symbol ON bt_signal_events(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_bt_signal_events_variant ON bt_signal_events(score_variant)",
        "CREATE INDEX IF NOT EXISTS idx_bt_signal_events_quality ON bt_signal_events(signal_quality)",
        "CREATE INDEX IF NOT EXISTS idx_bt_signal_events_regime ON bt_signal_events(regime)",
        "CREATE INDEX IF NOT EXISTS idx_bt_backtest_runs_variant ON bt_backtest_runs(score_variant)",
        "CREATE INDEX IF NOT EXISTS idx_bt_backtest_runs_hit_rate ON bt_backtest_runs(hit_rate)",
        "CREATE INDEX IF NOT EXISTS idx_bt_frequency_baseline_symbol ON bt_frequency_baseline(symbol)",
    ]
    for idx_sql in bt_indexes:
        try:
            conn.execute(idx_sql)
        except Exception:
            pass  # Index may already exist


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


# =============================================================================
# Polygon Ingestion State (Caching) Helpers
# =============================================================================

def check_ingestion_state(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    trade_date: date,
    data_source: str,
) -> bool:
    """Return True if symbol+date+source already fetched successfully."""
    result = conn.execute(
        """
        SELECT COUNT(*) as cnt FROM polygon_ingestion_state
        WHERE symbol = ? AND trade_date = ? AND data_source = ? AND status = 'COMPLETE'
        """,
        [symbol, trade_date, data_source],
    ).fetchone()
    return result[0] > 0 if result else False


def mark_ingestion_complete(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    trade_date: date,
    data_source: str,
    record_count: int,
) -> None:
    """Mark a symbol+date+source as successfully fetched."""
    now = datetime.utcnow()
    conn.execute(
        """
        INSERT INTO polygon_ingestion_state (symbol, trade_date, data_source, fetch_timestamp, record_count, status)
        VALUES (?, ?, ?, ?, ?, 'COMPLETE')
        ON CONFLICT (symbol, trade_date, data_source) DO UPDATE SET
            fetch_timestamp = EXCLUDED.fetch_timestamp,
            record_count = EXCLUDED.record_count,
            status = EXCLUDED.status
        """,
        [symbol, trade_date, data_source, now, record_count],
    )


def get_cached_symbols(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    trade_date: date,
    data_source: str,
) -> set[str]:
    """Return set of symbols that are already cached for a given date+source."""
    if not symbols:
        return set()
    placeholders = ", ".join(["?" for _ in symbols])
    result = conn.execute(
        f"""
        SELECT symbol FROM polygon_ingestion_state
        WHERE trade_date = ? AND data_source = ? AND status = 'COMPLETE'
        AND symbol IN ({placeholders})
        """,
        [trade_date, data_source] + symbols,
    ).fetchall()
    return {row[0] for row in result}


def get_uncached_symbols(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    trade_date: date,
    data_source: str,
) -> list[str]:
    """Return list of symbols that are NOT cached for a given date+source."""
    cached = get_cached_symbols(conn, symbols, trade_date, data_source)
    return [s for s in symbols if s not in cached]
