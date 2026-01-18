"""
Database Schema Module

Defines and creates bt_ prefixed tables for backtesting in the main darkpool database.
"""

import duckdb
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE CREATION SQL
# =============================================================================

CREATE_BT_SIGNAL_EVENTS = """
CREATE TABLE IF NOT EXISTS bt_signal_events (
    event_id INTEGER PRIMARY KEY,

    -- Core identity
    date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    sector_etf VARCHAR,

    -- Signal config
    score_variant VARCHAR NOT NULL,
    score_value FLOAT,
    buy_threshold INTEGER,
    sell_threshold INTEGER,

    -- Entry
    entry_price FLOAT,
    signal_close FLOAT,

    -- Trend context
    trend_state VARCHAR,
    sma_20_slope FLOAT,
    price_vs_sma_20_pct FLOAT,
    rsi_14 FLOAT,

    -- Options context (liquid names only)
    opt_put_call_ratio FLOAT,
    opt_iv_percentile FLOAT,
    opt_put_call_ratio_5d_chg FLOAT,

    -- Market regime
    regime VARCHAR,
    spy_vs_sma_50 FLOAT,
    vix_level FLOAT,

    -- Outcomes
    fwd_return_1d FLOAT,
    fwd_return_3d FLOAT,
    fwd_return_5d FLOAT,
    fwd_return_10d FLOAT,
    fwd_return_20d FLOAT,
    days_to_first_dist INTEGER,
    days_to_2x_consec_dist INTEGER,
    return_at_first_dist FLOAT,
    return_at_2x_consec_dist FLOAT,
    max_drawdown_before_dist FLOAT,
    max_gain_before_dist FLOAT,

    -- Flags
    hit_at_dist BOOLEAN,
    hit_5d BOOLEAN,
    hit_10d BOOLEAN,
    parent_etf_aligned BOOLEAN,
    signal_quality VARCHAR,
    failure_mode VARCHAR,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Composite key for deduplication
    UNIQUE (date, symbol, score_variant, buy_threshold, sell_threshold)
);
"""

CREATE_BT_BACKTEST_RUNS = """
CREATE TABLE IF NOT EXISTS bt_backtest_runs (
    run_id INTEGER PRIMARY KEY,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Configuration
    score_variant VARCHAR NOT NULL,
    short_z_source VARCHAR,
    weight_short FLOAT,
    weight_lit FLOAT,
    weight_price FLOAT,
    z_window INTEGER,
    buy_threshold INTEGER,
    sell_threshold INTEGER,
    exit_strategy VARCHAR,

    -- Universe & Regime
    ticker_universe VARCHAR,
    regime_filter VARCHAR,
    date_start DATE,
    date_end DATE,

    -- Results
    total_signals INTEGER,
    total_trades INTEGER,
    hit_rate FLOAT,
    avg_return FLOAT,
    median_return FLOAT,
    win_loss_ratio FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    avg_hold_days FLOAT,
    signals_per_month FLOAT,

    -- Diagnostic
    false_positive_rate FLOAT,
    high_quality_hit_rate FLOAT,

    -- Unique constraint on configuration
    UNIQUE (score_variant, z_window, buy_threshold, sell_threshold,
            exit_strategy, ticker_universe, regime_filter, date_start, date_end)
);
"""

CREATE_BT_FREQUENCY_BASELINE = """
CREATE TABLE IF NOT EXISTS bt_frequency_baseline (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    score_variant VARCHAR NOT NULL,
    date_start DATE NOT NULL,
    date_end DATE NOT NULL,

    accum_signals_count INTEGER,
    dist_signals_count INTEGER,
    avg_gap_accum_to_dist FLOAT,
    median_gap_accum_to_dist FLOAT,
    avg_gap_dist_to_accum FLOAT,
    signals_per_month FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (symbol, score_variant, date_start, date_end)
);
"""

# Index creation for performance
CREATE_INDEXES = """
-- Signal events indexes
CREATE INDEX IF NOT EXISTS idx_bt_signal_events_date ON bt_signal_events(date);
CREATE INDEX IF NOT EXISTS idx_bt_signal_events_symbol ON bt_signal_events(symbol);
CREATE INDEX IF NOT EXISTS idx_bt_signal_events_variant ON bt_signal_events(score_variant);
CREATE INDEX IF NOT EXISTS idx_bt_signal_events_quality ON bt_signal_events(signal_quality);
CREATE INDEX IF NOT EXISTS idx_bt_signal_events_regime ON bt_signal_events(regime);

-- Backtest runs indexes
CREATE INDEX IF NOT EXISTS idx_bt_backtest_runs_variant ON bt_backtest_runs(score_variant);
CREATE INDEX IF NOT EXISTS idx_bt_backtest_runs_hit_rate ON bt_backtest_runs(hit_rate);

-- Frequency baseline indexes
CREATE INDEX IF NOT EXISTS idx_bt_frequency_baseline_symbol ON bt_frequency_baseline(symbol);
"""


# =============================================================================
# SCHEMA MANAGEMENT FUNCTIONS
# =============================================================================

def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all bt_ prefixed tables if they don't exist."""
    logger.info("Creating bt_ backtest tables...")

    conn.execute(CREATE_BT_SIGNAL_EVENTS)
    conn.execute(CREATE_BT_BACKTEST_RUNS)
    conn.execute(CREATE_BT_FREQUENCY_BASELINE)

    # Create indexes (split into individual statements)
    for line in CREATE_INDEXES.strip().split(";"):
        line = line.strip()
        if line and not line.startswith("--"):
            try:
                conn.execute(line)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")

    logger.info("bt_ tables created successfully")


def drop_tables(conn: duckdb.DuckDBPyConnection, confirm: bool = False) -> None:
    """Drop all bt_ prefixed tables. Requires confirmation."""
    if not confirm:
        raise ValueError("Must pass confirm=True to drop tables")

    logger.warning("Dropping all bt_ backtest tables...")

    conn.execute("DROP TABLE IF EXISTS bt_signal_events")
    conn.execute("DROP TABLE IF EXISTS bt_backtest_runs")
    conn.execute("DROP TABLE IF EXISTS bt_frequency_baseline")

    logger.info("bt_ tables dropped")


def clear_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Clear all data from bt_ tables without dropping them."""
    logger.info("Clearing bt_ backtest tables...")

    conn.execute("DELETE FROM bt_signal_events")
    conn.execute("DELETE FROM bt_backtest_runs")
    conn.execute("DELETE FROM bt_frequency_baseline")

    logger.info("bt_ tables cleared")


def get_table_counts(conn: duckdb.DuckDBPyConnection) -> dict:
    """Get row counts for all bt_ tables."""
    counts = {}

    for table in ["bt_signal_events", "bt_backtest_runs", "bt_frequency_baseline"]:
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = result[0] if result else 0
        except Exception:
            counts[table] = None  # Table doesn't exist

    return counts


def ensure_tables_exist(db_path: Path) -> None:
    """Ensure all bt_ tables exist in the database."""
    conn = duckdb.connect(str(db_path))
    try:
        create_tables(conn)
    finally:
        conn.close()


# =============================================================================
# UPSERT FUNCTIONS
# =============================================================================

def upsert_signal_event(
    conn: duckdb.DuckDBPyConnection,
    event_data: dict,
) -> None:
    """Insert or update a signal event."""
    columns = list(event_data.keys())
    placeholders = ", ".join(["?" for _ in columns])
    columns_str = ", ".join(columns)

    # Use INSERT OR REPLACE for DuckDB
    sql = f"""
    INSERT OR REPLACE INTO bt_signal_events ({columns_str})
    VALUES ({placeholders})
    """

    conn.execute(sql, list(event_data.values()))


def upsert_backtest_run(
    conn: duckdb.DuckDBPyConnection,
    run_data: dict,
) -> int:
    """Insert or update a backtest run. Returns run_id."""
    columns = [k for k in run_data.keys() if k != "run_id"]
    placeholders = ", ".join(["?" for _ in columns])
    columns_str = ", ".join(columns)

    sql = f"""
    INSERT INTO bt_backtest_runs ({columns_str})
    VALUES ({placeholders})
    RETURNING run_id
    """

    result = conn.execute(sql, [run_data[c] for c in columns]).fetchone()
    return result[0] if result else None


def upsert_frequency_baseline(
    conn: duckdb.DuckDBPyConnection,
    baseline_data: dict,
) -> None:
    """Insert or update frequency baseline data."""
    columns = [k for k in baseline_data.keys() if k != "id"]
    placeholders = ", ".join(["?" for _ in columns])
    columns_str = ", ".join(columns)

    sql = f"""
    INSERT OR REPLACE INTO bt_frequency_baseline ({columns_str})
    VALUES ({placeholders})
    """

    conn.execute(sql, [baseline_data[c] for c in columns])


# =============================================================================
# QUERY HELPERS
# =============================================================================

def get_signal_events(
    conn: duckdb.DuckDBPyConnection,
    symbol: Optional[str] = None,
    score_variant: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    signal_quality: Optional[str] = None,
    regime: Optional[str] = None,
) -> list:
    """Query signal events with optional filters."""
    sql = "SELECT * FROM bt_signal_events WHERE 1=1"
    params = []

    if symbol:
        sql += " AND symbol = ?"
        params.append(symbol)
    if score_variant:
        sql += " AND score_variant = ?"
        params.append(score_variant)
    if date_start:
        sql += " AND date >= ?"
        params.append(date_start)
    if date_end:
        sql += " AND date <= ?"
        params.append(date_end)
    if signal_quality:
        sql += " AND signal_quality = ?"
        params.append(signal_quality)
    if regime:
        sql += " AND regime = ?"
        params.append(regime)

    sql += " ORDER BY date, symbol"

    return conn.execute(sql, params).fetchall()


def get_backtest_runs(
    conn: duckdb.DuckDBPyConnection,
    score_variant: Optional[str] = None,
    min_hit_rate: Optional[float] = None,
    order_by: str = "hit_rate DESC",
    limit: Optional[int] = None,
) -> list:
    """Query backtest runs with optional filters."""
    sql = "SELECT * FROM bt_backtest_runs WHERE 1=1"
    params = []

    if score_variant:
        sql += " AND score_variant = ?"
        params.append(score_variant)
    if min_hit_rate is not None:
        sql += " AND hit_rate >= ?"
        params.append(min_hit_rate)

    sql += f" ORDER BY {order_by}"

    if limit:
        sql += f" LIMIT {limit}"

    return conn.execute(sql, params).fetchall()


# =============================================================================
# CLI / STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Get database path from config
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from darkpool_analysis.config import load_config

    config = load_config()
    db_path = config.db_path

    print(f"Database: {db_path}")

    conn = duckdb.connect(str(db_path))
    try:
        # Check current state
        counts = get_table_counts(conn)
        print(f"Current table counts: {counts}")

        # Create tables if needed
        create_tables(conn)

        # Check again
        counts = get_table_counts(conn)
        print(f"After creation: {counts}")

    finally:
        conn.close()

    print("Schema setup complete!")
