"""
Backfill ITM/OTM premium breakdown for existing options_premium_summary data.

This script recomputes the ITM/OTM columns from existing per-strike data in
options_premium_daily, without needing to re-fetch from Polygon.

Usage:
    python backfill_itm_otm.py
    python backfill_itm_otm.py --dry-run
    python backfill_itm_otm.py --start-date 2025-01-01 --end-date 2025-12-31
"""
from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_db_path(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parent / "data" / "darkpool.duckdb"


def _ensure_columns_exist(conn: duckdb.DuckDBPyConnection) -> None:
    """Add ITM/OTM columns if they don't exist (schema migration)."""
    columns_to_add = [
        "otm_call_premium DOUBLE",
        "itm_call_premium DOUBLE",
        "otm_put_premium DOUBLE",
        "itm_put_premium DOUBLE",
        "directional_score DOUBLE",
    ]
    for column_def in columns_to_add:
        try:
            conn.execute(f"ALTER TABLE options_premium_summary ADD COLUMN IF NOT EXISTS {column_def}")
        except Exception as e:
            logger.debug("Column may already exist: %s", e)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def compute_itm_otm_for_group(
    detail_df: pd.DataFrame,
    atm_strike: float,
) -> dict:
    """
    Compute ITM/OTM breakdown for a single (symbol, trade_date, expiration_type) group.

    ITM/OTM determination (per WTD's methodology):
    - For calls: OTM = strike > atm_price, ITM = strike < atm_price
    - For puts: OTM = strike < atm_price, ITM = strike > atm_price
    """
    calls = detail_df[detail_df["option_type"] == "C"]
    puts = detail_df[detail_df["option_type"] == "P"]

    # ITM/OTM breakdown
    otm_call_premium = calls[calls["strike"] > atm_strike]["premium"].sum()
    itm_call_premium = calls[calls["strike"] < atm_strike]["premium"].sum()
    otm_put_premium = puts[puts["strike"] < atm_strike]["premium"].sum()
    itm_put_premium = puts[puts["strike"] > atm_strike]["premium"].sum()

    # Directional score (per WTD's insight):
    # OTM calls are bullish, OTM puts are bearish, ITM calls sold are bearish (hedge)
    directional_score = otm_call_premium - otm_put_premium - (itm_call_premium * 0.5)

    return {
        "otm_call_premium": otm_call_premium,
        "itm_call_premium": itm_call_premium,
        "otm_put_premium": otm_put_premium,
        "itm_put_premium": itm_put_premium,
        "directional_score": directional_score,
    }


def backfill_itm_otm(
    conn: duckdb.DuckDBPyConnection,
    start_date: date | None = None,
    end_date: date | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Backfill ITM/OTM columns in options_premium_summary from existing detail data.

    Returns dict with statistics.
    """
    stats = {"processed": 0, "updated": 0, "skipped": 0, "errors": 0}

    # Get all summary rows that need backfill
    date_filter = ""
    params = []
    if start_date and end_date:
        date_filter = "WHERE trade_date >= ? AND trade_date <= ?"
        params = [start_date, end_date]

    summary_query = f"""
        SELECT symbol, trade_date, expiration_type, atm_strike
        FROM options_premium_summary
        {date_filter}
        ORDER BY trade_date, symbol
    """
    summary_rows = conn.execute(summary_query, params).fetchall()

    if not summary_rows:
        logger.info("No summary rows found to backfill.")
        return stats

    logger.info("Found %d summary rows to process.", len(summary_rows))

    for symbol, trade_date, expiration_type, atm_strike in summary_rows:
        stats["processed"] += 1

        if atm_strike is None or atm_strike <= 0:
            logger.warning(
                "Skipping %s %s %s: invalid atm_strike=%s",
                symbol, trade_date, expiration_type, atm_strike
            )
            stats["skipped"] += 1
            continue

        # Get detail data for this group
        detail_query = """
            SELECT strike, option_type, premium
            FROM options_premium_daily
            WHERE symbol = ? AND trade_date = ? AND expiration_type = ?
        """
        detail_df = conn.execute(
            detail_query, [symbol, trade_date, expiration_type]
        ).df()

        if detail_df.empty:
            logger.warning(
                "Skipping %s %s %s: no detail data found",
                symbol, trade_date, expiration_type
            )
            stats["skipped"] += 1
            continue

        try:
            # Compute ITM/OTM breakdown
            breakdown = compute_itm_otm_for_group(detail_df, atm_strike)

            if dry_run:
                logger.info(
                    "[DRY RUN] Would update %s %s %s: OTM_C=%.2f, ITM_C=%.2f, OTM_P=%.2f, ITM_P=%.2f, DIR=%.2f",
                    symbol, trade_date, expiration_type,
                    breakdown["otm_call_premium"],
                    breakdown["itm_call_premium"],
                    breakdown["otm_put_premium"],
                    breakdown["itm_put_premium"],
                    breakdown["directional_score"],
                )
            else:
                # Update summary row
                conn.execute(
                    """
                    UPDATE options_premium_summary
                    SET otm_call_premium = ?,
                        itm_call_premium = ?,
                        otm_put_premium = ?,
                        itm_put_premium = ?,
                        directional_score = ?
                    WHERE symbol = ? AND trade_date = ? AND expiration_type = ?
                    """,
                    [
                        breakdown["otm_call_premium"],
                        breakdown["itm_call_premium"],
                        breakdown["otm_put_premium"],
                        breakdown["itm_put_premium"],
                        breakdown["directional_score"],
                        symbol,
                        trade_date,
                        expiration_type,
                    ],
                )

            stats["updated"] += 1

        except Exception as e:
            logger.error(
                "Error processing %s %s %s: %s",
                symbol, trade_date, expiration_type, e
            )
            stats["errors"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill ITM/OTM premium breakdown from existing detail data."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to DuckDB file (defaults to project DB).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Filter by start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Filter by end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes.",
    )
    args = parser.parse_args()

    db_path = _resolve_db_path(args.db)
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return 1

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    if (start_date and not end_date) or (end_date and not start_date):
        logger.error("Both --start-date and --end-date are required for range filtering.")
        return 1

    if start_date and end_date and start_date > end_date:
        logger.error("start-date must be <= end-date.")
        return 1

    conn = duckdb.connect(str(db_path))
    try:
        # Ensure ITM/OTM columns exist (schema migration)
        _ensure_columns_exist(conn)

        # Show current state
        summary_count = conn.execute(
            "SELECT COUNT(*) FROM options_premium_summary"
        ).fetchone()[0]
        null_count = conn.execute(
            "SELECT COUNT(*) FROM options_premium_summary WHERE otm_call_premium IS NULL"
        ).fetchone()[0]

        logger.info("Total summary rows: %d", summary_count)
        logger.info("Rows with NULL ITM/OTM data: %d", null_count)

        if start_date and end_date:
            logger.info("Date filter: %s to %s", start_date, end_date)

        if args.dry_run:
            logger.info("DRY RUN MODE - no changes will be made")

        # Run backfill
        stats = backfill_itm_otm(conn, start_date, end_date, args.dry_run)

        logger.info(
            "Backfill complete: processed=%d, updated=%d, skipped=%d, errors=%d",
            stats["processed"],
            stats["updated"],
            stats["skipped"],
            stats["errors"],
        )

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
