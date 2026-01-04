#--------------------------------------
### instruction to claar database without loosing raw data
#--------------------------------------

# python reset_daily_metrics.py
## simple reset of all daily metrics

# python reset_daily_metrics.py --mode reset-vwflow
## Keeps all rows, but sets vwbr, vwbr_z, and accumulation_short_z_source to NULL in daily_metrics.
## This forces a recompute of VW Flow without wiping other metrics.
## Prompts for DELETE to confirm.

# python reset_daily_metrics.py --mode delete --include-index --yes
## Deletes all rows in daily_metrics and index_constituent_short_agg_daily.
## --yes skips the confirmation prompt.

# python reset_daily_metrics.py --mode delete-keep-lit
## Deletes all non-lit tables but keeps:
## - polygon_equity_trades_raw (tick data)
## - lit_direction_daily (derived lit flow)
## - polygon_ingestion_state (cache keys)
## Prompts for DELETE to confirm.

### Example with dates
# python reset_daily_metrics.py --mode reset-vwflow --start-date 2025-11-01 --end-date 2025-12-31
# python reset_daily_metrics.py --mode delete --start-date 2025-11-01 --end-date 2025-12-31 --include-index
# python reset_daily_metrics.py --mode delete-keep-lit --start-date 2025-11-01 --end-date 2025-12-31
# python reset_daily_metrics.py --mode delete-keep-lit --start-date 2025-11-01 --end-date 2025-12-31
#----------------------------------------






from __future__ import annotations

# Usage examples:
# python reset_daily_metrics.py
# python reset_daily_metrics.py --mode reset-vwflow
# python reset_daily_metrics.py --mode delete --include-index --yes

import argparse
from datetime import date
import sys
from pathlib import Path

import duckdb


def _resolve_db_path(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser()
    # Script is in Special_tools/, database is in darkpool_analysis/data/
    return Path(__file__).resolve().parent.parent / "darkpool_analysis" / "data" / "darkpool.duckdb"


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def _get_table_stats(conn: duckdb.DuckDBPyConnection, table: str, date_col: str) -> tuple[int, str, str]:
    if not conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()[0]:
        return 0, "NA", "NA"
    row = conn.execute(
        f"SELECT COUNT(*), MIN({date_col}), MAX({date_col}) FROM {table}"
    ).fetchone()
    return int(row[0] or 0), str(row[1] or "NA"), str(row[2] or "NA")


def _confirm(action: str) -> bool:
    prompt = f"{action} Type DELETE to confirm: "
    return input(prompt).strip() == "DELETE"


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset daily_metrics safely without touching raw tables.")
    parser.add_argument("--db", type=str, default=None, help="Path to DuckDB file (defaults to project DB).")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["delete", "reset-vwflow", "delete-keep-lit"],
        default="delete",
        help=(
            "delete: remove daily_metrics rows; "
            "reset-vwflow: NULL vw flow fields; "
            "delete-keep-lit: clear all non-lit tables."
        ),
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
        "--include-index",
        action="store_true",
        help="Also clear index_constituent_short_agg_daily (delete mode only).",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()

    db_path = _resolve_db_path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    if (start_date and not end_date) or (end_date and not start_date):
        print("Both --start-date and --end-date are required for range filtering.")
        return 1
    if start_date and end_date and start_date > end_date:
        print("start-date must be <= end-date.")
        return 1

    conn = duckdb.connect(str(db_path))
    try:
        daily_count, daily_min, daily_max = _get_table_stats(conn, "daily_metrics", "date")
        index_count, index_min, index_max = _get_table_stats(
            conn, "index_constituent_short_agg_daily", "trade_date"
        )

        print(f"daily_metrics rows: {daily_count} (range: {daily_min} to {daily_max})")
        print(f"index_constituent_short_agg_daily rows: {index_count} (range: {index_min} to {index_max})")
        if start_date and end_date:
            print(f"date filter: {start_date} to {end_date}")

        if args.mode == "delete":
            action = "This will DELETE all rows in daily_metrics."
            if args.include_index:
                action += " It will also DELETE index_constituent_short_agg_daily."
            if not args.yes and not _confirm(action):
                print("Aborted.")
                return 1
            if start_date and end_date:
                conn.execute(
                    "DELETE FROM daily_metrics WHERE date >= ? AND date <= ?",
                    [start_date, end_date],
                )
            else:
                conn.execute("DELETE FROM daily_metrics")
            if args.include_index:
                if start_date and end_date:
                    conn.execute(
                        "DELETE FROM index_constituent_short_agg_daily WHERE trade_date >= ? AND trade_date <= ?",
                        [start_date, end_date],
                    )
                else:
                    conn.execute("DELETE FROM index_constituent_short_agg_daily")
            print("Delete complete.")
        elif args.mode == "delete-keep-lit":
            action = (
                "This will DELETE all non-lit tables (daily_metrics, index_constituent_short_agg_daily, "
                "finra_otc_weekly_raw, finra_short_daily_raw, "
                "polygon_daily_agg_raw, options_premium_daily, options_premium_summary, "
                "composite_signal). "
                "It will KEEP polygon_equity_trades_raw, lit_direction_daily, and polygon_ingestion_state."
            )
            if not args.yes and not _confirm(action):
                print("Aborted.")
                return 1

            tables_to_clear = [
                ("daily_metrics", "date"),
                ("index_constituent_short_agg_daily", "trade_date"),
                ("finra_otc_weekly_raw", "week_start_date"),
                ("finra_short_daily_raw", "trade_date"),
                ("polygon_daily_agg_raw", "trade_date"),
                ("options_premium_daily", "trade_date"),
                ("options_premium_summary", "trade_date"),
                ("composite_signal", "date"),
            ]
            for table, date_col in tables_to_clear:
                exists = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                    [table],
                ).fetchone()[0]
                if not exists:
                    continue
                if start_date and end_date:
                    conn.execute(
                        f"DELETE FROM {table} WHERE {date_col} >= ? AND {date_col} <= ?",
                        [start_date, end_date],
                    )
                else:
                    conn.execute(f"DELETE FROM {table}")
            print("Non-lit tables cleared.")
        else:
            action = "This will NULL vw flow fields in daily_metrics (vwbr, vwbr_z, accumulation_short_z_source)."
            if not args.yes and not _confirm(action):
                print("Aborted.")
                return 1
            if start_date and end_date:
                conn.execute(
                    """
                    UPDATE daily_metrics
                    SET vwbr = NULL,
                        vwbr_z = NULL,
                        accumulation_short_z_source = NULL
                    WHERE date >= ? AND date <= ?
                    """,
                    [start_date, end_date],
                )
            else:
                conn.execute(
                    """
                    UPDATE daily_metrics
                    SET vwbr = NULL,
                        vwbr_z = NULL,
                        accumulation_short_z_source = NULL
                    """
                )
            print("VW flow reset complete.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
