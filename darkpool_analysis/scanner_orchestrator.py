from __future__ import annotations

"""
FINRA short sale scanner (full list via CDN).

This scanner uses FINRA daily short sale data only (no tick-level trades).
"""

from datetime import timedelta
import logging

import pandas as pd

try:
    from .config import load_config
    from .db import get_connection, init_db, upsert_dataframe
    from .fetch_finra_short_cdn import fetch_finra_short_cdn_daily
    from .scanner_analytics import build_scanner_metrics
    from .scanner_renderer import render_scanner_outputs
except ImportError:
    from config import load_config
    from db import get_connection, init_db, upsert_dataframe
    from fetch_finra_short_cdn import fetch_finra_short_cdn_daily
    from scanner_analytics import build_scanner_metrics
    from scanner_renderer import render_scanner_outputs


def _ensure_dirs(config) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.scanner_output_dir.mkdir(parents=True, exist_ok=True)


def _load_scanner_history(conn, start_date, end_date) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT symbol,
               trade_date,
               short_volume,
               short_exempt_volume,
               total_volume
        FROM finra_short_daily_all_raw
        WHERE trade_date >= ? AND trade_date <= ?
        """,
        [start_date, end_date],
    ).fetchdf()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    _ensure_dirs(config)

    if not config.target_dates:
        raise RuntimeError("No target dates configured for scanner.")

    logging.info(
        "Scanner run: %d date(s), lookback=%d days, top_n=%d",
        len(config.target_dates),
        config.scanner_lookback_days,
        config.scanner_top_n,
    )

    min_date = min(config.target_dates)
    max_date = max(config.target_dates)
    history_start = min_date - timedelta(days=config.scanner_lookback_days)

    conn = get_connection(config.scanner_db_path)
    metrics_df = pd.DataFrame()

    try:
        init_db(conn)

        for run_date in config.target_dates:
            logging.info("Scanner ingest: %s", run_date.isoformat())
            try:
                short_df = fetch_finra_short_cdn_daily(config, run_date)
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("FINRA CDN fetch failed for %s: %s", run_date.isoformat(), exc)
                short_df = pd.DataFrame()

            if short_df.empty:
                logging.warning("No FINRA short sale rows for %s", run_date.isoformat())
            else:
                upsert_dataframe(
                    conn,
                    "finra_short_daily_all_raw",
                    short_df,
                    ["symbol", "trade_date", "market", "source"],
                )

        history_df = _load_scanner_history(conn, history_start, max_date)
        metrics_df = build_scanner_metrics(history_df, config.target_dates, config)
        if not metrics_df.empty:
            upsert_dataframe(conn, "scanner_daily_metrics", metrics_df, ["symbol", "date"])
        else:
            logging.warning("Scanner metrics are empty for %s - %s", min_date, max_date)

    finally:
        conn.close()

    if metrics_df.empty:
        return

    for run_date in config.target_dates:
        render_scanner_outputs(
            metrics_df,
            config.scanner_output_dir,
            run_date,
            config.scanner_top_n,
            config.scanner_export_full,
        )

    logging.info("Scanner complete. Output root: %s", config.scanner_output_dir)


if __name__ == "__main__":
    main()
