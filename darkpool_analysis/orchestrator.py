from __future__ import annotations

"""
FINRA does not publish trade direction for off-exchange volume.
Buy/Sell values are inferred estimates derived from lit-market equity trades
and applied proportionally to FINRA OTC volume.
"""

from datetime import date
import logging

import pandas as pd

try:
    from .analytics import build_darkpool_estimates, build_daily_summary
    from .config import load_config
    from .db import get_connection, init_db, upsert_dataframe
    from .fetch_finra import fetch_finra_otc_volume
    from .fetch_polygon_equity import fetch_polygon_trades
    from .infer_buy_sell import compute_lit_directional_flow
    from .plotter import plot_buy_ratio_series
    from .table_renderer import render_darkpool_table
except ImportError:
    from analytics import build_darkpool_estimates, build_daily_summary
    from config import load_config
    from db import get_connection, init_db, upsert_dataframe
    from fetch_finra import fetch_finra_otc_volume
    from fetch_polygon_equity import fetch_polygon_trades
    from infer_buy_sell import compute_lit_directional_flow
    from plotter import plot_buy_ratio_series
    from table_renderer import render_darkpool_table


def _ensure_dirs(config) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.table_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)


def _export_table(df: pd.DataFrame, path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _resolve_snapshot_date(fetch_mode: str, target_date: date, finra_week: date) -> date:
    """Derive snapshot date from fetch_mode.

    - 'weekly': use FINRA's week_start_date for output keying
    - 'daily' or 'single': use the actual target_date
    """
    if fetch_mode == "weekly":
        return finra_week
    return target_date  # 'daily' or 'single'


def _process_single_date(config, conn, run_date: date) -> list[str]:
    """Process a single date and return list of eligible symbols."""
    logging.info("Processing date: %s", run_date.isoformat())

    finra_all_df, finra_week_df, finra_week = fetch_finra_otc_volume(config, run_date)
    upsert_dataframe(conn, "finra_otc_volume_raw", finra_all_df, ["symbol", "week_start_date", "source"])

    finra_symbols = set(finra_week_df["symbol"].unique())
    missing_finra_symbols = sorted(set(config.finra_tickers) - finra_symbols)
    if missing_finra_symbols:
        logging.warning("Missing FINRA data for symbols: %s", ", ".join(missing_finra_symbols))

    # When include_polygon_only is True, fetch Polygon data for ALL tickers
    # Otherwise, only fetch for FINRA-eligible symbols
    if config.include_polygon_only_tickers:
        symbols_to_fetch = sorted(config.tickers)
        logging.info("Fetching Polygon data for all tickers (include_polygon_only=True)")
    else:
        symbols_to_fetch = sorted(finra_symbols)

    trades_df, failures = fetch_polygon_trades(config, symbols_to_fetch, run_date)
    if failures:
        logging.warning("Polygon failures for symbols: %s", ", ".join(failures))

    if not trades_df.empty:
        upsert_dataframe(conn, "equity_trades_raw", trades_df, ["symbol", "timestamp"])

    lit_flow_df = compute_lit_directional_flow(trades_df, symbols_to_fetch, run_date, config)
    upsert_dataframe(conn, "equity_lit_directional_flow", lit_flow_df, ["symbol", "date"])

    snapshot_date = _resolve_snapshot_date(config.fetch_mode, run_date, finra_week)
    estimated_flow_df = build_darkpool_estimates(
        finra_week_df,
        lit_flow_df,
        snapshot_date,
        config.inference_version,
        finra_week,
        include_polygon_only=config.include_polygon_only_tickers,
    )
    upsert_dataframe(conn, "darkpool_estimated_flow", estimated_flow_df, ["symbol", "date"])

    summary_df = build_daily_summary(estimated_flow_df, "weekly")
    upsert_dataframe(conn, "darkpool_daily_summary", summary_df, ["symbol", "date"])

    date_tag = snapshot_date.strftime("%Y%m%d")

    # Export CSVs if enabled
    if config.export_csv:
        _export_table(
            lit_flow_df,
            config.table_dir / f"equity_lit_directional_flow_{date_tag}.csv",
        )
        _export_table(
            estimated_flow_df,
            config.table_dir / f"darkpool_estimated_flow_{date_tag}.csv",
        )
        _export_table(
            summary_df,
            config.table_dir / f"darkpool_daily_summary_{date_tag}.csv",
        )

    return symbols_to_fetch, snapshot_date


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    _ensure_dirs(config)

    if not config.finra_tickers:
        raise RuntimeError("No FINRA-eligible tickers configured.")

    logging.info("Fetch mode: %s, dates to process: %d", config.fetch_mode, len(config.target_dates))

    all_symbols = set()
    processed_dates = []
    conn = get_connection(config.db_path)
    try:
        init_db(conn)

        for run_date in config.target_dates:
            try:
                symbols, snapshot_date = _process_single_date(config, conn, run_date)
                all_symbols.update(symbols)
                processed_dates.append(snapshot_date)
            except Exception as exc:
                logging.error("Failed to process %s: %s", run_date.isoformat(), exc)
                continue

    finally:
        conn.close()

    if all_symbols:
        # Generate buy ratio plots
        plot_buy_ratio_series(config.db_path, config.plot_dir, sorted(all_symbols), config.plot_mode)

        # Render combined dark pool table for all processed dates
        if processed_dates:
            try:
                render_darkpool_table(
                    db_path=config.db_path,
                    output_dir=config.table_dir,
                    dates=processed_dates,
                    tickers=config.tickers,
                    title="Dark Pool Volume",
                )
            except Exception as exc:
                logging.error("Failed to render table: %s", exc)

    logging.info("Dark pool analysis complete. Processed %d dates.", len(config.target_dates))


if __name__ == "__main__":
    main()
