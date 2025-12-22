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
except ImportError:
    from analytics import build_darkpool_estimates, build_daily_summary
    from config import load_config
    from db import get_connection, init_db, upsert_dataframe
    from fetch_finra import fetch_finra_otc_volume
    from fetch_polygon_equity import fetch_polygon_trades
    from infer_buy_sell import compute_lit_directional_flow
    from plotter import plot_buy_ratio_series


def _ensure_dirs(config) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.table_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)


def _export_table(df: pd.DataFrame, path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _resolve_snapshot_date(run_mode: str, target_date: date, finra_week: date) -> date:
    if run_mode == "daily":
        return target_date
    if run_mode == "weekly":
        return finra_week
    raise ValueError(f"Unsupported RUN_MODE: {run_mode}")


def _process_single_date(config, conn, run_date: date) -> list[str]:
    """Process a single date and return list of eligible symbols."""
    logging.info("Processing date: %s", run_date.isoformat())

    finra_all_df, finra_week_df, finra_week = fetch_finra_otc_volume(config, run_date)
    upsert_dataframe(conn, "finra_otc_volume_raw", finra_all_df, ["symbol", "week_start_date"])

    eligible_symbols = sorted(finra_week_df["symbol"].unique())
    missing_symbols = sorted(set(config.finra_tickers) - set(eligible_symbols))
    if missing_symbols:
        logging.warning("Missing FINRA data for symbols: %s", ", ".join(missing_symbols))

    trades_df, failures = fetch_polygon_trades(config, eligible_symbols, run_date)
    if failures:
        logging.warning("Polygon failures for symbols: %s", ", ".join(failures))

    if not trades_df.empty:
        upsert_dataframe(conn, "equity_trades_raw", trades_df, ["symbol", "timestamp"])

    lit_flow_df = compute_lit_directional_flow(trades_df, eligible_symbols, run_date, config)
    upsert_dataframe(conn, "equity_lit_directional_flow", lit_flow_df, ["symbol", "date"])

    snapshot_date = _resolve_snapshot_date(config.run_mode, run_date, finra_week)
    estimated_flow_df = build_darkpool_estimates(
        finra_week_df, lit_flow_df, snapshot_date, config.inference_version
    )
    upsert_dataframe(conn, "darkpool_estimated_flow", estimated_flow_df, ["symbol", "date"])

    summary_df = build_daily_summary(estimated_flow_df, "weekly")
    upsert_dataframe(conn, "darkpool_daily_summary", summary_df, ["symbol", "date"])

    date_tag = snapshot_date.strftime("%Y%m%d")
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

    return eligible_symbols


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    _ensure_dirs(config)

    if not config.finra_tickers:
        raise RuntimeError("No FINRA-eligible tickers configured.")

    logging.info("Fetch mode: %s, dates to process: %d", config.fetch_mode, len(config.target_dates))

    all_symbols = set()
    conn = get_connection(config.db_path)
    try:
        init_db(conn)

        for run_date in config.target_dates:
            try:
                symbols = _process_single_date(config, conn, run_date)
                all_symbols.update(symbols)
            except Exception as exc:
                logging.error("Failed to process %s: %s", run_date.isoformat(), exc)
                continue

    finally:
        conn.close()

    if all_symbols:
        plot_buy_ratio_series(config.db_path, config.plot_dir, sorted(all_symbols))
    logging.info("Dark pool analysis complete. Processed %d dates.", len(config.target_dates))


if __name__ == "__main__":
    main()
