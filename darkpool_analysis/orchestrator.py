from __future__ import annotations

"""
FINRA does not publish trade direction for off-exchange volume.
Buy/Sell values are inferred estimates derived from lit-market equity trades
and applied proportionally to FINRA OTC volume. Daily short sale data is a
separate pressure proxy and must never replace OTC weekly volume.
"""

import logging
from datetime import timedelta

import pandas as pd

try:
    from .analytics import build_daily_metrics, build_index_constituent_short_agg
    from .config import load_config
    from .db import get_connection, init_db, upsert_dataframe
    from .fetch_finra_otc import fetch_finra_otc_weekly
    from .fetch_finra_short import fetch_finra_short_daily
    from .fetch_polygon_agg import fetch_polygon_daily_agg
    from .fetch_polygon_equity import fetch_polygon_trades
    from .fetch_polygon_options import fetch_options_premium_batch
    from .infer_buy_sell import compute_lit_directional_flow
    from .plotter import render_metrics_plots
    from .plotter_chart_ATMprem import render_price_charts
    from .table_renderer import render_daily_metrics_table
    from .combination_plotter import render_combination_plot
except ImportError:
    from analytics import build_daily_metrics, build_index_constituent_short_agg
    from config import load_config
    from db import get_connection, init_db, upsert_dataframe
    from fetch_finra_otc import fetch_finra_otc_weekly
    from fetch_finra_short import fetch_finra_short_daily
    from fetch_polygon_agg import fetch_polygon_daily_agg
    from fetch_polygon_equity import fetch_polygon_trades
    from fetch_polygon_options import fetch_options_premium_batch
    from infer_buy_sell import compute_lit_directional_flow
    from plotter import render_metrics_plots
    from plotter_chart_ATMprem import render_price_charts
    from table_renderer import render_daily_metrics_table
    from combination_plotter import render_combination_plot


def _ensure_dirs(config) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.table_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    config.price_chart_dir.mkdir(parents=True, exist_ok=True)


def _export_table(df: pd.DataFrame, path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    # Filter to non-empty frames that have at least one non-NA value
    non_empty = [df for df in frames if not df.empty and not df.isna().all().all()]
    if not non_empty:
        # Preserve column structure from first frame with columns
        for df in frames:
            if len(df.columns) > 0:
                return df.head(0)  # Empty DataFrame with same columns
        return pd.DataFrame()
    if len(non_empty) == 1:
        return non_empty[0].reset_index(drop=True)
    return pd.concat(non_empty, ignore_index=True)


def _get_existing_metrics(conn, symbols: list[str], dates: list) -> pd.DataFrame:
    """Check if daily_metrics has complete data for symbols and dates."""
    if not symbols or not dates:
        return pd.DataFrame()

    symbol_placeholders = ", ".join(["?" for _ in symbols])
    date_placeholders = ", ".join(["?" for _ in dates])

    query = f"""
        SELECT * FROM daily_metrics
        WHERE symbol IN ({symbol_placeholders})
          AND date IN ({date_placeholders})
    """
    params = list(symbols) + list(dates)
    return conn.execute(query, params).fetchdf()


def _metrics_are_complete(
    existing_metrics: pd.DataFrame,
    expected_rows: int,
    short_z_source: str | None = None,
) -> bool:
    """Check if existing metrics have complete data (non-null key columns)."""
    if len(existing_metrics) != expected_rows:
        return False

    # Key columns that must have at least some non-null values for data to be "complete"
    key_columns = [
        "accumulation_score",
        "short_ratio",
        "lit_flow_imbalance",
        "combined_ratio",
        "vw_flow",
        "finra_buy_volume",
    ]

    for col in key_columns:
        if col in existing_metrics.columns:
            # Check if column has at least some non-null values
            if existing_metrics[col].notna().sum() == 0:
                return False

    if short_z_source:
        if "accumulation_short_z_source" not in existing_metrics.columns:
            return False
        sources = (
            existing_metrics["accumulation_short_z_source"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
            .tolist()
        )
        if not sources or any(source != short_z_source for source in sources):
            return False

    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    _ensure_dirs(config)

    if not config.tickers:
        raise RuntimeError("No tickers configured.")

    logging.info("Fetch mode: %s, dates to process: %d", config.fetch_mode, len(config.target_dates))
    logging.info("Polygon trades mode: %s", config.polygon_trades_mode)
    logging.info("Cache enabled: %s", config.skip_cached)
    logging.info(
        "Price charts: %s (timeframe=%s)",
        config.render_price_charts,
        config.price_bar_timeframe,
    )
    logging.info("Options premium fetch: %s", config.fetch_options_premium)

    max_date = max(config.target_dates)
    min_date = min(config.target_dates)
    conn = get_connection(config.db_path)
    short_frames = []
    lit_frames = []
    agg_frames = []

    try:
        init_db(conn)

        finra_all_df, _, _ = fetch_finra_otc_weekly(config, max_date, range_start=min_date)
        if finra_all_df.empty:
            logging.warning("No FINRA OTC weekly data returned.")
        else:
            upsert_dataframe(conn, "finra_otc_weekly_raw", finra_all_df, ["symbol", "week_start_date"])

        finra_symbols = set(finra_all_df["symbol"].unique()) if not finra_all_df.empty else set()
        missing_finra_symbols = sorted(set(config.finra_tickers) - finra_symbols)
        if missing_finra_symbols:
            logging.warning("Missing FINRA OTC data for symbols: %s", ", ".join(missing_finra_symbols))

        if config.include_polygon_only_tickers:
            symbols_to_fetch = sorted(config.tickers)
        else:
            symbols_to_fetch = sorted(finra_symbols)

        if not symbols_to_fetch:
            raise RuntimeError("No eligible symbols to fetch from Polygon.")

        for run_date in config.target_dates:
            logging.info("Processing date: %s", run_date.isoformat())

            try:
                short_df = fetch_finra_short_daily(config, run_date)
            except Exception as exc:
                logging.warning("Short sale fetch failed for %s: %s", run_date.isoformat(), exc)
                short_df = pd.DataFrame()
            if not short_df.empty:
                upsert_dataframe(conn, "finra_short_daily_raw", short_df, ["symbol", "trade_date"])
            short_frames.append(short_df)

            trades_df, failures, trades_cache_stats = fetch_polygon_trades(
                config, symbols_to_fetch, run_date, conn=conn
            )
            if trades_cache_stats["cached"] > 0 or trades_cache_stats["fetched"] > 0:
                logging.info(
                    "Trades cache: %d cached, %d fetched (%s mode)",
                    trades_cache_stats["cached"], trades_cache_stats["fetched"],
                    trades_cache_stats["data_source"]
                )
            if failures:
                logging.warning("Polygon trade fetch failures: %s", ", ".join(failures))
            if not trades_df.empty:
                upsert_dataframe(conn, "polygon_equity_trades_raw", trades_df, ["symbol", "timestamp", "data_source"])

            # Only compute lit direction for symbols actually fetched this run.
            # Avoid overwriting cached symbols with placeholder rows.
            fetched_symbols = []
            if not trades_df.empty and "symbol" in trades_df.columns:
                fetched_symbols = sorted(trades_df["symbol"].dropna().unique().tolist())
            if fetched_symbols:
                lit_df = compute_lit_directional_flow(trades_df, fetched_symbols, run_date, config)
                upsert_dataframe(conn, "lit_direction_daily", lit_df, ["symbol", "date"])
                lit_frames.append(lit_df)

            try:
                agg_df, agg_cache_stats = fetch_polygon_daily_agg(
                    config, symbols_to_fetch, run_date, conn=conn
                )
                if agg_cache_stats["cached"] > 0 or agg_cache_stats["fetched"] > 0:
                    logging.info(
                        "Daily agg cache: %d cached, %d fetched",
                        agg_cache_stats["cached"], agg_cache_stats["fetched"]
                    )
            except Exception as exc:
                logging.warning("Polygon daily agg fetch failed for %s: %s", run_date.isoformat(), exc)
                agg_df = pd.DataFrame()
            if not agg_df.empty:
                upsert_dataframe(conn, "polygon_daily_agg_raw", agg_df, ["symbol", "trade_date"])
            agg_frames.append(agg_df)

            # Fetch options premium if enabled
            if config.fetch_options_premium and not agg_df.empty:
                try:
                    # Build ATM prices dict from daily agg close prices
                    atm_prices = dict(zip(agg_df["symbol"], agg_df["close"]))
                    options_stats = fetch_options_premium_batch(
                        symbols=symbols_to_fetch,
                        trade_date=run_date,
                        atm_prices=atm_prices,
                        config=config,
                        conn=conn,
                    )
                    if options_stats["fetched"] > 0 or options_stats["cached"] > 0:
                        logging.info(
                            "Options premium: %d fetched, %d cached, %d skipped, %d errors",
                            options_stats["fetched"],
                            options_stats["cached"],
                            options_stats.get("skipped", 0),
                            options_stats["errors"],
                        )
                except Exception as exc:
                    logging.warning("Options premium fetch failed for %s: %s", run_date.isoformat(), exc)

        date_start = min(config.target_dates)
        date_end = max(config.target_dates)

        # Always load inputs from DB to avoid partial-frame recomputation when cache is enabled.
        # This ensures metrics use the full 60-day window even if only some days were fetched.
        symbol_placeholders = ", ".join(["?" for _ in config.tickers])
        symbols_params = list(config.tickers)

        finra_all_df = conn.execute(
            """
            SELECT *
            FROM finra_otc_weekly_raw
            WHERE week_start_date >= ? AND week_start_date <= ?
            """,
            [
                date_start - timedelta(days=date_start.weekday()),
                date_end - timedelta(days=date_end.weekday()),
            ],
        ).fetchdf()

        short_all = conn.execute(
            f"""
            SELECT symbol, trade_date, short_volume, short_exempt_volume, total_volume, market, source, source_file
            FROM finra_short_daily_raw
            WHERE trade_date >= ? AND trade_date <= ?
              AND symbol IN ({symbol_placeholders})
            """,
            [date_start, date_end] + symbols_params,
        ).fetchdf()

        agg_all = conn.execute(
            f"""
            SELECT symbol, trade_date, open, high, low, close, vwap, volume
            FROM polygon_daily_agg_raw
            WHERE trade_date >= ? AND trade_date <= ?
              AND symbol IN ({symbol_placeholders})
            """,
            [date_start, date_end] + symbols_params,
        ).fetchdf()

        # Read lit_direction_daily from DB (includes cached + newly computed)
        lit_all = conn.execute(
            f"""
            SELECT symbol, date, lit_buy_volume, lit_sell_volume, lit_buy_ratio,
                   log_buy_sell, classification_method, lit_coverage_pct, inference_version
            FROM lit_direction_daily
            WHERE date >= ? AND date <= ?
              AND symbol IN ({symbol_placeholders})
            """,
            [date_start, date_end] + symbols_params,
        ).fetchdf()

        # Check if metrics already exist in DB with complete data
        # This avoids recomputing and losing z-scores/OTC data when running with fewer dates
        existing_metrics = _get_existing_metrics(conn, config.tickers, config.target_dates)
        expected_rows = len(config.tickers) * len(config.target_dates)

        if _metrics_are_complete(
            existing_metrics,
            expected_rows,
            short_z_source=config.accumulation_short_z_source,
        ):
            logging.info("Using existing metrics from database (skipping recomputation)")
            daily_metrics_df = existing_metrics
        else:
            logging.info("Computing fresh metrics (existing data incomplete or missing)")
            daily_metrics_df = build_daily_metrics(
                finra_all_df,
                short_all,
                agg_all,
                lit_all,
                config.target_dates,
                config,
            )
            upsert_dataframe(conn, "daily_metrics", daily_metrics_df, ["symbol", "date"])

        index_agg_df = build_index_constituent_short_agg(
            daily_metrics_df,
            agg_all,
            config.target_dates,
            config,
        )
        if not index_agg_df.empty:
            upsert_dataframe(conn, "index_constituent_short_agg_daily", index_agg_df, ["index_symbol", "trade_date"])

        if config.export_csv:
            date_tag = max_date.strftime("%Y%m%d")
            _export_table(daily_metrics_df, config.table_dir / f"daily_metrics_{date_tag}.csv")
            _export_table(index_agg_df, config.table_dir / f"index_agg_{date_tag}.csv")

    finally:
        conn.close()

    try:
        render_daily_metrics_table(
            db_path=config.db_path,
            output_dir=config.table_dir,
            dates=config.target_dates,
            tickers=config.tickers,
            title="Institutional Pressure Metrics",
            table_style=config.table_style,
        )
    except Exception as exc:
        logging.error("Failed to render daily metrics table: %s", exc)

    try:
        recent_dates = sorted(config.target_dates)[-3:]
        render_daily_metrics_table(
            db_path=config.db_path,
            output_dir=config.table_dir,
            dates=recent_dates,
            tickers=config.tickers,
            title="Institutional Pressure Metrics (Last 3 Days)",
            table_style=config.table_style,
        )
    except Exception as exc:
        logging.error("Failed to render last-3-days metrics table: %s", exc)

    try:
        render_metrics_plots(
            db_path=config.db_path,
            output_dir=config.plot_dir,
            dates=config.target_dates,
            tickers=config.tickers,
            mode="layered",
            plot_trading_gaps=config.plot_trading_gaps,
            panel1_metric=config.panel1_metric,
        )
    except Exception as exc:
        logging.error("Failed to render metrics plots: %s", exc)

    try:
        render_combination_plot(
            db_path=config.db_path,
            output_dir=config.plot_dir,
            dates=config.target_dates,
            tickers=config.tickers,
            combination_plot=config.combination_plot,
        )
    except Exception as exc:
        logging.error("Failed to render combination plot: %s", exc)

    if config.render_price_charts:
        try:
            render_price_charts(
                db_path=config.db_path,
                output_dir=config.price_chart_dir,
                dates=config.target_dates,
                tickers=config.tickers,
                timeframe=config.price_bar_timeframe,
                min_premium_highlight=config.options_min_premium_highlight,
                options_premium_display_mode=config.options_premium_display_mode,
                itm_call_hedge_threshold=config.itm_call_hedge_threshold,
            )
        except Exception as exc:
            logging.error("Failed to render price charts: %s", exc)

    logging.info("Analysis complete. Processed %d dates.", len(config.target_dates))


if __name__ == "__main__":
    main()
