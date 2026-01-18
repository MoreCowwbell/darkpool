"""
Walk-Forward Validation Module

Implements rolling walk-forward validation for accumulation signal backtesting.
This prevents overfitting by ensuring out-of-sample testing.

Structure:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Jan-Jun 2025 (Train)  │ Jul-Aug (Test) │ Sep-Oct (Train) │ ...  │
    └─────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import logging
import duckdb

from .config import BacktestConfig, ParameterSpace, COARSE_SWEEP_SPACE, load_config
from .parameter_sweep import (
    SweepConfig,
    run_single_backtest,
    get_top_configs,
    generate_sweep_configs,
    count_sweep_configs,
)
from .signal_generator import load_daily_metrics
from .forward_returns import load_price_data
from .regime_classifier import load_spy_vix_data
from .metrics import BacktestMetrics, compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Defines a single train/test window."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    window_id: int


@dataclass
class WalkForwardResult:
    """Results from a walk-forward validation run."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_config: SweepConfig
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    config_id: str


@dataclass
class WalkForwardSummary:
    """Aggregated walk-forward results."""
    total_windows: int = 0
    avg_train_hit_rate: float = 0.0
    avg_test_hit_rate: float = 0.0
    avg_train_return: float = 0.0
    avg_test_return: float = 0.0
    train_test_hit_rate_gap: float = 0.0  # Overfitting indicator
    train_test_return_gap: float = 0.0
    consistent_configs: List[str] = field(default_factory=list)
    total_oos_signals: int = 0
    total_oos_trades: int = 0
    oos_hit_rate: float = 0.0
    oos_avg_return: float = 0.0


def generate_walk_forward_windows(
    start_date: str,
    end_date: str,
    train_months: int = 6,
    test_months: int = 2,
    step_months: int = 2,
) -> List[WalkForwardWindow]:
    """
    Generate walk-forward train/test windows.

    Args:
        start_date: Overall start date
        end_date: Overall end date
        train_months: Training window size in months
        test_months: Test window size in months
        step_months: Step size for rolling windows

    Returns:
        List of WalkForwardWindow objects
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    windows = []
    window_id = 0
    current = start

    while True:
        # Train window
        train_start = current
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)

        # Test window starts after train ends
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        # Check if test window is within bounds
        if test_end > end:
            break

        windows.append(WalkForwardWindow(
            train_start=train_start.strftime('%Y-%m-%d'),
            train_end=train_end.strftime('%Y-%m-%d'),
            test_start=test_start.strftime('%Y-%m-%d'),
            test_end=test_end.strftime('%Y-%m-%d'),
            window_id=window_id,
        ))

        window_id += 1
        current += pd.DateOffset(months=step_months)

    return windows


def run_walk_forward_validation(
    param_space: ParameterSpace,
    conn: duckdb.DuckDBPyConnection,
    symbols: List[str],
    start_date: str,
    end_date: str,
    train_months: int = 6,
    test_months: int = 2,
    step_months: int = 2,
    top_n_configs: int = 3,
    min_train_signals: int = 20,
    progress_callback: Optional[callable] = None,
) -> Tuple[List[WalkForwardResult], WalkForwardSummary]:
    """
    Run walk-forward validation across multiple windows.

    Args:
        param_space: Parameter space to search
        conn: DuckDB connection
        symbols: List of symbols
        start_date: Overall start date
        end_date: Overall end date
        train_months: Training window size
        test_months: Test window size
        step_months: Rolling step size
        top_n_configs: Number of top configs to test in each window
        min_train_signals: Minimum signals required in training
        progress_callback: Optional callback(window_id, total_windows, phase)

    Returns:
        Tuple of (list of WalkForwardResult, WalkForwardSummary)
    """
    # Generate windows
    windows = generate_walk_forward_windows(
        start_date, end_date, train_months, test_months, step_months
    )

    if not windows:
        logger.error("No valid walk-forward windows generated")
        return [], WalkForwardSummary()

    logger.info(f"Generated {len(windows)} walk-forward windows")

    # Load full data range once
    full_start = windows[0].train_start
    full_end = windows[-1].test_end

    logger.info(f"Loading data from {full_start} to {full_end}")
    daily_metrics_df = load_daily_metrics(conn, symbols, full_start, full_end)
    price_df = load_price_data(conn, symbols, full_start, full_end, buffer_days=30)
    spy_vix_df = load_spy_vix_data(conn, full_start, full_end)

    if daily_metrics_df.empty:
        logger.error("No data loaded")
        return [], WalkForwardSummary()

    results = []

    # Process each window
    for window in windows:
        if progress_callback:
            progress_callback(window.window_id, len(windows), "train")

        logger.info(f"\nWindow {window.window_id}: Train {window.train_start} to {window.train_end}")
        logger.info(f"                     Test  {window.test_start} to {window.test_end}")

        # Filter data for training period
        train_mask = (
            (daily_metrics_df['date'] >= window.train_start) &
            (daily_metrics_df['date'] <= window.train_end)
        )
        train_data = daily_metrics_df[train_mask].copy()

        train_price_mask = (
            (price_df['date'] >= window.train_start) &
            (price_df['date'] <= window.train_end)
        )
        train_prices = price_df[train_price_mask].copy()

        if train_data.empty or train_prices.empty:
            logger.warning(f"Window {window.window_id}: No training data")
            continue

        # Run parameter sweep on training data
        train_results = {}

        for sweep_config in generate_sweep_configs(param_space):
            try:
                metrics, _ = run_single_backtest(
                    sweep_config=sweep_config,
                    daily_metrics_df=train_data,
                    price_df=train_prices,
                    spy_vix_df=spy_vix_df,
                    date_start=window.train_start,
                    date_end=window.train_end,
                )

                if metrics.total_signals >= min_train_signals:
                    train_results[sweep_config.config_id] = (sweep_config, metrics)

            except Exception as e:
                logger.debug(f"Error in config {sweep_config.config_id}: {e}")
                continue

        if not train_results:
            logger.warning(f"Window {window.window_id}: No valid configs found")
            continue

        # Find best config by rank score
        best_config_id = max(
            train_results.keys(),
            key=lambda k: train_results[k][1].rank_score
        )
        best_config, train_metrics = train_results[best_config_id]

        logger.info(f"Best config: {best_config_id}")
        logger.info(f"Train hit rate: {train_metrics.hit_rate:.1%}, avg return: {train_metrics.avg_return:.2%}")

        # Test on out-of-sample period
        if progress_callback:
            progress_callback(window.window_id, len(windows), "test")

        test_mask = (
            (daily_metrics_df['date'] >= window.test_start) &
            (daily_metrics_df['date'] <= window.test_end)
        )
        test_data = daily_metrics_df[test_mask].copy()

        test_price_mask = (
            (price_df['date'] >= window.test_start) &
            (price_df['date'] <= window.test_end)
        )
        test_prices = price_df[test_price_mask].copy()

        if test_data.empty or test_prices.empty:
            logger.warning(f"Window {window.window_id}: No test data")
            continue

        try:
            test_metrics, _ = run_single_backtest(
                sweep_config=best_config,
                daily_metrics_df=test_data,
                price_df=test_prices,
                spy_vix_df=spy_vix_df,
                date_start=window.test_start,
                date_end=window.test_end,
            )

            logger.info(f"Test hit rate: {test_metrics.hit_rate:.1%}, avg return: {test_metrics.avg_return:.2%}")

            results.append(WalkForwardResult(
                window_id=window.window_id,
                train_start=window.train_start,
                train_end=window.train_end,
                test_start=window.test_start,
                test_end=window.test_end,
                best_config=best_config,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                config_id=best_config_id,
            ))

        except Exception as e:
            logger.error(f"Window {window.window_id}: Test error: {e}")
            continue

    # Compute summary
    summary = compute_walk_forward_summary(results)

    return results, summary


def compute_walk_forward_summary(results: List[WalkForwardResult]) -> WalkForwardSummary:
    """Compute aggregate summary from walk-forward results."""
    if not results:
        return WalkForwardSummary()

    summary = WalkForwardSummary()
    summary.total_windows = len(results)

    # Aggregate metrics
    train_hit_rates = [r.train_metrics.hit_rate for r in results]
    test_hit_rates = [r.test_metrics.hit_rate for r in results]
    train_returns = [r.train_metrics.avg_return for r in results]
    test_returns = [r.test_metrics.avg_return for r in results]

    summary.avg_train_hit_rate = np.mean(train_hit_rates)
    summary.avg_test_hit_rate = np.mean(test_hit_rates)
    summary.avg_train_return = np.mean(train_returns)
    summary.avg_test_return = np.mean(test_returns)

    # Overfitting indicators
    summary.train_test_hit_rate_gap = summary.avg_train_hit_rate - summary.avg_test_hit_rate
    summary.train_test_return_gap = summary.avg_train_return - summary.avg_test_return

    # Find consistent configs (appeared in multiple windows)
    config_counts = {}
    for r in results:
        config_id = r.config_id
        config_counts[config_id] = config_counts.get(config_id, 0) + 1

    summary.consistent_configs = [c for c, count in config_counts.items() if count > 1]

    # Out-of-sample aggregates
    summary.total_oos_signals = sum(r.test_metrics.total_signals for r in results)
    summary.total_oos_trades = sum(r.test_metrics.total_trades for r in results)

    if summary.total_oos_trades > 0:
        total_oos_wins = sum(r.test_metrics.profitable_trades for r in results)
        summary.oos_hit_rate = total_oos_wins / summary.total_oos_trades

        # Weighted average return by number of trades
        weighted_returns = sum(
            r.test_metrics.avg_return * r.test_metrics.total_trades
            for r in results
        )
        summary.oos_avg_return = weighted_returns / summary.total_oos_trades

    return summary


def results_to_dataframe(results: List[WalkForwardResult]) -> pd.DataFrame:
    """Convert walk-forward results to DataFrame."""
    rows = []
    for r in results:
        row = {
            'window_id': r.window_id,
            'train_start': r.train_start,
            'train_end': r.train_end,
            'test_start': r.test_start,
            'test_end': r.test_end,
            'config_id': r.config_id,
            'score_variant': r.best_config.score_variant,
            'z_window': r.best_config.z_window,
            'exit_strategy': r.best_config.exit_strategy,
            'train_signals': r.train_metrics.total_signals,
            'train_hit_rate': r.train_metrics.hit_rate,
            'train_avg_return': r.train_metrics.avg_return,
            'train_sharpe': r.train_metrics.sharpe_ratio,
            'test_signals': r.test_metrics.total_signals,
            'test_hit_rate': r.test_metrics.hit_rate,
            'test_avg_return': r.test_metrics.avg_return,
            'test_sharpe': r.test_metrics.sharpe_ratio,
            'hit_rate_decay': r.train_metrics.hit_rate - r.test_metrics.hit_rate,
            'return_decay': r.train_metrics.avg_return - r.test_metrics.avg_return,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_walk_forward_summary(summary: WalkForwardSummary):
    """Print formatted walk-forward summary."""
    print("\n" + "=" * 60)
    print(" Walk-Forward Validation Summary")
    print("=" * 60)
    print(f"  Total Windows:          {summary.total_windows}")
    print(f"  Avg Train Hit Rate:     {summary.avg_train_hit_rate:.1%}")
    print(f"  Avg Test Hit Rate:      {summary.avg_test_hit_rate:.1%}")
    print(f"  Hit Rate Gap:           {summary.train_test_hit_rate_gap:.1%} (train - test)")
    print(f"  Avg Train Return:       {summary.avg_train_return:.2%}")
    print(f"  Avg Test Return:        {summary.avg_test_return:.2%}")
    print(f"  Return Gap:             {summary.train_test_return_gap:.2%}")
    print(f"  Total OOS Trades:       {summary.total_oos_trades}")
    print(f"  OOS Hit Rate:           {summary.oos_hit_rate:.1%}")
    print(f"  OOS Avg Return:         {summary.oos_avg_return:.2%}")

    if summary.consistent_configs:
        print(f"\n  Consistent Configs (multiple windows):")
        for config in summary.consistent_configs[:5]:
            print(f"    - {config}")

    # Overfitting assessment
    print("\n  Overfitting Assessment:")
    if summary.train_test_hit_rate_gap < 0.05 and summary.train_test_return_gap < 0.005:
        print("    ✓ Low overfitting risk - test performance close to train")
    elif summary.train_test_hit_rate_gap < 0.10:
        print("    ⚠ Moderate overfitting - some performance decay")
    else:
        print("    ✗ High overfitting risk - significant train/test gap")

    print("=" * 60)


# =============================================================================
# CLI / STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    config = load_config()

    print("=" * 60)
    print(" Walk-Forward Validation Test")
    print("=" * 60)

    # Test window generation
    windows = generate_walk_forward_windows(
        start_date="2025-01-01",
        end_date="2025-12-31",
        train_months=6,
        test_months=2,
        step_months=2,
    )

    print(f"\nGenerated {len(windows)} windows:")
    for w in windows:
        print(f"  Window {w.window_id}: Train {w.train_start} to {w.train_end} | Test {w.test_start} to {w.test_end}")

    # Minimal test sweep
    minimal_space = ParameterSpace(
        score_variants=["B"],
        w_short=[0.55],
        w_lit=[0.30],
        w_price=[0.15],
        z_windows=[20],
        buy_thresholds=[70],
        sell_thresholds=[30],
        exit_strategies=["first_dist"],
    )

    print("\nRunning minimal walk-forward test...")
    conn = duckdb.connect(str(config.db_path))
    try:
        symbols = ["SPY", "QQQ"]

        def progress(window_id, total, phase):
            print(f"  Window {window_id}/{total} - {phase}")

        results, summary = run_walk_forward_validation(
            param_space=minimal_space,
            conn=conn,
            symbols=symbols,
            start_date="2025-01-01",
            end_date="2025-12-31",
            train_months=6,
            test_months=2,
            min_train_signals=5,
            progress_callback=progress,
        )

        if results:
            df = results_to_dataframe(results)
            print("\nResults:")
            print(df[['window_id', 'config_id', 'train_hit_rate', 'test_hit_rate', 'hit_rate_decay']].to_string())

            print_walk_forward_summary(summary)
        else:
            print("No results (check data availability)")

    finally:
        conn.close()
