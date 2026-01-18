"""
Parameter Sweep Module

Grid search over accumulation score configurations.
Supports coarse and fine-grained parameter sweeps.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
from pathlib import Path
from itertools import product
import logging
import duckdb

from .config import (
    BacktestConfig,
    ParameterSpace,
    COARSE_SWEEP_SPACE,
    SCORE_VARIANTS,
    WEIGHT_PRESETS,
    load_config,
)
from .signal_generator import SignalConfig, generate_signals, load_daily_metrics
from .forward_returns import add_forward_returns_to_signals, load_price_data
from .exit_detector import add_exit_metrics_to_signals
from .trend_classifier import add_trend_columns
from .regime_classifier import add_regime_to_dataframe, load_spy_vix_data
from .options_context import add_options_context
from .metrics import compute_all_metrics, BacktestMetrics, compare_configs
from .schema import upsert_backtest_run

logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Configuration for a single parameter sweep iteration."""
    score_variant: str
    w_short: float
    w_lit: float
    w_price: float
    z_window: int
    buy_threshold: int
    sell_threshold: int
    exit_strategy: str

    def to_signal_config(self) -> SignalConfig:
        """Convert to SignalConfig for signal generation."""
        return SignalConfig(
            score_variant=self.score_variant,
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold,
            w_short=self.w_short,
            w_lit=self.w_lit,
            w_price=self.w_price,
            z_window=self.z_window,
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'score_variant': self.score_variant,
            'w_short': self.w_short,
            'w_lit': self.w_lit,
            'w_price': self.w_price,
            'z_window': self.z_window,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'exit_strategy': self.exit_strategy,
        }

    @property
    def config_id(self) -> str:
        """Unique identifier for this configuration."""
        return (
            f"{self.score_variant}_"
            f"w{self.w_short:.2f}-{self.w_lit:.2f}-{self.w_price:.2f}_"
            f"z{self.z_window}_"
            f"t{self.buy_threshold}-{self.sell_threshold}_"
            f"{self.exit_strategy}"
        )


def generate_sweep_configs(
    param_space: ParameterSpace,
    weight_constraint: bool = True,
) -> Iterator[SweepConfig]:
    """
    Generate all parameter combinations from a ParameterSpace.

    Args:
        param_space: Parameter space definition
        weight_constraint: If True, filter configs where weights sum to ~1.0

    Yields:
        SweepConfig for each valid combination
    """
    for variant, w_short, w_lit, w_price, z_window, buy_thresh, sell_thresh, exit_strat in product(
        param_space.score_variants,
        param_space.w_short,
        param_space.w_lit,
        param_space.w_price,
        param_space.z_windows,
        param_space.buy_thresholds,
        param_space.sell_thresholds,
        param_space.exit_strategies,
    ):
        # Weight constraint: must sum to ~1.0 (within 0.05)
        if weight_constraint:
            weight_sum = w_short + w_lit + w_price
            if abs(weight_sum - 1.0) > 0.05:
                continue

        yield SweepConfig(
            score_variant=variant,
            w_short=w_short,
            w_lit=w_lit,
            w_price=w_price,
            z_window=z_window,
            buy_threshold=buy_thresh,
            sell_threshold=sell_thresh,
            exit_strategy=exit_strat,
        )


def count_sweep_configs(param_space: ParameterSpace, weight_constraint: bool = True) -> int:
    """Count total number of configurations in a sweep."""
    return sum(1 for _ in generate_sweep_configs(param_space, weight_constraint))


def run_single_backtest(
    sweep_config: SweepConfig,
    daily_metrics_df: pd.DataFrame,
    price_df: pd.DataFrame,
    spy_vix_df: Optional[pd.DataFrame] = None,
    options_df: Optional[pd.DataFrame] = None,
    date_start: str = None,
    date_end: str = None,
) -> Tuple[BacktestMetrics, pd.DataFrame]:
    """
    Run a single backtest for one configuration.

    Args:
        sweep_config: Configuration to test
        daily_metrics_df: Daily metrics data
        price_df: Price data for forward returns
        spy_vix_df: SPY/VIX data for regime classification
        options_df: Options data for context
        date_start: Start date for filtering
        date_end: End date for filtering

    Returns:
        Tuple of (BacktestMetrics, signals_df with outcomes)
    """
    # Convert to signal config
    signal_config = sweep_config.to_signal_config()

    # Generate signals
    signals_df = generate_signals(daily_metrics_df, signal_config)

    # Filter to accumulation signals only
    accum_signals = signals_df[signals_df['signal_type'] == 'ACCUMULATION'].copy()

    if accum_signals.empty:
        return BacktestMetrics(), pd.DataFrame()

    # Filter by date range if specified
    if date_start:
        accum_signals = accum_signals[accum_signals['date'] >= date_start]
    if date_end:
        accum_signals = accum_signals[accum_signals['date'] <= date_end]

    if accum_signals.empty:
        return BacktestMetrics(), pd.DataFrame()

    # Add trend context
    accum_signals = add_trend_columns(accum_signals, close_col='close')

    # Add regime context if SPY/VIX data available
    if spy_vix_df is not None:
        accum_signals = add_regime_to_dataframe(accum_signals, spy_vix_df)

    # Add options context if available
    if options_df is not None:
        accum_signals = add_options_context(accum_signals, options_df)

    # Add forward returns
    accum_signals = add_forward_returns_to_signals(
        signals_df=accum_signals,
        price_df=price_df,
        price_col='open',
        close_col='close',
    )

    # Add exit metrics
    accum_signals = add_exit_metrics_to_signals(
        signals_df=accum_signals,
        scores_df=signals_df,  # Full scores for exit detection
        price_df=price_df,
        strategy=sweep_config.exit_strategy,
        sell_threshold=sweep_config.sell_threshold,
    )

    # Compute metrics
    metrics = compute_all_metrics(accum_signals)

    return metrics, accum_signals


def run_parameter_sweep(
    param_space: ParameterSpace,
    conn: duckdb.DuckDBPyConnection,
    symbols: List[str],
    date_start: str,
    date_end: str,
    regime_filter: Optional[str] = None,
    save_signals: bool = False,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[callable] = None,
) -> pd.DataFrame:
    """
    Run full parameter sweep and return results.

    Args:
        param_space: Parameter space to sweep
        conn: DuckDB connection
        symbols: List of symbols to test
        date_start: Start date
        date_end: End date
        regime_filter: Optional regime filter ('RISK_ON', 'RISK_OFF', or None for all)
        save_signals: If True, save individual signal DataFrames
        output_dir: Directory for output files
        progress_callback: Optional callback(current, total, config_id) for progress

    Returns:
        DataFrame with results for each configuration
    """
    # Load data once
    logger.info("Loading daily metrics data...")
    daily_metrics_df = load_daily_metrics(conn, symbols, date_start, date_end)

    if daily_metrics_df.empty:
        logger.error("No daily metrics data found")
        return pd.DataFrame()

    logger.info(f"Loaded {len(daily_metrics_df)} daily metrics rows")

    # Load price data
    logger.info("Loading price data...")
    price_df = load_price_data(conn, symbols, date_start, date_end, buffer_days=30)
    logger.info(f"Loaded {len(price_df)} price rows")

    # Load SPY/VIX for regime classification
    logger.info("Loading SPY/VIX data...")
    spy_vix_df = load_spy_vix_data(conn, date_start, date_end)

    # Count configurations
    total_configs = count_sweep_configs(param_space)
    logger.info(f"Running sweep over {total_configs} configurations")

    # Run sweep
    results = []
    current = 0

    for sweep_config in generate_sweep_configs(param_space):
        current += 1

        if progress_callback:
            progress_callback(current, total_configs, sweep_config.config_id)

        try:
            metrics, signals_df = run_single_backtest(
                sweep_config=sweep_config,
                daily_metrics_df=daily_metrics_df,
                price_df=price_df,
                spy_vix_df=spy_vix_df,
                date_start=date_start,
                date_end=date_end,
            )

            # Apply regime filter if specified
            if regime_filter and 'regime' in signals_df.columns:
                signals_df = signals_df[signals_df['regime'] == regime_filter]
                metrics = compute_all_metrics(signals_df)

            # Build result row
            result = sweep_config.to_dict()
            result['config_id'] = sweep_config.config_id
            result['total_signals'] = metrics.total_signals
            result['total_trades'] = metrics.total_trades
            result['hit_rate'] = metrics.hit_rate
            result['avg_return'] = metrics.avg_return
            result['median_return'] = metrics.median_return
            result['win_loss_ratio'] = metrics.win_loss_ratio
            result['sharpe_ratio'] = metrics.sharpe_ratio
            result['max_drawdown'] = metrics.max_drawdown
            result['avg_hold_days'] = metrics.avg_hold_days
            result['signals_per_month'] = metrics.signals_per_month
            result['rank_score'] = metrics.rank_score

            results.append(result)

            # Save individual signals if requested
            if save_signals and output_dir and not signals_df.empty:
                signals_path = output_dir / f"signals_{sweep_config.config_id}.parquet"
                signals_df.to_parquet(signals_path)

        except Exception as e:
            logger.error(f"Error running config {sweep_config.config_id}: {e}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by rank score
    if 'rank_score' in results_df.columns:
        results_df = results_df.sort_values('rank_score', ascending=False)

    return results_df


def get_top_configs(
    results_df: pd.DataFrame,
    n: int = 10,
    sort_by: str = 'rank_score',
    min_signals: int = 10,
) -> pd.DataFrame:
    """
    Get top N configurations from sweep results.

    Args:
        results_df: Results DataFrame from sweep
        n: Number of top configs to return
        sort_by: Column to sort by
        min_signals: Minimum signals required

    Returns:
        Top N configurations
    """
    filtered = results_df[results_df['total_signals'] >= min_signals].copy()

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=False)

    return filtered.head(n)


def save_sweep_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    sweep_name: str = "parameter_sweep",
) -> None:
    """Save sweep results to multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as CSV
    csv_path = output_dir / f"{sweep_name}_results.csv"
    results_df.to_csv(csv_path, index=False)

    # Save as parquet for faster loading
    parquet_path = output_dir / f"{sweep_name}_results.parquet"
    results_df.to_parquet(parquet_path)

    # Save top 20 summary
    top_20 = get_top_configs(results_df, n=20)
    summary_path = output_dir / f"{sweep_name}_top20.csv"
    top_20.to_csv(summary_path, index=False)

    logger.info(f"Saved results to {output_dir}")


# =============================================================================
# CLI / STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys
    from datetime import datetime

    logging.basicConfig(level=logging.INFO)

    # Load config
    config = load_config()

    print("=" * 60)
    print(" Parameter Sweep Test")
    print("=" * 60)

    # Count configurations in coarse sweep
    coarse_count = count_sweep_configs(COARSE_SWEEP_SPACE)
    print(f"\nCoarse sweep: {coarse_count} configurations")

    # Show sample configurations
    print("\nSample configurations:")
    for i, sweep_config in enumerate(generate_sweep_configs(COARSE_SWEEP_SPACE)):
        if i >= 5:
            break
        print(f"  {sweep_config.config_id}")

    # Test with minimal sweep
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

    minimal_count = count_sweep_configs(minimal_space)
    print(f"\nMinimal sweep: {minimal_count} configuration(s)")

    # Run minimal sweep
    print("\nRunning minimal sweep...")
    conn = duckdb.connect(str(config.db_path))
    try:
        symbols = ["SPY", "QQQ"]
        date_start = "2025-10-01"
        date_end = "2025-12-31"

        def progress(current, total, config_id):
            print(f"  [{current}/{total}] {config_id}")

        results = run_parameter_sweep(
            param_space=minimal_space,
            conn=conn,
            symbols=symbols,
            date_start=date_start,
            date_end=date_end,
            progress_callback=progress,
        )

        if not results.empty:
            print("\nResults:")
            print(results[['config_id', 'total_signals', 'hit_rate', 'avg_return', 'rank_score']].to_string())
        else:
            print("No results (check data availability)")

    finally:
        conn.close()
