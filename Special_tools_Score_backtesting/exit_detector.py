"""
Exit Detector Module

Detects distribution-based exits and computes exit metrics.
Supports multiple exit strategies:
    - fixed_Nd: Exit after N trading days
    - first_dist: Exit on first distribution signal
    - 2x_consec_dist: Exit on 2 consecutive distribution days
    - score_decay: Exit if score < threshold for N consecutive days
    - hybrid: First of distribution OR max_days
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ExitStrategy(Enum):
    """Exit strategy types."""
    FIXED_5D = "fixed_5d"
    FIXED_10D = "fixed_10d"
    FIXED_20D = "fixed_20d"
    FIXED_30D = "fixed_30d"
    FIRST_DIST = "first_dist"
    TWO_CONSEC_DIST = "2x_consec_dist"
    SCORE_DECAY = "score_decay"
    HYBRID_30D = "hybrid_30d"


@dataclass
class ExitResult:
    """Result of exit detection for a single trade."""
    exit_date: Optional[pd.Timestamp]
    exit_price: Optional[float]
    exit_reason: str                    # 'DISTRIBUTION', 'FIXED_PERIOD', 'SCORE_DECAY', 'MAX_DAYS'
    hold_days: int
    return_pct: float
    days_to_first_dist: Optional[int]
    days_to_2x_consec_dist: Optional[int]
    return_at_first_dist: Optional[float]
    return_at_2x_consec_dist: Optional[float]
    max_gain_before_exit: float
    max_drawdown_before_exit: float


# Fixed period mappings
FIXED_PERIODS = {
    "fixed_5d": 5,
    "fixed_10d": 10,
    "fixed_20d": 20,
    "fixed_30d": 30,
}


def find_first_distribution(
    scores: pd.Series,
    start_idx: int,
    sell_threshold: float = 30.0,
    max_look_forward: int = 60,
) -> Optional[int]:
    """
    Find the first distribution signal after entry.

    Args:
        scores: Series of score values (0-100)
        start_idx: Index to start looking from (entry day)
        sell_threshold: Score threshold for distribution
        max_look_forward: Maximum days to look forward

    Returns:
        Index of first distribution day, or None if not found
    """
    end_idx = min(start_idx + max_look_forward, len(scores))

    for i in range(start_idx + 1, end_idx):
        if scores.iloc[i] <= sell_threshold:
            return i

    return None


def find_consecutive_distribution(
    scores: pd.Series,
    start_idx: int,
    sell_threshold: float = 30.0,
    consecutive_days: int = 2,
    max_look_forward: int = 60,
) -> Optional[int]:
    """
    Find first occurrence of N consecutive distribution days.

    Args:
        scores: Series of score values (0-100)
        start_idx: Index to start looking from
        sell_threshold: Score threshold for distribution
        consecutive_days: Required consecutive days
        max_look_forward: Maximum days to look forward

    Returns:
        Index of the Nth consecutive distribution day, or None
    """
    end_idx = min(start_idx + max_look_forward, len(scores))
    consecutive_count = 0

    for i in range(start_idx + 1, end_idx):
        if scores.iloc[i] <= sell_threshold:
            consecutive_count += 1
            if consecutive_count >= consecutive_days:
                return i
        else:
            consecutive_count = 0

    return None


def find_score_decay_exit(
    scores: pd.Series,
    start_idx: int,
    decay_threshold: float = 50.0,
    decay_days: int = 3,
    max_look_forward: int = 60,
) -> Optional[int]:
    """
    Find exit when score stays below threshold for N consecutive days.

    Args:
        scores: Series of score values (0-100)
        start_idx: Index to start looking from
        decay_threshold: Score threshold
        decay_days: Required consecutive days below threshold
        max_look_forward: Maximum days to look forward

    Returns:
        Index of exit day, or None
    """
    end_idx = min(start_idx + max_look_forward, len(scores))
    below_count = 0

    for i in range(start_idx + 1, end_idx):
        if scores.iloc[i] < decay_threshold:
            below_count += 1
            if below_count >= decay_days:
                return i
        else:
            below_count = 0

    return None


def compute_exit(
    entry_idx: int,
    entry_price: float,
    scores: pd.Series,
    prices: pd.Series,
    strategy: str,
    sell_threshold: float = 30.0,
    max_days: int = 30,
    decay_threshold: float = 50.0,
    decay_days: int = 3,
) -> ExitResult:
    """
    Compute exit details for a trade based on strategy.

    Args:
        entry_idx: Index of entry (next day after signal)
        entry_price: Entry price
        scores: Score series
        prices: Price series (close prices)
        strategy: Exit strategy name
        sell_threshold: Distribution threshold
        max_days: Maximum holding period
        decay_threshold: Score decay threshold
        decay_days: Consecutive days for score decay

    Returns:
        ExitResult with all exit metrics
    """
    # Find distribution points regardless of strategy
    first_dist_idx = find_first_distribution(
        scores, entry_idx - 1, sell_threshold, max_days + 10
    )
    two_consec_dist_idx = find_consecutive_distribution(
        scores, entry_idx - 1, sell_threshold, 2, max_days + 10
    )

    # Compute metrics at distribution points
    days_to_first_dist = None
    days_to_2x_consec_dist = None
    return_at_first_dist = None
    return_at_2x_consec_dist = None

    if first_dist_idx is not None:
        days_to_first_dist = first_dist_idx - entry_idx
        if first_dist_idx < len(prices):
            return_at_first_dist = (prices.iloc[first_dist_idx] - entry_price) / entry_price

    if two_consec_dist_idx is not None:
        days_to_2x_consec_dist = two_consec_dist_idx - entry_idx
        if two_consec_dist_idx < len(prices):
            return_at_2x_consec_dist = (prices.iloc[two_consec_dist_idx] - entry_price) / entry_price

    # Determine exit based on strategy
    exit_idx = None
    exit_reason = "MAX_DAYS"

    if strategy in FIXED_PERIODS:
        # Fixed holding period
        period = FIXED_PERIODS[strategy]
        exit_idx = min(entry_idx + period, len(prices) - 1)
        exit_reason = "FIXED_PERIOD"

    elif strategy == "first_dist":
        if first_dist_idx is not None:
            exit_idx = first_dist_idx
            exit_reason = "DISTRIBUTION"
        else:
            exit_idx = min(entry_idx + max_days, len(prices) - 1)
            exit_reason = "MAX_DAYS"

    elif strategy == "2x_consec_dist":
        if two_consec_dist_idx is not None:
            exit_idx = two_consec_dist_idx
            exit_reason = "DISTRIBUTION"
        else:
            exit_idx = min(entry_idx + max_days, len(prices) - 1)
            exit_reason = "MAX_DAYS"

    elif strategy == "score_decay":
        decay_idx = find_score_decay_exit(
            scores, entry_idx - 1, decay_threshold, decay_days, max_days + 10
        )
        if decay_idx is not None:
            exit_idx = decay_idx
            exit_reason = "SCORE_DECAY"
        else:
            exit_idx = min(entry_idx + max_days, len(prices) - 1)
            exit_reason = "MAX_DAYS"

    elif strategy == "hybrid_30d":
        # Exit on first distribution OR 30 days, whichever comes first
        if first_dist_idx is not None and first_dist_idx <= entry_idx + 30:
            exit_idx = first_dist_idx
            exit_reason = "DISTRIBUTION"
        else:
            exit_idx = min(entry_idx + 30, len(prices) - 1)
            exit_reason = "MAX_DAYS"

    else:
        # Default to max days
        exit_idx = min(entry_idx + max_days, len(prices) - 1)
        exit_reason = "MAX_DAYS"

    # Compute exit metrics
    if exit_idx is not None and exit_idx < len(prices):
        exit_price = prices.iloc[exit_idx]
        hold_days = exit_idx - entry_idx
        return_pct = (exit_price - entry_price) / entry_price

        # Get exit date if index exists
        if hasattr(prices, 'index'):
            exit_date = prices.index[exit_idx] if isinstance(prices.index[0], pd.Timestamp) else None
        else:
            exit_date = None

        # Max gain/drawdown during holding period
        hold_prices = prices.iloc[entry_idx:exit_idx + 1]
        if len(hold_prices) > 0:
            max_gain = ((hold_prices.max() - entry_price) / entry_price)
            max_drawdown = ((hold_prices.min() - entry_price) / entry_price)
        else:
            max_gain = 0.0
            max_drawdown = 0.0
    else:
        exit_date = None
        exit_price = None
        hold_days = 0
        return_pct = 0.0
        max_gain = 0.0
        max_drawdown = 0.0

    return ExitResult(
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=exit_reason,
        hold_days=hold_days,
        return_pct=return_pct,
        days_to_first_dist=days_to_first_dist,
        days_to_2x_consec_dist=days_to_2x_consec_dist,
        return_at_first_dist=return_at_first_dist,
        return_at_2x_consec_dist=return_at_2x_consec_dist,
        max_gain_before_exit=max_gain,
        max_drawdown_before_exit=max_drawdown,
    )


def add_exit_metrics_to_signals(
    signals_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    price_df: pd.DataFrame,
    strategy: str = "first_dist",
    sell_threshold: float = 30.0,
    max_days: int = 30,
    symbol_col: str = 'symbol',
    date_col: str = 'date',
    score_col: str = 'score_value',
    price_col: str = 'close',
) -> pd.DataFrame:
    """
    Add exit metrics to signals DataFrame.

    Args:
        signals_df: DataFrame with accumulation signals
        scores_df: DataFrame with daily scores (same structure as signals_df)
        price_df: DataFrame with price data
        strategy: Exit strategy name
        sell_threshold: Distribution threshold
        max_days: Maximum holding period
        symbol_col: Symbol column name
        date_col: Date column name
        score_col: Score column name
        price_col: Price column name

    Returns:
        signals_df with exit metric columns added
    """
    signals_df = signals_df.copy()

    # Initialize exit columns
    signals_df['exit_date'] = pd.NaT
    signals_df['exit_price'] = np.nan
    signals_df['exit_reason'] = ''
    signals_df['hold_days'] = 0
    signals_df['trade_return'] = np.nan
    signals_df['days_to_first_dist'] = np.nan
    signals_df['days_to_2x_consec_dist'] = np.nan
    signals_df['return_at_first_dist'] = np.nan
    signals_df['return_at_2x_consec_dist'] = np.nan
    signals_df['max_gain_before_exit'] = np.nan
    signals_df['max_drawdown_before_exit'] = np.nan

    # Process each symbol
    for symbol in signals_df[symbol_col].unique():
        symbol_signals = signals_df[signals_df[symbol_col] == symbol]
        symbol_scores = scores_df[scores_df[symbol_col] == symbol].sort_values(date_col)
        symbol_prices = price_df[price_df[symbol_col] == symbol].sort_values(date_col)

        if symbol_scores.empty or symbol_prices.empty:
            continue

        # Create date to index mappings
        scores_dates = symbol_scores[date_col].values
        prices_dates = symbol_prices[date_col].values

        scores_series = symbol_scores[score_col].reset_index(drop=True)
        prices_series = symbol_prices[price_col].reset_index(drop=True)

        date_to_score_idx = {d: i for i, d in enumerate(scores_dates)}
        date_to_price_idx = {d: i for i, d in enumerate(prices_dates)}

        # Process each signal
        for sig_idx in symbol_signals.index:
            sig_date = signals_df.loc[sig_idx, date_col]

            # Skip if date not in data
            if sig_date not in date_to_score_idx or sig_date not in date_to_price_idx:
                continue

            score_idx = date_to_score_idx[sig_date]
            price_idx = date_to_price_idx[sig_date]

            # Entry is next day
            entry_idx = price_idx + 1
            if entry_idx >= len(prices_series):
                continue

            entry_price = symbol_prices.iloc[entry_idx][price_col]

            # Compute exit
            exit_result = compute_exit(
                entry_idx=entry_idx,
                entry_price=entry_price,
                scores=scores_series,
                prices=prices_series,
                strategy=strategy,
                sell_threshold=sell_threshold,
                max_days=max_days,
            )

            # Get exit date
            if exit_result.hold_days > 0:
                exit_price_idx = entry_idx + exit_result.hold_days
                if exit_price_idx < len(symbol_prices):
                    exit_date = symbol_prices.iloc[exit_price_idx][date_col]
                    signals_df.loc[sig_idx, 'exit_date'] = exit_date

            signals_df.loc[sig_idx, 'exit_price'] = exit_result.exit_price
            signals_df.loc[sig_idx, 'exit_reason'] = exit_result.exit_reason
            signals_df.loc[sig_idx, 'hold_days'] = exit_result.hold_days
            signals_df.loc[sig_idx, 'trade_return'] = exit_result.return_pct
            signals_df.loc[sig_idx, 'days_to_first_dist'] = exit_result.days_to_first_dist
            signals_df.loc[sig_idx, 'days_to_2x_consec_dist'] = exit_result.days_to_2x_consec_dist
            signals_df.loc[sig_idx, 'return_at_first_dist'] = exit_result.return_at_first_dist
            signals_df.loc[sig_idx, 'return_at_2x_consec_dist'] = exit_result.return_at_2x_consec_dist
            signals_df.loc[sig_idx, 'max_gain_before_exit'] = exit_result.max_gain_before_exit
            signals_df.loc[sig_idx, 'max_drawdown_before_exit'] = exit_result.max_drawdown_before_exit

    return signals_df


def classify_failure_mode(
    entry_price: float,
    prices_during_hold: pd.Series,
    hold_days: int,
    trade_return: float,
    parent_etf_return: Optional[float] = None,
) -> Optional[str]:
    """
    Classify why a trade failed (returned negative).

    Args:
        entry_price: Entry price
        prices_during_hold: Price series during holding period
        hold_days: Number of days held
        trade_return: Final trade return
        parent_etf_return: Parent ETF return during same period

    Returns:
        Failure mode string, or None if trade was profitable
    """
    if trade_return >= 0:
        return None

    if len(prices_during_hold) < 2:
        return "INSUFFICIENT_DATA"

    # Check for immediate reversal (loss within first 2 days)
    early_prices = prices_during_hold.head(3)
    if len(early_prices) >= 2:
        early_return = (early_prices.iloc[-1] - entry_price) / entry_price
        if early_return < -0.02:  # More than 2% loss early
            return "IMMEDIATE_REVERSAL"

    # Check for gap down (first price much lower than entry)
    first_price = prices_during_hold.iloc[0]
    gap = (first_price - entry_price) / entry_price
    if gap < -0.015:  # More than 1.5% gap down
        return "GAP_DOWN"

    # Check for sector drag
    if parent_etf_return is not None and parent_etf_return < -0.02:
        return "SECTOR_DRAG"

    # Check for slow bleed (gradual decline)
    returns = (prices_during_hold - entry_price) / entry_price
    if returns.max() < 0.01:  # Never got to 1% profit
        return "SLOW_BLEED"

    # Default: false signal
    return "FALSE_SIGNAL"


# =============================================================================
# CLI / STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import duckdb
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from darkpool_analysis.config import load_config

    config = load_config()

    # Create sample data for testing
    dates = pd.date_range('2025-01-01', periods=60, freq='D')

    # Simulated scores and prices
    np.random.seed(42)
    scores = pd.Series(50 + 20 * np.sin(np.arange(60) * 0.2) + np.random.randn(60) * 5)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(60) * 0.01)))

    print("Testing exit detection...")
    print(f"Scores range: {scores.min():.1f} - {scores.max():.1f}")
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")

    # Test finding distribution points
    first_dist = find_first_distribution(scores, 5, sell_threshold=40)
    print(f"\nFirst distribution after day 5: day {first_dist}")

    two_consec = find_consecutive_distribution(scores, 5, sell_threshold=40, consecutive_days=2)
    print(f"First 2 consecutive distribution after day 5: day {two_consec}")

    # Test compute_exit
    for strategy in ["fixed_10d", "first_dist", "2x_consec_dist", "hybrid_30d"]:
        entry_idx = 10
        entry_price = prices.iloc[entry_idx]

        result = compute_exit(
            entry_idx=entry_idx,
            entry_price=entry_price,
            scores=scores,
            prices=prices,
            strategy=strategy,
            sell_threshold=40,
            max_days=30,
        )

        print(f"\nStrategy: {strategy}")
        print(f"  Exit reason: {result.exit_reason}")
        print(f"  Hold days: {result.hold_days}")
        print(f"  Return: {result.return_pct:.2%}")
        print(f"  Days to first dist: {result.days_to_first_dist}")
        print(f"  Max gain: {result.max_gain_before_exit:.2%}")
        print(f"  Max drawdown: {result.max_drawdown_before_exit:.2%}")
