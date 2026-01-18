"""
Forward Returns Module

Computes forward returns for accumulation signals.
Entry is assumed at next-day open (realistic timing).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import duckdb
from pathlib import Path


@dataclass
class ForwardReturnResult:
    """Forward return metrics for a single signal."""
    entry_date: pd.Timestamp
    entry_price: float
    fwd_return_1d: float
    fwd_return_3d: float
    fwd_return_5d: float
    fwd_return_10d: float
    fwd_return_20d: float
    max_gain: float           # Max gain from entry before exit
    max_drawdown: float       # Max drawdown from entry before exit
    hit_5d: bool              # Profitable at 5d
    hit_10d: bool             # Profitable at 10d


# Forward return periods to compute
FORWARD_PERIODS = [1, 3, 5, 10, 20]


def compute_forward_returns(
    prices: pd.Series,
    signal_idx: int,
    periods: List[int] = None,
) -> Dict[str, float]:
    """
    Compute forward returns from a signal point.

    Entry is at next-day open (signal_idx + 1).

    Args:
        prices: Series of prices (typically 'open' or 'close')
        signal_idx: Index of the signal day
        periods: List of forward periods [1, 3, 5, 10, 20]

    Returns:
        Dictionary with forward returns for each period
    """
    if periods is None:
        periods = FORWARD_PERIODS

    results = {}

    # Entry is at next day (signal_idx + 1)
    entry_idx = signal_idx + 1

    if entry_idx >= len(prices):
        # No next-day data
        for p in periods:
            results[f'fwd_return_{p}d'] = np.nan
        return results

    entry_price = prices.iloc[entry_idx]

    if pd.isna(entry_price) or entry_price <= 0:
        for p in periods:
            results[f'fwd_return_{p}d'] = np.nan
        return results

    for p in periods:
        exit_idx = entry_idx + p
        if exit_idx < len(prices):
            exit_price = prices.iloc[exit_idx]
            if pd.notna(exit_price) and exit_price > 0:
                results[f'fwd_return_{p}d'] = (exit_price - entry_price) / entry_price
            else:
                results[f'fwd_return_{p}d'] = np.nan
        else:
            results[f'fwd_return_{p}d'] = np.nan

    return results


def compute_max_gain_drawdown(
    prices: pd.Series,
    signal_idx: int,
    max_hold_days: int = 30,
) -> Tuple[float, float]:
    """
    Compute maximum gain and drawdown from entry before max holding period.

    Args:
        prices: Series of prices
        signal_idx: Index of signal day
        max_hold_days: Maximum days to look forward

    Returns:
        Tuple of (max_gain, max_drawdown) as percentages
    """
    entry_idx = signal_idx + 1

    if entry_idx >= len(prices):
        return np.nan, np.nan

    entry_price = prices.iloc[entry_idx]

    if pd.isna(entry_price) or entry_price <= 0:
        return np.nan, np.nan

    # Get prices from entry to max hold period
    end_idx = min(entry_idx + max_hold_days + 1, len(prices))
    future_prices = prices.iloc[entry_idx:end_idx]

    if len(future_prices) == 0:
        return np.nan, np.nan

    # Compute returns from entry
    returns = (future_prices - entry_price) / entry_price

    max_gain = returns.max()
    max_drawdown = returns.min()  # Most negative is the drawdown

    return max_gain, max_drawdown


def add_forward_returns_to_signals(
    signals_df: pd.DataFrame,
    price_df: pd.DataFrame,
    price_col: str = 'open',
    close_col: str = 'close',
    symbol_col: str = 'symbol',
    date_col: str = 'date',
    max_hold_days: int = 30,
) -> pd.DataFrame:
    """
    Add forward return columns to a signals DataFrame.

    Args:
        signals_df: DataFrame with signal events
        price_df: DataFrame with price data (must have open, close, date, symbol)
        price_col: Column to use for entry price (typically 'open')
        close_col: Column to use for exit price
        symbol_col: Symbol column name
        date_col: Date column name
        max_hold_days: Max holding period for gain/drawdown calculation

    Returns:
        signals_df with forward return columns added
    """
    signals_df = signals_df.copy()

    # Initialize return columns
    for p in FORWARD_PERIODS:
        signals_df[f'fwd_return_{p}d'] = np.nan
    signals_df['max_gain_before_exit'] = np.nan
    signals_df['max_drawdown_before_exit'] = np.nan
    signals_df['entry_price'] = np.nan
    signals_df['hit_5d'] = False
    signals_df['hit_10d'] = False

    # Process each symbol separately
    for symbol in signals_df[symbol_col].unique():
        symbol_mask = signals_df[symbol_col] == symbol
        symbol_prices = price_df[price_df[symbol_col] == symbol].copy()

        if symbol_prices.empty:
            continue

        # Ensure date sorting
        symbol_prices = symbol_prices.sort_values(date_col).reset_index(drop=True)

        # Create date to index mapping
        symbol_prices['_idx'] = range(len(symbol_prices))
        date_to_idx = dict(zip(symbol_prices[date_col], symbol_prices['_idx']))

        # Get price series
        open_prices = symbol_prices[price_col]
        close_prices = symbol_prices[close_col]

        # Process each signal for this symbol
        signal_indices = signals_df[symbol_mask].index

        for sig_idx in signal_indices:
            sig_date = signals_df.loc[sig_idx, date_col]

            # Find index in price data
            if sig_date not in date_to_idx:
                continue

            price_idx = date_to_idx[sig_date]

            # Entry is at next day's open
            entry_idx = price_idx + 1
            if entry_idx >= len(open_prices):
                continue

            entry_price = open_prices.iloc[entry_idx]
            signals_df.loc[sig_idx, 'entry_price'] = entry_price

            # Compute forward returns using close prices
            fwd_returns = compute_forward_returns(
                close_prices,
                price_idx,
                FORWARD_PERIODS,
            )

            for key, val in fwd_returns.items():
                signals_df.loc[sig_idx, key] = val

            # Compute max gain/drawdown
            max_gain, max_dd = compute_max_gain_drawdown(
                close_prices,
                price_idx,
                max_hold_days,
            )
            signals_df.loc[sig_idx, 'max_gain_before_exit'] = max_gain
            signals_df.loc[sig_idx, 'max_drawdown_before_exit'] = max_dd

            # Hit flags
            ret_5d = signals_df.loc[sig_idx, 'fwd_return_5d']
            ret_10d = signals_df.loc[sig_idx, 'fwd_return_10d']
            signals_df.loc[sig_idx, 'hit_5d'] = (pd.notna(ret_5d) and ret_5d > 0)
            signals_df.loc[sig_idx, 'hit_10d'] = (pd.notna(ret_10d) and ret_10d > 0)

    return signals_df


def compute_forward_returns_vectorized(
    df: pd.DataFrame,
    close_col: str = 'close',
    open_col: str = 'open',
) -> pd.DataFrame:
    """
    Vectorized forward return computation for a single symbol's data.

    Assumes df is sorted by date and has continuous trading days.
    Computes forward returns for ALL rows (filtering happens later).

    Args:
        df: DataFrame with OHLC data, sorted by date
        close_col: Close price column
        open_col: Open price column (for entry)

    Returns:
        DataFrame with forward return columns added
    """
    df = df.copy()

    # Entry price is next day's open
    df['entry_price'] = df[open_col].shift(-1)

    # Forward returns from entry (next-day open) to future closes
    for p in FORWARD_PERIODS:
        # Exit price is p days after entry (so p+1 days after signal)
        exit_price = df[close_col].shift(-(p + 1))
        df[f'fwd_return_{p}d'] = (exit_price - df['entry_price']) / df['entry_price']

    # Compute rolling max and min for gain/drawdown
    # This is approximate - exact requires per-signal computation
    max_hold = 20  # Use 20-day window for gain/drawdown
    df['max_future_close'] = df[close_col].shift(-1).rolling(window=max_hold, min_periods=1).max()
    df['min_future_close'] = df[close_col].shift(-1).rolling(window=max_hold, min_periods=1).min()

    # Shift to align with signal day
    df['max_future_close'] = df['max_future_close'].shift(-max_hold + 1)
    df['min_future_close'] = df['min_future_close'].shift(-max_hold + 1)

    df['max_gain_approx'] = (df['max_future_close'] - df['entry_price']) / df['entry_price']
    df['max_drawdown_approx'] = (df['min_future_close'] - df['entry_price']) / df['entry_price']

    # Hit flags
    df['hit_5d'] = df['fwd_return_5d'] > 0
    df['hit_10d'] = df['fwd_return_10d'] > 0

    return df


def load_price_data(
    conn: duckdb.DuckDBPyConnection,
    symbols: List[str],
    date_start: str,
    date_end: str,
    buffer_days: int = 30,
) -> pd.DataFrame:
    """
    Load price data for forward return computation.

    Loads extra days at the end to compute forward returns for signals
    near the end of the date range.

    Args:
        conn: DuckDB connection
        symbols: List of symbols
        date_start: Start date
        date_end: End date
        buffer_days: Extra days to load after date_end

    Returns:
        DataFrame with OHLCV data
    """
    placeholders = ", ".join(["?" for _ in symbols])

    # Add buffer days to end date for forward return computation
    query = f"""
        SELECT
            trade_date as date,
            symbol,
            open,
            high,
            low,
            close,
            volume
        FROM polygon_daily_agg_raw
        WHERE symbol IN ({placeholders})
          AND trade_date >= ?
          AND trade_date <= DATE ? + INTERVAL '{buffer_days}' DAY
        ORDER BY symbol, trade_date
    """

    params = symbols + [date_start, date_end]
    df = conn.execute(query, params).df()

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])

    return df


def get_entry_price(
    price_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    symbol: str,
    price_col: str = 'open',
) -> Optional[float]:
    """
    Get entry price for a signal (next-day open).

    Args:
        price_df: Price data DataFrame
        signal_date: Date of the signal
        symbol: Symbol
        price_col: Price column to use (default 'open')

    Returns:
        Entry price or None if not available
    """
    symbol_prices = price_df[price_df['symbol'] == symbol].copy()
    symbol_prices = symbol_prices.sort_values('date')

    # Find signal date index
    dates = symbol_prices['date'].values
    signal_idx = np.where(dates == signal_date)[0]

    if len(signal_idx) == 0:
        return None

    signal_idx = signal_idx[0]
    entry_idx = signal_idx + 1

    if entry_idx >= len(symbol_prices):
        return None

    return symbol_prices.iloc[entry_idx][price_col]


# =============================================================================
# CLI / STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from darkpool_analysis.config import load_config

    config = load_config()

    conn = duckdb.connect(str(config.db_path))
    try:
        # Test with sample data
        symbols = ["SPY", "QQQ"]
        date_start = "2025-10-01"
        date_end = "2025-12-31"

        print(f"Loading price data for {symbols}...")
        price_df = load_price_data(conn, symbols, date_start, date_end)

        if price_df.empty:
            print("No price data found.")
        else:
            print(f"Loaded {len(price_df)} price rows")

            # Test vectorized computation
            for symbol in symbols:
                sym_df = price_df[price_df['symbol'] == symbol].copy()
                sym_df = sym_df.sort_values('date').reset_index(drop=True)

                result = compute_forward_returns_vectorized(sym_df)

                print(f"\n{symbol} sample forward returns:")
                cols = ['date', 'close', 'entry_price', 'fwd_return_1d',
                        'fwd_return_5d', 'fwd_return_10d', 'hit_5d', 'hit_10d']
                print(result[cols].head(10).to_string())

    finally:
        conn.close()
