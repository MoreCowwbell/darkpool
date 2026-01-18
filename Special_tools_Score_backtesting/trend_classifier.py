"""
Trend Classification Module

Classifies price trends as UP, DOWN, or SIDEWAYS based on SMA and slope analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TrendResult:
    """Result of trend classification for a single row."""
    trend_state: str          # 'UP', 'DOWN', 'SIDEWAYS'
    sma_20: float             # 20-day SMA value
    sma_20_slope: float       # Normalized slope of SMA
    price_vs_sma_pct: float   # Price relative to SMA (%)
    rsi_14: Optional[float]   # RSI value if computed


def compute_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Compute Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def compute_sma_slope(sma: pd.Series, lookback: int = 1) -> pd.Series:
    """
    Compute normalized slope of SMA.

    Returns (sma - sma_prev) / sma_prev as a percentage change.
    """
    sma_prev = sma.shift(lookback)
    slope = (sma - sma_prev) / sma_prev
    return slope


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Use exponential moving average for smoothing
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def classify_trend(
    close: float,
    sma_20: float,
    sma_20_prev: float,
    slope_threshold: float = 0.001,
) -> str:
    """
    Classify trend state based on price position relative to SMA and SMA slope.

    Args:
        close: Current closing price
        sma_20: Current 20-day SMA
        sma_20_prev: Previous day's 20-day SMA
        slope_threshold: Minimum slope magnitude to determine direction

    Returns:
        'UP': Close > SMA AND slope > threshold
        'DOWN': Close < SMA AND slope < -threshold
        'SIDEWAYS': Neither condition met

    Signal Interpretation:
        - UP + Accumulation = Continuation signal (lower risk, potentially lower return)
        - DOWN + Accumulation = Reversal signal (higher risk, potentially higher return)
        - SIDEWAYS + Accumulation = Breakout setup
    """
    if pd.isna(close) or pd.isna(sma_20) or pd.isna(sma_20_prev):
        return 'SIDEWAYS'

    # Compute normalized slope
    slope = (sma_20 - sma_20_prev) / sma_20_prev if sma_20_prev != 0 else 0

    # Classification logic
    if close > sma_20 and slope > slope_threshold:
        return 'UP'
    elif close < sma_20 and slope < -slope_threshold:
        return 'DOWN'
    else:
        return 'SIDEWAYS'


def add_trend_columns(
    df: pd.DataFrame,
    close_col: str = 'close',
    sma_period: int = 20,
    rsi_period: int = 14,
    slope_threshold: float = 0.001,
) -> pd.DataFrame:
    """
    Add trend classification columns to a DataFrame.

    Adds columns:
        - sma_20: 20-day simple moving average
        - sma_20_slope: Normalized slope of SMA
        - price_vs_sma_20_pct: Price relative to SMA (%)
        - rsi_14: 14-day RSI
        - trend_state: 'UP', 'DOWN', or 'SIDEWAYS'

    Args:
        df: DataFrame with at least a close price column
        close_col: Name of the close price column
        sma_period: Period for SMA calculation
        rsi_period: Period for RSI calculation
        slope_threshold: Threshold for trend direction classification

    Returns:
        DataFrame with trend columns added
    """
    df = df.copy()

    # Ensure close column exists
    if close_col not in df.columns:
        raise ValueError(f"Close column '{close_col}' not found in DataFrame")

    close = df[close_col]

    # Compute SMA
    df['sma_20'] = compute_sma(close, period=sma_period)

    # Compute SMA slope
    df['sma_20_slope'] = compute_sma_slope(df['sma_20'], lookback=1)

    # Compute price vs SMA percentage
    df['price_vs_sma_20_pct'] = (close - df['sma_20']) / df['sma_20'] * 100

    # Compute RSI
    df['rsi_14'] = compute_rsi(close, period=rsi_period)

    # Classify trend for each row
    sma_prev = df['sma_20'].shift(1)
    df['trend_state'] = df.apply(
        lambda row: classify_trend(
            close=row[close_col],
            sma_20=row['sma_20'],
            sma_20_prev=sma_prev.loc[row.name] if row.name in sma_prev.index else np.nan,
            slope_threshold=slope_threshold,
        ),
        axis=1,
    )

    return df


def get_trend_for_date(
    df: pd.DataFrame,
    date: pd.Timestamp,
    close_col: str = 'close',
) -> TrendResult:
    """
    Get trend classification for a specific date.

    Args:
        df: DataFrame with trend columns (or will be computed)
        date: Date to get trend for
        close_col: Name of close price column

    Returns:
        TrendResult with trend state and supporting metrics
    """
    # Add trend columns if not present
    if 'trend_state' not in df.columns:
        df = add_trend_columns(df, close_col=close_col)

    # Get row for date
    if date not in df.index:
        # Try to find by date column
        if 'date' in df.columns:
            row = df[df['date'] == date]
            if row.empty:
                return TrendResult(
                    trend_state='SIDEWAYS',
                    sma_20=np.nan,
                    sma_20_slope=np.nan,
                    price_vs_sma_pct=np.nan,
                    rsi_14=np.nan,
                )
            row = row.iloc[0]
        else:
            return TrendResult(
                trend_state='SIDEWAYS',
                sma_20=np.nan,
                sma_20_slope=np.nan,
                price_vs_sma_pct=np.nan,
                rsi_14=np.nan,
            )
    else:
        row = df.loc[date]

    return TrendResult(
        trend_state=row.get('trend_state', 'SIDEWAYS'),
        sma_20=row.get('sma_20', np.nan),
        sma_20_slope=row.get('sma_20_slope', np.nan),
        price_vs_sma_pct=row.get('price_vs_sma_20_pct', np.nan),
        rsi_14=row.get('rsi_14', np.nan),
    )


def compute_trend_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for trend states in a DataFrame.

    Returns:
        Dictionary with trend distribution and related metrics
    """
    if 'trend_state' not in df.columns:
        df = add_trend_columns(df)

    trend_counts = df['trend_state'].value_counts()
    total = len(df)

    return {
        'total_days': total,
        'up_days': trend_counts.get('UP', 0),
        'down_days': trend_counts.get('DOWN', 0),
        'sideways_days': trend_counts.get('SIDEWAYS', 0),
        'up_pct': trend_counts.get('UP', 0) / total * 100 if total > 0 else 0,
        'down_pct': trend_counts.get('DOWN', 0) / total * 100 if total > 0 else 0,
        'sideways_pct': trend_counts.get('SIDEWAYS', 0) / total * 100 if total > 0 else 0,
        'avg_rsi': df['rsi_14'].mean() if 'rsi_14' in df.columns else np.nan,
    }


# =============================================================================
# CLI / STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example usage
    import duckdb
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from darkpool_analysis.config import load_config

    config = load_config()

    # Get sample price data
    conn = duckdb.connect(str(config.db_path))
    try:
        df = conn.execute("""
            SELECT trade_date as date, symbol, open, high, low, close, volume
            FROM polygon_daily_agg_raw
            WHERE symbol = 'SPY'
            ORDER BY trade_date
            LIMIT 100
        """).df()

        if not df.empty:
            print(f"Loaded {len(df)} rows for SPY")

            # Add trend columns
            df = add_trend_columns(df, close_col='close')

            # Show last 10 rows
            print("\nLast 10 rows with trend classification:")
            print(df[['date', 'close', 'sma_20', 'sma_20_slope', 'rsi_14', 'trend_state']].tail(10))

            # Compute stats
            stats = compute_trend_stats(df)
            print(f"\nTrend Stats: {stats}")
        else:
            print("No data found for SPY")

    finally:
        conn.close()
