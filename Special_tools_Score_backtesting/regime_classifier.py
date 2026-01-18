"""
Market Regime Classification Module

Classifies market regime as RISK_ON, RISK_OFF, or TRANSITIONAL based on
SPY position relative to its 50-day SMA and VIX level.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import duckdb
from pathlib import Path


@dataclass
class RegimeResult:
    """Result of regime classification for a single date."""
    regime: str              # 'RISK_ON', 'RISK_OFF', 'TRANSITIONAL'
    spy_close: float         # SPY closing price
    spy_sma_50: float        # SPY 50-day SMA
    spy_vs_sma_50_pct: float # SPY position relative to SMA (%)
    vix_close: float         # VIX closing value


# Regime definitions:
# RISK_ON: SPY > 50d SMA AND VIX < 20
# RISK_OFF: SPY < 50d SMA AND VIX > 25
# TRANSITIONAL: Everything else (mixed signals)

DEFAULT_VIX_RISK_ON_MAX = 20.0
DEFAULT_VIX_RISK_OFF_MIN = 25.0
DEFAULT_SPY_SMA_PERIOD = 50


def classify_regime(
    spy_close: float,
    spy_sma_50: float,
    vix_close: float,
    vix_risk_on_max: float = DEFAULT_VIX_RISK_ON_MAX,
    vix_risk_off_min: float = DEFAULT_VIX_RISK_OFF_MIN,
) -> str:
    """
    Classify market regime based on SPY and VIX.

    Args:
        spy_close: SPY closing price
        spy_sma_50: SPY 50-day SMA
        vix_close: VIX closing value
        vix_risk_on_max: Max VIX for RISK_ON (default 20)
        vix_risk_off_min: Min VIX for RISK_OFF (default 25)

    Returns:
        'RISK_ON': Bullish regime - SPY above SMA, low volatility
        'RISK_OFF': Bearish regime - SPY below SMA, high volatility
        'TRANSITIONAL': Mixed signals - neither fully risk-on nor risk-off

    Why Segment by Regime:
        - Optimal accumulation score config might be *different* per regime
        - RISK_ON: Trend continuation signals work better
        - RISK_OFF: Reversal signals may work better (or avoid trading entirely)
        - Testing across regimes blindly produces "compromise" config mediocre in both
    """
    if pd.isna(spy_close) or pd.isna(spy_sma_50) or pd.isna(vix_close):
        return 'TRANSITIONAL'

    spy_above_sma = spy_close > spy_sma_50
    vix_low = vix_close < vix_risk_on_max
    vix_high = vix_close > vix_risk_off_min

    # RISK_ON: SPY above SMA AND VIX low
    if spy_above_sma and vix_low:
        return 'RISK_ON'

    # RISK_OFF: SPY below SMA AND VIX high
    if not spy_above_sma and vix_high:
        return 'RISK_OFF'

    # Everything else is transitional
    return 'TRANSITIONAL'


def get_spy_vix_data(
    conn: duckdb.DuckDBPyConnection,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    Fetch SPY and VIX data from the database.

    Args:
        conn: DuckDB connection
        date_start: Start date (YYYY-MM-DD)
        date_end: End date (YYYY-MM-DD)

    Returns:
        DataFrame with date, spy_close, vix_close columns
    """
    # Get SPY data
    spy_query = """
        SELECT trade_date as date, close as spy_close
        FROM polygon_daily_agg_raw
        WHERE symbol = 'SPY'
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """
    spy_df = conn.execute(spy_query, [date_start, date_end]).df()

    # Get VIX data (try different possible symbols)
    vix_query = """
        SELECT trade_date as date, close as vix_close
        FROM polygon_daily_agg_raw
        WHERE symbol IN ('VIX', 'VIXY', '$VIX')
          AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """
    vix_df = conn.execute(vix_query, [date_start, date_end]).df()

    if spy_df.empty:
        raise ValueError("No SPY data found in database")

    # Merge on date
    if not vix_df.empty:
        df = spy_df.merge(vix_df, on='date', how='left')
    else:
        df = spy_df.copy()
        df['vix_close'] = np.nan  # VIX data not available

    df['date'] = pd.to_datetime(df['date'])
    return df


def add_regime_columns(
    df: pd.DataFrame,
    spy_close_col: str = 'spy_close',
    vix_close_col: str = 'vix_close',
    sma_period: int = DEFAULT_SPY_SMA_PERIOD,
    vix_risk_on_max: float = DEFAULT_VIX_RISK_ON_MAX,
    vix_risk_off_min: float = DEFAULT_VIX_RISK_OFF_MIN,
) -> pd.DataFrame:
    """
    Add regime classification columns to a DataFrame.

    Adds columns:
        - spy_sma_50: 50-day SMA of SPY
        - spy_vs_sma_50_pct: SPY position relative to SMA (%)
        - regime: 'RISK_ON', 'RISK_OFF', or 'TRANSITIONAL'

    Args:
        df: DataFrame with SPY and VIX close columns
        spy_close_col: Name of SPY close column
        vix_close_col: Name of VIX close column
        sma_period: Period for SPY SMA calculation
        vix_risk_on_max: Max VIX for RISK_ON
        vix_risk_off_min: Min VIX for RISK_OFF

    Returns:
        DataFrame with regime columns added
    """
    df = df.copy()

    # Ensure required columns exist
    if spy_close_col not in df.columns:
        raise ValueError(f"SPY close column '{spy_close_col}' not found")

    spy_close = df[spy_close_col]

    # Compute SPY SMA
    df['spy_sma_50'] = spy_close.rolling(window=sma_period, min_periods=sma_period).mean()

    # Compute SPY vs SMA percentage
    df['spy_vs_sma_50_pct'] = (spy_close - df['spy_sma_50']) / df['spy_sma_50'] * 100

    # Get VIX values (may be NaN if not available)
    vix_close = df.get(vix_close_col, pd.Series([np.nan] * len(df)))

    # Classify regime for each row
    df['regime'] = df.apply(
        lambda row: classify_regime(
            spy_close=row[spy_close_col],
            spy_sma_50=row['spy_sma_50'],
            vix_close=row.get(vix_close_col, np.nan) if vix_close_col in df.columns else np.nan,
            vix_risk_on_max=vix_risk_on_max,
            vix_risk_off_min=vix_risk_off_min,
        ),
        axis=1,
    )

    return df


def build_regime_lookup(
    conn: duckdb.DuckDBPyConnection,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    Build a lookup table of market regime for each trading day.

    Args:
        conn: DuckDB connection
        date_start: Start date
        date_end: End date

    Returns:
        DataFrame with date and regime columns, indexed by date
    """
    # Get SPY/VIX data
    df = get_spy_vix_data(conn, date_start, date_end)

    # Add regime columns
    df = add_regime_columns(df)

    # Return just the columns we need for lookup
    result = df[['date', 'spy_close', 'spy_sma_50', 'spy_vs_sma_50_pct', 'vix_close', 'regime']].copy()

    # Handle missing VIX
    if 'vix_close' not in df.columns:
        result['vix_close'] = np.nan

    result = result.set_index('date')
    return result


def get_regime_for_date(
    regime_lookup: pd.DataFrame,
    date: pd.Timestamp,
) -> RegimeResult:
    """
    Get regime classification for a specific date.

    Args:
        regime_lookup: DataFrame from build_regime_lookup
        date: Date to get regime for

    Returns:
        RegimeResult with regime and supporting metrics
    """
    if date not in regime_lookup.index:
        return RegimeResult(
            regime='TRANSITIONAL',
            spy_close=np.nan,
            spy_sma_50=np.nan,
            spy_vs_sma_50_pct=np.nan,
            vix_close=np.nan,
        )

    row = regime_lookup.loc[date]

    return RegimeResult(
        regime=row.get('regime', 'TRANSITIONAL'),
        spy_close=row.get('spy_close', np.nan),
        spy_sma_50=row.get('spy_sma_50', np.nan),
        spy_vs_sma_50_pct=row.get('spy_vs_sma_50_pct', np.nan),
        vix_close=row.get('vix_close', np.nan),
    )


def compute_regime_stats(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for regime distribution.

    Returns:
        Dictionary with regime distribution and related metrics
    """
    if 'regime' not in df.columns:
        df = add_regime_columns(df)

    regime_counts = df['regime'].value_counts()
    total = len(df)

    return {
        'total_days': total,
        'risk_on_days': regime_counts.get('RISK_ON', 0),
        'risk_off_days': regime_counts.get('RISK_OFF', 0),
        'transitional_days': regime_counts.get('TRANSITIONAL', 0),
        'risk_on_pct': regime_counts.get('RISK_ON', 0) / total * 100 if total > 0 else 0,
        'risk_off_pct': regime_counts.get('RISK_OFF', 0) / total * 100 if total > 0 else 0,
        'transitional_pct': regime_counts.get('TRANSITIONAL', 0) / total * 100 if total > 0 else 0,
        'avg_vix': df['vix_close'].mean() if 'vix_close' in df.columns else np.nan,
        'avg_spy_vs_sma': df['spy_vs_sma_50_pct'].mean() if 'spy_vs_sma_50_pct' in df.columns else np.nan,
    }


def get_regime_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify continuous periods of each regime.

    Returns:
        DataFrame with regime, start_date, end_date, duration_days columns
    """
    if 'regime' not in df.columns:
        df = add_regime_columns(df)

    # Ensure date is a column
    if 'date' not in df.columns:
        df = df.reset_index()

    # Identify regime changes
    df = df.sort_values('date')
    df['regime_change'] = df['regime'] != df['regime'].shift(1)
    df['regime_group'] = df['regime_change'].cumsum()

    # Group by regime periods
    periods = df.groupby('regime_group').agg({
        'regime': 'first',
        'date': ['min', 'max', 'count'],
    }).reset_index(drop=True)

    periods.columns = ['regime', 'start_date', 'end_date', 'duration_days']

    return periods


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
        # Build regime lookup for available date range
        date_start = "2025-01-01"
        date_end = "2025-12-31"

        try:
            regime_lookup = build_regime_lookup(conn, date_start, date_end)

            print(f"Built regime lookup for {len(regime_lookup)} trading days")
            print(f"\nDate range: {regime_lookup.index.min()} to {regime_lookup.index.max()}")

            # Compute stats
            stats = compute_regime_stats(regime_lookup.reset_index())
            print(f"\nRegime Stats:")
            for k, v in stats.items():
                print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

            # Show last 10 days
            print("\nLast 10 days:")
            print(regime_lookup.tail(10))

            # Show regime periods
            periods = get_regime_periods(regime_lookup.reset_index())
            print(f"\nRegime Periods ({len(periods)} total):")
            print(periods.tail(10))

        except ValueError as e:
            print(f"Error: {e}")
            print("Make sure SPY data is loaded in the database")

    finally:
        conn.close()
