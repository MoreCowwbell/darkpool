"""
Options Context Module

Extracts options premium metrics (put/call ratio, IV percentile, etc.)
for use in signal quality assessment.

Key Hypothesis:
    Accumulation ≥ 70 + Put/Call ratio declining + IV elevated → Higher conviction
    Institution selling puts during accumulation → Reversal signal strength
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass
import duckdb
from pathlib import Path


@dataclass
class OptionsContext:
    """Options metrics for a single symbol on a single date."""
    symbol: str
    date: pd.Timestamp
    put_call_ratio: Optional[float]       # Put volume / Call volume
    put_call_ratio_5d_chg: Optional[float] # 5-day change in put/call ratio
    atm_iv: Optional[float]                # At-the-money implied volatility
    iv_percentile_30d: Optional[float]     # IV rank over 30 days (0-100)
    total_call_volume: Optional[float]
    total_put_volume: Optional[float]
    total_call_premium: Optional[float]    # In $M
    total_put_premium: Optional[float]     # In $M


# Liquid tickers that have reliable options data
# Only use options context for these (high volume = reliable data)
LIQUID_OPTIONS_TICKERS = {
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "USO",
    "XLF", "XLK", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU",
    # MAG8
    "MSFT", "AAPL", "GOOGL", "AMZN", "NVDA", "AVGO", "META", "TSLA",
    # High volume speculative
    "AMD", "JPM", "PLTR",
}


def is_liquid_options_ticker(symbol: str) -> bool:
    """Check if ticker has liquid enough options for reliable metrics."""
    return symbol.upper() in LIQUID_OPTIONS_TICKERS


def get_options_summary_data(
    conn: duckdb.DuckDBPyConnection,
    symbols: List[str],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    Fetch options summary data from the database.

    Uses the options_premium_summary table if available.

    Args:
        conn: DuckDB connection
        symbols: List of symbols to fetch
        date_start: Start date (YYYY-MM-DD)
        date_end: End date (YYYY-MM-DD)

    Returns:
        DataFrame with options metrics per symbol/date
    """
    # Filter to liquid tickers only
    liquid_symbols = [s for s in symbols if is_liquid_options_ticker(s)]

    if not liquid_symbols:
        return pd.DataFrame()

    # Check if options_premium_summary table exists
    table_check = conn.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_name = 'options_premium_summary'
    """).fetchone()[0]

    if table_check == 0:
        # Table doesn't exist, return empty
        return pd.DataFrame()

    placeholders = ", ".join(["?" for _ in liquid_symbols])

    query = f"""
        SELECT
            symbol,
            trade_date as date,
            total_call_volume,
            total_put_volume,
            total_call_premium,
            total_put_premium,
            atm_call_iv,
            atm_put_iv
        FROM options_premium_summary
        WHERE symbol IN ({placeholders})
          AND trade_date BETWEEN ? AND ?
        ORDER BY symbol, trade_date
    """

    params = liquid_symbols + [date_start, date_end]
    df = conn.execute(query, params).df()

    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_put_call_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute put/call ratio from volume data.

    Returns DataFrame with put_call_ratio column added.
    """
    df = df.copy()

    # Compute put/call ratio (handle division by zero)
    df['put_call_ratio'] = np.where(
        df['total_call_volume'] > 0,
        df['total_put_volume'] / df['total_call_volume'],
        np.nan
    )

    return df


def compute_put_call_ratio_change(
    df: pd.DataFrame,
    lookback: int = 5,
) -> pd.DataFrame:
    """
    Compute N-day change in put/call ratio.

    Args:
        df: DataFrame with put_call_ratio column
        lookback: Number of days for change calculation

    Returns DataFrame with put_call_ratio_Nd_chg column added.
    """
    df = df.copy()

    if 'put_call_ratio' not in df.columns:
        df = compute_put_call_ratio(df)

    # Compute change per symbol
    df['put_call_ratio_5d_chg'] = df.groupby('symbol')['put_call_ratio'].transform(
        lambda x: x - x.shift(lookback)
    )

    return df


def compute_iv_percentile(
    df: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute IV percentile over rolling window.

    Args:
        df: DataFrame with atm_call_iv or atm_put_iv column
        window: Rolling window for percentile calculation

    Returns DataFrame with iv_percentile_30d column added.
    """
    df = df.copy()

    # Use average of call and put IV as ATM IV
    if 'atm_iv' not in df.columns:
        df['atm_iv'] = (df.get('atm_call_iv', 0) + df.get('atm_put_iv', 0)) / 2

    # Compute percentile per symbol
    def rolling_percentile(series):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i < window - 1:
                result.iloc[i] = np.nan
            else:
                window_data = series.iloc[max(0, i - window + 1):i + 1]
                current_val = series.iloc[i]
                percentile = (window_data < current_val).sum() / len(window_data) * 100
                result.iloc[i] = percentile
        return result

    df['iv_percentile_30d'] = df.groupby('symbol')['atm_iv'].transform(rolling_percentile)

    return df


def build_options_lookup(
    conn: duckdb.DuckDBPyConnection,
    symbols: List[str],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    Build a lookup table of options metrics for each symbol/date.

    Args:
        conn: DuckDB connection
        symbols: List of symbols
        date_start: Start date
        date_end: End date

    Returns:
        DataFrame indexed by (symbol, date) with options metrics
    """
    # Get raw data
    df = get_options_summary_data(conn, symbols, date_start, date_end)

    if df.empty:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'symbol', 'date', 'put_call_ratio', 'put_call_ratio_5d_chg',
            'atm_iv', 'iv_percentile_30d', 'total_call_volume', 'total_put_volume',
            'total_call_premium', 'total_put_premium',
        ])

    # Compute derived metrics
    df = compute_put_call_ratio(df)
    df = compute_put_call_ratio_change(df, lookback=5)
    df = compute_iv_percentile(df, window=30)

    # Set multi-index
    df = df.set_index(['symbol', 'date'])

    return df


def get_options_context_for_signal(
    options_lookup: pd.DataFrame,
    symbol: str,
    date: pd.Timestamp,
) -> OptionsContext:
    """
    Get options context for a specific signal.

    Args:
        options_lookup: DataFrame from build_options_lookup
        symbol: Ticker symbol
        date: Signal date

    Returns:
        OptionsContext with metrics (or None values if not available)
    """
    # Check if ticker is liquid
    if not is_liquid_options_ticker(symbol):
        return OptionsContext(
            symbol=symbol,
            date=date,
            put_call_ratio=None,
            put_call_ratio_5d_chg=None,
            atm_iv=None,
            iv_percentile_30d=None,
            total_call_volume=None,
            total_put_volume=None,
            total_call_premium=None,
            total_put_premium=None,
        )

    # Try to find in lookup
    try:
        if (symbol, date) in options_lookup.index:
            row = options_lookup.loc[(symbol, date)]
            return OptionsContext(
                symbol=symbol,
                date=date,
                put_call_ratio=row.get('put_call_ratio'),
                put_call_ratio_5d_chg=row.get('put_call_ratio_5d_chg'),
                atm_iv=row.get('atm_iv'),
                iv_percentile_30d=row.get('iv_percentile_30d'),
                total_call_volume=row.get('total_call_volume'),
                total_put_volume=row.get('total_put_volume'),
                total_call_premium=row.get('total_call_premium'),
                total_put_premium=row.get('total_put_premium'),
            )
    except KeyError:
        pass

    # Not found
    return OptionsContext(
        symbol=symbol,
        date=date,
        put_call_ratio=None,
        put_call_ratio_5d_chg=None,
        atm_iv=None,
        iv_percentile_30d=None,
        total_call_volume=None,
        total_put_volume=None,
        total_call_premium=None,
        total_put_premium=None,
    )


def assess_options_signal_quality(context: OptionsContext) -> Dict:
    """
    Assess signal quality based on options context.

    Key patterns:
        - Low put/call ratio (< 0.7) = bullish sentiment
        - Declining put/call ratio = improving sentiment
        - High IV percentile (> 70) + accumulation = potential squeeze
        - Institution selling puts during accumulation = reversal strength

    Returns:
        Dictionary with quality flags and scores
    """
    result = {
        'has_options_data': False,
        'bullish_sentiment': False,
        'improving_sentiment': False,
        'high_iv_environment': False,
        'options_quality_boost': 0.0,
    }

    # Check if we have data
    if context.put_call_ratio is None:
        return result

    result['has_options_data'] = True

    # Bullish sentiment: low put/call ratio
    if context.put_call_ratio < 0.7:
        result['bullish_sentiment'] = True
        result['options_quality_boost'] += 0.1

    # Improving sentiment: declining put/call ratio
    if context.put_call_ratio_5d_chg is not None and context.put_call_ratio_5d_chg < -0.1:
        result['improving_sentiment'] = True
        result['options_quality_boost'] += 0.1

    # High IV environment: IV percentile > 70
    if context.iv_percentile_30d is not None and context.iv_percentile_30d > 70:
        result['high_iv_environment'] = True
        result['options_quality_boost'] += 0.05

    return result


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
        # Test with SPY
        symbols = ["SPY", "QQQ", "AAPL", "NVDA"]
        date_start = "2025-01-01"
        date_end = "2025-12-31"

        print(f"Building options lookup for {symbols}...")
        options_lookup = build_options_lookup(conn, symbols, date_start, date_end)

        if not options_lookup.empty:
            print(f"Built options lookup with {len(options_lookup)} rows")
            print(f"\nSample data:")
            print(options_lookup.head(10))

            # Get context for a specific signal
            test_date = options_lookup.index.get_level_values('date')[0]
            test_symbol = options_lookup.index.get_level_values('symbol')[0]

            context = get_options_context_for_signal(options_lookup, test_symbol, test_date)
            print(f"\nOptions context for {test_symbol} on {test_date}:")
            print(f"  Put/Call Ratio: {context.put_call_ratio}")
            print(f"  5d Change: {context.put_call_ratio_5d_chg}")
            print(f"  IV Percentile: {context.iv_percentile_30d}")

            # Assess quality
            quality = assess_options_signal_quality(context)
            print(f"\nSignal Quality Assessment:")
            for k, v in quality.items():
                print(f"  {k}: {v}")
        else:
            print("No options data found. Make sure options_premium_summary table exists and has data.")

    finally:
        conn.close()
