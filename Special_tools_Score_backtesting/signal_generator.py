"""
Signal Generator Module

Generates accumulation/distribution signals with configurable parameters.
This is the core module that recomputes scores with arbitrary configurations
for backtesting different signal variants.

Score Variants:
    A: finra_buy_volume_z (pure institutional volume)
    B: short_buy_sell_ratio_z (directional bias - baseline)
    C: 0.5*A + 0.5*B (blended)
    D: zscore(imbalance * log1p(volume)) (conviction-weighted)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import duckdb
from pathlib import Path


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    score_variant: str = "B"           # 'A', 'B', 'C', 'D'
    buy_threshold: int = 70            # Score >= this = accumulation signal
    sell_threshold: int = 30           # Score <= this = distribution signal
    w_short: float = 0.55              # Weight for short component
    w_lit: float = 0.30                # Weight for lit component
    w_price: float = 0.15              # Weight for price component
    z_window: int = 20                 # Rolling window for z-scores
    intensity_scale_min: float = 0.7   # OTC intensity scaling min
    intensity_scale_max: float = 1.3   # OTC intensity scaling max


@dataclass
class SignalResult:
    """Result for a single accumulation signal."""
    date: pd.Timestamp
    symbol: str
    score_variant: str
    score_value: float                 # 0-100 display score
    score_raw: float                   # -1 to +1 raw score
    signal_type: str                   # 'ACCUMULATION', 'DISTRIBUTION', 'NEUTRAL'
    short_z: float
    lit_z: float
    price_z: float
    otc_z: float
    confidence: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute rolling z-score with safe division."""
    if min_periods is None:
        min_periods = max(window // 2, 5)

    rolling = series.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std(ddof=0)

    # Avoid division by zero
    std = std.replace(0, np.nan)

    return (series - mean) / std


def compute_imbalance_signal(
    finra_buy_volume: pd.Series,
    short_sell_volume: pd.Series,
) -> pd.Series:
    """
    Compute imbalance-weighted signal for Score D.

    Formula: imbalance = (B - S) / (B + S)
             signal = imbalance × log1p(B + S)

    This captures both direction AND magnitude of conviction.
    """
    B = finra_buy_volume.fillna(0)
    S = short_sell_volume.fillna(0)
    total = B + S

    # Compute imbalance (-1 to +1)
    imbalance = np.where(total > 0, (B - S) / total, 0)

    # Scale by log of volume
    signal = imbalance * np.log1p(total)

    return pd.Series(signal, index=finra_buy_volume.index)


def compute_score(
    short_z: np.ndarray,
    lit_z: np.ndarray,
    price_z: np.ndarray,
    otc_z: np.ndarray,
    config: SignalConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute accumulation score using the composite formula.

    Formula:
        raw_score = w_short * tanh(short_z * 0.5)
                  + w_lit * tanh(lit_z * 0.5)
                  + w_price * tanh(price_z * 0.3)

        intensity = scale_min + (scale_max - scale_min) * sigmoid(otc_z)

        final_score = clip(raw_score * intensity, -1, 1)
        display_score = (final_score + 1) * 50  # Maps to 0-100

    Returns:
        Tuple of (raw_scores, display_scores)
    """
    # Replace NaN with 0 for calculation
    short_z = np.nan_to_num(short_z, nan=0.0)
    lit_z = np.nan_to_num(lit_z, nan=0.0)
    price_z = np.nan_to_num(price_z, nan=0.0)
    otc_z = np.nan_to_num(otc_z, nan=0.0)

    # Weighted combination with tanh compression
    raw_score = (
        config.w_short * np.tanh(short_z * 0.5)
        + config.w_lit * np.tanh(lit_z * 0.5)
        + config.w_price * np.tanh(price_z * 0.3)
    )

    # Intensity modulation via OTC participation
    intensity_range = config.intensity_scale_max - config.intensity_scale_min
    intensity = config.intensity_scale_min + intensity_range * sigmoid(otc_z)

    # Final score
    final_score = np.clip(raw_score * intensity, -1.0, 1.0)

    # Convert to display scale (0-100)
    display_score = (final_score + 1.0) * 50.0

    return final_score, display_score


def get_short_z_for_variant(
    df: pd.DataFrame,
    variant: str,
    z_window: int,
) -> pd.Series:
    """
    Get the appropriate short_z column based on score variant.

    Variants:
        A: finra_buy_volume_z (pure volume)
        B: short_buy_sell_ratio_z (ratio - baseline)
        C: 0.5 * A + 0.5 * B (blended)
        D: zscore(imbalance_signal) (conviction-weighted)
    """
    min_periods = max(z_window // 2, 5)

    if variant == "A":
        # Pure institutional buy volume
        if 'finra_buy_volume_z' in df.columns and df['finra_buy_volume_z'].notna().any():
            return df['finra_buy_volume_z']
        elif 'finra_buy_volume' in df.columns:
            return rolling_zscore(df['finra_buy_volume'], z_window, min_periods)
        else:
            return pd.Series(np.nan, index=df.index)

    elif variant == "B":
        # Short buy/sell ratio (baseline)
        if 'short_buy_sell_ratio_z' in df.columns and df['short_buy_sell_ratio_z'].notna().any():
            return df['short_buy_sell_ratio_z']
        elif 'short_buy_sell_ratio' in df.columns:
            return rolling_zscore(df['short_buy_sell_ratio'], z_window, min_periods)
        else:
            return pd.Series(np.nan, index=df.index)

    elif variant == "C":
        # Blended: 50% A + 50% B
        z_a = get_short_z_for_variant(df, "A", z_window)
        z_b = get_short_z_for_variant(df, "B", z_window)
        return 0.5 * z_a.fillna(0) + 0.5 * z_b.fillna(0)

    elif variant == "D":
        # Imbalance-weighted
        if 'finra_buy_volume' in df.columns and 'short_sell_volume' in df.columns:
            imbalance_signal = compute_imbalance_signal(
                df['finra_buy_volume'],
                df['short_sell_volume'],
            )
            return rolling_zscore(imbalance_signal, z_window, min_periods)
        else:
            return pd.Series(np.nan, index=df.index)

    else:
        raise ValueError(f"Unknown score variant: {variant}")


def generate_signals(
    df: pd.DataFrame,
    config: SignalConfig,
) -> pd.DataFrame:
    """
    Generate accumulation/distribution signals from daily_metrics data.

    Args:
        df: DataFrame with daily_metrics columns
        config: SignalConfig with parameters

    Returns:
        DataFrame with signal columns added:
            - score_raw: Raw score (-1 to +1)
            - score_value: Display score (0-100)
            - signal_type: 'ACCUMULATION', 'DISTRIBUTION', or 'NEUTRAL'
            - short_z, lit_z, price_z, otc_z: Component z-scores
    """
    df = df.copy()

    # Get short_z based on variant
    df['short_z'] = get_short_z_for_variant(df, config.score_variant, config.z_window)

    # Get lit_z (lit flow imbalance z-score)
    if 'lit_flow_imbalance_z' in df.columns:
        df['lit_z'] = df['lit_flow_imbalance_z']
    elif 'lit_flow_imbalance' in df.columns:
        df['lit_z'] = rolling_zscore(df['lit_flow_imbalance'], config.z_window)
    else:
        df['lit_z'] = 0.0

    # Get price_z (return z-score)
    if 'return_z' in df.columns:
        df['price_z'] = df['return_z']
    elif 'return_1d' in df.columns:
        df['price_z'] = rolling_zscore(df['return_1d'], config.z_window)
    else:
        df['price_z'] = 0.0

    # Get otc_z (OTC participation z-score)
    if 'otc_participation_z' in df.columns:
        df['otc_z'] = df['otc_participation_z']
    else:
        df['otc_z'] = 0.0

    # Compute scores
    raw_scores, display_scores = compute_score(
        short_z=df['short_z'].to_numpy(),
        lit_z=df['lit_z'].to_numpy(),
        price_z=df['price_z'].to_numpy(),
        otc_z=df['otc_z'].to_numpy(),
        config=config,
    )

    df['score_raw'] = raw_scores
    df['score_value'] = display_scores

    # Classify signal type
    df['signal_type'] = 'NEUTRAL'
    df.loc[df['score_value'] >= config.buy_threshold, 'signal_type'] = 'ACCUMULATION'
    df.loc[df['score_value'] <= config.sell_threshold, 'signal_type'] = 'DISTRIBUTION'

    # Add config info
    df['score_variant'] = config.score_variant
    df['buy_threshold'] = config.buy_threshold
    df['sell_threshold'] = config.sell_threshold

    return df


def get_accumulation_signals(
    df: pd.DataFrame,
    config: SignalConfig,
) -> pd.DataFrame:
    """
    Get only accumulation signals (score >= buy_threshold).

    Args:
        df: DataFrame with daily_metrics
        config: SignalConfig

    Returns:
        DataFrame filtered to accumulation signals only
    """
    signals_df = generate_signals(df, config)
    return signals_df[signals_df['signal_type'] == 'ACCUMULATION'].copy()


def get_distribution_signals(
    df: pd.DataFrame,
    config: SignalConfig,
) -> pd.DataFrame:
    """
    Get only distribution signals (score <= sell_threshold).

    Args:
        df: DataFrame with daily_metrics
        config: SignalConfig

    Returns:
        DataFrame filtered to distribution signals only
    """
    signals_df = generate_signals(df, config)
    return signals_df[signals_df['signal_type'] == 'DISTRIBUTION'].copy()


def load_daily_metrics(
    conn: duckdb.DuckDBPyConnection,
    symbols: List[str],
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    Load daily_metrics data for signal generation.

    Args:
        conn: DuckDB connection
        symbols: List of symbols
        date_start: Start date
        date_end: End date

    Returns:
        DataFrame with daily_metrics data
    """
    placeholders = ", ".join(["?" for _ in symbols])

    query = f"""
        SELECT *
        FROM daily_metrics
        WHERE symbol IN ({placeholders})
          AND date BETWEEN ? AND ?
        ORDER BY symbol, date
    """

    params = symbols + [date_start, date_end]
    df = conn.execute(query, params).df()

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])

    return df


def generate_all_variant_signals(
    df: pd.DataFrame,
    buy_threshold: int = 70,
    sell_threshold: int = 30,
    z_window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    Generate signals for all score variants (A, B, C, D).

    Args:
        df: DataFrame with daily_metrics
        buy_threshold: Accumulation threshold
        sell_threshold: Distribution threshold
        z_window: Z-score rolling window

    Returns:
        Dictionary mapping variant name to signals DataFrame
    """
    results = {}

    for variant in ["A", "B", "C", "D"]:
        config = SignalConfig(
            score_variant=variant,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            z_window=z_window,
        )
        results[variant] = generate_signals(df, config)

    return results


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
        # Test with a few symbols
        symbols = ["SPY", "QQQ", "XLE"]
        date_start = "2025-10-01"
        date_end = "2025-12-31"

        print(f"Loading daily_metrics for {symbols}...")
        df = load_daily_metrics(conn, symbols, date_start, date_end)

        if df.empty:
            print("No data found. Make sure daily_metrics table is populated.")
        else:
            print(f"Loaded {len(df)} rows")

            # Generate signals for each variant
            for variant in ["A", "B", "C", "D"]:
                signal_config = SignalConfig(
                    score_variant=variant,
                    buy_threshold=70,
                    sell_threshold=30,
                )

                signals_df = generate_signals(df, signal_config)

                accum_count = (signals_df['signal_type'] == 'ACCUMULATION').sum()
                dist_count = (signals_df['signal_type'] == 'DISTRIBUTION').sum()

                print(f"\nVariant {variant}:")
                print(f"  Accumulation signals: {accum_count}")
                print(f"  Distribution signals: {dist_count}")

                # Show sample accumulation signals
                accum = signals_df[signals_df['signal_type'] == 'ACCUMULATION'][
                    ['date', 'symbol', 'score_value', 'short_z', 'lit_z']
                ].head(3)
                if not accum.empty:
                    print(f"  Sample signals:\n{accum.to_string()}")

    finally:
        conn.close()
