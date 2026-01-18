"""
Performance Metrics Module

Computes backtesting performance metrics:
    - Hit rate (win rate)
    - Average/Median return
    - Win/Loss ratio
    - Sharpe ratio
    - Max drawdown
    - Signal efficiency
    - Composite ranking score
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BacktestMetrics:
    """Complete set of backtest performance metrics."""

    # Core metrics
    total_signals: int = 0
    total_trades: int = 0              # Signals that resulted in trades
    profitable_trades: int = 0
    losing_trades: int = 0

    # Return metrics
    hit_rate: float = 0.0              # Winning trades / total trades
    avg_return: float = 0.0            # Mean return per trade
    median_return: float = 0.0         # Median return per trade
    total_return: float = 0.0          # Sum of all returns
    std_return: float = 0.0            # Standard deviation of returns

    # Win/Loss metrics
    avg_win: float = 0.0               # Average winning return
    avg_loss: float = 0.0              # Average losing return (absolute)
    win_loss_ratio: float = 0.0        # avg_win / avg_loss
    profit_factor: float = 0.0         # Sum of wins / sum of losses

    # Risk metrics
    sharpe_ratio: float = 0.0          # (avg_return - rf) / std_return
    sortino_ratio: float = 0.0         # (avg_return - rf) / downside_std
    max_drawdown: float = 0.0          # Maximum peak-to-trough decline
    calmar_ratio: float = 0.0          # CAGR / max_drawdown

    # Holding period metrics
    avg_hold_days: float = 0.0
    median_hold_days: float = 0.0
    max_hold_days: int = 0
    min_hold_days: int = 0

    # Signal frequency
    signals_per_month: float = 0.0
    trading_days: int = 0

    # Quality metrics
    false_positive_rate: float = 0.0   # Signals losing > 5% within 3 days
    high_quality_hit_rate: float = 0.0 # Hit rate for HIGH quality signals only

    # Composite score
    rank_score: float = 0.0            # Composite ranking metric


# Annualization factor (trading days per year)
TRADING_DAYS_PER_YEAR = 252


def compute_hit_rate(returns: pd.Series) -> float:
    """Compute win rate (% of profitable trades)."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def compute_avg_return(returns: pd.Series) -> float:
    """Compute mean return."""
    if len(returns) == 0:
        return 0.0
    return returns.mean()


def compute_win_loss_ratio(returns: pd.Series) -> float:
    """Compute average win / average loss ratio."""
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    avg_win = wins.mean()
    avg_loss = abs(losses.mean())

    if avg_loss == 0:
        return float('inf') if avg_win > 0 else 0.0

    return avg_win / avg_loss


def compute_profit_factor(returns: pd.Series) -> float:
    """Compute sum of wins / sum of losses."""
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float('inf') if wins > 0 else 0.0

    return wins / losses


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Series of trade returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Convert annual rf to per-period
    rf_per_period = risk_free_rate / periods_per_year

    excess_return = returns.mean() - rf_per_period
    annualized_excess = excess_return * periods_per_year
    annualized_std = returns.std() * np.sqrt(periods_per_year)

    if annualized_std == 0:
        return 0.0

    return annualized_excess / annualized_std


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
) -> float:
    """
    Compute Sortino ratio (penalizes only downside volatility).

    Args:
        returns: Series of trade returns
        risk_free_rate: Annual risk-free rate
        target_return: Minimum acceptable return

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return float('inf') if returns.mean() > risk_free_rate else 0.0

    downside_std = downside_returns.std()
    if downside_std == 0:
        return float('inf') if returns.mean() > risk_free_rate else 0.0

    return (returns.mean() - risk_free_rate / 252) / downside_std


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute maximum drawdown from an equity curve.

    Args:
        equity_curve: Cumulative equity series

    Returns:
        Maximum drawdown as a positive percentage
    """
    if len(equity_curve) == 0:
        return 0.0

    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak

    return abs(drawdown.min())


def compute_equity_curve(
    returns: pd.Series,
    initial_capital: float = 100.0,
) -> pd.Series:
    """
    Compute equity curve from trade returns.

    Assumes sequential trades with no overlapping positions.

    Args:
        returns: Series of trade returns
        initial_capital: Starting capital

    Returns:
        Equity curve series
    """
    if len(returns) == 0:
        return pd.Series([initial_capital])

    # Compound returns
    cumulative = (1 + returns).cumprod()
    equity = initial_capital * cumulative

    return equity


def compute_false_positive_rate(
    returns: pd.Series,
    early_returns: pd.Series,
    loss_threshold: float = -0.05,
) -> float:
    """
    Compute false positive rate.

    False positive = signal that loses > threshold within first few days.

    Args:
        returns: Final trade returns
        early_returns: Returns at day 2-3
        loss_threshold: Threshold for early loss

    Returns:
        False positive rate
    """
    if len(early_returns) == 0:
        return 0.0

    false_positives = (early_returns < loss_threshold).sum()
    return false_positives / len(early_returns)


def compute_rank_score(
    hit_rate: float,
    avg_return: float,
    win_loss_ratio: float,
    signals_per_month: float,
    weight_hit_rate: float = 1.0,
    weight_return: float = 1.0,
    weight_wl_ratio: float = 0.5,
    signal_penalty: float = 0.5,
) -> float:
    """
    Compute composite ranking score for parameter comparison.

    Formula:
        score = (hit_rate × avg_return) × win_loss_ratio × (1 / sqrt(signals_per_month))

    Higher is better. Penalizes high signal frequency to avoid overfitting.

    Args:
        hit_rate: Win rate
        avg_return: Average return per trade
        win_loss_ratio: Average win / average loss
        signals_per_month: Signal frequency
        weight_*: Weights for each component
        signal_penalty: Penalty factor for high frequency

    Returns:
        Composite rank score
    """
    if hit_rate <= 0 or avg_return <= 0 or win_loss_ratio <= 0:
        return 0.0

    # Normalize signals per month (target ~5-10 signals/month)
    freq_penalty = 1.0 / (1.0 + signal_penalty * max(0, signals_per_month - 5))

    score = (
        (hit_rate ** weight_hit_rate)
        * (avg_return ** weight_return)
        * (win_loss_ratio ** weight_wl_ratio)
        * freq_penalty
    )

    return score


def compute_all_metrics(
    signals_df: pd.DataFrame,
    returns_col: str = 'trade_return',
    hold_days_col: str = 'hold_days',
    quality_col: str = 'signal_quality',
    early_return_col: str = 'fwd_return_3d',
    date_col: str = 'date',
) -> BacktestMetrics:
    """
    Compute all backtest metrics from a signals DataFrame.

    Args:
        signals_df: DataFrame with trade signals and outcomes
        returns_col: Column with trade returns
        hold_days_col: Column with holding days
        quality_col: Column with signal quality
        early_return_col: Column with early returns (for false positive rate)
        date_col: Date column

    Returns:
        BacktestMetrics with all computed values
    """
    metrics = BacktestMetrics()

    # Filter to trades with valid returns
    valid_trades = signals_df[signals_df[returns_col].notna()].copy()

    metrics.total_signals = len(signals_df)
    metrics.total_trades = len(valid_trades)

    if len(valid_trades) == 0:
        return metrics

    returns = valid_trades[returns_col]

    # Basic counts
    metrics.profitable_trades = (returns > 0).sum()
    metrics.losing_trades = (returns < 0).sum()

    # Return metrics
    metrics.hit_rate = compute_hit_rate(returns)
    metrics.avg_return = compute_avg_return(returns)
    metrics.median_return = returns.median()
    metrics.total_return = returns.sum()
    metrics.std_return = returns.std()

    # Win/Loss metrics
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(wins) > 0:
        metrics.avg_win = wins.mean()
    if len(losses) > 0:
        metrics.avg_loss = abs(losses.mean())

    metrics.win_loss_ratio = compute_win_loss_ratio(returns)
    metrics.profit_factor = compute_profit_factor(returns)

    # Risk metrics
    metrics.sharpe_ratio = compute_sharpe_ratio(returns)
    metrics.sortino_ratio = compute_sortino_ratio(returns)

    equity_curve = compute_equity_curve(returns)
    metrics.max_drawdown = compute_max_drawdown(equity_curve)

    if metrics.max_drawdown > 0:
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(returns)) - 1
        metrics.calmar_ratio = cagr / metrics.max_drawdown

    # Holding period metrics
    if hold_days_col in valid_trades.columns:
        hold_days = valid_trades[hold_days_col]
        metrics.avg_hold_days = hold_days.mean()
        metrics.median_hold_days = hold_days.median()
        metrics.max_hold_days = int(hold_days.max())
        metrics.min_hold_days = int(hold_days.min())

    # Signal frequency
    if date_col in valid_trades.columns:
        dates = pd.to_datetime(valid_trades[date_col])
        if len(dates) > 0:
            date_range_days = (dates.max() - dates.min()).days
            if date_range_days > 0:
                months = date_range_days / 30
                metrics.signals_per_month = len(valid_trades) / months if months > 0 else 0
            metrics.trading_days = date_range_days

    # Quality metrics
    if early_return_col in valid_trades.columns:
        early_returns = valid_trades[early_return_col]
        metrics.false_positive_rate = compute_false_positive_rate(
            returns, early_returns, loss_threshold=-0.05
        )

    # High quality hit rate
    if quality_col in valid_trades.columns:
        high_quality = valid_trades[valid_trades[quality_col] == 'HIGH']
        if len(high_quality) > 0:
            metrics.high_quality_hit_rate = compute_hit_rate(high_quality[returns_col])

    # Composite rank score
    metrics.rank_score = compute_rank_score(
        hit_rate=metrics.hit_rate,
        avg_return=metrics.avg_return,
        win_loss_ratio=metrics.win_loss_ratio,
        signals_per_month=metrics.signals_per_month,
    )

    return metrics


def metrics_to_dict(metrics: BacktestMetrics) -> Dict:
    """Convert BacktestMetrics to dictionary for storage."""
    return {
        'total_signals': metrics.total_signals,
        'total_trades': metrics.total_trades,
        'profitable_trades': metrics.profitable_trades,
        'losing_trades': metrics.losing_trades,
        'hit_rate': metrics.hit_rate,
        'avg_return': metrics.avg_return,
        'median_return': metrics.median_return,
        'total_return': metrics.total_return,
        'std_return': metrics.std_return,
        'avg_win': metrics.avg_win,
        'avg_loss': metrics.avg_loss,
        'win_loss_ratio': metrics.win_loss_ratio,
        'profit_factor': metrics.profit_factor,
        'sharpe_ratio': metrics.sharpe_ratio,
        'sortino_ratio': metrics.sortino_ratio,
        'max_drawdown': metrics.max_drawdown,
        'calmar_ratio': metrics.calmar_ratio,
        'avg_hold_days': metrics.avg_hold_days,
        'median_hold_days': metrics.median_hold_days,
        'max_hold_days': metrics.max_hold_days,
        'min_hold_days': metrics.min_hold_days,
        'signals_per_month': metrics.signals_per_month,
        'trading_days': metrics.trading_days,
        'false_positive_rate': metrics.false_positive_rate,
        'high_quality_hit_rate': metrics.high_quality_hit_rate,
        'rank_score': metrics.rank_score,
    }


def compare_configs(
    results: Dict[str, BacktestMetrics],
    sort_by: str = 'rank_score',
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Compare multiple configuration results.

    Args:
        results: Dictionary mapping config name to BacktestMetrics
        sort_by: Column to sort by
        ascending: Sort order

    Returns:
        DataFrame with comparison
    """
    rows = []
    for config_name, metrics in results.items():
        row = metrics_to_dict(metrics)
        row['config'] = config_name
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns
    cols = ['config'] + [c for c in df.columns if c != 'config']
    df = df[cols]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    return df


def print_metrics_summary(metrics: BacktestMetrics, title: str = "Backtest Results"):
    """Print formatted metrics summary."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"  Total Signals:     {metrics.total_signals}")
    print(f"  Total Trades:      {metrics.total_trades}")
    print(f"  Win Rate:          {metrics.hit_rate:.1%}")
    print(f"  Avg Return:        {metrics.avg_return:.2%}")
    print(f"  Median Return:     {metrics.median_return:.2%}")
    print(f"  Win/Loss Ratio:    {metrics.win_loss_ratio:.2f}")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:      {metrics.max_drawdown:.1%}")
    print(f"  Avg Hold Days:     {metrics.avg_hold_days:.1f}")
    print(f"  Signals/Month:     {metrics.signals_per_month:.1f}")
    print(f"  Rank Score:        {metrics.rank_score:.4f}")
    print(f"{'='*60}")


# =============================================================================
# CLI / STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create sample trade data
    np.random.seed(42)
    n_trades = 100

    # Simulate trades with realistic distribution
    returns = pd.Series(np.random.normal(0.015, 0.05, n_trades))  # 1.5% mean, 5% std
    hold_days = pd.Series(np.random.randint(3, 20, n_trades))
    dates = pd.date_range('2025-01-01', periods=n_trades, freq='3D')

    signals_df = pd.DataFrame({
        'date': dates,
        'trade_return': returns,
        'hold_days': hold_days,
        'fwd_return_3d': returns * 0.3 + np.random.normal(0, 0.02, n_trades),
        'signal_quality': np.random.choice(['HIGH', 'MEDIUM', 'LOW'], n_trades, p=[0.2, 0.5, 0.3]),
    })

    print("Testing metrics computation...")
    print(f"Sample trades: {len(signals_df)}")
    print(f"Return range: {returns.min():.2%} to {returns.max():.2%}")

    metrics = compute_all_metrics(signals_df)
    print_metrics_summary(metrics, "Sample Backtest")

    # Test comparison
    print("\n\nTesting config comparison...")
    configs = {
        "variant_A": metrics,
        "variant_B": BacktestMetrics(hit_rate=0.52, avg_return=0.012, win_loss_ratio=1.8),
        "variant_C": BacktestMetrics(hit_rate=0.60, avg_return=0.010, win_loss_ratio=2.2),
    }

    comparison = compare_configs(configs)
    print(comparison[['config', 'hit_rate', 'avg_return', 'win_loss_ratio', 'rank_score']].to_string())
