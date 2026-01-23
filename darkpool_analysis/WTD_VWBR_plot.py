#!/usr/bin/env python3
"""
WTD VWBR Plot - FINRA Buy Volume Analysis

Generates FINRA Buy Volume plots with buy/sell signals for multiple tickers.
Can be run standalone or imported and called from main.py.

Usage:
    python WTD_VWBR_plot.py                    # Uses config.py tickers
    python WTD_VWBR_plot.py --tickers SPY QQQ  # Override tickers
    python WTD_VWBR_plot.py --help             # Show options
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Add project root to path
project_root = Path(__file__).parent.parent
if (project_root / 'darkpool_analysis').exists():
    sys.path.insert(0, str(project_root))

from darkpool_analysis.config import load_config

# =============================================================================
# CONFIGURATION
# =============================================================================
# Load config for default tickers and paths
_config = load_config()

# Ticker configuration - None means use config.tickers
TICKERS_OVERRIDE: list[str] | None = None

# Date configuration
START_DATE = '2025-12-14'
END_DATE = 'Today'  # 'Today' or 'YYYY-MM-DD' format

# Output configuration
OUTPUT_BASE_DIR = _config.output_dir / 'WTD_VWBR'
DB_PATH = _config.db_path

# Buy/sell signal configuration
BUY_SELL_SIGNAL = 'Rolling_Zscore'  # 'Mean_Threshold' or 'Rolling_Zscore'

# Median-threshold settings
THRESHOLD_K = 1.0
THRESHOLD_K_NEG = 1.0

# Rolling Z-score settings
ROLLING_LOOKBACK_DAYS = 20
ROLLING_WINDOW_DAYS = 20
ZSCORE_K_BUY = 1.1
ZSCORE_K_SELL = 1.1
ZSCORE_MIN_PERIODS = 0

# Rolling mean line settings
ROLLING_MEAN_WINDOW_DAYS = ROLLING_WINDOW_DAYS
ROLLING_MEAN_SHIFT_DAYS = 1

# Plot display settings
AUTO_TEXT_SCALE = True
BASE_FIG_WIDTH = 12.0
TEXT_SCALE_MIN = 1.0
TEXT_SCALE_MAX = 3.5
TEXT_SCALE_POWER = 6
MAX_X_LABELS = 90
FIG_HEIGHT_RATIO = 1.0
FIG_DPI = 100
FIGSIZE_PX = None

# =============================================================================
# STYLING CONSTANTS
# =============================================================================
COLORS = {
    'background': '#0f0f10',
    'panel_bg': '#141416',
    'text': '#e6e6e6',
    'text_muted': '#8b8b8b',
    'grid': '#2a2a2d',
    'white': '#ffffff',
    'green': '#00ff88',
    'red': '#ff6b6b',
    'yellow': '#ffd700',
    'cyan': '#00d4ff',
    'orange': '#ff9f43',
    'purple': '#b026ff',
    'neutral': '#6b6b6b',
    'blue': '#4aa3ff',
}
GRID_ALPHA = 0.18
OHLC_LINE_WIDTH = 1.2

# Global text scale (set during plotting)
TEXT_SCALE = 1.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_tickers(tickers_override: list[str] | None = None) -> list[str]:
    """Get tickers from config or use override if provided."""
    if tickers_override:
        return tickers_override
    return _config.tickers


def parse_end_date(end_date_str: str) -> str:
    """Parse END_DATE, supporting 'Today' keyword."""
    if end_date_str.lower() == 'today':
        return pd.Timestamp.now().strftime('%Y-%m-%d')
    return end_date_str


def get_output_dir(base_dir: Path, end_date: str) -> Path:
    """Create and return dated output directory."""
    date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    output_dir = base_dir / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _compute_fig_width(day_count: int) -> float:
    if day_count <= 0:
        return 12.0
    buckets = int(np.ceil(day_count / 50))
    return 12.0 * max(1, buckets)


def _apply_axis_style(ax):
    scale = TEXT_SCALE
    ax.set_facecolor(COLORS['panel_bg'])
    ax.tick_params(colors=COLORS['text'], labelsize=9 * scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.grid(True, alpha=GRID_ALPHA, color=COLORS['grid'], linestyle='--')


def _apply_accum_axis_style(ax):
    scale = TEXT_SCALE
    ax.set_facecolor(COLORS['panel_bg'])
    ax.tick_params(colors=COLORS['text'], labelsize=9 * scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.grid(True, axis='y', alpha=GRID_ALPHA, color=COLORS['grid'], linestyle='-')
    ax.grid(False, axis='x')


def _format_plain_int(value) -> str:
    if pd.isna(value):
        return 'NA'
    return f'{value:,.0f}'.replace(',', '')


def _plot_smooth_line(ax, x_values, values, color, valid_mask, linewidth=1.0, alpha=0.7, zorder=3, linestyle='-'):
    """Plot a smooth PCHIP-interpolated line through valid data points."""
    mask = np.asarray(valid_mask)
    if mask.sum() >= 3:
        valid_x = x_values[mask]
        valid_values = values[mask]

        if pd.api.types.is_datetime64_any_dtype(valid_x):
            x_nums = mdates.date2num(pd.to_datetime(valid_x))
            x_dense = np.linspace(x_nums.min(), x_nums.max(), 150)
            x_smooth = np.unique(np.concatenate([x_nums, x_dense]))
            interp = PchipInterpolator(x_nums, np.asarray(valid_values, dtype=float))
            y_smooth = interp(x_smooth)
            x_plot = mdates.num2date(x_smooth)
        else:
            x_nums = np.asarray(valid_x, dtype=float)
            x_dense = np.linspace(x_nums.min(), x_nums.max(), 150)
            x_smooth = np.unique(np.concatenate([x_nums, x_dense]))
            interp = PchipInterpolator(x_nums, np.asarray(valid_values, dtype=float))
            y_smooth = interp(x_smooth)
            x_plot = x_smooth

        ax.plot(x_plot, y_smooth, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, linestyle=linestyle)
    else:
        ax.plot(x_values[mask], values[mask], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, linestyle=linestyle)


def _nice_volume_ticks(max_val, target_ticks=6):
    if max_val is None or max_val <= 0 or not np.isfinite(max_val):
        return np.array([0.0, 1.0])
    raw_step = max_val / max(target_ticks - 1, 1)
    magnitude = 10 ** int(math.floor(math.log10(raw_step)))
    while raw_step < magnitude:
        magnitude /= 10
    steps = [1, 2, 5, 10]
    step = steps[0] * magnitude
    for s in steps:
        candidate = s * magnitude
        if candidate <= raw_step:
            step = candidate
    top = math.ceil(max_val / step) * step
    return np.arange(0, top + step, step)


def _plot_ohlc_bars(ax, df_ohlc, x_indices):
    """Plot OHLC bars."""
    bar_width = 0.6
    half_width = bar_width / 2

    for xi, (_, row) in zip(x_indices, df_ohlc.iterrows()):
        open_ = row['open']
        high = row['high']
        low = row['low']
        close = row['close']
        if pd.isna(open_) or pd.isna(close) or pd.isna(high) or pd.isna(low):
            continue
        color = COLORS['blue'] if close >= open_ else COLORS['orange']
        ax.vlines(xi, low, high, color=color, linewidth=OHLC_LINE_WIDTH, zorder=3)
        ax.hlines(open_, xi - half_width, xi, color=color, linewidth=OHLC_LINE_WIDTH, zorder=3)
        ax.hlines(close, xi, xi + half_width, color=color, linewidth=OHLC_LINE_WIDTH, zorder=3)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_ticker_data(ticker: str, start_date: str, end_date: str, db_path: Path,
                     rolling_lookback_days: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load metrics and OHLC data for a single ticker."""
    query_start_date = start_date
    if rolling_lookback_days and int(rolling_lookback_days) > 0:
        query_start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=int(rolling_lookback_days))).strftime('%Y-%m-%d')

    query = '''
        SELECT
            date, symbol, finra_buy_volume, short_sell_volume,
            short_buy_sell_ratio, short_buy_sell_ratio_z,
            lit_buy_volume, lit_sell_volume, lit_flow_imbalance,
            lit_flow_imbalance_z, return_z, otc_participation_z, confidence
        FROM daily_metrics
        WHERE symbol = ? AND date BETWEEN ? AND ?
        ORDER BY date
    '''

    ohlc_query = '''
        SELECT trade_date AS date, open, high, low, close, volume
        FROM polygon_daily_agg_raw
        WHERE symbol = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    '''

    with duckdb.connect(str(db_path), read_only=True) as conn:
        df = conn.execute(query, [ticker.upper(), query_start_date, end_date]).df()
        df_ohlc = conn.execute(ohlc_query, [ticker.upper(), query_start_date, end_date]).df()

    if df.empty:
        raise ValueError(f'No data found for {ticker} between {start_date} and {end_date}.')

    df['date'] = pd.to_datetime(df['date'])
    if 'finra_buy_volume' in df.columns:
        df['finra_buy_volume'] = pd.to_numeric(df['finra_buy_volume'], errors='coerce')

    if not df_ohlc.empty:
        df_ohlc['date'] = pd.to_datetime(df_ohlc['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_ohlc.columns:
                df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')
        df = df.merge(df_ohlc, on='date', how='left')
    else:
        df['open'] = np.nan
        df['high'] = np.nan
        df['low'] = np.nan
        df['close'] = np.nan
        df['volume'] = np.nan

    return df, df_ohlc


def prepare_plot_data(df: pd.DataFrame, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for plotting by filtering to date range."""
    full_df = df.sort_values('date').copy()
    market_mask = full_df['close'].notna()
    full_df = full_df.loc[market_mask].copy()

    if full_df.empty:
        raise ValueError('No market days with OHLC data in the selected range.')

    plot_start = pd.to_datetime(start_date)
    plot_end = pd.to_datetime(end_date)
    plot_df = full_df[(full_df['date'] >= plot_start) & (full_df['date'] <= plot_end)].copy()
    plot_df = plot_df.reset_index(drop=True)

    if plot_df.empty:
        raise ValueError('No market days with OHLC data in the selected range.')

    if 'finra_buy_volume' not in plot_df.columns:
        raise ValueError('finra_buy_volume missing from dataset.')

    full_df['vw_accum'] = pd.to_numeric(full_df['finra_buy_volume'], errors='coerce')
    plot_df['vw_accum'] = pd.to_numeric(plot_df['finra_buy_volume'], errors='coerce')

    return full_df, plot_df


# =============================================================================
# SIGNAL COMPUTATION
# =============================================================================
def _compute_signal_indices(accum_series, valid_mask_series, full_accum_series=None,
                           full_valid_mask=None, plot_dates=None, signal_config=None):
    """Compute buy/sell signal indices based on configuration."""
    if signal_config is None:
        signal_config = {}

    buy_sell_signal = signal_config.get('buy_sell_signal', BUY_SELL_SIGNAL)
    threshold_k = signal_config.get('threshold_k', THRESHOLD_K)
    threshold_k_neg = signal_config.get('threshold_k_neg', THRESHOLD_K_NEG)
    rolling_window_days = signal_config.get('rolling_window_days', ROLLING_WINDOW_DAYS)
    zscore_k_buy = signal_config.get('zscore_k_buy', ZSCORE_K_BUY)
    zscore_k_sell = signal_config.get('zscore_k_sell', ZSCORE_K_SELL)
    zscore_min_periods = signal_config.get('zscore_min_periods', ZSCORE_MIN_PERIODS)

    if buy_sell_signal == 'Mean_Threshold':
        mean_val = accum_series[valid_mask_series].mean()
        std_val = accum_series[valid_mask_series].std(ddof=0)

        pos_threshold = mean_val + (threshold_k * std_val if std_val and std_val > 0 else 0.0)
        pos_idx = accum_series[valid_mask_series][accum_series[valid_mask_series] >= pos_threshold].index
        if len(pos_idx) == 0:
            top_n = min(5, len(accum_series[valid_mask_series]))
            pos_idx = accum_series[valid_mask_series].nlargest(top_n).index

        neg_threshold = mean_val - (threshold_k_neg * std_val if std_val and std_val > 0 else 0.0)
        neg_idx = accum_series[valid_mask_series][accum_series[valid_mask_series] <= neg_threshold].index
        if len(neg_idx) == 0:
            bottom_n = min(5, len(accum_series[valid_mask_series]))
            neg_idx = accum_series[valid_mask_series].nsmallest(bottom_n).index

        return pos_idx, neg_idx, mean_val, None

    if buy_sell_signal == 'Rolling_Zscore':
        base_series = full_accum_series if full_accum_series is not None else accum_series
        base_valid = full_valid_mask if full_valid_mask is not None else valid_mask_series
        window = int(rolling_window_days)
        min_periods = window if zscore_min_periods is None or zscore_min_periods == 0 else int(zscore_min_periods)
        prior = base_series.shift(1)
        rolling_mean = prior.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = prior.rolling(window=window, min_periods=min_periods).std(ddof=0)
        z_scores = (base_series - rolling_mean) / rolling_std
        ok_mask = base_valid & rolling_mean.notna() & rolling_std.notna() & (rolling_std > 0)
        pos_mask = ok_mask & (z_scores >= zscore_k_buy)
        neg_mask = ok_mask & (z_scores <= -zscore_k_sell)
        pos_dates = base_series[pos_mask].index
        neg_dates = base_series[neg_mask].index
        if plot_dates is not None:
            pos_idx = plot_dates.index[plot_dates.isin(pos_dates)]
            neg_idx = plot_dates.index[plot_dates.isin(neg_dates)]
        else:
            pos_idx = pos_dates
            neg_idx = neg_dates
        return pos_idx, neg_idx, None, rolling_mean

    raise ValueError(f"Unknown BUY_SELL_SIGNAL mode: {buy_sell_signal}")


# =============================================================================
# PLOTTING FUNCTION
# =============================================================================
def plot_wtd_vwbr(ticker: str, full_df: pd.DataFrame, plot_df: pd.DataFrame,
                  start_date: str, end_date: str, output_dir: Path,
                  signal_config: dict = None, plot_config: dict = None,
                  show_plot: bool = True) -> tuple[plt.Figure, pd.DataFrame]:
    """Generate WTD VWBR plot for a single ticker."""
    global TEXT_SCALE

    if signal_config is None:
        signal_config = {
            'buy_sell_signal': BUY_SELL_SIGNAL,
            'threshold_k': THRESHOLD_K,
            'threshold_k_neg': THRESHOLD_K_NEG,
            'rolling_window_days': ROLLING_WINDOW_DAYS,
            'zscore_k_buy': ZSCORE_K_BUY,
            'zscore_k_sell': ZSCORE_K_SELL,
            'zscore_min_periods': ZSCORE_MIN_PERIODS,
            'rolling_mean_window_days': ROLLING_MEAN_WINDOW_DAYS,
            'rolling_mean_shift_days': ROLLING_MEAN_SHIFT_DAYS,
        }

    if plot_config is None:
        plot_config = {
            'auto_text_scale': AUTO_TEXT_SCALE,
            'base_fig_width': BASE_FIG_WIDTH,
            'text_scale_min': TEXT_SCALE_MIN,
            'text_scale_max': TEXT_SCALE_MAX,
            'text_scale_power': TEXT_SCALE_POWER,
            'max_x_labels': MAX_X_LABELS,
            'fig_height_ratio': FIG_HEIGHT_RATIO,
            'fig_dpi': FIG_DPI,
            'figsize_px': FIGSIZE_PX,
        }

    x_labels = plot_df['date'].dt.strftime('%Y%m%d').tolist()
    accum_series = plot_df['finra_buy_volume']

    if accum_series.isna().all():
        raise ValueError(f'finra_buy_volume is empty for {ticker} in the selected range.')

    full_accum_series = full_df.set_index('date')['vw_accum']
    full_valid_mask = ~full_accum_series.isna()
    plot_dates = plot_df['date']
    valid_mask = ~plot_df['vw_accum'].isna()
    accum = plot_df['vw_accum']

    # Figure sizing
    fig_width = _compute_fig_width(len(plot_df))
    fig_height = fig_width * plot_config['fig_height_ratio']
    figsize = (fig_width, fig_height)
    if plot_config['figsize_px'] is not None:
        fig_width = plot_config['figsize_px'][0] / plot_config['fig_dpi']
        fig_height = plot_config['figsize_px'][1] / plot_config['fig_dpi']
        figsize = (fig_width, fig_height)

    text_scale = 1.0
    if plot_config['auto_text_scale']:
        base_width = float(plot_config['base_fig_width'])
        if base_width > 0:
            text_scale = (fig_width / base_width) ** plot_config['text_scale_power']
        text_scale = max(plot_config['text_scale_min'], min(plot_config['text_scale_max'], text_scale))
    TEXT_SCALE = text_scale

    plt.style.use('dark_background')
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(COLORS['background'])

    height_ratios = [1.3, 2]
    fig_gs = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=0.15, left=0.1, right=0.94, top=0.9, bottom=0.1)

    ax_price = fig.add_subplot(fig_gs[0])
    ax_accum = fig.add_subplot(fig_gs[1], sharex=ax_price)

    # Panel 1: OHLC
    _apply_axis_style(ax_price)
    x_vals = np.arange(len(plot_df))
    _plot_ohlc_bars(ax_price, plot_df, x_vals)
    ax_price.set_ylabel('Price', color=COLORS['white'], fontsize=10 * text_scale)
    ax_price.set_title('Price Action (OHLC)', color=COLORS['white'], fontsize=11 * text_scale, fontweight='bold', loc='left')
    ax_price.tick_params(axis='x', labelbottom=False)
    if len(x_vals) > 0:
        ax_price.set_xlim(x_vals.min() - 0.6, x_vals.max() + 0.6)

    # Panel 2: Accumulation
    _apply_accum_axis_style(ax_accum)

    if valid_mask.any():
        _plot_smooth_line(ax_accum, x_vals, accum, COLORS['green'], valid_mask, linewidth=1.4, alpha=0.9, zorder=3)
        ax_accum.scatter(x_vals[valid_mask], accum.loc[valid_mask], color=COLORS['green'], s=80, zorder=4)

    max_val = accum[valid_mask].max() if valid_mask.any() else 1.0
    if pd.isna(max_val) or max_val <= 0:
        max_val = 1.0

    y_min = 0.0
    y_ticks = _nice_volume_ticks(max_val * 1.1)
    ax_accum.set_ylim(y_min, float(y_ticks[-1]))
    ax_accum.set_yticks(y_ticks)
    ax_accum.grid(True, axis='y', linestyle='-', color=COLORS['grid'], alpha=0.9, linewidth=0.8)
    ax_accum.set_axisbelow(True)
    ax_accum.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _format_plain_int(x)))
    ax_accum.tick_params(axis='y', colors=COLORS['green'])
    ax_accum.set_ylabel('FINRA Buy Volume', color=COLORS['cyan'], fontsize=12 * text_scale)
    ax_accum.set_title('FINRA Buy Volume', color=COLORS['cyan'], fontsize=12 * text_scale, fontweight='bold', loc='center', pad=10 * text_scale)

    # Compute signals
    pos_idx, neg_idx = [], []
    if valid_mask.any():
        pos_idx, neg_idx, mean_val, rolling_mean = _compute_signal_indices(
            accum, valid_mask, full_accum_series=full_accum_series,
            full_valid_mask=full_valid_mask, plot_dates=plot_dates, signal_config=signal_config
        )

        if signal_config['buy_sell_signal'] == 'Mean_Threshold' and mean_val is not None:
            ax_accum.axhline(mean_val, color=COLORS['orange'], linestyle='--', linewidth=1.0, alpha=0.8, zorder=2)

        if signal_config['buy_sell_signal'] == 'Rolling_Zscore':
            rolling_prior = full_accum_series.shift(int(signal_config['rolling_mean_shift_days']))
            rolling_mean_line_full = rolling_prior.rolling(
                window=int(signal_config['rolling_mean_window_days']),
                min_periods=int(signal_config['rolling_mean_window_days'])
            ).mean()
            rolling_mean_line = rolling_mean_line_full.reindex(plot_dates)
            rolling_mask = rolling_mean_line.notna()
            if rolling_mask.any():
                _plot_smooth_line(ax_accum, x_vals, rolling_mean_line, COLORS['orange'], rolling_mask,
                                 linewidth=1.0, alpha=0.8, zorder=2, linestyle='--')

        # Signal markers on price panel
        price_lows = plot_df['low'].to_numpy()
        price_highs = plot_df['high'].to_numpy()
        price_min = np.nanmin(price_lows) if np.isfinite(price_lows).any() else np.nan
        price_max = np.nanmax(price_highs) if np.isfinite(price_highs).any() else np.nan
        if np.isfinite(price_min) and np.isfinite(price_max):
            offset = (price_max - price_min) * 0.03 or 1.0
            if len(pos_idx) > 0:
                ax_price.scatter(x_vals[pos_idx], plot_df.loc[pos_idx, 'low'].to_numpy() - offset,
                                s=60, color=COLORS['green'], zorder=5)
            if len(neg_idx) > 0:
                ax_price.scatter(x_vals[neg_idx], plot_df.loc[neg_idx, 'high'].to_numpy() + offset,
                                s=60, color=COLORS['red'], zorder=5)

        # Vertical lines on accumulation panel
        for idx in pos_idx:
            ax_accum.vlines(x_vals[idx], y_min, accum.loc[idx], color="#6bd5ff", linestyle='--', linewidth=1.0, alpha=0.8, zorder=2)
        for idx in neg_idx:
            ax_accum.vlines(x_vals[idx], y_min, accum.loc[idx], color="#ff0000", linestyle='--', linewidth=1.0, alpha=0.8, zorder=2)

    # X-axis labels
    label_step = 1
    if len(x_vals) > plot_config['max_x_labels']:
        label_step = int(np.ceil(len(x_vals) / plot_config['max_x_labels']))
    x_tick_positions = x_vals[::label_step]
    x_tick_labels = x_labels[::label_step]
    if len(x_tick_positions) > 0 and x_tick_positions[-1] != x_vals[-1]:
        x_tick_positions = np.append(x_tick_positions, x_vals[-1])
        x_tick_labels.append(x_labels[-1])

    ax_accum.set_xticks(x_tick_positions)
    ax_accum.set_xticklabels(x_tick_labels, rotation=90, ha='center', fontsize=8 * text_scale, color=COLORS['text_muted'])
    ax_accum.tick_params(axis='x', labelbottom=True, colors=COLORS['text_muted'])
    ax_accum.set_xlabel('Date', color=COLORS['cyan'], fontsize=12 * text_scale)

    # Annotations
    last_update = plot_df['date'].max()
    if signal_config['buy_sell_signal'] == 'Mean_Threshold':
        signal_note = f"Signal: Mean Threshold (mean +/- k*std). k_buy={signal_config['threshold_k']:.2f}, k_sell={signal_config['threshold_k_neg']:.2f}"
    else:
        min_p = signal_config['zscore_min_periods'] or signal_config['rolling_window_days']
        signal_note = f"Signal: Rolling Z-Score (window={signal_config['rolling_window_days']}, min={min_p}, K Buy +{signal_config['zscore_k_buy']:.2f} / K Sell -{signal_config['zscore_k_sell']:.2f})"

    last_update_str = f"{last_update.month}/{last_update.day}/{last_update.year}" if pd.notna(last_update) else 'NA'

    fig.text(0.02, 0.96, f"{ticker}: FINRA Buy Volume", color=COLORS['cyan'], fontsize=16 * text_scale, fontweight='bold', ha='left', va='top')
    fig.text(0.98, 0.96, f"Last Update {last_update_str}", color=COLORS['white'], fontsize=11 * text_scale, ha='right', va='top')
    fig.text(0.02, 0.02, signal_note, color=COLORS['text_muted'], fontsize=8 * text_scale, ha='left', va='bottom')
    fig.text(0.94, 0.02, 'all credit to: next', color=COLORS['text_muted'], fontsize=10 * text_scale, ha='right', va='bottom')
    fig.text(0.98, 0.02, 'Signals', color=COLORS['cyan'], fontsize=10 * text_scale, ha='right', va='bottom')

    # Save figure
    start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    output_path = output_dir / f"{ticker.lower()}_wtd_vwbr_{start_str}_{end_str}.png"
    fig.savefig(output_path, dpi=150, facecolor=COLORS['background'], edgecolor='none')

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # Build signal report
    buy_dates = plot_df.loc[pos_idx, 'date'] if len(pos_idx) > 0 else pd.Series(dtype='datetime64[ns]')
    sell_dates = plot_df.loc[neg_idx, 'date'] if len(neg_idx) > 0 else pd.Series(dtype='datetime64[ns]')
    report_df = pd.DataFrame({
        'date': pd.concat([buy_dates, sell_dates], ignore_index=True),
        'signal': ['BUY'] * len(buy_dates) + ['SELL'] * len(sell_dates),
    }).sort_values('date')

    return fig, report_df


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def run_wtd_vwbr(tickers: list[str] | None = None,
                 start_date: str = None,
                 end_date: str = None,
                 output_base_dir: Path = None,
                 db_path: Path = None,
                 show_plots: bool = True) -> tuple[list[str], list[tuple], pd.DataFrame | None]:
    """
    Run WTD VWBR analysis for multiple tickers.

    Args:
        tickers: List of ticker symbols (None = use config.tickers)
        start_date: Start date (None = use START_DATE)
        end_date: End date (None = use END_DATE, supports 'Today')
        output_base_dir: Base output directory (None = use OUTPUT_BASE_DIR)
        db_path: Database path (None = use DB_PATH)
        show_plots: Whether to display plots inline

    Returns:
        tuple: (successful_tickers, failed_tickers, combined_report_df)
    """
    tickers = get_tickers(tickers)
    start_date = start_date or START_DATE
    end_date = parse_end_date(end_date or END_DATE)
    output_base_dir = output_base_dir or OUTPUT_BASE_DIR
    db_path = db_path or DB_PATH

    output_dir = get_output_dir(output_base_dir, end_date)

    print(f"\n{'='*60}")
    print(f"Processing {len(tickers)} tickers")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    signal_config = {
        'buy_sell_signal': BUY_SELL_SIGNAL,
        'threshold_k': THRESHOLD_K,
        'threshold_k_neg': THRESHOLD_K_NEG,
        'rolling_window_days': ROLLING_WINDOW_DAYS,
        'zscore_k_buy': ZSCORE_K_BUY,
        'zscore_k_sell': ZSCORE_K_SELL,
        'zscore_min_periods': ZSCORE_MIN_PERIODS,
        'rolling_mean_window_days': ROLLING_MEAN_WINDOW_DAYS,
        'rolling_mean_shift_days': ROLLING_MEAN_SHIFT_DAYS,
    }

    plot_config = {
        'auto_text_scale': AUTO_TEXT_SCALE,
        'base_fig_width': BASE_FIG_WIDTH,
        'text_scale_min': TEXT_SCALE_MIN,
        'text_scale_max': TEXT_SCALE_MAX,
        'text_scale_power': TEXT_SCALE_POWER,
        'max_x_labels': MAX_X_LABELS,
        'fig_height_ratio': FIG_HEIGHT_RATIO,
        'fig_dpi': FIG_DPI,
        'figsize_px': FIGSIZE_PX,
    }

    all_reports = []
    successful_tickers = []
    failed_tickers = []

    for ticker in tickers:
        print(f"\n{'─'*40}")
        print(f"Processing: {ticker}")
        print(f"{'─'*40}")

        try:
            lookback = ROLLING_LOOKBACK_DAYS if BUY_SELL_SIGNAL == 'Rolling_Zscore' else 0
            df, df_ohlc = load_ticker_data(ticker, start_date, end_date, db_path, lookback)
            full_df, plot_df = prepare_plot_data(df, start_date, end_date)

            fig, report_df = plot_wtd_vwbr(
                ticker, full_df, plot_df, start_date, end_date, output_dir,
                signal_config=signal_config, plot_config=plot_config, show_plot=show_plots
            )

            if not report_df.empty:
                report_df['ticker'] = ticker
                all_reports.append(report_df)

            successful_tickers.append(ticker)
            print(f"  Saved: {output_dir / f'{ticker.lower()}_wtd_vwbr_{start_date}_{end_date}.png'}")

        except ValueError as e:
            print(f"  SKIPPED: {e}")
            failed_tickers.append((ticker, str(e)))
        except Exception as e:
            print(f"  ERROR: {e}")
            failed_tickers.append((ticker, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {len(successful_tickers)} tickers")
    print(f"Failed: {len(failed_tickers)} tickers")
    if failed_tickers:
        print("\nFailed tickers:")
        for t, err in failed_tickers:
            print(f"  - {t}: {err}")
    print(f"\nOutput saved to: {output_dir}")

    # Combine reports
    combined_report = None
    if all_reports:
        combined_report = pd.concat(all_reports, ignore_index=True)
        combined_report = combined_report.sort_values(['date', 'ticker'])
        combined_report['date'] = combined_report['date'].dt.strftime('%Y-%m-%d')

        # Save combined report
        report_path = output_dir / f"signal_report_{start_date}_{end_date}.csv"
        combined_report[['date', 'ticker', 'signal']].to_csv(report_path, index=False)
        print(f"\nSignal report saved to: {report_path}")

    return successful_tickers, failed_tickers, combined_report


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate FINRA Buy Volume plots with buy/sell signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python WTD_VWBR_plot.py                      # Use config.py tickers
    python WTD_VWBR_plot.py --tickers SPY QQQ   # Override tickers
    python WTD_VWBR_plot.py --start 2025-01-01  # Custom start date
    python WTD_VWBR_plot.py --no-show           # Don't display plots
        """
    )
    parser.add_argument('--tickers', nargs='+', help='Ticker symbols (overrides config.py)')
    parser.add_argument('--start', default=START_DATE, help=f'Start date (default: {START_DATE})')
    parser.add_argument('--end', default=END_DATE, help=f'End date (default: {END_DATE}, supports "Today")')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots (save only)')

    args = parser.parse_args()

    run_wtd_vwbr(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        show_plots=not args.no_show
    )


if __name__ == '__main__':
    main()
