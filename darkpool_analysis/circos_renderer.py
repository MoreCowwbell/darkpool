#!/usr/bin/env python
"""
Circos-style chord diagram renderer for institutional money flow visualization.

This module extracts the logic from Special_tools/circos_v2.ipynb into a callable
Python module that can be integrated into the main pipeline.

Usage:
    from circos_renderer import render_circos
    output_path = render_circos()  # Uses default config

    # Or with custom config
    from circos_renderer import CircosConfig, render_circos
    config = CircosConfig(ticker_type=["SECTOR", "MAG8"], flow_period_days=10)
    output_path = render_circos(config=config, output_dir=Path("output/circos"))
"""
from __future__ import annotations

import importlib.util
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Wedge, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class CircosConfig:
    """Configuration for circos rendering with sensible defaults."""

    # Universe selection (ticker groups)
    ticker_type: List[str] = field(default_factory=lambda: [
        "SECTOR", "THEMATIC", "GLOBAL", "COMMODITIES", "MAG8", "CRYPTO", "SPECULATIVE"
    ])

    # Time window
    end_date: Optional[str] = None  # None = auto-detect max date in DB
    flow_period_days: int = 5
    lookback_days: int = 20  # For z-score stats

    # Flow selection
    top_k_winners: int = 15
    top_k_losers: int = 15
    min_edge_flow: float = 0.0
    distribution_mode: str = "demand_weighted"  # "equal" or "demand_weighted"

    # Chord density
    metric_band_mode: str = "proportional"  # "equal" or "proportional"
    max_edges_per_metric: int = 80
    edge_ribbon_splits: int = 1
    edge_ribbon_max: int = 100
    chord_arc_fraction: float = 1.0
    chord_radius: float = 0.78
    band_gap_frac: float = 0.0
    dir_gap_frac: float = 0.0
    category_gap_deg: float = 3.0

    # Ring layout
    time_slice_bins: int = 30
    ring_base_thickness: float = 0.005
    ring_thickness_scale: float = 0.1
    ring1_thickness_mult: float = 0.8
    ring2_thickness_mult: float = 0.8
    ring3_thickness_mult: float = 0.8
    ring_gap: float = 0.01
    ticker_outer: float = 1.02  # Outer radius where rings start (labels inside this)

    # Fanned chord layout
    ribbon_gap_rad: float = 0.001
    ribbon_width_scale_by_flow: bool = True
    ribbon_centered: bool = True
    ribbon_converge_to_point: bool = False
    ribbon_anchor_to_center: bool = True

    # Render quality
    render_mode: str = "balanced"  # "fast", "balanced", "quality"
    chord_fill_alpha: float = 0.55
    chord_line_alpha: float = 0.8
    chord_color_soften: float = 0.25

    # Layer toggles
    show_accum_flow: bool = False
    show_lit_flow: bool = False
    show_short_net_flow: bool = False
    show_vwbr_z: bool = True
    show_finra_flow: bool = False
    show_volume_ring: bool = True
    ring3_color_mode: str = "VWBR_Z"  # "SHORT_RATIO", "VWBR", "VWBR_Z"
    ring3_zscore_span: float = 2.0
    ring3_vwbr_span: float = 1.0
    ring3_brightness_min: float = 0.2
    ring3_brightness_max: float = 0.9

    # Time-fade chords
    show_time_fade_chords: bool = False
    time_fade_use_daily_edges: bool = True
    time_fade_min_alpha: float = 0.2
    time_fade_max_alpha: float = 1.0
    time_fade_power: float = 10.0

    # Figure layout
    figure_size: Tuple[int, int] = (20, 20)
    plot_center_x: float = 0.55
    plot_center_y: float = 0.52
    main_ax_size: float = 0.82

    # Watermark
    watermark_path: Optional[str] = None
    watermark_width: float = 0.15
    watermark_alpha: float = 0.6

    # Edge multiplier for display
    edge_multiplier: float = 1000.0

    # Table column widths
    table_col1_width: int = 25
    table_col2_width: int = 20
    table_col3_width: int = 20

    def __post_init__(self):
        """Apply render mode presets and time-fade adjustments."""
        # Figure scale for font sizes
        self.figure_scale = min(self.figure_size) / 12.0
        self.title_fontsize = 14 * self.figure_scale
        self.subtitle_fontsize = 10 * self.figure_scale
        self.ticker_fontsize = 9 * self.figure_scale
        self.legend_title_fontsize = 10 * self.figure_scale
        self.legend_label_fontsize = 8 * self.figure_scale
        self.ring_title_fontsize = 9 * self.figure_scale
        self.ring_label_fontsize = 7 * self.figure_scale
        self.table_fontsize = 9 * self.figure_scale
        self.legend_linewidth = 4 * self.figure_scale

        if self.render_mode == "fast":
            self.use_gradient_fill = False
            self.chord_gradient_steps = 8
            self.chord_arc_points = 8
            self.chord_curve_points = 30
        elif self.render_mode == "quality":
            self.use_gradient_fill = True
            self.chord_gradient_steps = 32
            self.chord_arc_points = 18
            self.chord_curve_points = 70
        else:  # balanced
            self.use_gradient_fill = True
            self.chord_gradient_steps = 18
            self.chord_arc_points = 12
            self.chord_curve_points = 50

        # Reduce density when time-fade is enabled
        if self.show_time_fade_chords:
            self.top_k_winners = self.top_k_winners // 2
            self.top_k_losers = self.top_k_losers // 2
            self.max_edges_per_metric = self.max_edges_per_metric // 2

        # Per-metric ribbon min widths
        self.ribbon_min_width_rad = {
            'accum': 0.001, 'short': 0.001, 'lit': 0.001,
            'finra_buy': 0.001, 'vwbr_z': 0.001,
        }
        self.ribbon_center_offset = {
            'accum': 0.0, 'short': 0.0, 'lit': 0.0,
            'finra_buy': 0.0, 'vwbr_z': 0.0,
        }


# ---------------------------------------------------------------------------
# Color and style constants
# ---------------------------------------------------------------------------

BG_COLOR = '#0b0f1a'

CATEGORY_LABELS = {
    'GLOBAL_MACRO': 'GLOBAL MACRO',
    'MAG8': 'MAG8',
    'THEMATIC_SECTORS': 'THEMATIC SECTORS',
    'SECTOR_CORE': 'SECTORS',
    'COMMODITIES': 'COMMODITIES',
    'RATES_CREDIT': 'RATES/CREDIT',
    'SPECULATIVE': 'SPECULATIVE',
    'CRYPTO': 'CRYPTO',
}

CATEGORY_PALETTE = {
    'GLOBAL_MACRO': "#0059FF",
    'MAG8': "#D38CFA",
    'THEMATIC_SECTORS': "#00E4C5",
    'SECTOR_CORE': "#FAAF00F2",
    'COMMODITIES': "#FFFC2F",
    'RATES_CREDIT': "#FF9966",
    'SPECULATIVE': "#FF7A45",
    'CRYPTO': "#4CC9F0",
    'UNKNOWN': "#8F8E8E",
}

CHORD_METRIC_ORDER = ['accum', 'short', 'lit', 'finra_buy', 'vwbr_z']
BAND_ORDER = ['lit', 'accum', 'short', 'finra_buy', 'vwbr_z']
RING_METRIC_ORDER = ['accum', 'dark_lit', 'finra_buy']

METRIC_LABELS = {
    'accum': 'Accumulation',
    'short': 'Daily Short',
    'lit': 'Lit',
    'finra_buy': 'Finra Buy',
    'vwbr_z': 'VWBR Z (Distribution -> Accumulation)',
    'dark_lit': 'Dark/Lit Ratio',
}

METRIC_COLORS = {
    'accum': {'sell': "#8304B9", 'buy': "#26FF00"},
    'short': {'sell': "#280042", 'buy': "#00AEFF"},
    'lit': {'sell': "#FF0B0B", 'buy': "#1E7237"},
    'finra_buy': {'low': "#63238D", 'high': "#00E4C5"},
    'vwbr_z': {'sell': "#63238D", 'buy': "#00E4C5"},
}

RING_COLORS = {
    'accum': {'negative': "#8304B9", 'positive': "#26FF00"},
    'dark_lit': {'lit': "#4488FF", 'neutral': "#888888", 'dark': "#FF4444"},
    'finra_buy': {},
}


# ---------------------------------------------------------------------------
# Database functions
# ---------------------------------------------------------------------------

def connect_db(db_path: Path) -> Tuple[duckdb.DuckDBPyConnection, str]:
    """Connect to DuckDB database in read-only mode."""
    db_path = Path(db_path)
    return duckdb.connect(database=str(db_path), read_only=True), 'duckdb'


def get_tables(conn, db_type: str) -> List[str]:
    """Get list of tables in database."""
    if db_type == 'duckdb':
        return [r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]


def get_columns(conn, db_type: str, table: str) -> List[str]:
    """Get column names for a table."""
    if db_type == 'duckdb':
        return [r[0] for r in conn.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}'"
        ).fetchall()]
    return [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')").fetchall()]


def pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Find first matching column from candidates (case-insensitive)."""
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def quote_ident(name: str) -> str:
    """Quote identifier for SQL."""
    return f'"{name}"'


# ---------------------------------------------------------------------------
# Ticker universe functions
# ---------------------------------------------------------------------------

def _load_ticker_dictionary() -> Any:
    """Load ticker_dictionary.py from known locations."""
    candidates = [
        Path.cwd() / "ticker_dictionary.py",
        Path.cwd() / "Special_tools" / "ticker_dictionary.py",
        Path.cwd() / "darkpool_analysis" / "ticker_dictionary.py",
        Path.cwd().parent / "darkpool_analysis" / "ticker_dictionary.py",
        Path(__file__).parent / "ticker_dictionary.py",
        Path(__file__).parent.parent / "Special_tools" / "ticker_dictionary.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("ticker_dictionary", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    raise FileNotFoundError("ticker_dictionary.py not found")


def _normalize_ticker_types(value: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize ticker type input to list of uppercase strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).upper() for v in value]
    return [str(value).upper()]


def build_ticker_universe(ticker_types: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Build ordered ticker universe from ticker_dictionary.py.

    Returns:
        Tuple of (ordered_tickers, category_map)
    """
    ticker_dict = _load_ticker_dictionary()

    # Map ticker types to dictionary attribute names
    type_to_attr = {
        "GLOBAL": "GLOBAL_MACRO_TICKERS",
        "MAG8": "MAG8_TICKERS",
        "THEMATIC": "THEMATIC_SECTORS_TICKERS",
        "SECTOR": "SECTOR_CORE_TICKERS",
        "COMMODITIES": "COMMODITIES_TICKERS",
        "RATES": "RATES_CREDIT_TICKERS",
        "SPECULATIVE": "SPECULATIVE_TICKERS",
        "CRYPTO": "CRYPTO_TICKERS",
    }

    type_to_category = {
        "GLOBAL": "GLOBAL_MACRO",
        "MAG8": "MAG8",
        "THEMATIC": "THEMATIC_SECTORS",
        "SECTOR": "SECTOR_CORE",
        "COMMODITIES": "COMMODITIES",
        "RATES": "RATES_CREDIT",
        "SPECULATIVE": "SPECULATIVE",
        "CRYPTO": "CRYPTO",
    }

    normalized_types = _normalize_ticker_types(ticker_types)
    if "ALL" in normalized_types:
        normalized_types = list(type_to_attr.keys())

    seen = set()
    ordered = []
    categories = {}

    for ticker_type in normalized_types:
        attr_name = type_to_attr.get(ticker_type)
        if not attr_name:
            continue
        tickers = getattr(ticker_dict, attr_name, [])
        category = type_to_category.get(ticker_type, "UNKNOWN")
        for t in tickers:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
                categories[t] = category

    return ordered, categories


# ---------------------------------------------------------------------------
# Geometry and color functions
# ---------------------------------------------------------------------------

def blend_color(c1, c2, t: float = 0.5):
    """Linear interpolation between two colors."""
    a = np.array(to_rgba(c1))
    b = np.array(to_rgba(c2))
    return a * (1 - t) + b * t


def soften_color(color, amount: float, base=BG_COLOR):
    """Reduce saturation by blending toward background."""
    return blend_color(color, base, max(0.0, min(1.0, amount)))


def darken_color(color, factor: float):
    """Darken a color by factor (0=black, 1=original)."""
    rgba = np.array(to_rgba(color))
    rgba[:3] = rgba[:3] * max(0.0, min(1.0, factor))
    return rgba


def arc_points(a0: float, a1: float, r: float, n: int = 12) -> np.ndarray:
    """Generate points along an arc."""
    angles = np.linspace(a0, a1, n)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])


def bezier_curve(p0, p1, p2, n: int = 50) -> np.ndarray:
    """Generate quadratic Bezier curve points."""
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def add_gradient_curve(ax, points, color_start, color_end, lw: float, alpha: float):
    """Draw a gradient-colored curve."""
    if len(points) < 2:
        return
    segments = np.stack([points[:-1], points[1:]], axis=1)
    c0 = np.array(to_rgba(color_start))
    c1 = np.array(to_rgba(color_end))
    t = np.linspace(0, 1, len(segments))[:, None]
    colors = c0 * (1 - t) + c1 * t
    colors[:, 3] = colors[:, 3] * alpha
    lc = LineCollection(segments, colors=colors, linewidths=lw, capstyle='round')
    ax.add_collection(lc)


def ribbon_patch(a0, a1, b0, b1, r, color_start, color_end, alpha, arc_points_n=16):
    """Create a chord ribbon patch."""
    arc1 = arc_points(a0, a1, r, n=arc_points_n)
    arc2 = arc_points(b0, b1, r, n=arc_points_n)
    curve1 = bezier_curve(arc1[-1], np.array([0.0, 0.0]), arc2[0], n=24)
    curve2 = bezier_curve(arc2[-1], np.array([0.0, 0.0]), arc1[0], n=24)
    poly = np.vstack([arc1, curve1, arc2, curve2])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(poly) - 1)
    path = MplPath(poly, codes)
    mid = blend_color(color_start, color_end, 0.5)
    return PathPatch(path, facecolor=mid, edgecolor='none', alpha=alpha)


def gradient_fill_collection(a0, a1, b0, b1, r, color_start, color_end, alpha, steps=18):
    """Create gradient-filled polygon collection for ribbon."""
    arc1 = arc_points(a0, a1, r, n=4)
    arc2 = arc_points(b0, b1, r, n=4)
    p_a0, p_a1 = arc1[0], arc1[-1]
    p_b0, p_b1 = arc2[0], arc2[-1]
    left = bezier_curve(p_a0, np.array([0.0, 0.0]), p_b0, n=steps + 1)
    right = bezier_curve(p_a1, np.array([0.0, 0.0]), p_b1, n=steps + 1)
    polys = []
    colors = []
    for i in range(steps):
        quad = np.vstack([left[i], left[i + 1], right[i + 1], right[i]])
        t = (i + 0.5) / steps
        color = blend_color(color_start, color_end, t)
        color[3] = color[3] * alpha
        polys.append(quad)
        colors.append(color)
    return PolyCollection(polys, facecolors=colors, edgecolors='none')


def draw_ribbon(
    ax,
    a0, a1, b0, b1, r,
    color_start, color_end,
    fill_alpha, line_alpha, lw,
    use_gradient_fill=True,
    gradient_steps=18,
):
    """Draw a chord ribbon with optional gradient fill."""
    if use_gradient_fill:
        clip = ribbon_patch(a0, a1, b0, b1, r, color_start, color_end, alpha=1.0)
        clip.set_facecolor('none')
        clip.set_edgecolor('none')
        ax.add_patch(clip)
        fill = gradient_fill_collection(a0, a1, b0, b1, r, color_start, color_end, fill_alpha, steps=gradient_steps)
        fill.set_clip_path(clip)
        ax.add_collection(fill)
    else:
        patch = ribbon_patch(a0, a1, b0, b1, r, color_start, color_end, alpha=fill_alpha)
        ax.add_patch(patch)

    mid_a = (a0 + a1) / 2
    mid_b = (b0 + b1) / 2
    p0 = np.array([r * np.cos(mid_a), r * np.sin(mid_a)])
    p2 = np.array([r * np.cos(mid_b), r * np.sin(mid_b)])
    curve = bezier_curve(p0, np.array([0.0, 0.0]), p2)
    add_gradient_curve(ax, curve, color_start, color_end, lw=lw, alpha=line_alpha)


# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------

def parse_accum(value: Any) -> float:
    """Parse accumulation score value to float."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        if np.isnan(value):
            return np.nan
        return float(value)
    s = str(value).strip()
    if not s:
        return np.nan
    s = s.replace('%', '').replace(',', '')
    m = re.search(r'[-+]?\d+\.?\d*', s)
    return float(m.group(0)) if m else np.nan


def load_daily_metrics(
    conn,
    db_type: str,
    ticker_list: List[str],
    end_date: datetime,
    lookback_days: int,
) -> pd.DataFrame:
    """Load daily_metrics data for given tickers and date range."""
    start_date = end_date - timedelta(days=lookback_days + 30)

    placeholders = ','.join(['?'] * len(ticker_list))
    query = f"""
        SELECT
            UPPER(symbol) AS ticker,
            CAST(date AS DATE) AS date,
            accumulation_score,
            lit_buy_volume,
            lit_sell_volume,
            finra_buy_volume,
            short_ratio,
            vw_flow,
            vw_flow_z,
            lit_buy_ratio
        FROM daily_metrics
        WHERE UPPER(symbol) IN ({placeholders})
          AND date >= ?
          AND date <= ?
        ORDER BY date
    """
    params = ticker_list + [start_date.date(), end_date.date()]

    if db_type == 'duckdb':
        df = conn.execute(query, params).df()
    else:
        df = pd.read_sql_query(query, conn, params=params)

    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df['accumulation_score'] = df['accumulation_score'].apply(parse_accum)

    # Compute lit_total
    if 'lit_buy_volume' in df.columns and 'lit_sell_volume' in df.columns:
        df['lit_total'] = df['lit_buy_volume'].fillna(0) + df['lit_sell_volume'].fillna(0)
    else:
        df['lit_total'] = 0

    return df


def compute_stats(df: pd.DataFrame, ticker_list: List[str], value_col: str, prefix: str) -> pd.DataFrame:
    """Compute mean and std statistics per ticker."""
    if df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=['ticker', f'{prefix}_mean', f'{prefix}_std'])

    stats = df.groupby('ticker')[value_col].agg(['mean', 'std']).reset_index()
    stats.columns = ['ticker', f'{prefix}_mean', f'{prefix}_std']
    return stats


def tail_n_days(df: pd.DataFrame, ticker: str, n: int, end_date) -> pd.DataFrame:
    """Get last N trading days for a ticker up to end_date."""
    df_ticker = df[df['ticker'] == ticker].copy()
    df_ticker = df_ticker[df_ticker['date'] <= end_date].sort_values('date')
    return df_ticker.tail(n)


def make_time_bins(dates: List, bins: int) -> List[List]:
    """Split dates into N bins for time-series ring representation."""
    if not dates:
        return []
    if bins is None or bins <= 0:
        return [dates]
    bins = min(bins, len(dates))
    split = np.array_split(dates, bins)
    return [list(s) for s in split if len(s)]


# ---------------------------------------------------------------------------
# Edge computation functions
# ---------------------------------------------------------------------------

def build_edges_from_value(
    df: pd.DataFrame,
    value_col: str,
    top_k_winners: int,
    top_k_losers: int,
    distribution_mode: str,
    min_edge_flow: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    """Build edges from signed value column (positive=winners, negative=losers)."""
    df = df[['ticker', value_col]].dropna().copy()
    winners = df[df[value_col] > 0].nlargest(top_k_winners, value_col)
    losers = df[df[value_col] < 0].copy()
    losers['supply'] = -losers[value_col]
    losers = losers.nlargest(top_k_losers, 'supply')

    edges = []
    total_demand = winners[value_col].sum() if not winners.empty else 0.0
    total_supply = losers['supply'].sum() if not losers.empty else 0.0

    if winners.empty or losers.empty:
        return pd.DataFrame(edges), winners, losers, total_demand, total_supply

    if distribution_mode == 'equal':
        for _, loser in losers.iterrows():
            flow_each = loser['supply'] / len(winners)
            for _, winner in winners.iterrows():
                if flow_each >= min_edge_flow:
                    edges.append({
                        'source': loser['ticker'],
                        'dest': winner['ticker'],
                        'flow': float(flow_each),
                    })
    else:  # demand_weighted
        if total_demand > 0:
            for _, loser in losers.iterrows():
                for _, winner in winners.iterrows():
                    flow = loser['supply'] * (winner[value_col] / total_demand)
                    if flow >= min_edge_flow:
                        edges.append({
                            'source': loser['ticker'],
                            'dest': winner['ticker'],
                            'flow': float(flow),
                        })

    edges_df = pd.DataFrame(edges)
    if not edges_df.empty:
        edges_df = edges_df.sort_values('flow', ascending=False).reset_index(drop=True)
    return edges_df, winners, losers, total_demand, total_supply


def build_edges_from_positive_value(
    df: pd.DataFrame,
    value_col: str,
    top_k_high: int,
    top_k_low: int,
    min_edge_flow: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    """Build edges for metrics that are always positive (like finra_buy_volume).
    Flow goes from low-value tickers to high-value tickers."""
    df = df[['ticker', value_col]].dropna().copy()
    df = df[df[value_col] > 0]  # Only positive values
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

    median_val = df[value_col].median()
    high_tickers = df[df[value_col] >= median_val].nlargest(top_k_high, value_col)
    low_tickers = df[df[value_col] < median_val].nsmallest(top_k_low, value_col)

    if high_tickers.empty or low_tickers.empty:
        return pd.DataFrame(), high_tickers, low_tickers, 0.0, 0.0

    total_high = high_tickers[value_col].sum()
    total_low = low_tickers[value_col].sum()

    edges = []
    for _, low_row in low_tickers.iterrows():
        for _, high_row in high_tickers.iterrows():
            flow = (high_row[value_col] - low_row[value_col]) * (low_row[value_col] / total_low) if total_low > 0 else 0
            if flow > min_edge_flow:
                edges.append({
                    'source': low_row['ticker'],
                    'dest': high_row['ticker'],
                    'flow': float(flow),
                })

    edges_df = pd.DataFrame(edges)
    if not edges_df.empty:
        edges_df = edges_df.sort_values('flow', ascending=False).reset_index(drop=True)
    return edges_df, high_tickers, low_tickers, total_high, total_low


def filter_edges(edges_df: pd.DataFrame, max_edges: int) -> pd.DataFrame:
    """Filter to top N edges by flow."""
    if edges_df is None or edges_df.empty:
        return edges_df
    if max_edges and max_edges > 0:
        return edges_df.nlargest(max_edges, 'flow')
    return edges_df


def expand_edges(edges_df: pd.DataFrame, splits: int, edge_ribbon_max: int) -> pd.DataFrame:
    """Multiply edges based on flow magnitude for visual density."""
    if edges_df is None or edges_df.empty:
        return edges_df
    if not splits or splits <= 1:
        return edges_df
    max_flow = edges_df['flow'].max() if 'flow' in edges_df.columns and not edges_df.empty else 0.0
    if max_flow <= 0:
        return edges_df
    rows = []
    for row in edges_df.itertuples():
        date_val = getattr(row, 'date', None)
        n = max(1, int(round(splits * (row.flow / max_flow))))
        if edge_ribbon_max and edge_ribbon_max > 0:
            n = min(n, edge_ribbon_max)
        flow = row.flow / n
        for _ in range(n):
            rows.append({'source': row.source, 'dest': row.dest, 'flow': flow, 'date': date_val})
    return pd.DataFrame(rows)


def compute_metric_totals(edges_df: pd.DataFrame) -> Dict[str, float]:
    """Compute total flow per ticker from edges."""
    totals = {}
    if edges_df is None or edges_df.empty:
        return totals
    for row in edges_df.itertuples():
        totals[row.source] = totals.get(row.source, 0.0) + row.flow
        totals[row.dest] = totals.get(row.dest, 0.0) + row.flow
    return totals


# ---------------------------------------------------------------------------
# Layout functions
# ---------------------------------------------------------------------------

def compute_ticker_angles(
    ticker_order: List[str],
    categories: Dict[str, str],
    category_gap_deg: float,
    chord_arc_fraction: float,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, List[str]]]:
    """
    Compute angular positions for each ticker around the circle.

    Returns:
        Tuple of (spans dict {ticker: (start, end)}, grouped dict {category: [tickers]})
    """
    # Group tickers by category
    grouped = {}
    for t in ticker_order:
        cat = categories.get(t, 'UNKNOWN')
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(t)

    n_categories = len(grouped)
    n_tickers = len(ticker_order)

    if n_tickers == 0:
        return {}, grouped

    # Total arc available
    total_arc = 2 * math.pi * chord_arc_fraction

    # Subtract category gaps
    total_gap = n_categories * math.radians(category_gap_deg)
    available_arc = total_arc - total_gap

    # Arc per ticker
    arc_per_ticker = available_arc / n_tickers if n_tickers > 0 else 0

    spans = {}
    current_angle = 0

    for cat in grouped:
        for t in grouped[cat]:
            spans[t] = (current_angle, current_angle + arc_per_ticker)
            current_angle += arc_per_ticker
        current_angle += math.radians(category_gap_deg)

    return spans, grouped


def metric_visible(metric_key: str, config: CircosConfig) -> bool:
    """Determine if a metric should be displayed based on configuration."""
    if metric_key == "accum":
        return config.show_accum_flow
    if metric_key == "short":
        return config.show_short_net_flow
    if metric_key == "lit":
        return config.show_lit_flow
    if metric_key == "finra_buy":
        return config.show_finra_flow
    if metric_key == "vwbr_z":
        return config.show_vwbr_z
    return True


# ---------------------------------------------------------------------------
# Score computation for summary table
# ---------------------------------------------------------------------------

def compute_score_maps(
    flow_map: Dict[str, float],
    trend_map: Dict[str, float],
    anomaly_map: Dict[str, float],
    ticker_order: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Combine flow, trend, and anomaly metrics into buy/sell scores."""
    if flow_map is None:
        return {}, {}
    flow_vals = [abs(v) for v in flow_map.values() if v is not None and np.isfinite(v)]
    flow_scale = np.std(flow_vals) if len(flow_vals) > 1 else 1.0
    if flow_scale == 0:
        flow_scale = 1.0
    trend_vals = [v for v in trend_map.values() if v is not None and np.isfinite(v)]
    trend_scale = np.std(trend_vals) if len(trend_vals) > 1 else 1.0
    if trend_scale == 0:
        trend_scale = 1.0
    buy_scores = {}
    sell_scores = {}
    for ticker in ticker_order:
        flow = flow_map.get(ticker)
        if flow is None or not np.isfinite(flow) or flow == 0:
            continue
        direction = 1 if flow > 0 else -1
        flow_norm = abs(flow) / flow_scale
        trend_norm = (trend_map.get(ticker, 0.0) / trend_scale) if trend_scale else 0.0
        anomaly = anomaly_map.get(ticker, 0.0)
        base = flow_norm + anomaly
        if direction > 0:
            buy_scores[ticker] = base + max(trend_norm, 0.0)
        else:
            sell_scores[ticker] = base + max(-trend_norm, 0.0)
    return buy_scores, sell_scores


def top_tickers_from_scores(score_map: Dict[str, float], k: int = 3) -> str:
    """Get top k tickers from score map as comma-separated string."""
    if not score_map:
        return 'n/a'
    items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return ', '.join([t for t, _ in items[:k]]) or 'n/a'


def top_tickers(net_map: Dict[str, float], positive: bool = True, k: int = 3) -> str:
    """Get top k tickers from net flow map."""
    if not net_map:
        return 'n/a'
    items = [(t, v) for t, v in net_map.items() if v is not None and np.isfinite(v)]
    if positive:
        items = sorted([x for x in items if x[1] > 0], key=lambda x: x[1], reverse=True)
    else:
        items = sorted([x for x in items if x[1] < 0], key=lambda x: x[1])
    return ', '.join([t for t, _ in items[:k]]) or 'n/a'


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_circos(
    config: Optional[CircosConfig] = None,
    output_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Render circos-style chord diagram for institutional money flow.

    Args:
        config: CircosConfig instance (uses defaults if None)
        output_dir: Output directory for PNG (uses default if None)
        db_path: Database path (auto-detects from config if None)

    Returns:
        Path to saved PNG file, or None if rendering failed
    """
    if config is None:
        config = CircosConfig()

    logger.info("Starting circos render with config: ticker_type=%s, flow_period_days=%d",
                 config.ticker_type, config.flow_period_days)

    # Resolve database path
    if db_path is None:
        try:
            from .config import load_config
            main_config = load_config()
            db_path = main_config.db_path
        except ImportError:
            try:
                from config import load_config
                main_config = load_config()
                db_path = main_config.db_path
            except ImportError:
                try:
                    from .db_path import get_db_path
                    db_path = get_db_path()
                except ImportError:
                    from db_path import get_db_path
                    db_path = get_db_path()

    # Resolve output directory
    if output_dir is None:
        # Default to darkpool_analysis/output/circos_plot relative to this file
        output_dir = Path(__file__).parent / "output" / "circos_plot"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build ticker universe
    try:
        ticker_order, categories = build_ticker_universe(config.ticker_type)
    except FileNotFoundError as e:
        logger.error("Failed to load ticker dictionary: %s", e)
        return None

    if not ticker_order:
        logger.warning("No tickers found for types: %s", config.ticker_type)
        return None

    logger.info("Ticker universe: %d tickers from %d categories",
                 len(ticker_order), len(set(categories.values())))

    # Connect to database
    try:
        conn, db_type = connect_db(db_path)
    except Exception as e:
        logger.error("Failed to connect to database: %s", e)
        return None

    try:
        # Determine end date
        if config.end_date:
            end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
        else:
            result = conn.execute("SELECT MAX(date) FROM daily_metrics").fetchone()
            if result and result[0]:
                end_date = pd.to_datetime(result[0])
            else:
                end_date = datetime.now()

        end_date_dt = end_date if isinstance(end_date, datetime) else datetime.combine(end_date, datetime.min.time())
        start_date = end_date_dt - timedelta(days=config.flow_period_days + 10)
        lookback_start = end_date_dt - timedelta(days=config.lookback_days + 30)

        logger.info("Date range: %s to %s", start_date.date(), end_date_dt.date())

        # Load daily metrics
        df = load_daily_metrics(conn, db_type, ticker_order, end_date_dt, config.flow_period_days + config.lookback_days)

        if df.empty:
            logger.warning("No data found for tickers in date range")
            return None

        # Get window dates (last N trading days)
        all_dates = sorted(df['date'].unique())
        window_dates = [d for d in all_dates if d >= start_date.date() and d <= end_date_dt.date()]
        window_dates = sorted(window_dates)[-config.flow_period_days:]

        if not window_dates:
            logger.warning("No window dates found")
            return None

        # Create time bins for rings
        time_bins = make_time_bins(window_dates, config.time_slice_bins)
        time_bins = list(reversed(time_bins))  # Most recent first

        # Compute 20-day statistics for ring sizing
        accum_stats = compute_stats(df, ticker_order, 'accumulation_score', 'accum')
        lit_stats = compute_stats(df, ticker_order, 'lit_total', 'lit_vol')
        finra_stats = compute_stats(df, ticker_order, 'finra_buy_volume', 'finra_buy')

        # Compute accumulation delta for each ticker over the period
        accum_delta = {}
        lit_delta = {}
        vwbr_z_map = {}

        for ticker in ticker_order:
            df_t = tail_n_days(df, ticker, config.flow_period_days, end_date_dt.date())
            if len(df_t) >= 2:
                accum_delta[ticker] = df_t['accumulation_score'].iloc[-1] - df_t['accumulation_score'].iloc[0]
                if 'lit_buy_ratio' in df_t.columns:
                    lit_delta[ticker] = (df_t['lit_buy_ratio'].iloc[-1] - 0.5) * 100
            elif len(df_t) == 1:
                accum_delta[ticker] = df_t['accumulation_score'].iloc[0] - 50
                if 'lit_buy_ratio' in df_t.columns:
                    lit_delta[ticker] = (df_t['lit_buy_ratio'].iloc[0] - 0.5) * 100

            # Get latest vw_flow_z
            if 'vw_flow_z' in df_t.columns and len(df_t) > 0:
                vwbr_z_map[ticker] = df_t['vw_flow_z'].iloc[-1]

        # Build delta dataframes
        delta_df = pd.DataFrame([
            {'ticker': t, 'accum_delta': v}
            for t, v in accum_delta.items() if not np.isnan(v)
        ])

        vwbr_z_df = pd.DataFrame([
            {'ticker': t, 'vwbr_z': v}
            for t, v in vwbr_z_map.items() if v is not None and np.isfinite(v)
        ])

        if delta_df.empty and vwbr_z_df.empty:
            logger.warning("No valid data for edge computation")
            return None

        # Build edges for visible metrics
        all_edges = {}

        # VWBR Z edges (primary visible metric)
        if config.show_vwbr_z and not vwbr_z_df.empty:
            edges_df, _, _, _, _ = build_edges_from_value(
                vwbr_z_df, 'vwbr_z',
                config.top_k_winners, config.top_k_losers,
                config.distribution_mode, config.min_edge_flow
            )
            all_edges['vwbr_z'] = filter_edges(edges_df, config.max_edges_per_metric)

        # Accumulation edges
        if config.show_accum_flow and not delta_df.empty:
            edges_df, _, _, _, _ = build_edges_from_value(
                delta_df, 'accum_delta',
                config.top_k_winners, config.top_k_losers,
                config.distribution_mode, config.min_edge_flow
            )
            all_edges['accum'] = filter_edges(edges_df, config.max_edges_per_metric)

        # Check if we have any edges
        has_edges = any(e is not None and not e.empty for e in all_edges.values())
        if not has_edges:
            logger.warning("No edges computed from data")
            return None

        logger.info("Computed edges for metrics: %s", list(all_edges.keys()))

        # Compute layout
        spans, grouped = compute_ticker_angles(
            ticker_order, categories,
            config.category_gap_deg, config.chord_arc_fraction
        )

        # Create figure
        fig = plt.figure(figsize=config.figure_size, facecolor=BG_COLOR)
        ax = fig.add_axes([
            (1 - config.main_ax_size) / 2,
            (1 - config.main_ax_size) / 2,
            config.main_ax_size,
            config.main_ax_size
        ])
        ax.set_facecolor(BG_COLOR)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.axis('off')
        ax.set_aspect('equal')

        # Draw circular reference line at r=1.0
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color='#39424e', linewidth=1.0, alpha=0.6)

        # Draw ticker labels (inside the rings, at ~0.876)
        label_r = config.chord_radius + (config.ticker_outer - config.chord_radius) * 0.4
        for ticker in ticker_order:
            a0, a1 = spans.get(ticker, (0, 0))
            mid_angle = (a0 + a1) / 2
            x = label_r * math.cos(mid_angle)
            y = label_r * math.sin(mid_angle)
            rotation = math.degrees(mid_angle)
            if 90 < rotation < 270:
                rotation += 180
            ax.text(x, y, ticker, fontsize=config.ticker_fontsize, color='white', ha='center', va='center',
                    rotation=rotation, rotation_mode='anchor')

        # Draw rings (time-series bars)
        if config.show_volume_ring:
            # Build ring data
            ring_bin_data = {m: {} for m in RING_METRIC_ORDER}
            ring_max_mag = {}

            for m in RING_METRIC_ORDER:
                max_val = 0.0
                for t in ticker_order:
                    data_by_bin = []
                    for bin_dates in time_bins:
                        mask = (df['ticker'] == t) & (df['date'].isin(bin_dates))

                        if m == 'accum':
                            val = float(df.loc[mask, 'accumulation_score'].mean()) if mask.any() else 50.0
                            data_by_bin.append({'value': val, 'extra': 0.0})
                            max_val = max(max_val, abs(val - 50))
                        elif m == 'dark_lit':
                            lit_vol = float(df.loc[mask, 'lit_total'].sum()) if mask.any() else 0.0
                            lit_ratio = float(df.loc[mask, 'lit_buy_ratio'].mean()) if mask.any() and 'lit_buy_ratio' in df.columns else 0.5
                            data_by_bin.append({'value': lit_vol, 'extra': lit_ratio})
                            max_val = max(max_val, lit_vol)
                        elif m == 'finra_buy':
                            val = float(df.loc[mask, 'finra_buy_volume'].sum()) if mask.any() else 0.0
                            vwbr_z_val = float(df.loc[mask, 'vw_flow_z'].mean()) if mask.any() and 'vw_flow_z' in df.columns else 0.0
                            data_by_bin.append({'value': val, 'extra': {'vwbr_z': vwbr_z_val}})
                            max_val = max(max_val, val)
                        else:
                            data_by_bin.append({'value': 0.0, 'extra': 0.0})

                    ring_bin_data[m][t] = data_by_bin
                ring_max_mag[m] = max_val if max_val > 0 else 1.0

            # Draw rings
            track_span = config.ring_base_thickness + config.ring_thickness_scale + config.ring_gap
            for idx, m in enumerate(RING_METRIC_ORDER):
                inner_base = config.ticker_outer + 0.02 + idx * track_span
                max_mag = ring_max_mag.get(m, 1.0)

                # Get stats for z-score sizing
                if m == 'accum':
                    stats_df = accum_stats
                    mult = config.ring1_thickness_mult
                elif m == 'dark_lit':
                    stats_df = lit_stats
                    mult = config.ring2_thickness_mult
                else:
                    stats_df = finra_stats
                    mult = config.ring3_thickness_mult

                for t, (a0, a1) in spans.items():
                    bin_data = ring_bin_data[m].get(t, [])
                    if not bin_data:
                        continue
                    n_bins = len(bin_data)
                    arc_len = a1 - a0
                    slice_gap = arc_len * 0.02 / max(1, n_bins)
                    slice_len = (arc_len - slice_gap * (n_bins - 1)) / n_bins if n_bins > 0 else arc_len
                    cursor = a0

                    for bd in bin_data:
                        val = bd['value']
                        extra = bd['extra']

                        # Compute thickness based on z-score deviation
                        scale = config.ring_thickness_scale * mult
                        if not stats_df.empty and t in stats_df['ticker'].values:
                            row = stats_df[stats_df['ticker'] == t].iloc[0]
                            if m == 'accum':
                                mean_val = row.get('accum_mean', 50)
                                std_val = row.get('accum_std', 1)
                            elif m == 'dark_lit':
                                mean_val = row.get('lit_vol_mean', 0)
                                std_val = row.get('lit_vol_std', 1)
                            else:
                                mean_val = row.get('finra_buy_mean', 0)
                                std_val = row.get('finra_buy_std', 1)

                            if std_val and std_val > 0:
                                z = abs(val - mean_val) / std_val
                                thickness = config.ring_base_thickness + scale * min(z / 2.0, 1.0)
                            else:
                                thickness = config.ring_base_thickness
                        else:
                            # Fallback when no stats available (matches notebook)
                            if m == 'accum':
                                # Dynamic based on deviation from 50
                                deviation = abs(val - 50)
                                thickness = config.ring_base_thickness + scale * (deviation / 50.0)
                            else:
                                # dark_lit and finra_buy: minimal thickness
                                thickness = config.ring_base_thickness

                        # Compute color
                        if m == 'accum':
                            if val >= 70:
                                color = RING_COLORS['accum']['positive']
                            elif val <= 30:
                                color = RING_COLORS['accum']['negative']
                            else:
                                t_blend = (val - 30) / 40.0
                                color = blend_color(RING_COLORS['accum']['negative'], RING_COLORS['accum']['positive'], t_blend)
                        elif m == 'dark_lit':
                            lit_ratio = extra if isinstance(extra, (int, float)) else 0.5
                            normalized = (lit_ratio - 0.4) / 0.2
                            normalized = max(0.0, min(1.0, normalized))
                            if normalized < 0.5:
                                color = blend_color('#FF6666', '#888888', normalized * 2)
                            else:
                                color = blend_color('#888888', '#66FF66', (normalized - 0.5) * 2)
                        else:  # finra_buy
                            extra_vals = extra if isinstance(extra, dict) else {}
                            vwbr_z_val = extra_vals.get('vwbr_z', 0.0)
                            if vwbr_z_val is None or not np.isfinite(vwbr_z_val):
                                vwbr_z_val = 0.0
                            normalized = 0.5 + (vwbr_z_val / config.ring3_zscore_span)
                            normalized = max(0.0, min(1.0, normalized))
                            cat_color = CATEGORY_PALETTE.get(categories.get(t, 'UNKNOWN'), '#A0A0A0')
                            brightness = config.ring3_brightness_min + (config.ring3_brightness_max - config.ring3_brightness_min) * normalized
                            color = darken_color(cat_color, brightness)

                        wedge = Wedge(
                            (0, 0), inner_base + thickness, math.degrees(cursor), math.degrees(cursor + slice_len),
                            width=thickness,
                            facecolor=color, edgecolor='none', alpha=0.85,
                        )
                        ax.add_patch(wedge)
                        cursor += slice_len + slice_gap

        # Draw chord ribbons
        chord_r = config.chord_radius
        for metric_key, edges_df in all_edges.items():
            if edges_df is None or edges_df.empty:
                continue

            metric_colors = METRIC_COLORS.get(metric_key, {'sell': '#888', 'buy': '#888'})
            color_sell = metric_colors.get('sell', metric_colors.get('low', '#888'))
            color_buy = metric_colors.get('buy', metric_colors.get('high', '#888'))

            for _, row in edges_df.iterrows():
                src, dst = row['source'], row['dest']
                if src not in spans or dst not in spans:
                    continue

                src_a0, src_a1 = spans[src]
                dst_a0, dst_a1 = spans[dst]

                a0 = src_a0 + (src_a1 - src_a0) * 0.3
                a1 = src_a0 + (src_a1 - src_a0) * 0.7
                b0 = dst_a0 + (dst_a1 - dst_a0) * 0.3
                b1 = dst_a0 + (dst_a1 - dst_a0) * 0.7

                color_start = soften_color(color_sell, config.chord_color_soften)
                color_end = soften_color(color_buy, config.chord_color_soften)

                draw_ribbon(
                    ax, a0, a1, b0, b1, chord_r,
                    color_start, color_end,
                    config.chord_fill_alpha, config.chord_line_alpha,
                    lw=1.0,
                    use_gradient_fill=config.use_gradient_fill,
                    gradient_steps=config.chord_gradient_steps,
                )

        # Draw title
        title_text = "Institutional Money Flow Tracker"
        ax.text(0.5, 1.02, title_text, fontsize=config.title_fontsize, color='white', ha='center', va='bottom',
                transform=ax.transAxes, fontweight='bold')

        date_text = f"Date Range: {window_dates[0].strftime('%Y-%m-%d')} -> {window_dates[-1].strftime('%Y-%m-%d')}"
        ax.text(0.5, 0.98, date_text, fontsize=config.subtitle_fontsize, color='#888888', ha='center', va='bottom',
                transform=ax.transAxes)

        # Draw chords legend (upper-left)
        leg = fig.add_axes([0.08, 0.75, 0.30, 0.18])
        leg.axis('off')
        leg.set_facecolor('none')
        leg.set_xlim(0, 1)
        leg.set_ylim(0, 1)

        y = 0.95
        leg.text(0.0, y, 'Chords', color='white', fontsize=config.legend_title_fontsize, fontweight='bold', va='top')
        y -= 0.15

        chord_items = [
            ('vwbr_z', METRIC_LABELS.get('vwbr_z', 'VWBR Z'), METRIC_COLORS['vwbr_z']['sell'], METRIC_COLORS['vwbr_z']['buy']),
        ]
        for key, label, c_start, c_end in chord_items:
            if not metric_visible(key, config):
                continue
            xs = np.linspace(0.0, 0.20, 30)
            points = np.column_stack([xs, np.full_like(xs, y)])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            c0 = np.array(to_rgba(c_start))
            c1 = np.array(to_rgba(c_end))
            t_arr = np.linspace(0, 1, len(segments))[:, None]
            colors = c0 * (1 - t_arr) + c1 * t_arr
            leg.add_collection(LineCollection(segments, colors=colors, linewidths=config.legend_linewidth))
            leg.text(0.24, y, label, color='white', fontsize=config.legend_label_fontsize, va='center')
            y -= 0.15

        # Draw category legend (upper-right)
        present_categories = [cat for cat in ['GLOBAL_MACRO', 'MAG8', 'THEMATIC_SECTORS', 'SECTOR_CORE', 'COMMODITIES', 'RATES_CREDIT', 'SPECULATIVE', 'CRYPTO']
                             if grouped.get(cat)]

        if present_categories:
            cat_leg = fig.add_axes([0.85, 0.75, 0.18, 0.18])
            cat_leg.axis('off')
            cat_leg.set_facecolor('none')
            cat_leg.set_xlim(0, 1)
            cat_leg.set_ylim(0, 1)

            y = 0.95
            cat_leg.text(0.0, y, 'Categories', color='white', fontsize=config.legend_title_fontsize, fontweight='bold', va='top')
            y -= 0.12

            for cat in present_categories:
                base_color = CATEGORY_PALETTE.get(cat, '#A0A0A0')
                dark_color = darken_color(base_color, 0.2)

                for i in range(10):
                    t = i / 9.0
                    color = blend_color(dark_color, base_color, t)
                    rect = Rectangle((i * 0.008, y - 0.04), 0.008, 0.08, facecolor=color, edgecolor='none')
                    cat_leg.add_patch(rect)

                cat_leg.text(0.12, y, CATEGORY_LABELS.get(cat, cat), color='white', fontsize=config.legend_label_fontsize, va='center')
                y -= 0.16

        # Draw ring legend (bottom-left)
        ring_leg = fig.add_axes([0.08, 0.00, 0.6, 0.22])
        ring_leg.axis('off')
        ring_leg.set_facecolor('none')
        ring_leg.set_xlim(0, 1)
        ring_leg.set_ylim(0, 1)

        y = 0.98
        ring_leg.text(0.0, y, 'Ring Sizing - how unusual is today', color='white', fontsize=config.ring_title_fontsize, fontweight='bold', va='top')
        y -= 0.12

        sizing_items = [
            ('Inner Ring 1', 'Accumulation score vs. 20-day average'),
            ('Center Ring 2', 'Lit total volume (buy + sell) vs. 20-day average'),
            ('Outer Ring 3', 'FINRA buy volume vs. 20-day average'),
        ]
        for ring, desc in sizing_items:
            ring_leg.text(0.02, y, f'{ring}:', color='#888888', fontsize=config.ring_label_fontsize, va='center')
            ring_leg.text(0.14, y, desc, color='#C9D1D9', fontsize=config.ring_label_fontsize, va='center')
            y -= 0.09

        ring_leg.text(0.02, y, 'Each bar = 1 day', color='#C9D1D9', fontsize=config.ring_label_fontsize, va='center')
        y -= 0.13

        ring_leg.text(0.0, y, 'Ring Coloring - what is the direction of the flow today', color='white', fontsize=config.ring_title_fontsize, fontweight='bold', va='top')
        y -= 0.12

        coloring_items = [
            ('Ring 1', 'Acc score 70 -> 30', RING_COLORS['accum']['positive'], RING_COLORS['accum']['negative']),
            ('Ring 2', 'Lit buy ratio (buy -> sell)', '#66FF66', '#FF6666'),
            ('Ring 3', 'VWBR Z (category tint bright->dark)', '#E6E6E6', '#444444'),
        ]
        for ring, desc, c_start, c_end in coloring_items:
            xs = np.linspace(0.0, 0.10, 30)
            points = np.column_stack([xs, np.full_like(xs, y)])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            c0 = np.array(to_rgba(c_start))
            c1 = np.array(to_rgba(c_end))
            t_arr = np.linspace(0, 1, len(segments))[:, None]
            colors_arr = c0 * (1 - t_arr) + c1 * t_arr
            ring_leg.add_collection(LineCollection(segments, colors=colors_arr, linewidths=config.legend_linewidth))
            ring_leg.text(0.12, y, f'{ring}: {desc}', color='#C9D1D9', fontsize=config.ring_label_fontsize, va='center')
            y -= 0.09

        # Draw summary table (bottom-right)
        accum_buy_scores, accum_sell_scores = compute_score_maps(accum_delta, {}, {}, ticker_order)
        lit_buy_scores, lit_sell_scores = compute_score_maps(lit_delta, {}, {}, ticker_order)
        vwbr_z_buy_scores, vwbr_z_sell_scores = compute_score_maps(vwbr_z_map, {}, {}, ticker_order)

        table_rows = [
            ('Accumulation Score', top_tickers_from_scores(accum_buy_scores), top_tickers_from_scores(accum_sell_scores)),
            ('Lit Buy/Sell Ratio', top_tickers_from_scores(lit_buy_scores), top_tickers_from_scores(lit_sell_scores)),
            ('VWBR Z', top_tickers_from_scores(vwbr_z_buy_scores), top_tickers_from_scores(vwbr_z_sell_scores)),
        ]
        col1, col2, col3 = config.table_col1_width, config.table_col2_width, config.table_col3_width
        table_lines = [f"{'Metric':<{col1}}{'Buy (Top)':<{col2}}{'Sell (Top)':<{col3}}", '-' * (col1 + col2 + col3)]
        for label, buy_str, sell_str in table_rows:
            table_lines.append(f"{label:<{col1}}{buy_str:<{col2}}{sell_str:<{col3}}")
        fig.text(0.55, 0.05, '\n'.join(table_lines), ha='left', va='bottom', color='#C9D1D9',
                 fontsize=config.table_fontsize, fontfamily='monospace', linespacing=1.2)

        # Optional watermark
        if config.watermark_path:
            try:
                wm_img = plt.imread(config.watermark_path)
                fig_w, fig_h = fig.get_size_inches()
                wm_w = config.watermark_width
                # Calculate aspect ratio from image shape
                wm_h = wm_w * wm_img.shape[0] / wm_img.shape[1]
                wm_left = 0.85 - wm_w / 2
                wm_bottom = 0.05
                wm_ax = fig.add_axes([wm_left, wm_bottom, wm_w, wm_h / (fig_h / fig_w)])
                wm_ax.imshow(wm_img, alpha=config.watermark_alpha)
                wm_ax.axis('off')
            except Exception as e:
                logger.warning("Could not load watermark: %s", e)

        # Save figure
        date_tag = end_date_dt.strftime("%Y-%m-%d")
        output_path = output_dir / f"circos_{date_tag}.png"
        fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)

        logger.info("Circos plot saved to: %s", output_path)
        return output_path

    except Exception as e:
        logger.error("Error rendering circos: %s", e, exc_info=True)
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Run with default config
    output = render_circos()
    if output:
        print(f"Output: {output}")
    else:
        print("Render failed")
