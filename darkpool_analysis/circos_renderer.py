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

    def __post_init__(self):
        """Apply render mode presets and time-fade adjustments."""
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
    'vwbr_z': 'VWBR Z',
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
    start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer for weekends

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
            vw_flow_z
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

    return df


def tail_n_days(df: pd.DataFrame, ticker: str, n: int, end_date) -> pd.DataFrame:
    """Get last N trading days for a ticker up to end_date."""
    df_ticker = df[df['ticker'] == ticker].copy()
    df_ticker = df_ticker[df_ticker['date'] <= end_date].sort_values('date')
    return df_ticker.tail(n)


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


def filter_edges(edges_df: pd.DataFrame, max_edges: int) -> pd.DataFrame:
    """Filter to top N edges by flow."""
    if edges_df is None or edges_df.empty:
        return edges_df
    if max_edges and max_edges > 0:
        return edges_df.nlargest(max_edges, 'flow')
    return edges_df


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
# Layout functions
# ---------------------------------------------------------------------------

def compute_ticker_angles(
    ticker_order: List[str],
    categories: Dict[str, str],
    category_gap_deg: float,
    chord_arc_fraction: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute angular positions for each ticker around the circle.

    Returns:
        Tuple of (angle_starts, angle_spans) dicts
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
        return {}, {}

    # Total arc available
    total_arc = 2 * math.pi * chord_arc_fraction

    # Subtract category gaps
    total_gap = n_categories * math.radians(category_gap_deg)
    available_arc = total_arc - total_gap

    # Arc per ticker
    arc_per_ticker = available_arc / n_tickers if n_tickers > 0 else 0

    angles = {}
    spans = {}
    current_angle = 0

    for cat in grouped:
        for t in grouped[cat]:
            angles[t] = current_angle
            spans[t] = arc_per_ticker
            current_angle += arc_per_ticker
        current_angle += math.radians(category_gap_deg)

    return angles, spans


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

    logging.info("Starting circos render with config: ticker_type=%s, flow_period_days=%d",
                 config.ticker_type, config.flow_period_days)

    # Resolve database path
    if db_path is None:
        try:
            from config import load_config
            main_config = load_config()
            db_path = main_config.db_path
        except ImportError:
            try:
                from db_path import get_db_path
                db_path = get_db_path()
            except ImportError:
                logging.error("Could not resolve database path")
                return None

    # Resolve output directory
    if output_dir is None:
        output_dir = Path(db_path).parent.parent / "darkpool_analysis" / "output" / "circos_plot"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build ticker universe
    try:
        ticker_order, categories = build_ticker_universe(config.ticker_type)
    except FileNotFoundError as e:
        logging.error("Failed to load ticker dictionary: %s", e)
        return None

    if not ticker_order:
        logging.warning("No tickers found for types: %s", config.ticker_type)
        return None

    logging.info("Ticker universe: %d tickers from %d categories",
                 len(ticker_order), len(set(categories.values())))

    # Connect to database
    try:
        conn, db_type = connect_db(db_path)
    except Exception as e:
        logging.error("Failed to connect to database: %s", e)
        return None

    try:
        # Determine end date
        if config.end_date:
            end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
        else:
            # Auto-detect max date in database
            result = conn.execute("SELECT MAX(date) FROM daily_metrics").fetchone()
            if result and result[0]:
                end_date = pd.to_datetime(result[0])
            else:
                end_date = datetime.now()

        end_date_dt = end_date if isinstance(end_date, datetime) else datetime.combine(end_date, datetime.min.time())
        start_date = end_date_dt - timedelta(days=config.flow_period_days + 10)

        logging.info("Date range: %s to %s", start_date.date(), end_date_dt.date())

        # Load daily metrics
        df = load_daily_metrics(conn, db_type, ticker_order, end_date_dt, config.flow_period_days + 20)

        if df.empty:
            logging.warning("No data found for tickers in date range")
            return None

        # Compute accumulation delta for each ticker over the period
        accum_delta = {}
        for ticker in ticker_order:
            df_t = tail_n_days(df, ticker, config.flow_period_days, end_date_dt.date())
            if len(df_t) >= 2:
                accum_delta[ticker] = df_t['accumulation_score'].iloc[-1] - df_t['accumulation_score'].iloc[0]
            elif len(df_t) == 1:
                accum_delta[ticker] = df_t['accumulation_score'].iloc[0] - 50  # vs neutral

        delta_df = pd.DataFrame([
            {'ticker': t, 'accum_delta': v}
            for t, v in accum_delta.items() if not np.isnan(v)
        ])

        if delta_df.empty:
            logging.warning("No valid accumulation deltas computed")
            return None

        # Build edges
        edges_df, winners, losers, total_demand, total_supply = build_edges_from_value(
            delta_df, 'accum_delta',
            config.top_k_winners, config.top_k_losers,
            config.distribution_mode, config.min_edge_flow
        )

        edges_df = filter_edges(edges_df, config.max_edges_per_metric)

        if edges_df.empty:
            logging.warning("No edges computed from accumulation data")
            return None

        logging.info("Computed %d edges between %d winners and %d losers",
                     len(edges_df), len(winners), len(losers))

        # Compute layout
        angles, spans = compute_ticker_angles(
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
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        ax.set_aspect('equal')

        # Draw category bands
        for ticker in ticker_order:
            cat = categories.get(ticker, 'UNKNOWN')
            color = CATEGORY_PALETTE.get(cat, CATEGORY_PALETTE['UNKNOWN'])
            a0 = angles.get(ticker, 0)
            span = spans.get(ticker, 0)

            wedge = Wedge(
                (0, 0), 0.95, math.degrees(a0), math.degrees(a0 + span),
                width=0.05, facecolor=color, edgecolor='none', alpha=0.8
            )
            ax.add_patch(wedge)

            # Ticker label
            mid_angle = a0 + span / 2
            label_r = 1.02
            x = label_r * math.cos(mid_angle)
            y = label_r * math.sin(mid_angle)
            rotation = math.degrees(mid_angle)
            if 90 < rotation < 270:
                rotation += 180
            ax.text(x, y, ticker, fontsize=8, color='white', ha='center', va='center',
                    rotation=rotation, rotation_mode='anchor')

        # Draw chord ribbons
        chord_r = config.chord_radius
        for _, row in edges_df.iterrows():
            src, dst = row['source'], row['dest']
            if src not in angles or dst not in angles:
                continue

            a0 = angles[src] + spans[src] * 0.3
            a1 = angles[src] + spans[src] * 0.7
            b0 = angles[dst] + spans[dst] * 0.3
            b1 = angles[dst] + spans[dst] * 0.7

            # Color based on flow direction
            src_cat = categories.get(src, 'UNKNOWN')
            dst_cat = categories.get(dst, 'UNKNOWN')
            color_start = soften_color(CATEGORY_PALETTE.get(src_cat, '#888'), config.chord_color_soften)
            color_end = soften_color(CATEGORY_PALETTE.get(dst_cat, '#888'), config.chord_color_soften)

            draw_ribbon(
                ax, a0, a1, b0, b1, chord_r,
                color_start, color_end,
                config.chord_fill_alpha, config.chord_line_alpha,
                lw=1.0,
                use_gradient_fill=config.use_gradient_fill,
                gradient_steps=config.chord_gradient_steps,
            )

        # Title
        title_text = f"Institutional Money Flow - {config.flow_period_days}D"
        ax.text(0, 1.12, title_text, fontsize=14, color='white', ha='center', va='bottom',
                transform=ax.transAxes, fontweight='bold')

        date_text = f"{start_date.strftime('%Y-%m-%d')} to {end_date_dt.strftime('%Y-%m-%d')}"
        ax.text(0.5, 1.08, date_text, fontsize=10, color='#888888', ha='center', va='bottom',
                transform=ax.transAxes)

        # Save figure
        date_tag = end_date_dt.strftime("%Y-%m-%d")
        output_path = output_dir / f"circos_{date_tag}.png"
        fig.savefig(output_path, dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
        plt.close(fig)

        logging.info("Circos plot saved to: %s", output_path)
        return output_path

    except Exception as e:
        logging.error("Error rendering circos: %s", e, exc_info=True)
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
