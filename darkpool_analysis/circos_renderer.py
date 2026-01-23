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
    ribbon_min_width_rad: Optional[Dict[str, float]] = None
    ribbon_center_offset: Optional[Dict[str, float]] = None

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
        if self.ribbon_min_width_rad is None:
            self.ribbon_min_width_rad = {
                'accum': 0.001, 'short': 0.001, 'lit': 0.001,
                'finra_buy': 0.001, 'vwbr_z': 0.001,
            }
        if self.ribbon_center_offset is None:
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
    'GLOBAL_MACRO': "#2D73FF",
    'MAG8': "#D38CFA",
    'THEMATIC_SECTORS': "#00E47D",
    'SECTOR_CORE': "#FAAF00F2",
    'COMMODITIES': "#FFFC2F",
    'RATES_CREDIT': "#FF9966",
    'SPECULATIVE': "#FF7A45",
    'CRYPTO': "#00D0FF",
    'UNKNOWN': "#B9B9B9",
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
# Column name candidates (auto-detection)
# ---------------------------------------------------------------------------

TABLE_CANDIDATES = ['daily_metrics', 'scanner_daily_metrics', 'darkpool_metrics', 'metrics', 'darkpool', 'accumulation']
TICKER_COL_CANDIDATES = ['ticker', 'symbol', 'stock', 'name']
DATE_COL_CANDIDATES = ['date', 'trade_date', 'dt', 'timestamp']
ACCUM_COL_CANDIDATES = [
    'accumulation_score_display', 'accum_score_display',
    'accumulation_score', 'accum_score', 'accumulation', 'accum'
]

SHORT_BUY_CANDIDATES = ['finra_buy_volume', 'short_buy_volume', 'short_buy', 'short_buy_vol']
SHORT_SELL_CANDIDATES = ['short_sell_volume', 'short_sell', 'short_sell_vol']
LIT_BUY_CANDIDATES = ['lit_buy_volume', 'lit_buy', 'lit_buy_vol']
LIT_SELL_CANDIDATES = ['lit_sell_volume', 'lit_sell', 'lit_sell_vol']
LIT_BUY_RATIO_CANDIDATES = ['lit_buy_ratio', 'lit_buy_ratio_pct']
OTC_VOLUME_CANDIDATES = ['otc_off_exchange_volume', 'otc_volume', 'dark_volume']
LIT_TOTAL_CANDIDATES = ['lit_total_volume', 'lit_volume', 'lit_total']
FINRA_BUY_CANDIDATES = ['finra_buy_volume', 'finra_buy', 'finra_buy_vol']
FINRA_BUY_Z_CANDIDATES = ['finra_buy_volume_z', 'finra_buy_z', 'finra_buy_vol_z']
SHORT_RATIO_CANDIDATES = ['short_ratio']
SHORT_BUY_SELL_RATIO_CANDIDATES = ['short_buy_sell_ratio', 'sbsr', 'short_bs_ratio']
VWBR_CANDIDATES = ['vw_flow', 'vwbr', 'vw_buy_ratio', 'vw_buy_sell_ratio']
VWBR_Z_CANDIDATES = ['vw_flow_z', 'vwbr_z', 'vw_buy_ratio_z', 'finra_buy_volume_z']

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


def _extract_group_tickers(value: Any, prefer_mag8: bool = False) -> List[str]:
    """Extract ticker list from list-like or dict-like group definitions."""
    if value is None:
        return []
    if isinstance(value, dict):
        if prefer_mag8 and "MAG8" in value and isinstance(value.get("MAG8"), (list, tuple, set)):
            return list(value["MAG8"])
        return list(value.keys())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def build_ticker_universe(ticker_types: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Build ordered ticker universe from ticker_dictionary.py.

    Returns:
        Tuple of (ordered_tickers, category_map)
    """
    ticker_dict = _load_ticker_dictionary()

    type_to_group = {
        "GLOBAL": ("GLOBAL_MACRO", ["GLOBAL_MACRO_TICKERS", "GLOBAL_MACRO"]),
        "MAG8": ("MAG8", ["MAG8_TICKERS", "MAG8"]),
        "THEMATIC": ("THEMATIC_SECTORS", ["THEMATIC_SECTORS_TICKERS", "THEMATIC_SECTORS"]),
        "SECTOR": ("SECTOR_CORE", ["SECTOR_CORE_TICKERS", "SECTOR_CORE"]),
        "COMMODITIES": ("COMMODITIES", ["COMMODITIES_TICKERS", "COMMODITIES"]),
        "RATES": ("RATES_CREDIT", ["RATES_CREDIT_TICKERS", "RATES_CREDIT"]),
        "SPECULATIVE": ("SPECULATIVE", ["SPECULATIVE_TICKERS"]),
        "CRYPTO": ("CRYPTO", ["CRYPTO_TICKERS"]),
    }

    normalized_types = _normalize_ticker_types(ticker_types)
    if "ALL" in normalized_types:
        normalized_types = list(type_to_group.keys())

    seen = set()
    ordered = []
    categories = {}

    for ticker_type in normalized_types:
        group_info = type_to_group.get(ticker_type)
        if not group_info:
            continue
        category, attr_candidates = group_info
        tickers: List[str] = []
        for attr_name in attr_candidates:
            if hasattr(ticker_dict, attr_name):
                value = getattr(ticker_dict, attr_name)
                tickers = _extract_group_tickers(value, prefer_mag8=(attr_name == "MAG8"))
                if tickers:
                    break
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

def detect_table_and_columns(conn, db_type: str) -> Dict[str, Optional[str]]:
    """Detect table and key column names from available tables."""
    tables = get_tables(conn, db_type)
    select_table = None
    for cand in TABLE_CANDIDATES:
        matching = [t for t in tables if t.lower() == cand.lower()]
        if matching:
            select_table = matching[0]
            break
    if not select_table and tables:
        select_table = tables[0]
    if not select_table:
        raise ValueError("No suitable table found in database.")

    columns = get_columns(conn, db_type, select_table)
    ticker_col = pick_column(columns, TICKER_COL_CANDIDATES)
    date_col = pick_column(columns, DATE_COL_CANDIDATES)
    accum_col = pick_column(columns, ACCUM_COL_CANDIDATES)

    if not ticker_col or not date_col:
        raise ValueError(f"Missing required ticker/date columns in {select_table}")

    volume_info = {
        "table": select_table,
        "ticker_col": ticker_col,
        "date_col": date_col,
        "accum_col": accum_col,
        "lit_buy_col": pick_column(columns, LIT_BUY_CANDIDATES),
        "lit_sell_col": pick_column(columns, LIT_SELL_CANDIDATES),
        "lit_buy_ratio_col": pick_column(columns, LIT_BUY_RATIO_CANDIDATES),
        "short_buy_col": pick_column(columns, SHORT_BUY_CANDIDATES),
        "short_sell_col": pick_column(columns, SHORT_SELL_CANDIDATES),
        "otc_vol_col": pick_column(columns, OTC_VOLUME_CANDIDATES),
        "lit_total_col": pick_column(columns, LIT_TOTAL_CANDIDATES),
        "finra_buy_col": pick_column(columns, FINRA_BUY_CANDIDATES),
        "finra_buy_z_col": pick_column(columns, FINRA_BUY_Z_CANDIDATES),
        "short_ratio_col": pick_column(columns, SHORT_RATIO_CANDIDATES),
        "short_buy_sell_ratio_col": pick_column(columns, SHORT_BUY_SELL_RATIO_CANDIDATES),
        "vwbr_col": pick_column(columns, VWBR_CANDIDATES),
        "vwbr_z_col": pick_column(columns, VWBR_Z_CANDIDATES),
    }

    return volume_info


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
    volume_info: Dict[str, Optional[str]],
    ticker_list: List[str],
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Load daily metrics data for given tickers and date range with auto-detected columns."""
    table = volume_info["table"]
    ticker_col = volume_info["ticker_col"]
    date_col = volume_info["date_col"]
    accum_col = volume_info.get("accum_col")

    num_cast = "TRY_CAST" if db_type == "duckdb" else "CAST"
    date_expr = f"CAST({quote_ident(date_col)} AS DATE)" if db_type == "duckdb" else f"DATE({quote_ident(date_col)})"

    select_cols = [
        f"UPPER({quote_ident(ticker_col)}) AS ticker",
        f"{date_expr} AS date",
    ]
    if accum_col:
        select_cols.append(f"{num_cast}({quote_ident(accum_col)} AS DOUBLE) AS accumulation_score")
    else:
        select_cols.append("NULL AS accumulation_score")

    def _add(col_key: str, alias: str):
        col_name = volume_info.get(col_key)
        if col_name:
            select_cols.append(f"{num_cast}({quote_ident(col_name)} AS DOUBLE) AS {alias}")

    _add("lit_buy_col", "lit_buy")
    _add("lit_sell_col", "lit_sell")
    _add("lit_buy_ratio_col", "lit_buy_ratio")
    _add("short_buy_col", "short_buy")
    _add("short_sell_col", "short_sell")
    _add("finra_buy_col", "finra_buy")
    _add("finra_buy_z_col", "finra_buy_z")
    _add("vwbr_col", "vwbr")
    _add("vwbr_z_col", "vwbr_z")
    _add("short_ratio_col", "short_ratio")
    _add("short_buy_sell_ratio_col", "short_buy_sell_ratio")
    _add("otc_vol_col", "otc_volume")
    _add("lit_total_col", "lit_total")

    placeholders = ",".join(["?"] * len(ticker_list))
    query = f"""
        SELECT {', '.join(select_cols)}
        FROM {quote_ident(table)}
        WHERE UPPER({quote_ident(ticker_col)}) IN ({placeholders})
          AND {date_expr} >= ?
          AND {date_expr} <= ?
        ORDER BY date
    """
    params = ticker_list + [start_date.date(), end_date.date()]

    if db_type == "duckdb":
        df = conn.execute(query, params).df()
    else:
        df = pd.read_sql_query(query, conn, params=params)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["accumulation_score"] = df["accumulation_score"].apply(parse_accum)

    # Standardize missing columns
    for col in [
        "lit_buy", "lit_sell", "short_buy", "short_sell",
        "finra_buy", "finra_buy_z", "vwbr", "vwbr_z",
        "short_ratio", "short_buy_sell_ratio", "otc_volume", "lit_total",
        "lit_buy_ratio",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Derive lit_total if not provided
    if df["lit_total"].isna().all():
        if df["lit_buy"].notna().any() or df["lit_sell"].notna().any():
            df["lit_total"] = df["lit_buy"].fillna(0) + df["lit_sell"].fillna(0)
        else:
            df["lit_total"] = 0.0

    # Derive lit_buy_ratio if missing
    if df["lit_buy_ratio"].isna().all():
        total = df["lit_buy"].fillna(0) + df["lit_sell"].fillna(0)
        df["lit_buy_ratio"] = (df["lit_buy"] / total.replace(0, np.nan)).fillna(0.5)

    # Prefer finra_buy_z as vwbr_z when vwbr_z missing
    if df["vwbr_z"].isna().all() and df["finra_buy_z"].notna().any():
        df["vwbr_z"] = df["finra_buy_z"]

    # Derive short_buy_sell_ratio if possible
    if df["short_buy_sell_ratio"].isna().all():
        denom = df["short_sell"].replace(0, np.nan)
        df["short_buy_sell_ratio"] = (df["short_buy"] / denom).replace([np.inf, -np.inf], np.nan)

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


def build_edges_by_date(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    top_k_winners: int,
    top_k_losers: int,
    distribution_mode: str,
    min_edge_flow: float,
    positive_only: bool = False,
) -> pd.DataFrame:
    """Build edges per day and return concatenated edges with date column."""
    if df is None or df.empty:
        return pd.DataFrame()

    edges_all = []
    for date_val, group in df.groupby(date_col):
        if positive_only:
            edges_df, _, _, _, _ = build_edges_from_positive_value(
                group, value_col, top_k_winners, top_k_losers, min_edge_flow
            )
        else:
            edges_df, _, _, _, _ = build_edges_from_value(
                group, value_col, top_k_winners, top_k_losers, distribution_mode, min_edge_flow
            )
        if edges_df is None or edges_df.empty:
            continue
        edges_df = edges_df.copy()
        edges_df["date"] = date_val
        edges_all.append(edges_df)

    if not edges_all:
        return pd.DataFrame()
    return pd.concat(edges_all, ignore_index=True)


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
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], Dict[str, List[str]]]:
    """
    Compute angular positions for each ticker around the circle.

    Returns:
        Tuple of (angles dict {ticker: angle}, spans dict {ticker: (start, end)}, grouped dict {category: [tickers]})
    """
    # Group tickers by category
    grouped = {}
    for t in ticker_order:
        cat = categories.get(t, 'UNKNOWN')
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(t)

    n_tickers = len(ticker_order)

    if n_tickers == 0:
        return {}, {}, grouped

    gap = math.radians(category_gap_deg)
    total_gap = gap * len([g for g in grouped.values() if g])
    usable = 2 * math.pi - total_gap
    if usable <= 0:
        usable = 2 * math.pi

    step = usable / n_tickers if n_tickers > 0 else 0.0
    arc_span = step * 0.85

    angles: Dict[str, float] = {}
    spans: Dict[str, Tuple[float, float]] = {}
    angle = 0.0

    ordered_categories = [
        'GLOBAL_MACRO', 'MAG8', 'THEMATIC_SECTORS', 'SECTOR_CORE',
        'COMMODITIES', 'RATES_CREDIT', 'SPECULATIVE', 'CRYPTO', 'UNKNOWN',
    ]
    for cat in ordered_categories:
        if not grouped.get(cat):
            continue
        angle += gap / 2
        for t in grouped[cat]:
            angles[t] = angle
            spans[t] = (angle - arc_span / 2, angle + arc_span / 2)
            angle += step
        angle += gap / 2

    return angles, spans, grouped


def allocate_intervals(
    edges_df: pd.DataFrame,
    band_map: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    metric_key: str,
    ticker_order: List[str],
    config: CircosConfig,
    centered: bool = True,
    center_offset: float = 0.0,
) -> List[Dict[str, Any]]:
    """Allocate chord ribbon intervals within each ticker band."""
    if edges_df is None or edges_df.empty:
        return []

    min_width_rad = config.ribbon_min_width_rad.get(metric_key, 0.003)
    out_counts = edges_df.groupby('source').size().to_dict()
    in_counts = edges_df.groupby('dest').size().to_dict()
    max_flow = edges_df['flow'].max() if not edges_df.empty else 1.0

    out_flow_totals = edges_df.groupby('source')['flow'].apply(lambda s: s.abs().sum()).to_dict()
    in_flow_totals = edges_df.groupby('dest')['flow'].apply(lambda s: s.abs().sum()).to_dict()
    total_flow_by_ticker = {t: out_flow_totals.get(t, 0.0) + in_flow_totals.get(t, 0.0) for t in ticker_order}
    max_total_flow = max(total_flow_by_ticker.values()) if total_flow_by_ticker else 0.0

    use_time_order = config.show_time_fade_chords and config.time_fade_use_daily_edges and 'date' in edges_df.columns
    out_order: Dict[int, int] = {}
    in_order: Dict[int, int] = {}
    edges_with_id = edges_df
    if use_time_order:
        edges_with_id = edges_df.copy()
        edges_with_id['_row_id'] = range(len(edges_with_id))
        for src, group in edges_with_id.groupby('source'):
            group_sorted = group.sort_values(['date', 'flow'], ascending=[False, False], na_position='first')
            for idx, row_id in enumerate(group_sorted['_row_id'].tolist()):
                out_order[row_id] = idx
        for dst, group in edges_with_id.groupby('dest'):
            group_sorted = group.sort_values(['date', 'flow'], ascending=[False, False], na_position='first')
            for idx, row_id in enumerate(group_sorted['_row_id'].tolist()):
                in_order[row_id] = idx

    out_slot_info: Dict[str, Dict[str, Any]] = {}
    in_slot_info: Dict[str, Dict[str, Any]] = {}

    def build_slot_info(range_start, range_end, n, anchor=None, flow_scale=None):
        arc = range_end - range_start
        if n <= 0 or arc <= 0:
            return None
        anchor_mode = centered and config.ribbon_anchor_to_center and anchor is not None
        effective_arc = arc
        if anchor_mode:
            left_cap = max(0.0, anchor - range_start)
            right_cap = max(0.0, range_end - anchor)
            effective_arc = 2 * min(left_cap, right_cap)
        if flow_scale is not None:
            flow_scale = max(0.0, min(1.0, flow_scale))
            effective_arc = effective_arc * flow_scale
        if effective_arc <= 0:
            return None

        total_gap = config.ribbon_gap_rad * (n - 1)
        usable = max(0.0, effective_arc - total_gap)
        slot_width = usable / n if n > 0 else 0.0
        if slot_width < min_width_rad and n > 1:
            slot_width = min_width_rad
            total_gap = max(0.0, effective_arc - n * slot_width)
        if n > 0 and slot_width * n > effective_arc:
            slot_width = effective_arc / n if n else 0.0
            total_gap = 0.0

        actual_gap = total_gap / max(1, n - 1) if n > 1 else 0.0
        total_width = n * slot_width + (n - 1) * actual_gap

        if centered:
            if anchor_mode:
                start_pos = anchor - total_width / 2
            else:
                arc_center = (range_start + range_end) / 2 + center_offset
                start_pos = arc_center - total_width / 2
        else:
            start_pos = range_start

        if not anchor_mode:
            start_pos = max(range_start, min(start_pos, range_end - total_width))

        return {
            'slot_width': slot_width,
            'cursor': start_pos,
            'gap': actual_gap,
            'anchor': anchor if anchor_mode else None,
            'range': (range_start, range_end),
        }

    for ticker in ticker_order:
        spans = band_map.get(ticker, {}).get(metric_key)
        if not spans:
            continue

        out_range = spans['out']
        in_range = spans['in']
        band_center = spans.get('center')
        if band_center is None:
            band_center = (out_range[0] + in_range[1]) / 2
        band_center = band_center + center_offset

        n_out = out_counts.get(ticker, 0)
        n_in = in_counts.get(ticker, 0)
        total_flow = total_flow_by_ticker.get(ticker, 0.0)
        flow_scale = (total_flow / max_total_flow) if max_total_flow > 0 else 0.0

        if centered and config.ribbon_anchor_to_center:
            full_start = out_range[0]
            full_end = in_range[1]
            anchor = max(full_start, min(band_center, full_end))
            out_info = build_slot_info(full_start, full_end, n_out, anchor, flow_scale)
            in_info = build_slot_info(full_start, full_end, n_in, anchor, flow_scale)
        else:
            out_info = build_slot_info(out_range[0], out_range[1], n_out, None, flow_scale)
            in_info = build_slot_info(in_range[0], in_range[1], n_in, None, flow_scale)

        if out_info:
            out_slot_info[ticker] = out_info
        if in_info:
            in_slot_info[ticker] = in_info

    intervals: List[Dict[str, Any]] = []
    edge_iter = edges_with_id.itertuples() if use_time_order else edges_df.sort_values('flow', ascending=False).itertuples()
    for row in edge_iter:
        date_val = getattr(row, 'date', None)
        row_id = getattr(row, '_row_id', None)
        src, dst, flow = row.source, row.dest, row.flow
        if src not in out_slot_info or dst not in in_slot_info:
            continue

        out_info = out_slot_info[src]
        in_info = in_slot_info[dst]

        if config.ribbon_width_scale_by_flow:
            flow_scale = (flow / max_flow) ** 0.5 if max_flow > 0 else 1.0
            out_width = max(min_width_rad * 0.5, out_info['slot_width'] * flow_scale)
            in_width = max(min_width_rad * 0.5, in_info['slot_width'] * flow_scale)
        else:
            out_width = out_info['slot_width']
            in_width = in_info['slot_width']

        out_width = min(out_width, out_info['slot_width'])
        in_width = min(in_width, in_info['slot_width'])

        if config.ribbon_converge_to_point:
            if out_info.get('anchor') is not None:
                a0 = out_info['anchor'] - out_width / 2
                a1 = out_info['anchor'] + out_width / 2
            else:
                a0 = out_info['cursor']
                a1 = a0 + out_width
            if in_info.get('anchor') is not None:
                b0 = in_info['anchor'] - in_width / 2
                b1 = in_info['anchor'] + in_width / 2
            else:
                b0 = in_info['cursor']
                b1 = b0 + in_width
        else:
            if use_time_order and row_id is not None:
                out_idx = out_order.get(row_id, 0)
                in_idx = in_order.get(row_id, 0)
                out_step = out_info['slot_width'] + out_info['gap']
                in_step = in_info['slot_width'] + in_info['gap']
                a0 = out_info['cursor'] + out_idx * out_step
                a1 = a0 + out_width
                b0 = in_info['cursor'] + in_idx * in_step
                b1 = b0 + in_width
            else:
                a0 = out_info['cursor']
                a1 = a0 + out_width
                b0 = in_info['cursor']
                b1 = b0 + in_width
                out_info['cursor'] = a0 + out_info['slot_width'] + out_info['gap']
                in_info['cursor'] = b0 + in_info['slot_width'] + in_info['gap']

        if a1 > a0 and b1 > b0:
            intervals.append({
                'source': src, 'dest': dst, 'flow': flow,
                'a0': a0, 'a1': a1, 'b0': b0, 'b1': b1,
                'date': date_val,
            })

    return intervals


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


def compute_trend_map(df_daily: pd.DataFrame, value_col: str, dates: List[datetime.date]) -> Dict[str, float]:
    """Compute trend map between early/late windows."""
    if df_daily is None or df_daily.empty or not dates:
        return {}
    df = df_daily[["ticker", "date", value_col]].dropna()
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[df["date"].isin(dates)]
    if df.empty:
        return {}
    dates_sorted = sorted(set(dates))
    split = max(1, len(dates_sorted) // 2)
    early_dates = set(dates_sorted[:split])
    late_dates = set(dates_sorted[split:])
    early = df[df["date"].isin(early_dates)].groupby("ticker")[value_col].mean()
    late = df[df["date"].isin(late_dates)].groupby("ticker")[value_col].mean()
    trend = (late - early).dropna()
    return trend.to_dict()


def compute_anomaly_map_from_stats(
    df_daily: pd.DataFrame,
    value_col: str,
    stats_df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    latest_date: Optional[datetime.date],
) -> Dict[str, float]:
    """Compute anomaly map from per-ticker stats."""
    if df_daily is None or df_daily.empty or stats_df is None or stats_df.empty or latest_date is None:
        return {}
    df = df_daily[df_daily["date"] == latest_date][["ticker", value_col]].dropna()
    if df.empty:
        return {}
    stats = stats_df.set_index("ticker")
    anomaly = {}
    for row in df.itertuples():
        if row.ticker not in stats.index:
            continue
        mean_val = stats.loc[row.ticker, mean_col]
        std_val = stats.loc[row.ticker, std_col]
        if std_val and std_val > 0:
            anomaly[row.ticker] = abs((getattr(row, value_col) - mean_val) / std_val)
    return anomaly


def compute_abs_latest_map(
    df_daily: pd.DataFrame,
    value_col: str,
    latest_date: Optional[datetime.date],
) -> Dict[str, float]:
    """Compute absolute latest value map."""
    if df_daily is None or df_daily.empty or latest_date is None:
        return {}
    df = df_daily[df_daily["date"] == latest_date][["ticker", value_col]].dropna()
    if df.empty:
        return {}
    return df.set_index("ticker")[value_col].abs().to_dict()


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
        volume_info = detect_table_and_columns(conn, db_type)
        if not volume_info.get("accum_col"):
            logger.error("Could not detect accumulation column in %s", volume_info.get("table"))
            return None

        # Determine end date
        if config.end_date:
            end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
        else:
            date_col = volume_info["date_col"]
            date_expr = f"CAST({quote_ident(date_col)} AS DATE)" if db_type == "duckdb" else f"DATE({quote_ident(date_col)})"
            result = conn.execute(
                f"SELECT MAX({date_expr}) FROM {quote_ident(volume_info['table'])}"
            ).fetchone()
            if result and result[0]:
                end_date = pd.to_datetime(result[0])
            else:
                end_date = datetime.now()

        end_date_dt = end_date if isinstance(end_date, datetime) else datetime.combine(end_date, datetime.min.time())
        start_date = end_date_dt - timedelta(days=config.flow_period_days + config.lookback_days + 10)

        logger.info("Date range: %s to %s", start_date.date(), end_date_dt.date())

        # Load daily metrics
        df = load_daily_metrics(conn, db_type, volume_info, ticker_order, start_date, end_date_dt)

        if df.empty:
            logger.warning("No data found for tickers in date range")
            return None

        df["ticker"] = df["ticker"].str.upper()
        df = df[df["ticker"].isin([t.upper() for t in ticker_order])]
        if df.empty:
            logger.warning("No matching tickers found in data")
            return None

        # Get window dates (last N trading days)
        all_dates = sorted(df["date"].dropna().unique())
        window_dates = [d for d in all_dates if d <= end_date_dt.date()]
        window_dates = sorted(window_dates)[-config.flow_period_days:]

        if not window_dates:
            logger.warning("No window dates found")
            return None

        window_dates_sorted = sorted(window_dates)
        df_window = df[df["date"].isin(window_dates_sorted)].copy()

        # Compute 20-day statistics for ring sizing
        accum_stats = compute_stats(df, ticker_order, 'accumulation_score', 'accum')
        lit_stats = compute_stats(df, ticker_order, 'lit_total', 'lit_vol')
        finra_stats = compute_stats(df, ticker_order, 'finra_buy', 'finra_buy')

        # Compute accumulation delta for each ticker over the period
        accum_delta = {}
        lit_delta = {}
        vwbr_z_map = {}

        for ticker in ticker_order:
            df_t = tail_n_days(df_window, ticker, config.flow_period_days, end_date_dt.date())
            if len(df_t) >= 2:
                accum_delta[ticker] = df_t['accumulation_score'].iloc[-1] - df_t['accumulation_score'].iloc[0]
                if 'lit_buy_ratio' in df_t.columns:
                    lit_delta[ticker] = (df_t['lit_buy_ratio'].iloc[-1] - 0.5) * 100
            elif len(df_t) == 1:
                accum_delta[ticker] = df_t['accumulation_score'].iloc[0] - 50
                if 'lit_buy_ratio' in df_t.columns:
                    lit_delta[ticker] = (df_t['lit_buy_ratio'].iloc[0] - 0.5) * 100

            # Get latest VWBR Z (finra_buy_volume_z preferred)
            if 'vwbr_z' in df_t.columns and len(df_t) > 0:
                vwbr_z_map[ticker] = df_t['vwbr_z'].iloc[-1]

        # Build delta dataframes
        delta_df = pd.DataFrame([
            {'ticker': t, 'accum_delta': v}
            for t, v in accum_delta.items() if v is not None and np.isfinite(v)
        ])

        vwbr_z_df = pd.DataFrame([
            {'ticker': t, 'vwbr_z': v}
            for t, v in vwbr_z_map.items() if v is not None and np.isfinite(v)
        ])

        # ------------------------------------------------------------------
        # Volume aggregation for lit/short/finra edges
        # ------------------------------------------------------------------
        df_volume = pd.DataFrame()
        df_lit_daily = pd.DataFrame()
        df_short_daily = pd.DataFrame()
        df_finra_daily = pd.DataFrame()

        if not df_window.empty:
            df_vol_raw = df_window.copy()
            for col in ["lit_buy", "lit_sell", "short_buy", "short_sell", "finra_buy"]:
                if col in df_vol_raw.columns:
                    df_vol_raw[col] = df_vol_raw[col].fillna(0)

            agg_dict: Dict[str, str] = {}
            if df_vol_raw["lit_buy"].notna().any():
                agg_dict["lit_buy"] = "sum"
            if df_vol_raw["lit_sell"].notna().any():
                agg_dict["lit_sell"] = "sum"
            if df_vol_raw["short_buy"].notna().any():
                agg_dict["short_buy"] = "sum"
            if df_vol_raw["short_sell"].notna().any():
                agg_dict["short_sell"] = "sum"
            if df_vol_raw["finra_buy"].notna().any():
                agg_dict["finra_buy"] = "sum"
            if df_vol_raw["finra_buy_z"].notna().any():
                agg_dict["finra_buy_z"] = "mean"
            elif df_vol_raw["vwbr_z"].notna().any():
                agg_dict["vwbr_z"] = "mean"
            if df_vol_raw["vwbr"].notna().any():
                agg_dict["vwbr"] = "mean"
            if df_vol_raw["short_ratio"].notna().any():
                agg_dict["short_ratio"] = "mean"
            if df_vol_raw["short_buy_sell_ratio"].notna().any():
                agg_dict["short_buy_sell_ratio"] = "mean"

            if agg_dict:
                df_volume = df_vol_raw.groupby("ticker", as_index=False).agg(agg_dict)
                rename_map = {}
                if "lit_buy" in df_volume.columns:
                    rename_map["lit_buy"] = "lit_buy_sum"
                if "lit_sell" in df_volume.columns:
                    rename_map["lit_sell"] = "lit_sell_sum"
                if "short_buy" in df_volume.columns:
                    rename_map["short_buy"] = "short_buy_sum"
                if "short_sell" in df_volume.columns:
                    rename_map["short_sell"] = "short_sell_sum"
                if "finra_buy" in df_volume.columns:
                    rename_map["finra_buy"] = "finra_buy_sum"
                if "finra_buy_z" in df_volume.columns:
                    rename_map["finra_buy_z"] = "finra_buy_z_mean"
                if "vwbr_z" in df_volume.columns and "finra_buy_z_mean" not in rename_map.values():
                    rename_map["vwbr_z"] = "finra_buy_z_mean"
                if "vwbr" in df_volume.columns:
                    rename_map["vwbr"] = "vwbr_mean"
                if "short_ratio" in df_volume.columns:
                    rename_map["short_ratio"] = "short_ratio_mean"
                if "short_buy_sell_ratio" in df_volume.columns:
                    rename_map["short_buy_sell_ratio"] = "short_buy_sell_ratio_mean"
                if rename_map:
                    df_volume = df_volume.rename(columns=rename_map)

            if df_vol_raw["lit_buy"].notna().any() or df_vol_raw["lit_sell"].notna().any():
                df_lit_daily = (
                    df_vol_raw.groupby(["ticker", "date"], as_index=False)
                    .agg({"lit_buy": "sum", "lit_sell": "sum"})
                )
                df_lit_daily["lit_net"] = df_lit_daily["lit_buy"] - df_lit_daily["lit_sell"]
                df_lit_daily["lit_total"] = df_lit_daily["lit_buy"] + df_lit_daily["lit_sell"]
                df_lit_daily["lit_buy_ratio"] = (
                    df_lit_daily["lit_buy"] / df_lit_daily["lit_total"].replace(0, np.nan)
                ).fillna(0.5)

            if df_vol_raw["short_buy"].notna().any() or df_vol_raw["short_sell"].notna().any():
                df_short_daily = (
                    df_vol_raw.groupby(["ticker", "date"], as_index=False)
                    .agg({"short_buy": "sum", "short_sell": "sum"})
                )
                df_short_daily["short_net"] = df_short_daily["short_buy"] - df_short_daily["short_sell"]

            if df_vol_raw["finra_buy"].notna().any():
                df_finra_daily = (
                    df_vol_raw.groupby(["ticker", "date"], as_index=False)
                    .agg({"finra_buy": "sum"})
                )
                if df_vol_raw["finra_buy_z"].notna().any():
                    df_vwbr_z_daily = (
                        df_vol_raw.groupby(["ticker", "date"], as_index=False)
                        .agg({"finra_buy_z": "mean"})
                        .rename(columns={"finra_buy_z": "vwbr_z"})
                    )
                    df_finra_daily = df_finra_daily.merge(df_vwbr_z_daily, on=["ticker", "date"], how="left")
                elif df_vol_raw["vwbr_z"].notna().any():
                    df_vwbr_z_daily = (
                        df_vol_raw.groupby(["ticker", "date"], as_index=False)
                        .agg({"vwbr_z": "mean"})
                    )
                    df_finra_daily = df_finra_daily.merge(df_vwbr_z_daily, on=["ticker", "date"], how="left")
                if df_vol_raw["vwbr"].notna().any():
                    df_vwbr_daily = (
                        df_vol_raw.groupby(["ticker", "date"], as_index=False)
                        .agg({"vwbr": "mean"})
                    )
                    df_finra_daily = df_finra_daily.merge(df_vwbr_daily, on=["ticker", "date"], how="left")
                if df_vol_raw["short_ratio"].notna().any():
                    df_short_ratio_daily = (
                        df_vol_raw.groupby(["ticker", "date"], as_index=False)
                        .agg({"short_ratio": "mean"})
                    )
                    df_finra_daily = df_finra_daily.merge(df_short_ratio_daily, on=["ticker", "date"], how="left")
                if df_vol_raw["short_buy"].notna().any() and df_vol_raw["short_sell"].notna().any():
                    df_short_bs = (
                        df_vol_raw.groupby(["ticker", "date"], as_index=False)
                        .agg({"short_buy": "sum", "short_sell": "sum"})
                    )
                    denom = df_short_bs["short_sell"].replace(0, np.nan)
                    df_short_bs["short_buy_sell_ratio"] = (df_short_bs["short_buy"] / denom).replace(
                        [np.inf, -np.inf], np.nan
                    )
                    df_short_bs = df_short_bs[["ticker", "date", "short_buy_sell_ratio"]]
                    df_finra_daily = df_finra_daily.merge(df_short_bs, on=["ticker", "date"], how="left")

        # Accumulation level data (for time fade and summary)
        df_accum_level = df_window[["ticker", "date", "accumulation_score"]].copy()
        if not df_accum_level.empty:
            df_accum_level = df_accum_level.rename(columns={"accumulation_score": "value"})

        # Build edges
        accum_edges_plot = pd.DataFrame()
        lit_edges_plot = pd.DataFrame()
        short_edges_plot = pd.DataFrame()
        finra_edges_plot = pd.DataFrame()
        vwbr_z_edges_plot = pd.DataFrame()

        if not delta_df.empty:
            accum_edges_plot, _, _, _, _ = build_edges_from_value(
                delta_df, "accum_delta",
                config.top_k_winners, config.top_k_losers,
                config.distribution_mode, config.min_edge_flow
            )
            accum_edges_plot = filter_edges(accum_edges_plot, config.max_edges_per_metric)

        if not df_volume.empty:
            if "lit_buy_sum" in df_volume.columns and "lit_sell_sum" in df_volume.columns:
                df_volume["lit_net"] = df_volume["lit_buy_sum"] - df_volume["lit_sell_sum"]
                lit_edges_plot, _, _, _, _ = build_edges_from_value(
                    df_volume, "lit_net",
                    config.top_k_winners, config.top_k_losers,
                    config.distribution_mode, config.min_edge_flow
                )
                lit_edges_plot = filter_edges(lit_edges_plot, config.max_edges_per_metric)
            if "short_buy_sum" in df_volume.columns and "short_sell_sum" in df_volume.columns:
                df_volume["short_net"] = df_volume["short_buy_sum"] - df_volume["short_sell_sum"]
                short_edges_plot, _, _, _, _ = build_edges_from_value(
                    df_volume, "short_net",
                    config.top_k_winners, config.top_k_losers,
                    config.distribution_mode, config.min_edge_flow
                )
                short_edges_plot = filter_edges(short_edges_plot, config.max_edges_per_metric)
            if "finra_buy_sum" in df_volume.columns:
                finra_edges_plot, _, _, _, _ = build_edges_from_positive_value(
                    df_volume, "finra_buy_sum",
                    config.top_k_winners, config.top_k_losers,
                    config.min_edge_flow
                )
                finra_edges_plot = filter_edges(finra_edges_plot, config.max_edges_per_metric)
            if "finra_buy_z_mean" in df_volume.columns:
                vwbr_z_edges_plot, _, _, _, _ = build_edges_from_value(
                    df_volume, "finra_buy_z_mean",
                    config.top_k_winners, config.top_k_losers,
                    config.distribution_mode, config.min_edge_flow
                )
                vwbr_z_edges_plot = filter_edges(vwbr_z_edges_plot, config.max_edges_per_metric)

        if vwbr_z_edges_plot.empty and not vwbr_z_df.empty:
            vwbr_z_edges_plot, _, _, _, _ = build_edges_from_value(
                vwbr_z_df, "vwbr_z",
                config.top_k_winners, config.top_k_losers,
                config.distribution_mode, config.min_edge_flow
            )
            vwbr_z_edges_plot = filter_edges(vwbr_z_edges_plot, config.max_edges_per_metric)

        # Time-faded edges per day (optional)
        accum_time_edges_df = pd.DataFrame()
        lit_time_edges_df = pd.DataFrame()
        short_time_edges_df = pd.DataFrame()
        finra_time_edges_df = pd.DataFrame()
        vwbr_z_time_edges_df = pd.DataFrame()

        if config.show_time_fade_chords and config.time_fade_use_daily_edges:
            if not df_accum_level.empty:
                df_accum_daily_centered = df_accum_level.copy()
                df_accum_daily_centered["date"] = pd.to_datetime(
                    df_accum_daily_centered["date"], errors="coerce"
                ).dt.date
                daily_means = df_accum_daily_centered.groupby("date")["value"].mean()
                df_accum_daily_centered["accum_centered"] = df_accum_daily_centered["value"] - df_accum_daily_centered["date"].map(daily_means)
                accum_time_edges_df = build_edges_by_date(
                    df_accum_daily_centered, "date", "accum_centered",
                    config.top_k_winners, config.top_k_losers,
                    config.distribution_mode, config.min_edge_flow
                )
            if not df_lit_daily.empty:
                lit_time_edges_df = build_edges_by_date(
                    df_lit_daily, "date", "lit_net",
                    config.top_k_winners, config.top_k_losers,
                    config.distribution_mode, config.min_edge_flow
                )
            if not df_short_daily.empty:
                short_time_edges_df = build_edges_by_date(
                    df_short_daily, "date", "short_net",
                    config.top_k_winners, config.top_k_losers,
                    config.distribution_mode, config.min_edge_flow
                )
            if not df_finra_daily.empty and "finra_buy" in df_finra_daily.columns:
                finra_time_edges_df = build_edges_by_date(
                    df_finra_daily, "date", "finra_buy",
                    config.top_k_winners, config.top_k_losers,
                    config.distribution_mode, config.min_edge_flow,
                    positive_only=True
                )
            if not df_window.empty and df_window["finra_buy_z"].notna().any():
                df_vwbr_z_daily = df_window.groupby(["ticker", "date"], as_index=False).agg({"finra_buy_z": "mean"})
                vwbr_z_time_edges_df = build_edges_by_date(
                    df_vwbr_z_daily, "date", "finra_buy_z",
                    config.top_k_winners, config.top_k_losers,
                    config.distribution_mode, config.min_edge_flow
                )
            elif not df_window.empty and df_window["vwbr_z"].notna().any():
                df_vwbr_z_daily = df_window.groupby(["ticker", "date"], as_index=False).agg({"vwbr_z": "mean"})
                vwbr_z_time_edges_df = build_edges_by_date(
                    df_vwbr_z_daily, "date", "vwbr_z",
                    config.top_k_winners, config.top_k_losers,
                    config.distribution_mode, config.min_edge_flow
                )

        def choose_time_edges(time_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
            if config.show_time_fade_chords and config.time_fade_use_daily_edges and time_df is not None and not time_df.empty:
                return time_df
            return base_df

        metric_edges = {
            "accum": choose_time_edges(accum_time_edges_df, accum_edges_plot),
            "short": choose_time_edges(short_time_edges_df, short_edges_plot),
            "lit": choose_time_edges(lit_time_edges_df, lit_edges_plot),
            "finra_buy": choose_time_edges(finra_time_edges_df, finra_edges_plot),
            "vwbr_z": choose_time_edges(vwbr_z_time_edges_df, vwbr_z_edges_plot),
        }

        metric_edges_draw = {m: expand_edges(metric_edges.get(m), config.edge_ribbon_splits, config.edge_ribbon_max) for m in CHORD_METRIC_ORDER}
        metric_totals = {m: compute_metric_totals(metric_edges.get(m)) for m in CHORD_METRIC_ORDER}
        metric_nets = {
            "accum": accum_delta,
            "short": df_volume.set_index("ticker")["short_net"].to_dict() if not df_volume.empty and "short_net" in df_volume.columns else {},
            "lit": df_volume.set_index("ticker")["lit_net"].to_dict() if not df_volume.empty and "lit_net" in df_volume.columns else {},
            "finra_buy": df_volume.set_index("ticker")["finra_buy_sum"].to_dict() if not df_volume.empty and "finra_buy_sum" in df_volume.columns else {},
            "vwbr_z": df_volume.set_index("ticker")["finra_buy_z_mean"].to_dict() if not df_volume.empty and "finra_buy_z_mean" in df_volume.columns else vwbr_z_map,
            "vwbr": df_volume.set_index("ticker")["vwbr_mean"].to_dict() if not df_volume.empty and "vwbr_mean" in df_volume.columns else {},
        }

        # Compute layout
        angles, spans, grouped = compute_ticker_angles(
            ticker_order, categories,
            config.category_gap_deg, config.chord_arc_fraction
        )

        visible_band_order = [m for m in BAND_ORDER if metric_visible(m, config)]
        if not visible_band_order:
            visible_band_order = BAND_ORDER

        # Metric bands per ticker (for chords)
        band_map: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
        for t, (a0, a1) in spans.items():
            max_span = (a1 - a0)
            chord_span = min(max_span, max_span * config.chord_arc_fraction)
            chord_center = (a0 + a1) / 2
            band_gap = chord_span * config.band_gap_frac

            if config.metric_band_mode == "proportional":
                weights = {m: metric_totals.get(m, {}).get(t, 0.0) for m in visible_band_order}
                if sum(weights.values()) <= 0:
                    weights = {m: 1.0 for m in visible_band_order}
            else:
                weights = {m: 1.0 for m in visible_band_order}

            total_w = sum(weights.values())
            if total_w <= 0:
                weights = {m: 1.0 for m in visible_band_order}
                total_w = sum(weights.values())

            available = chord_span - band_gap * max(len(visible_band_order) - 1, 0)
            if available <= 0:
                band_gap = 0.0
                available = chord_span

            lengths = {m: max(0.0, available * (weights[m] / total_w)) for m in visible_band_order}

            total_len = sum(lengths.values()) + band_gap * max(len(visible_band_order) - 1, 0)
            start = chord_center - total_len / 2

            metric_spans: Dict[str, Tuple[float, float]] = {}
            cursor = start
            for idx, m in enumerate(visible_band_order):
                m_len = lengths.get(m, 0.0)
                m_start = cursor
                m_end = m_start + m_len
                metric_spans[m] = (m_start, m_end)
                if idx < len(visible_band_order) - 1:
                    cursor = m_end + band_gap
                else:
                    cursor = m_end

            def band_slices(start_angle, end_angle):
                span = end_angle - start_angle
                dir_gap = span * config.dir_gap_frac
                if config.ribbon_anchor_to_center:
                    dir_gap = 0.0
                dir_gap = min(dir_gap, span * 0.5)
                if dir_gap < 0:
                    dir_gap = 0.0
                mid = (start_angle + end_angle) / 2
                out_span = (start_angle, mid - dir_gap / 2)
                in_span = (mid + dir_gap / 2, end_angle)
                return {"out": out_span, "in": in_span, "center": mid}

            band_map[t] = {m: band_slices(*metric_spans[m]) for m in visible_band_order}

        # Prepare intervals per metric with per-metric center offset
        metric_intervals: Dict[str, List[Dict[str, Any]]] = {}
        for m in CHORD_METRIC_ORDER:
            if not metric_visible(m, config):
                metric_intervals[m] = []
                continue
            offset = config.ribbon_center_offset.get(m, 0.0)
            metric_intervals[m] = allocate_intervals(
                metric_edges_draw.get(m), band_map, m,
                ticker_order, config,
                centered=config.ribbon_centered,
                center_offset=offset,
            )

        # Time bins for outer ring
        time_bins = make_time_bins(window_dates_sorted, config.time_slice_bins)
        time_bins = list(reversed(time_bins))

        time_fade_alpha: Dict[datetime.date, float] = {}
        vwbr_z_brightness: Dict[Tuple[str, datetime.date], float] = {}
        if config.show_time_fade_chords and config.time_fade_use_daily_edges and config.show_vwbr_z:
            if not df_finra_daily.empty and "vwbr_z" in df_finra_daily.columns:
                df_vwbr_z_bright = df_finra_daily[["ticker", "date", "vwbr_z"]].dropna()
                df_vwbr_z_bright["date"] = pd.to_datetime(df_vwbr_z_bright["date"], errors="coerce").dt.date
                for row in df_vwbr_z_bright.itertuples():
                    z_val = getattr(row, "vwbr_z", None)
                    if z_val is None or not np.isfinite(z_val):
                        continue
                    normalized = 0.5 + (z_val / config.ring3_zscore_span)
                    normalized = max(0.0, min(1.0, normalized))
                    vwbr_z_brightness[(row.ticker, row.date)] = normalized

        if config.show_time_fade_chords and config.time_fade_use_daily_edges and window_dates_sorted:
            count = len(window_dates_sorted)
            for idx, dt in enumerate(window_dates_sorted):
                t = idx / max(1, count - 1)
                t = t ** config.time_fade_power
                time_fade_alpha[dt] = config.time_fade_min_alpha + t * (config.time_fade_max_alpha - config.time_fade_min_alpha)

        # Create figure
        fig = plt.figure(figsize=config.figure_size, facecolor=BG_COLOR)
        ax_left = config.plot_center_x - config.main_ax_size / 2
        ax_bottom = config.plot_center_y - config.main_ax_size / 2
        ax = fig.add_axes([ax_left, ax_bottom, config.main_ax_size, config.main_ax_size])
        ax.set_facecolor(BG_COLOR)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.axis('off')
        ax.set_aspect('equal')

        # Draw circular reference line at r=1.0
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color='#39424e', linewidth=1.0, alpha=0.6)

        # Ticker labels
        ticker_outer = config.ticker_outer
        for t, ang in angles.items():
            x, y = math.cos(ang), math.sin(ang)
            r = config.chord_radius + (ticker_outer - config.chord_radius) * 0.4
            rot = math.degrees(ang)
            if math.pi / 2 < ang < 3 * math.pi / 2:
                rot += 180
            ax.text(r * x, r * y, t, color='#FFFFFF', fontsize=config.ticker_fontsize, fontweight='bold',
                    ha='center', va='center', rotation=rot, rotation_mode='anchor')

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
                        mask = (df_window['ticker'] == t) & (df_window['date'].isin(bin_dates))

                        if m == 'accum':
                            val = float(df_window.loc[mask, 'accumulation_score'].mean()) if mask.any() else 50.0
                            data_by_bin.append({'value': val, 'extra': 0.0})
                            max_val = max(max_val, abs(val - 50))
                        elif m == 'dark_lit':
                            lit_vol = float(df_window.loc[mask, 'lit_total'].sum()) if mask.any() else 0.0
                            lit_ratio = float(df_window.loc[mask, 'lit_buy_ratio'].mean()) if mask.any() else 0.5
                            if not np.isfinite(lit_ratio):
                                lit_ratio = 0.5
                            data_by_bin.append({'value': lit_vol, 'extra': lit_ratio})
                            max_val = max(max_val, lit_vol)
                        elif m == 'finra_buy':
                            val = float(df_window.loc[mask, 'finra_buy'].sum()) if mask.any() else 0.0
                            vwbr_z_val = float(df_window.loc[mask, 'vwbr_z'].mean()) if mask.any() else 0.0
                            vwbr_val = float(df_window.loc[mask, 'vwbr'].mean()) if mask.any() else None
                            short_ratio_val = float(df_window.loc[mask, 'short_ratio'].mean()) if mask.any() else None
                            short_bs_val = float(df_window.loc[mask, 'short_buy_sell_ratio'].mean()) if mask.any() else None
                            data_by_bin.append({'value': val, 'extra': {
                                'vwbr_z': vwbr_z_val,
                                'vwbr': vwbr_val,
                                'short_ratio': short_ratio_val,
                                'short_buy_sell_ratio': short_bs_val,
                            }})
                            max_val = max(max_val, val)
                        else:
                            data_by_bin.append({'value': 0.0, 'extra': 0.0})

                    ring_bin_data[m][t] = data_by_bin
                ring_max_mag[m] = max_val if max_val > 0 else 1.0

            # Draw rings
            track_span = config.ring_base_thickness + config.ring_thickness_scale + config.ring_gap
            for idx, m in enumerate(RING_METRIC_ORDER):
                inner_base = ticker_outer + 0.02 + idx * track_span
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
                            if not np.isfinite(lit_ratio):
                                lit_ratio = 0.5
                            normalized = (lit_ratio - 0.4) / 0.2
                            normalized = max(0.0, min(1.0, normalized))
                            if normalized < 0.5:
                                color = blend_color('#FF6666', '#888888', normalized * 2)
                            else:
                                color = blend_color('#888888', '#66FF66', (normalized - 0.5) * 2)
                        else:  # finra_buy
                            extra_vals = extra if isinstance(extra, dict) else {}
                            color_mode = str(config.ring3_color_mode).upper()
                            if color_mode == 'VWBR_Z':
                                vwbr_z_val = extra_vals.get('vwbr_z', 0.0)
                                if vwbr_z_val is None or not np.isfinite(vwbr_z_val):
                                    vwbr_z_val = 0.0
                                normalized = 0.5 + (vwbr_z_val / config.ring3_zscore_span)
                                normalized = max(0.0, min(1.0, normalized))
                            elif color_mode == 'VWBR':
                                vwbr_val = extra_vals.get('vwbr', None)
                                if vwbr_val is None or not np.isfinite(vwbr_val):
                                    vwbr_val = 0.5
                                normalized = 0.5 + ((vwbr_val - 0.5) / config.ring3_vwbr_span)
                                normalized = max(0.0, min(1.0, normalized))
                            else:
                                short_bs = extra_vals.get('short_buy_sell_ratio', None)
                                if short_bs and short_bs > 0:
                                    log_ratio = np.log(short_bs)
                                    normalized = 0.5 + (log_ratio / 1.0)
                                    normalized = max(0.0, min(1.0, normalized))
                                else:
                                    normalized = 0.5

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

        # Draw chords per metric
        for m in CHORD_METRIC_ORDER:
            if not metric_visible(m, config):
                continue
            intervals = metric_intervals.get(m, [])
            if not intervals:
                continue

            if m == 'finra_buy':
                raw_start = METRIC_COLORS[m]['low']
                raw_end = METRIC_COLORS[m]['high']
            else:
                raw_start = METRIC_COLORS[m]['sell']
                raw_end = METRIC_COLORS[m]['buy']

            edges_df = metric_edges.get(m)
            max_flow = edges_df['flow'].max() if edges_df is not None and not edges_df.empty else 1.0
            intervals_to_draw = intervals
            if config.show_time_fade_chords and config.time_fade_use_daily_edges and time_fade_alpha:
                if m == 'vwbr_z' and vwbr_z_brightness:
                    def _draw_key(edge):
                        edge_date = edge.get('date')
                        src_val = vwbr_z_brightness.get((edge['source'], edge_date))
                        dst_val = vwbr_z_brightness.get((edge['dest'], edge_date))
                        vals = [v for v in (src_val, dst_val) if v is not None]
                        if vals:
                            return sum(vals) / len(vals)
                        return time_fade_alpha.get(edge_date, config.time_fade_min_alpha)
                    intervals_to_draw = sorted(intervals, key=_draw_key)
                else:
                    intervals_to_draw = sorted(intervals, key=lambda e: time_fade_alpha.get(e.get('date'), config.time_fade_min_alpha))

            for edge in intervals_to_draw:
                flow = edge['flow']
                lw = 0.6 + 2.2 * ((flow / max_flow) ** 0.6) if max_flow > 0 else 1.0
                edge_date = edge.get('date')
                alpha_factor = 1.0
                start_brightness = 1.0
                end_brightness = 1.0

                if config.show_time_fade_chords and config.time_fade_use_daily_edges and time_fade_alpha:
                    if edge_date in time_fade_alpha:
                        alpha_factor = time_fade_alpha[edge_date]
                        start_brightness = alpha_factor
                        end_brightness = alpha_factor

                if m == 'vwbr_z' and vwbr_z_brightness and edge_date is not None:
                    src_val = vwbr_z_brightness.get((edge['source'], edge_date))
                    dst_val = vwbr_z_brightness.get((edge['dest'], edge_date))
                    if src_val is not None:
                        start_brightness = src_val
                    if dst_val is not None:
                        end_brightness = dst_val

                start_brightness = max(0.0, min(1.0, start_brightness))
                end_brightness = max(0.0, min(1.0, end_brightness))
                start_soften = min(1.0, config.chord_color_soften + (1.0 - start_brightness) * 0.6)
                end_soften = min(1.0, config.chord_color_soften + (1.0 - end_brightness) * 0.6)
                edge_color_start = soften_color(raw_start, start_soften)
                edge_color_end = soften_color(raw_end, end_soften)

                draw_ribbon(
                    ax, edge['a0'], edge['a1'], edge['b0'], edge['b1'], config.chord_radius,
                    edge_color_start, edge_color_end,
                    config.chord_fill_alpha * alpha_factor,
                    config.chord_line_alpha * alpha_factor,
                    lw=lw,
                    use_gradient_fill=config.use_gradient_fill,
                    gradient_steps=config.chord_gradient_steps,
                )

        # Title and subtitle
        date_range = f"{window_dates_sorted[0]} -> {window_dates_sorted[-1]}" if window_dates_sorted else 'n/a'
        fig.text(0.55, 0.97, 'Institutional Money Flow Tracker', ha='center', va='top',
                 color='white', fontsize=config.title_fontsize, fontweight='bold')
        fig.text(0.55, 0.945, f'Date Range: {date_range}', ha='center', va='top',
                 color='#C9D1D9', fontsize=config.subtitle_fontsize)

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
            ('accum', 'Accumulation', METRIC_COLORS['accum']['sell'], METRIC_COLORS['accum']['buy']),
            ('short', 'Short', METRIC_COLORS['short']['sell'], METRIC_COLORS['short']['buy']),
            ('lit', 'Lit', METRIC_COLORS['lit']['sell'], METRIC_COLORS['lit']['buy']),
            ('finra_buy', 'Finra Buy', METRIC_COLORS['finra_buy']['low'], METRIC_COLORS['finra_buy']['high']),
            ('vwbr_z', 'VWBR Z', METRIC_COLORS['vwbr_z']['sell'], METRIC_COLORS['vwbr_z']['buy']),
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
            leg.text(0.24, y, METRIC_LABELS.get(key, label), color='white', fontsize=config.legend_label_fontsize, va='center')
            y -= 0.15

        # Draw category legend (upper-right)
        present_categories = [cat for cat in ['GLOBAL_MACRO', 'MAG8', 'THEMATIC_SECTORS', 'SECTOR_CORE', 'COMMODITIES', 'RATES_CREDIT', 'SPECULATIVE', 'CRYPTO']
                             if grouped.get(cat)]

        if present_categories:
            cat_leg = fig.add_axes([0.90, 0.75, 0.18, 0.18])
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
                    rect = Rectangle(
                        (i * 0.008, y - 0.04),
                        0.008,
                        0.08,
                        facecolor=color,
                        edgecolor='none',
                        clip_on=False,
                    )
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
        y -= 0.09

        y -= 0.04
        ring_leg.text(0.0, y, 'Ring Coloring - what is the direction of the flow today', color='white', fontsize=config.ring_title_fontsize, fontweight='bold', va='top')
        y -= 0.12

        ring3_mode = str(config.ring3_color_mode).upper()
        if ring3_mode == 'VWBR_Z':
            ring3_desc = 'VWBR Z (category tint bright->dark)'
        elif ring3_mode == 'VWBR':
            ring3_desc = 'VWBR buy ratio (category tint bright->dark)'
        else:
            ring3_desc = 'FINRA short buy/sell ratio (category tint bright->dark)'

        coloring_items = [
            ('Ring 1', 'Acc score 70 -> 30', RING_COLORS['accum']['positive'], RING_COLORS['accum']['negative']),
            ('Ring 2', 'Lit buy ratio (buy -> sell)', '#66FF66', '#FF6666'),
            ('Ring 3', ring3_desc, '#E6E6E6', '#444444'),
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
        latest_date = max(window_dates_sorted) if window_dates_sorted else None

        accum_trend_map = {}
        accum_anomaly_map = {}
        if not df_accum_level.empty:
            df_accum_daily_centered = df_accum_level.copy()
            df_accum_daily_centered['date'] = pd.to_datetime(df_accum_daily_centered['date'], errors='coerce').dt.date
            daily_means = df_accum_daily_centered.groupby('date')['value'].mean()
            df_accum_daily_centered['accum_centered'] = df_accum_daily_centered['value'] - df_accum_daily_centered['date'].map(daily_means)
            accum_trend_map = compute_trend_map(df_accum_daily_centered, 'accum_centered', window_dates_sorted)
            accum_anomaly_map = compute_anomaly_map_from_stats(df_accum_daily_centered, 'value', accum_stats, 'accum_mean', 'accum_std', latest_date)

        accum_buy_scores, accum_sell_scores = compute_score_maps(metric_nets.get('accum', {}), accum_trend_map, accum_anomaly_map, ticker_order)

        lit_trend_map = compute_trend_map(df_lit_daily, 'lit_net', window_dates_sorted) if not df_lit_daily.empty else {}
        lit_anomaly_map = compute_anomaly_map_from_stats(df_lit_daily, 'lit_total', lit_stats, 'lit_vol_mean', 'lit_vol_std', latest_date) if not df_lit_daily.empty else {}
        lit_buy_scores, lit_sell_scores = compute_score_maps(metric_nets.get('lit', {}), lit_trend_map, lit_anomaly_map, ticker_order)

        vwbr_z_trend_map = compute_trend_map(df_finra_daily, 'vwbr_z', window_dates_sorted) if not df_finra_daily.empty and 'vwbr_z' in df_finra_daily.columns else {}
        vwbr_z_anomaly_map = compute_abs_latest_map(df_finra_daily, 'vwbr_z', latest_date) if not df_finra_daily.empty and 'vwbr_z' in df_finra_daily.columns else {}
        vwbr_z_buy_scores, vwbr_z_sell_scores = compute_score_maps(metric_nets.get('vwbr_z', {}), vwbr_z_trend_map, vwbr_z_anomaly_map, ticker_order)

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

        # Watermark
        if config.watermark_path is None:
            candidate = Path(__file__).resolve().parent.parent / "cowwbell_waterrmark.png"
            if candidate.exists():
                config.watermark_path = str(candidate)
        if config.watermark_path:
            try:
                watermark_img = plt.imread(config.watermark_path)
                wm_h, wm_w = watermark_img.shape[:2]
                wm_aspect = (wm_h / wm_w) if wm_w else 1.0
                wm_w_frac = config.watermark_width
                wm_h_frac = wm_w_frac * wm_aspect
                wm_left = 1 - wm_w_frac - 0.03
                wm_bottom = 0.13
                wm_ax = fig.add_axes([wm_left, wm_bottom, wm_w_frac, wm_h_frac])
                wm_ax.axis('off')
                wm_ax.imshow(watermark_img, alpha=config.watermark_alpha)
            except Exception as e:
                logger.warning("Could not load watermark: %s", e)

        # Save figure
        date_tag = end_date_dt.strftime("%Y-%m-%d")
        dated_output_dir = output_dir / date_tag
        dated_output_dir.mkdir(parents=True, exist_ok=True)
        if window_dates_sorted:
            start_dt = str(window_dates_sorted[0]).replace('-', '')
            end_dt = str(window_dates_sorted[-1]).replace('-', '')
            filename = f"circos_{start_dt}_to_{end_dt}.png"
        else:
            filename = "circos_plot.png"
        output_path = dated_output_dir / filename
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
