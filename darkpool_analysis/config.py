from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import importlib.util
import json
import os
from pathlib import Path
from typing import Optional
from copy import deepcopy

import pytz
from dotenv import load_dotenv

try:
    from .market_calendar import get_past_trading_days
except ImportError:
    from market_calendar import get_past_trading_days


def _load_ticker_dictionary():
    # Try darkpool_analysis first, then Special_tools for backward compatibility
    ticker_path = Path(__file__).resolve().parent / "ticker_dictionary.py"
    if not ticker_path.exists():
        ticker_path = Path(__file__).resolve().parent.parent / "Special_tools" / "ticker_dictionary.py"
    if not ticker_path.exists():
        raise FileNotFoundError(f"ticker_dictionary.py not found in darkpool_analysis/ or Special_tools/")
    spec = importlib.util.spec_from_file_location("ticker_dictionary", ticker_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ticker_dictionary.py from {ticker_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expand_with_constituents(tickers: list[str], all_ticker_maps: list[dict]) -> list[str]:
    """Expand tickers to include constituents from ticker dictionaries.

    Looks up each ticker in all category maps (SECTOR_CORE, THEMATIC_SECTORS,
    GLOBAL_MACRO, COMMODITIES, MAG8, RATES_CREDIT) and adds any found constituents.
    """
    expanded = []
    seen = set()

    for ticker in tickers:
        # Add the ticker itself
        if ticker not in seen:
            expanded.append(ticker)
            seen.add(ticker)

        # Look for constituents in all category maps
        for ticker_map in all_ticker_maps:
            if ticker in ticker_map:
                constituents = ticker_map[ticker]
                # Handle both list and dict formats
                if isinstance(constituents, list):
                    for constituent in constituents:
                        if constituent not in seen:
                            expanded.append(constituent)
                            seen.add(constituent)

    return expanded


# =============================================================================
# Default Configuration (can be overridden via .env)
# =============================================================================

# -----------------------------------------------------------------------------
# Ticker groups (used when TICKERS_TYPE is set)
# -----------------------------------------------------------------------------
_TICKER_DICT = _load_ticker_dictionary()

# Load flat dictionaries directly (no SECTOR_ZOOM_MAP wrapper)
SECTOR_CORE = _TICKER_DICT.SECTOR_CORE
THEMATIC_SECTORS = _TICKER_DICT.THEMATIC_SECTORS
GLOBAL_MACRO = _TICKER_DICT.GLOBAL_MACRO
COMMODITIES = _TICKER_DICT.COMMODITIES
MAG8 = _TICKER_DICT.MAG8
RATES_CREDIT = _TICKER_DICT.RATES_CREDIT

# Extract ticker lists from flat dictionaries
SECTOR_CORE_TICKERS = list(SECTOR_CORE.keys())
THEMATIC_SECTORS_TICKERS = list(THEMATIC_SECTORS.keys())
GLOBAL_MACRO_TICKERS = list(GLOBAL_MACRO.keys())
COMMODITIES_TICKERS = list(COMMODITIES.keys())
MAG8_TICKERS = list(MAG8["MAG8"])
RATES_CREDIT_TICKERS = list(RATES_CREDIT.keys())
SPECULATIVE_TICKERS = list(_TICKER_DICT.SPECULATIVE_TICKERS)
CRYPTO_TICKERS = list(_TICKER_DICT.CRYPTO_TICKERS)

# Combined list for constituent expansion
ALL_TICKER_MAPS = [SECTOR_CORE, THEMATIC_SECTORS, GLOBAL_MACRO, COMMODITIES, MAG8, RATES_CREDIT]

EXCLUDED_FINRA_TICKERS = {"SPXW"}  # Options symbols, not equities

# -----------------------------------------------------------------------------
# User-facing defaults (most commonly edited)
# -----------------------------------------------------------------------------

TICKERS_TYPE =  ["ALL"]  # ["SECTOR", "THEMATIC", "GLOBAL", "COMMODITIES", "MAG8", "RATES", "CRYPTO", "SPECULATIVE"], ["SINGLE"], ["ALL"]
DEFAULT_TICKERS = ["XLP" ]
FETCH_INDICES_CONSTITUENTS = False  # When True, also fetch constituents of index/ETF tickers

DEFAULT_TARGET_DATE = "TODAY"  # "TODAY" for current date, or specific date like "2026-01-08"
DEFAULT_FETCH_MODE = "daily"  # "single", "daily", or "weekly"
DEFAULT_BACKFILL_COUNT = 30  # Number of periods to fetch (days for daily, weeks for weekly)

DEFAULT_MARKET_TZ = "US/Eastern"
DEFAULT_RTH_START = "09:30"
DEFAULT_RTH_END = "16:15"
DEFAULT_MIN_LIT_VOLUME = 10000  # Minimum lit volume for ratios to be valid
DEFAULT_INCLUDE_POLYGON_ONLY_TICKERS = True  # Include tickers without FINRA coverage

# -----------------------------------------------------------------------------
# Polygon Trades Mode (controls data granularity for lit direction inference)
# -----------------------------------------------------------------------------
# - "tick"  : Fetch individual trades with bid/ask for NBBO classification.
#             Most accurate but slowest (1000s of records, paginated requests).
# - "minute": Fetch 1-minute bars, synthesize bid/ask from OHLC.
#             Faster but less accurate (bid/ask approximated from bar range).
# - "daily" : Skip lit inference entirely, use only short-sale ratio.
#             Fastest option when you only need short pressure signals.
DEFAULT_POLYGON_TRADES_MODE = "minute"

# -----------------------------------------------------------------------------
# Caching and output behavior
# -----------------------------------------------------------------------------
# When True, bypass all output settings (no files saved to disk).
# Overrides: RENDER_METRICS_PLOTS, RENDER_PRICE_CHARTS, RENDER_SUMMARY_DASHBOARD,
#            RENDER_TABLES, COMBINATION_PLOT to False.
BYPASS_OUTPUT_SETTING = False

# When True, check polygon_ingestion_state before fetching and skip symbols
# that have already been fetched for a given date+source. No TTL (cache forever).
DEFAULT_SKIP_CACHED = True
DEFAULT_PRICE_BAR_TIMEFRAME = "daily"  # daily, weekly, monthly
DEFAULT_PANEL1_METRIC = "finra_buy_volume"  # "vw_flow", "combined_ratio", or "finra_buy_volume"
DEFAULT_RENDER_METRICS_PLOTS = True  # Render metrics plots (plotter.py)
DEFAULT_RENDER_PRICE_CHARTS = True # Render OHLC price charts
DEFAULT_RENDER_SUMMARY_DASHBOARD = True  # Render sector summary dashboard
DEFAULT_RENDER_TABLES = False  # Render daily metrics tables (HTML/PNG)
DEFAULT_COMBINATION_PLOT = False  # Render combined multi-ticker plot
DEFAULT_PLOT_TRADING_GAPS = False  # Keep weekend/holiday gaps in plots
DEFAULT_EXPORT_CSV = False  # Export tables to CSV files


# -----------------------------------------------------------------------------
# Provenance and scoring controls
# -----------------------------------------------------------------------------
# This string is stored in the database to track which algorithm version
# produced the data. Useful for debugging and reproducibility.
DEFAULT_INFERENCE_VERSION = "PhaseA_v1"
DEFAULT_ACCUMULATION_SHORT_Z_SOURCE = "short_buy_sell_ratio_z"  # "short_buy_sell_ratio_z", "vw_flow_z", or "finra_buy_volume_z"

# Z-score windows and thresholds
DEFAULT_SHORT_Z_WINDOW = 20
DEFAULT_RETURN_Z_WINDOW = 20
DEFAULT_ZSCORE_MIN_PERIODS = 5
DEFAULT_SHORT_Z_HIGH = 1.0
DEFAULT_SHORT_Z_LOW = -1.0
DEFAULT_RETURN_Z_MIN = 0.5
DEFAULT_SHORT_SALE_PREFERRED_SOURCE = "CNMS"  # Preferred FINRA market code
DEFAULT_INDEX_CONSTITUENTS_DIR = "data/constituents"  # Index member files
DEFAULT_INDEX_PROXY_MAP = {"SPX": "SPY"}  # Index return proxy map

# Composite Score Weights (Phase 4)
DEFAULT_COMPOSITE_W_SHORT = 0.55  # Short sale primary signal
DEFAULT_COMPOSITE_W_LIT = 0.30    # Lit flow confirmation
DEFAULT_COMPOSITE_W_PRICE = 0.15  # Price momentum

# Intensity Scale Bounds (OTC participation modulation)
DEFAULT_INTENSITY_SCALE_MIN = 0.7  # Low OTC dampens signal
DEFAULT_INTENSITY_SCALE_MAX = 1.3  # High OTC amplifies signal

# -----------------------------------------------------------------------------
# Options premium panel
# -----------------------------------------------------------------------------
# Tickers with daily (0DTE) options expiration
DAILY_EXPIRATION_TICKERS = {"SPY", "SPX", "SPXW", "QQQ", "IWM", "XSP"}
DEFAULT_OPTIONS_STRIKE_COUNT = 20  # 30 contracts from ATM
DEFAULT_OPTIONS_MIN_PREMIUM_HIGHLIGHT = 4.0  # $M threshold for highlighting
DEFAULT_FETCH_OPTIONS_PREMIUM = True  # Enable/disable options premium fetching

# Options premium display mode (affects visualization only, all data is always stored)
# - "TOTAL": Current behavior - total call/put premium (backwards compatible)
# - "WTD_STYLE": OTM focus with ITM call hedge warning (WTD's recommended approach)
# - "FULL_BREAKDOWN": Show all 4 categories (OTM/ITM x Call/Put)
DEFAULT_OPTIONS_PREMIUM_DISPLAY_MODE = "WTD_STYLE"

# Threshold for ITM call hedge warning (% of total call premium)
# When ITM calls exceed this ratio, shows warning marker in WTD_STYLE mode
DEFAULT_ITM_CALL_HEDGE_THRESHOLD = 0.30  # 30%

DEFAULT_TABLE_STYLE = {
    "mode": "scan",
    "font_family": '"Segoe UI", Arial, sans-serif',
    "font_family_numeric": '"Consolas", "Courier New", monospace',
    "base_font_size": 20,
    "header_font_size": 18,
    "numeric_font_scale": 1.0,
    "row_padding_y": 10,
    "row_padding_x": 8,
    "neutral_text_opacity": 0.78,
    "header_opacity": 0.72,
    "signal_opacity_strong": 0.95,
    "signal_opacity_muted": 0.6,
    "group_alt_strength": 0.12,
    "group_separator_px": 2,
    "group_separator_opacity": 0.35,
    "group_separator_padding": 2,
    "zone_tint_alpha": 0.035,
    "gridline_every": 5,
    "strong_signal_columns": ["short_z"],
    "muted_signal_columns": ["return_pct", "lit_buy_z"],
    "status_glyphs": {
        "pressure": {"Accumulating": "up", "Distribution": "down", "Neutral": "dot", "NA": "dot"},
        "otc_status": {"Anchored": "dot", "Staled": "dot", "None": "dot", "NA": "dot"},
    },
    "palette": {
        "background": "#0f0f10",
        "header_bg": "#141518",
        "panel_bg": "#141416",
        "row_bg": "#0f1012",
        "row_alt_bg": "#15161a",
        "text": "#e6e6e6",
        "text_muted": "#8b8b8b",
        "border": "#26262a",
        "green": "#3cbf8a",
        "red": "#d06c6c",
        "yellow": "#d6b35b",
        "orange": "#972727",
        "bright_green": "#42f5a7",
        "bright_red": "#ff5c5c",
        "cyan": "#7ab6ff",
        "white": "#ffffff",
    },
    "zones": {
        "id": "#121319",
        "volume": "#10151a",
        "ratio": "#141117",
        "status": "#121213",
    },
    "modes": {
        "scan": {"base_font_size": 21, "row_padding_y": 9, "neutral_text_opacity": 0.72},
        "analysis": {"base_font_size": 23, "row_padding_y": 12, "neutral_text_opacity": 0.85},
    },
}

# -----------------------------------------------------------------------------
# API endpoints (not secrets)
# -----------------------------------------------------------------------------
DEFAULT_POLYGON_BASE_URL = "https://api.polygon.io"

# FINRA OTC endpoint (returns all tiers: T1, T2, OTC via tierIdentifier field)
DEFAULT_FINRA_OTC_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
DEFAULT_FINRA_SHORT_SALE_URL = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
DEFAULT_FINRA_TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
DEFAULT_FINRA_REQUEST_METHOD = "POST"
DEFAULT_FINRA_CDN_URL = "https://cdn.finra.org/equity/regsho/daily"

def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return date.fromisoformat(value)


def _parse_time(value: Optional[str], fallback: time) -> time:
    if not value:
        return fallback
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def _parse_json_env(name: str) -> Optional[dict]:
    raw = os.getenv(name)
    if not raw:
        return None
    return json.loads(raw)


def _parse_csv_env(name: str) -> Optional[list[str]]:
    raw = os.getenv(name)
    if not raw:
        return None
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _resolve_path(root_dir: Path, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root_dir / path
    return str(path)


def _get_past_fridays(count: int, from_date: date) -> list[date]:
    """Return the last N Fridays up to and including from_date."""
    fridays = []
    current = from_date

    # Find the most recent Friday (or from_date if it's Friday)
    days_since_friday = (current.weekday() - 4) % 7
    if days_since_friday > 0:
        current = current - timedelta(days=days_since_friday)

    # Collect N Fridays going backwards
    while len(fridays) < count:
        fridays.append(current)
        current -= timedelta(days=7)

    return fridays  # Most recent first


def _get_past_trading_days(count: int, from_date: date) -> list[date]:
    """Return the last N trading days (excludes weekends and holidays)."""
    return get_past_trading_days(count, from_date)


@dataclass(frozen=True)
class Config:
    root_dir: Path
    data_dir: Path
    output_dir: Path
    table_dir: Path
    plot_dir: Path
    price_chart_dir: Path
    db_path: Path
    tickers: list[str]
    finra_tickers: list[str]
    min_lit_volume: float
    fetch_mode: str
    backfill_count: int
    target_date: date
    target_dates: list[date]
    market_tz: str
    rth_start: time
    rth_end: time
    inference_version: str
    polygon_trades_mode: str  # "tick", "minute", or "daily"
    skip_cached: bool  # Skip fetching if already in DB
    bypass_output_setting: bool  # When True, disable all output file generation
    export_csv: bool
    render_metrics_plots: bool
    render_tables: bool
    render_price_charts: bool
    render_summary_dashboard: bool
    price_bar_timeframe: str
    combination_plot: bool
    plot_trading_gaps: bool
    panel1_metric: str
    polygon_api_key: Optional[str]
    polygon_base_url: str
    polygon_trades_file: Optional[str]
    polygon_daily_agg_file: Optional[str]
    polygon_daily_agg_dir: Optional[str]
    finra_otc_url: Optional[str]
    finra_otc_file: Optional[str]
    finra_otc_dir: Optional[str]
    finra_short_sale_url: Optional[str]
    finra_short_sale_file: Optional[str]
    finra_short_sale_dir: Optional[str]
    finra_api_key: Optional[str]
    finra_api_secret: Optional[str]
    finra_token_url: Optional[str]
    finra_cdn_url: str
    finra_request_method: str
    finra_request_json: Optional[dict]
    finra_request_params: Optional[dict]
    finra_date_field: Optional[str]
    finra_symbol_field: Optional[str]
    finra_volume_field: Optional[str]
    finra_trade_count_field: Optional[str]
    include_polygon_only_tickers: bool
    fetch_indices_constituents: bool  # When True, expand index/ETF tickers to include constituents
    short_z_window: int
    return_z_window: int
    zscore_min_periods: int
    short_z_high: float
    short_z_low: float
    return_z_min: float
    short_sale_preferred_source: str
    index_constituents_dir: Path
    index_constituents_file: Optional[str]
    index_proxy_map: dict
    table_style: dict
    composite_w_short: float
    composite_w_lit: float
    composite_w_price: float
    intensity_scale_min: float
    intensity_scale_max: float
    accumulation_short_z_source: str
    # Options premium panel
    options_strike_count: int
    options_min_premium_highlight: float
    fetch_options_premium: bool
    options_premium_display_mode: str  # "TOTAL", "WTD_STYLE", or "FULL_BREAKDOWN"
    itm_call_hedge_threshold: float  # Threshold for ITM call hedge warning


def load_config() -> Config:
    root_dir = Path(__file__).resolve().parent
    project_root = root_dir.parent  # Go up one level to darkpool/
    load_dotenv(project_root / ".env")

    market_tz = os.getenv("MARKET_TZ", DEFAULT_MARKET_TZ)
    tz = pytz.timezone(market_tz)

    # Handle "TODAY" as a special value for target date
    target_date_str = os.getenv("TARGET_DATE", DEFAULT_TARGET_DATE)
    if target_date_str and target_date_str.upper() == "TODAY":
        target_date = datetime.now(tz).date()
    else:
        target_date = _parse_date(target_date_str)
        if not target_date:
            target_date = datetime.now(tz).date()

    fetch_mode = os.getenv("FETCH_MODE", DEFAULT_FETCH_MODE).lower()
    backfill_count = int(os.getenv("BACKFILL_COUNT", str(DEFAULT_BACKFILL_COUNT)))

    if fetch_mode == "weekly":
        target_dates = _get_past_fridays(backfill_count, target_date)
    elif fetch_mode == "daily":
        target_dates = _get_past_trading_days(backfill_count, target_date)
    else:  # "single"
        target_dates = [target_date]

    # Map ticker type names to their lists
    ticker_type_map = {
        "SINGLE": DEFAULT_TICKERS,
        "SECTOR": SECTOR_CORE_TICKERS,
        "THEMATIC": THEMATIC_SECTORS_TICKERS,
        "GLOBAL": GLOBAL_MACRO_TICKERS,
        "COMMODITIES": COMMODITIES_TICKERS,
        "MAG8": MAG8_TICKERS,
        "RATES": RATES_CREDIT_TICKERS,
        "SPECULATIVE": SPECULATIVE_TICKERS,
        "CRYPTO": CRYPTO_TICKERS,
    }

    # Compute ALL by combining all groups (excluding SINGLE), preserving order
    all_tickers = []
    seen_all = set()
    for key in ["SECTOR", "THEMATIC", "GLOBAL", "COMMODITIES", "MAG8", "RATES", "SPECULATIVE", "CRYPTO"]:
        for ticker in ticker_type_map[key]:
            if ticker not in seen_all:
                all_tickers.append(ticker)
                seen_all.add(ticker)
    ticker_type_map["ALL"] = all_tickers

    # Support both string and list for TICKERS_TYPE
    tickers_type_raw = os.getenv("TICKERS_TYPE") or TICKERS_TYPE

    if isinstance(tickers_type_raw, list):
        # Multiple types: combine all tickers, preserving order and removing duplicates
        selected_tickers = []
        seen = set()
        for t in tickers_type_raw:
            t_upper = t.upper()
            if t_upper not in ticker_type_map:
                raise ValueError(f"Unknown TICKERS_TYPE: {t_upper}")
            for ticker in ticker_type_map[t_upper]:
                if ticker not in seen:
                    selected_tickers.append(ticker)
                    seen.add(ticker)
    else:
        # Single type (string)
        t_upper = tickers_type_raw.upper()
        if t_upper not in ticker_type_map:
            raise ValueError(f"Unknown TICKERS_TYPE: {t_upper}")
        selected_tickers = ticker_type_map[t_upper]

    tickers = _parse_csv_env("TICKERS") or selected_tickers

    # Expand with constituents if enabled
    fetch_indices_constituents = os.getenv(
        "FETCH_INDICES_CONSTITUENTS", str(FETCH_INDICES_CONSTITUENTS)
    ).lower() in ("true", "1", "yes")
    if fetch_indices_constituents:
        tickers = _expand_with_constituents(tickers, ALL_TICKER_MAPS)

    finra_tickers = [ticker for ticker in tickers if ticker not in EXCLUDED_FINRA_TICKERS]

    rth_start = _parse_time(os.getenv("RTH_START", DEFAULT_RTH_START), time(9, 30))
    rth_end = _parse_time(os.getenv("RTH_END", DEFAULT_RTH_END), time(16, 15))
    index_constituents_dir = Path(
        os.getenv("INDEX_CONSTITUENTS_DIR", DEFAULT_INDEX_CONSTITUENTS_DIR)
    )
    if not index_constituents_dir.is_absolute():
        index_constituents_dir = root_dir / index_constituents_dir

    polygon_trades_file = _resolve_path(root_dir, os.getenv("POLYGON_TRADES_FILE"))
    polygon_daily_agg_file = _resolve_path(root_dir, os.getenv("POLYGON_DAILY_AGG_FILE"))
    polygon_daily_agg_dir = _resolve_path(root_dir, os.getenv("POLYGON_DAILY_AGG_DIR"))
    finra_otc_file = _resolve_path(root_dir, os.getenv("FINRA_OTC_FILE"))
    finra_otc_dir = _resolve_path(root_dir, os.getenv("FINRA_OTC_DIR"))
    finra_short_sale_file = _resolve_path(root_dir, os.getenv("FINRA_SHORT_SALE_FILE"))
    finra_short_sale_dir = _resolve_path(root_dir, os.getenv("FINRA_SHORT_SALE_DIR"))
    index_constituents_file = _resolve_path(root_dir, os.getenv("INDEX_CONSTITUENTS_FILE"))

    # Bypass output setting - when True, disables all file output generation
    bypass_output_setting = os.getenv(
        "BYPASS_OUTPUT_SETTING", str(BYPASS_OUTPUT_SETTING)
    ).lower() in ("true", "1", "yes")

    return Config(
        root_dir=root_dir,
        data_dir=root_dir / "data",
        output_dir=root_dir / "output",
        table_dir=root_dir / "output" / "tables",
        plot_dir=root_dir / "output" / "plots",
        price_chart_dir=root_dir / "output" / "price_charts",
        db_path=root_dir / "data" / "darkpool.duckdb",
        tickers=tickers,
        finra_tickers=finra_tickers,
        min_lit_volume=float(os.getenv("MIN_LIT_VOLUME", str(DEFAULT_MIN_LIT_VOLUME))),
        fetch_mode=fetch_mode,
        backfill_count=backfill_count,
        target_date=target_date,
        target_dates=target_dates,
        market_tz=market_tz,
        rth_start=rth_start,
        rth_end=rth_end,
        inference_version=os.getenv("INFERENCE_VERSION", DEFAULT_INFERENCE_VERSION),
        polygon_trades_mode=os.getenv("POLYGON_TRADES_MODE", DEFAULT_POLYGON_TRADES_MODE).lower(),
        skip_cached=os.getenv("SKIP_CACHED", str(DEFAULT_SKIP_CACHED)).lower() in ("true", "1", "yes"),
        bypass_output_setting=bypass_output_setting,
        export_csv=os.getenv("EXPORT_CSV", str(DEFAULT_EXPORT_CSV)).lower() in ("true", "1", "yes"),
        render_metrics_plots=False if bypass_output_setting else os.getenv(
            "RENDER_METRICS_PLOTS", str(DEFAULT_RENDER_METRICS_PLOTS)
        ).lower() in ("true", "1", "yes"),
        render_tables=False if bypass_output_setting else os.getenv("RENDER_TABLES", str(DEFAULT_RENDER_TABLES)).lower() in ("true", "1", "yes"),
        render_price_charts=False if bypass_output_setting else os.getenv(
            "RENDER_PRICE_CHARTS", str(DEFAULT_RENDER_PRICE_CHARTS)
        ).lower() in ("true", "1", "yes"),
        render_summary_dashboard=False if bypass_output_setting else os.getenv(
            "RENDER_SUMMARY_DASHBOARD", str(DEFAULT_RENDER_SUMMARY_DASHBOARD)
        ).lower() in ("true", "1", "yes"),
        price_bar_timeframe=os.getenv(
            "PRICE_BAR_TIMEFRAME", DEFAULT_PRICE_BAR_TIMEFRAME
        ).lower(),
        combination_plot=False if bypass_output_setting else os.getenv(
            "COMBINATION_PLOT", str(DEFAULT_COMBINATION_PLOT)
        ).lower() in ("true", "1", "yes"),
        plot_trading_gaps=os.getenv(
            "PLOT_TRADING_GAPS", str(DEFAULT_PLOT_TRADING_GAPS)
        ).lower() in ("true", "1", "yes"),
        panel1_metric=os.getenv("PANEL1_METRIC", DEFAULT_PANEL1_METRIC).lower(),
        polygon_api_key=os.getenv("POLYGON_API_KEY"),
        polygon_base_url=os.getenv("POLYGON_BASE_URL", DEFAULT_POLYGON_BASE_URL),
        polygon_trades_file=polygon_trades_file,
        polygon_daily_agg_file=polygon_daily_agg_file,
        polygon_daily_agg_dir=polygon_daily_agg_dir,
        finra_otc_url=os.getenv("FINRA_OTC_URL", DEFAULT_FINRA_OTC_URL),
        finra_otc_file=finra_otc_file,
        finra_otc_dir=finra_otc_dir,
        finra_short_sale_url=os.getenv("FINRA_SHORT_SALE_URL", DEFAULT_FINRA_SHORT_SALE_URL),
        finra_short_sale_file=finra_short_sale_file,
        finra_short_sale_dir=finra_short_sale_dir,
        finra_api_key=os.getenv("FINRA_API_KEY"),
        finra_api_secret=os.getenv("FINRA_API_SECRET"),
        finra_token_url=os.getenv("FINRA_TOKEN_URL", DEFAULT_FINRA_TOKEN_URL),
        finra_cdn_url=os.getenv("FINRA_CDN_URL", DEFAULT_FINRA_CDN_URL),
        finra_request_method=os.getenv("FINRA_REQUEST_METHOD", DEFAULT_FINRA_REQUEST_METHOD).upper(),
        finra_request_json=_parse_json_env("FINRA_REQUEST_JSON"),
        finra_request_params=_parse_json_env("FINRA_REQUEST_PARAMS"),
        finra_date_field=os.getenv("FINRA_DATE_FIELD"),
        finra_symbol_field=os.getenv("FINRA_SYMBOL_FIELD"),
        finra_volume_field=os.getenv("FINRA_VOLUME_FIELD"),
        finra_trade_count_field=os.getenv("FINRA_TRADE_COUNT_FIELD"),
        include_polygon_only_tickers=os.getenv(
            "INCLUDE_POLYGON_ONLY_TICKERS", str(DEFAULT_INCLUDE_POLYGON_ONLY_TICKERS)
        ).lower() in ("true", "1", "yes"),
        fetch_indices_constituents=fetch_indices_constituents,
        short_z_window=int(os.getenv("SHORT_Z_WINDOW", str(DEFAULT_SHORT_Z_WINDOW))),
        return_z_window=int(os.getenv("RETURN_Z_WINDOW", str(DEFAULT_RETURN_Z_WINDOW))),
        zscore_min_periods=int(os.getenv("ZSCORE_MIN_PERIODS", str(DEFAULT_ZSCORE_MIN_PERIODS))),
        short_z_high=float(os.getenv("SHORT_Z_HIGH", str(DEFAULT_SHORT_Z_HIGH))),
        short_z_low=float(os.getenv("SHORT_Z_LOW", str(DEFAULT_SHORT_Z_LOW))),
        return_z_min=float(os.getenv("RETURN_Z_MIN", str(DEFAULT_RETURN_Z_MIN))),
        short_sale_preferred_source=os.getenv(
            "SHORT_SALE_PREFERRED_SOURCE", DEFAULT_SHORT_SALE_PREFERRED_SOURCE
        ).upper(),
        index_constituents_dir=index_constituents_dir,
        index_constituents_file=index_constituents_file,
        index_proxy_map=_parse_json_env("INDEX_PROXY_MAP") or DEFAULT_INDEX_PROXY_MAP,
        table_style=deepcopy(DEFAULT_TABLE_STYLE),
        composite_w_short=float(os.getenv("COMPOSITE_W_SHORT", str(DEFAULT_COMPOSITE_W_SHORT))),
        composite_w_lit=float(os.getenv("COMPOSITE_W_LIT", str(DEFAULT_COMPOSITE_W_LIT))),
        composite_w_price=float(os.getenv("COMPOSITE_W_PRICE", str(DEFAULT_COMPOSITE_W_PRICE))),
        intensity_scale_min=float(os.getenv("INTENSITY_SCALE_MIN", str(DEFAULT_INTENSITY_SCALE_MIN))),
        intensity_scale_max=float(os.getenv("INTENSITY_SCALE_MAX", str(DEFAULT_INTENSITY_SCALE_MAX))),
        accumulation_short_z_source=os.getenv(
            "ACCUMULATION_SHORT_Z_SOURCE", DEFAULT_ACCUMULATION_SHORT_Z_SOURCE
        ).lower(),
        # Options premium panel
        options_strike_count=int(os.getenv("OPTIONS_STRIKE_COUNT", str(DEFAULT_OPTIONS_STRIKE_COUNT))),
        options_min_premium_highlight=float(os.getenv("OPTIONS_MIN_PREMIUM_HIGHLIGHT", str(DEFAULT_OPTIONS_MIN_PREMIUM_HIGHLIGHT))),
        fetch_options_premium=os.getenv("FETCH_OPTIONS_PREMIUM", str(DEFAULT_FETCH_OPTIONS_PREMIUM)).lower() in ("true", "1", "yes"),
        options_premium_display_mode=os.getenv("OPTIONS_PREMIUM_DISPLAY_MODE", DEFAULT_OPTIONS_PREMIUM_DISPLAY_MODE).upper(),
        itm_call_hedge_threshold=float(os.getenv("ITM_CALL_HEDGE_THRESHOLD", str(DEFAULT_ITM_CALL_HEDGE_THRESHOLD))),
    )
