from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import json
import os
from pathlib import Path
from typing import Optional

import pytz
from dotenv import load_dotenv

# =============================================================================
# Default Configuration (can be overridden via .env)
# =============================================================================
DEFAULT_TICKERS = ["AMZN"]
EXCLUDED_FINRA_TICKERS = {"SPXW"}  # Options symbols, not equities

### US_SECTOR_CORE
# "XLK",  // Technology (platforms, software, mega-cap growth)
# "SMH",  // Semiconductors (hardware + capex cycle)
# "XLF",  // Financials (money-center banks, insurers)
# "KRE",  // Regional Banks (rates, liquidity stress)
# "XLE",  // Energy (commodities, inflation hedge)
# "XLI",  // Industrials (cyclicals, defense, capex)
# "XLY",  // Consumer Discretionary (growth beta)
# "XLP",  // Consumer Staples (defensive)
# "XLV",  // Health Care (defensive + policy)
# "XLU",  // Utilities (rates, yield proxy)

### Global macro / index rotation set (ETF proxies)
# "SPY",   // US large-cap core (S&P 500)
# "QQQ",   // US growth / tech beta
# "IWM",   // US small caps (domestic liquidity)
# "EFA",   // Developed markets ex-US (EU + Japan)
# "EEM",   // Emerging markets (global risk / China beta)
# "EWJ",   // Japan (yield-curve / FX-sensitive)
# "FXI",   // China large-cap (policy + growth stress)
# "VGK",   // Europe (value / banks / energy tilt)
# "TLT",   // US long rates (risk-off / duration)
# "VIXY",  // Volatility (risk regime)
# "UUP",   // US dollar (global liquidity / stress)



# Analysis defaults
DEFAULT_TARGET_DATE = "2025-12-22"  # Last trading day (Monday)
DEFAULT_FETCH_MODE = "daily"  # "single", "daily", or "weekly"
DEFAULT_BACKFILL_COUNT = 27  # Number of periods to fetch (days for daily, weeks for weekly)
DEFAULT_MIN_LIT_VOLUME = 10000
DEFAULT_MARKET_TZ = "US/Eastern"
DEFAULT_RTH_START = "09:30"
DEFAULT_RTH_END = "16:15"
DEFAULT_INFERENCE_VERSION = "PhaseA_v1"
DEFAULT_EXPORT_CSV = False  # Export tables to CSV files
DEFAULT_INCLUDE_POLYGON_ONLY_TICKERS = True  # Include tickers without FINRA data (Polygon-only)
DEFAULT_SHORT_Z_WINDOW = 20
DEFAULT_RETURN_Z_WINDOW = 20
DEFAULT_ZSCORE_MIN_PERIODS = 5
DEFAULT_SHORT_Z_HIGH = 1.0
DEFAULT_SHORT_Z_LOW = -1.0
DEFAULT_RETURN_Z_MIN = 0.5
DEFAULT_SHORT_SALE_PREFERRED_SOURCE = "CNMS"
DEFAULT_INDEX_CONSTITUENTS_DIR = "data/constituents"
DEFAULT_INDEX_PROXY_MAP = {"SPX": "SPY"}

# API endpoints (not secrets)
DEFAULT_POLYGON_BASE_URL = "https://api.polygon.io"

# FINRA OTC endpoint (returns all tiers: T1, T2, OTC via tierIdentifier field)
DEFAULT_FINRA_OTC_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
DEFAULT_FINRA_SHORT_SALE_URL = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
DEFAULT_FINRA_TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
DEFAULT_FINRA_REQUEST_METHOD = "POST"


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
    """Return the last N trading days (excludes weekends)."""
    days = []
    current = from_date

    while len(days) < count:
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            days.append(current)
        current -= timedelta(days=1)

    return days  # Most recent first


@dataclass(frozen=True)
class Config:
    root_dir: Path
    data_dir: Path
    output_dir: Path
    table_dir: Path
    plot_dir: Path
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
    export_csv: bool
    polygon_api_key: Optional[str]
    polygon_base_url: str
    polygon_trades_file: Optional[str]
    polygon_daily_agg_file: Optional[str]
    polygon_daily_agg_dir: Optional[str]
    finra_otc_url: Optional[str]
    finra_otc_file: Optional[str]
    finra_short_sale_url: Optional[str]
    finra_short_sale_file: Optional[str]
    finra_short_sale_dir: Optional[str]
    finra_api_key: Optional[str]
    finra_api_secret: Optional[str]
    finra_token_url: Optional[str]
    finra_request_method: str
    finra_request_json: Optional[dict]
    finra_request_params: Optional[dict]
    finra_date_field: Optional[str]
    finra_symbol_field: Optional[str]
    finra_volume_field: Optional[str]
    finra_trade_count_field: Optional[str]
    include_polygon_only_tickers: bool
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


def load_config() -> Config:
    root_dir = Path(__file__).resolve().parent
    project_root = root_dir.parent  # Go up one level to darkpool/
    load_dotenv(project_root / ".env")

    market_tz = os.getenv("MARKET_TZ", DEFAULT_MARKET_TZ)
    tz = pytz.timezone(market_tz)
    target_date = _parse_date(os.getenv("TARGET_DATE")) or _parse_date(DEFAULT_TARGET_DATE)
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

    tickers = _parse_csv_env("TICKERS") or DEFAULT_TICKERS
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
    finra_short_sale_file = _resolve_path(root_dir, os.getenv("FINRA_SHORT_SALE_FILE"))
    finra_short_sale_dir = _resolve_path(root_dir, os.getenv("FINRA_SHORT_SALE_DIR"))
    index_constituents_file = _resolve_path(root_dir, os.getenv("INDEX_CONSTITUENTS_FILE"))

    return Config(
        root_dir=root_dir,
        data_dir=root_dir / "data",
        output_dir=root_dir / "output",
        table_dir=root_dir / "output" / "tables",
        plot_dir=root_dir / "output" / "plots",
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
        export_csv=os.getenv("EXPORT_CSV", str(DEFAULT_EXPORT_CSV)).lower() in ("true", "1", "yes"),
        polygon_api_key=os.getenv("POLYGON_API_KEY"),
        polygon_base_url=os.getenv("POLYGON_BASE_URL", DEFAULT_POLYGON_BASE_URL),
        polygon_trades_file=polygon_trades_file,
        polygon_daily_agg_file=polygon_daily_agg_file,
        polygon_daily_agg_dir=polygon_daily_agg_dir,
        finra_otc_url=os.getenv("FINRA_OTC_URL", DEFAULT_FINRA_OTC_URL),
        finra_otc_file=finra_otc_file,
        finra_short_sale_url=os.getenv("FINRA_SHORT_SALE_URL", DEFAULT_FINRA_SHORT_SALE_URL),
        finra_short_sale_file=finra_short_sale_file,
        finra_short_sale_dir=finra_short_sale_dir,
        finra_api_key=os.getenv("FINRA_API_KEY"),
        finra_api_secret=os.getenv("FINRA_API_SECRET"),
        finra_token_url=os.getenv("FINRA_TOKEN_URL", DEFAULT_FINRA_TOKEN_URL),
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
    )
