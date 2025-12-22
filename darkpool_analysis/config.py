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
DEFAULT_TICKERS = ["TQQQ"]
EXCLUDED_FINRA_TICKERS = {"SPXW"}  # Options symbols, not equities
# ["XLK", "SMH", "XLF", "KRE", "XLE", "XLI", "XLY", "XLP", "XLV", "XLU"]


# Analysis defaults
DEFAULT_TARGET_DATE = "2025-12-19"  # Last trading day (Friday)
DEFAULT_FETCH_MODE = "daily"  # "single", "daily", or "weekly"
DEFAULT_BACKFILL_COUNT = 10  # Number of periods to fetch (days for daily, weeks for weekly)
DEFAULT_MIN_LIT_VOLUME = 10000
DEFAULT_MARKET_TZ = "US/Eastern"
DEFAULT_RTH_START = "09:30"
DEFAULT_RTH_END = "16:00"
DEFAULT_INFERENCE_VERSION = "OptionB_v1"
DEFAULT_EXPORT_CSV = False  # Export tables to CSV files

# API endpoints (not secrets)
DEFAULT_POLYGON_BASE_URL = "https://api.polygon.io"

# FINRA OTC endpoint (returns all tiers: T1, T2, OTC via tierIdentifier field)
DEFAULT_FINRA_OTC_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
DEFAULT_FINRA_TOKEN_URL = ""  # Empty = use direct API key auth (no OAuth needed)
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
    finra_otc_url: Optional[str]
    finra_otc_file: Optional[str]
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
    rth_end = _parse_time(os.getenv("RTH_END", DEFAULT_RTH_END), time(16, 0))

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
        polygon_trades_file=os.getenv("POLYGON_TRADES_FILE"),
        finra_otc_url=os.getenv("FINRA_OTC_URL", DEFAULT_FINRA_OTC_URL),
        finra_otc_file=os.getenv("FINRA_OTC_FILE"),
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
    )
