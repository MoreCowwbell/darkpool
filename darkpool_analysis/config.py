from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import json
import os
from pathlib import Path
from typing import Optional

import pytz
from dotenv import load_dotenv

DEFAULT_TICKERS = ["TQQQ", "SPY", "QQQ", "IWM", "NVDA"]  # actual ETFs ["XLK", "SMH", "XLF", "KRE", "XLE", "XLI", "XLY", "XLP", "XLV", "XLU"]
EXCLUDED_FINRA_TICKERS = {"SPXW"}


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
    run_mode: str
    target_date: date
    market_tz: str
    rth_start: time
    rth_end: time
    inference_version: str
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

    market_tz = os.getenv("MARKET_TZ", "US/Eastern")
    tz = pytz.timezone(market_tz)
    target_date = _parse_date(os.getenv("TARGET_DATE"))
    if not target_date:
        target_date = datetime.now(tz).date()

    tickers = _parse_csv_env("TICKERS") or DEFAULT_TICKERS
    finra_tickers = [ticker for ticker in tickers if ticker not in EXCLUDED_FINRA_TICKERS]

    rth_start = _parse_time(os.getenv("RTH_START"), time(9, 30))
    rth_end = _parse_time(os.getenv("RTH_END"), time(16, 0))

    return Config(
        root_dir=root_dir,
        data_dir=root_dir / "data",
        output_dir=root_dir / "output",
        table_dir=root_dir / "output" / "tables",
        plot_dir=root_dir / "output" / "plots",
        db_path=root_dir / "data" / "darkpool.duckdb",
        tickers=tickers,
        finra_tickers=finra_tickers,
        min_lit_volume=float(os.getenv("MIN_LIT_VOLUME", "10000")),
        run_mode=os.getenv("RUN_MODE", "daily").lower(),
        target_date=target_date,
        market_tz=market_tz,
        rth_start=rth_start,
        rth_end=rth_end,
        inference_version=os.getenv("INFERENCE_VERSION", "OptionB_v1"),
        polygon_api_key=os.getenv("POLYGON_API_KEY"),
        polygon_base_url=os.getenv("POLYGON_BASE_URL", "https://api.polygon.io"),
        polygon_trades_file=os.getenv("POLYGON_TRADES_FILE"),
        finra_otc_url=os.getenv("FINRA_OTC_URL"),
        finra_otc_file=os.getenv("FINRA_OTC_FILE"),
        finra_api_key=os.getenv("FINRA_API_KEY"),
        finra_api_secret=os.getenv("FINRA_API_SECRET"),
        finra_token_url=os.getenv("FINRA_TOKEN_URL"),
        finra_request_method=os.getenv("FINRA_REQUEST_METHOD", "GET").upper(),
        finra_request_json=_parse_json_env("FINRA_REQUEST_JSON"),
        finra_request_params=_parse_json_env("FINRA_REQUEST_PARAMS"),
        finra_date_field=os.getenv("FINRA_DATE_FIELD"),
        finra_symbol_field=os.getenv("FINRA_SYMBOL_FIELD"),
        finra_volume_field=os.getenv("FINRA_VOLUME_FIELD"),
        finra_trade_count_field=os.getenv("FINRA_TRADE_COUNT_FIELD"),
    )
