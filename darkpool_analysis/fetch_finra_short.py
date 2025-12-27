from __future__ import annotations

from datetime import date
import logging
from pathlib import Path
import re
from typing import Optional

import pandas as pd
import requests

try:
    from .config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)


def _build_finra_headers(config: Config) -> dict:
    headers = {"Accept": "application/json"}
    if config.finra_token_url:
        # OAuth 2.0 client_credentials flow - get access token from FINRA Identity Platform
        response = requests.post(
            config.finra_token_url,
            auth=(config.finra_api_key, config.finra_api_secret),
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise ValueError("FINRA token response missing access_token.")
        headers["Authorization"] = f"Bearer {token}"
        return headers
    # Fallback: direct API key auth (may not work for all FINRA endpoints)
    if config.finra_api_key and config.finra_api_secret:
        headers["X-API-KEY"] = config.finra_api_key
        headers["X-API-SECRET"] = config.finra_api_secret
    return headers


def _infer_source_from_filename(name: str) -> Optional[str]:
    match = re.match(r"([A-Z]+)SHVOL", name.upper())
    return match.group(1) if match else None


def _normalize_short_sale_df(df: pd.DataFrame, source_file: Optional[str]) -> pd.DataFrame:
    columns = {col.lower(): col for col in df.columns}
    # Support both file format (Date, Symbol, etc.) and API format (tradeReportDate, etc.)
    date_col = columns.get("date") or columns.get("tradereportdate")
    symbol_col = columns.get("symbol") or columns.get("securitiesinformationprocessorsymbolidentifier")
    short_col = columns.get("shortvolume") or columns.get("short_volume") or columns.get("shortparquantity")
    short_exempt_col = columns.get("shortexemptvolume") or columns.get("short_exempt_volume") or columns.get("shortexemptparquantity")
    total_col = columns.get("totalvolume") or columns.get("total_volume") or columns.get("totalparquantity")
    market_col = columns.get("market") or columns.get("marketcode")

    if not date_col or not symbol_col or not short_col:
        raise ValueError("Short sale file missing required columns: Date, Symbol, ShortVolume")

    trade_date = pd.to_datetime(
        df[date_col].astype(str), format="%Y%m%d", errors="coerce"
    )
    fallback_date = pd.to_datetime(df[date_col], errors="coerce")
    trade_date = trade_date.fillna(fallback_date).dt.date
    normalized = pd.DataFrame(
        {
            "trade_date": trade_date,
            "symbol": df[symbol_col].astype(str).str.upper(),
            "short_volume": pd.to_numeric(df[short_col], errors="coerce"),
            "short_exempt_volume": pd.to_numeric(df[short_exempt_col], errors="coerce")
            if short_exempt_col
            else pd.NA,
            "total_volume": pd.to_numeric(df[total_col], errors="coerce") if total_col else pd.NA,
            "market": df[market_col].astype(str) if market_col else pd.NA,
        }
    )
    normalized["source_file"] = source_file
    normalized["source"] = _infer_source_from_filename(source_file or "") or pd.NA
    normalized = normalized.dropna(subset=["trade_date", "symbol", "short_volume"])
    return normalized[
        [
            "symbol",
            "trade_date",
            "short_volume",
            "short_exempt_volume",
            "total_volume",
            "market",
            "source",
            "source_file",
        ]
    ]


def normalize_short_sale_df(df: pd.DataFrame, source_file: Optional[str]) -> pd.DataFrame:
    return _normalize_short_sale_df(df, source_file)


def _load_short_sale_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    return _normalize_short_sale_df(df, Path(path).name)


def _load_short_sale_dir(directory: str) -> pd.DataFrame:
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Short sale directory not found: {directory}")
    frames = []
    for file_path in sorted(path.glob("*.txt")):
        try:
            frames.append(_load_short_sale_file(str(file_path)))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_path.name, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_short_sale_from_api(config: Config, target_date: date) -> pd.DataFrame:
    if not config.finra_short_sale_url:
        raise ValueError("FINRA_SHORT_SALE_URL is required for API fetch.")
    headers = _build_finra_headers(config)
    payload = {
        "limit": 10000,
        "dateRangeFilters": [
            {
                "fieldName": "tradeReportDate",
                "startDate": target_date.isoformat(),
                "endDate": target_date.isoformat(),
            }
        ],
    }
    # Add symbol filtering to avoid pagination issues (API returns max 10000 rows)
    # FINRA Query API uses domainFilters with a "values" list (like SQL IN clause)
    # See: https://developer.finra.org/docs
    if config.finra_tickers:
        payload["domainFilters"] = [
            {
                "fieldName": "securitiesInformationProcessorSymbolIdentifier",
                "values": config.finra_tickers,
            }
        ]
    response = requests.post(
        config.finra_short_sale_url,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and "data" in data:
        df = pd.DataFrame(data["data"])
        logger.info("FINRA short sale API returned %d rows", len(df))
    elif isinstance(data, list):
        df = pd.DataFrame(data)
        logger.info("FINRA short sale API returned %d rows (list)", len(df))
    else:
        raise ValueError("Unsupported FINRA short sale API response shape.")

    if df.empty:
        logger.warning("FINRA short sale API returned empty dataframe")
        return df

    return _normalize_short_sale_df(df, source_file=None)


def fetch_finra_short_daily(config: Config, target_date: date) -> pd.DataFrame:
    if config.finra_short_sale_file:
        logger.info("Loading FINRA short sale file: %s", config.finra_short_sale_file)
        raw_df = _load_short_sale_file(config.finra_short_sale_file)
    elif config.finra_short_sale_dir:
        logger.info("Loading FINRA short sale files from dir: %s", config.finra_short_sale_dir)
        raw_df = _load_short_sale_dir(config.finra_short_sale_dir)
    else:
        logger.info("Fetching FINRA short sale data from API for %s", target_date.isoformat())
        raw_df = _load_short_sale_from_api(config, target_date)

    if raw_df.empty:
        logger.warning("FINRA short sale data returned no rows.")
        return raw_df

    if config.finra_tickers:
        raw_df = raw_df[raw_df["symbol"].isin(config.finra_tickers)].copy()

    raw_df = raw_df[raw_df["trade_date"] == target_date].copy()
    if raw_df.empty:
        logger.warning("FINRA short sale data has no rows for %s.", target_date.isoformat())
    return raw_df
