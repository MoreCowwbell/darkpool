from __future__ import annotations

from datetime import date, timedelta
import io
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

try:
    from .config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)


def _resolve_column(df: pd.DataFrame, preferred: Optional[str], candidates: list[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"Missing required column. Tried: {candidates}")


def _resolve_optional_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _fetch_finra_token(config: Config) -> str:
    """OAuth 2.0 client_credentials flow - get access token from FINRA Identity Platform."""
    if not config.finra_token_url:
        raise ValueError("FINRA_TOKEN_URL is required to fetch an auth token.")
    if not config.finra_api_key or not config.finra_api_secret:
        raise ValueError("FINRA_API_KEY and FINRA_API_SECRET are required for token auth.")
    response = requests.post(
        config.finra_token_url,
        auth=(config.finra_api_key, config.finra_api_secret),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if "access_token" not in payload:
        raise ValueError("FINRA token response missing access_token.")
    return payload["access_token"]


def _build_finra_headers(config: Config) -> dict:
    headers = {"Accept": "application/json"}
    if config.finra_token_url:
        token = _fetch_finra_token(config)
        headers["Authorization"] = f"Bearer {token}"
        return headers
    if config.finra_api_key and config.finra_api_secret:
        headers["X-API-KEY"] = config.finra_api_key
        headers["X-API-SECRET"] = config.finra_api_secret
    return headers


def _load_finra_from_file(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "data" in payload:
            return pd.DataFrame(payload["data"])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        raise ValueError("Unsupported FINRA JSON file structure.")

    sep = "|" if file_path.suffix.lower() in {".txt", ".pipe"} else ","
    return pd.read_csv(path, sep=sep)


def _load_finra_from_api(
    config: Config,
    symbols: list[str] | None = None,
    target_date: date | None = None,
) -> pd.DataFrame:
    if not config.finra_otc_url:
        raise ValueError("FINRA_OTC_URL is required when FINRA_OTC_FILE is not set.")
    headers = _build_finra_headers(config)
    method = config.finra_request_method
    params = config.finra_request_params or {}

    payload = config.finra_request_json.copy() if config.finra_request_json else {}
    payload.setdefault("limit", 10000)

    if symbols:
        payload["domainFilters"] = [
            {
                "fieldName": "issueSymbolIdentifier",
                "values": [sym.upper() for sym in symbols],
            }
        ]

    if target_date:
        week_start = target_date - timedelta(days=target_date.weekday())
        start_date = week_start - timedelta(weeks=8)
        payload["dateRangeFilters"] = [
            {
                "fieldName": "weekStartDate",
                "startDate": start_date.isoformat(),
                "endDate": week_start.isoformat(),
            }
        ]

    if method == "POST":
        response = requests.post(
            config.finra_otc_url,
            headers=headers,
            params=params,
            json=payload,
            timeout=60,
        )
    else:
        response = requests.get(
            config.finra_otc_url,
            headers=headers,
            params=params,
            timeout=60,
        )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    text = response.text.strip()
    if "application/json" in content_type or text.startswith("{") or text.startswith("["):
        payload = response.json()
        if isinstance(payload, dict) and "data" in payload:
            return pd.DataFrame(payload["data"])
        if isinstance(payload, dict) and "results" in payload:
            return pd.DataFrame(payload["results"])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        raise ValueError("Unsupported FINRA JSON response shape.")
    return pd.read_csv(io.StringIO(response.text))


def _normalize_finra_columns(df: pd.DataFrame, config: Config, source_file: Optional[str]) -> pd.DataFrame:
    symbol_col = _resolve_column(
        df,
        config.finra_symbol_field,
        ["symbol", "issueSymbolIdentifier", "issueSymbol", "issue_symbol", "Symbol"],
    )
    date_col = _resolve_column(
        df,
        config.finra_date_field,
        [
            "week_start_date",
            "weekStartDate",
            "week_start",
            "weekStart",
            "tradeDate",
            "trade_date",
            "date",
            "lastUpdateDate",
        ],
    )
    volume_col = _resolve_column(
        df,
        config.finra_volume_field,
        [
            "off_exchange_volume",
            "totalWeeklyShareQuantity",
            "weeklyShareQuantity",
            "total_weekly_share_quantity",
            "totalShareQuantity",
        ],
    )
    trade_count_col = None
    if config.finra_trade_count_field or any(
        candidate in df.columns
        for candidate in ["trade_count", "weeklyTradeCount", "totalWeeklyTradeCount", "tradeCount"]
    ):
        trade_count_col = _resolve_column(
            df,
            config.finra_trade_count_field,
            ["trade_count", "weeklyTradeCount", "totalWeeklyTradeCount", "tradeCount"],
        )

    normalized = pd.DataFrame(
        {
            "symbol": df[symbol_col].astype(str).str.upper(),
            "week_start_date": pd.to_datetime(df[date_col]).dt.date,
            "off_exchange_volume": pd.to_numeric(df[volume_col], errors="coerce"),
        }
    )
    normalized["trade_count"] = (
        pd.to_numeric(df[trade_count_col], errors="coerce")
        if trade_count_col
        else pd.NA
    )

    tier_identifier_col = _resolve_optional_column(df, ["tierIdentifier", "tier_identifier", "tier"])
    tier_description_col = _resolve_optional_column(df, ["tierDescription", "tier_description"])
    issue_name_col = _resolve_optional_column(df, ["issueName", "issue_name"])
    mp_name_col = _resolve_optional_column(df, ["marketParticipantName", "market_participant_name"])
    mpid_col = _resolve_optional_column(df, ["MPID", "mpid"])
    last_update_col = _resolve_optional_column(df, ["lastUpdateDate", "last_update_date"])

    normalized["tier_identifier"] = (
        df[tier_identifier_col].astype(str).str.upper() if tier_identifier_col else pd.NA
    )
    normalized["tier_description"] = (
        df[tier_description_col].astype(str) if tier_description_col else pd.NA
    )
    normalized["issue_name"] = df[issue_name_col].astype(str) if issue_name_col else pd.NA
    normalized["market_participant_name"] = (
        df[mp_name_col].astype(str) if mp_name_col else pd.NA
    )
    normalized["mpid"] = df[mpid_col].astype(str).str.upper() if mpid_col else pd.NA
    normalized["last_update_date"] = (
        pd.to_datetime(df[last_update_col]).dt.date if last_update_col else pd.NA
    )
    normalized["source_file"] = source_file

    normalized = normalized.dropna(
        subset=["symbol", "week_start_date", "off_exchange_volume"]
    )
    return normalized[
        [
            "symbol",
            "week_start_date",
            "off_exchange_volume",
            "trade_count",
            "tier_identifier",
            "tier_description",
            "issue_name",
            "market_participant_name",
            "mpid",
            "last_update_date",
            "source_file",
        ]
    ]


def fetch_finra_otc_weekly(
    config: Config, target_date: date
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[date]]:
    if config.finra_otc_file:
        logger.info("Loading FINRA OTC data from file: %s", config.finra_otc_file)
        raw_df = _load_finra_from_file(config.finra_otc_file)
        source_file = Path(config.finra_otc_file).name
    else:
        logger.info("Fetching FINRA OTC data from API for symbols: %s", config.finra_tickers)
        raw_df = _load_finra_from_api(config, symbols=config.finra_tickers, target_date=target_date)
        source_file = None
        logger.info("Fetched %d rows from FINRA API", len(raw_df))

    normalized = _normalize_finra_columns(raw_df, config, source_file)
    normalized = normalized[normalized["symbol"].isin(config.finra_tickers)].copy()
    if normalized.empty:
        logger.warning("FINRA OTC data returned no rows for configured tickers.")
        return pd.DataFrame(), pd.DataFrame(), None

    available_weeks = sorted(normalized["week_start_date"].unique())
    eligible_weeks = [week for week in available_weeks if week <= target_date]
    if not eligible_weeks:
        raise RuntimeError("No FINRA week_start_date available on or before target_date.")
    latest_week = max(eligible_weeks)
    week_df = normalized[normalized["week_start_date"] == latest_week].copy()
    if week_df.empty:
        raise RuntimeError("Latest FINRA week filter returned no rows.")

    return normalized, week_df, latest_week
