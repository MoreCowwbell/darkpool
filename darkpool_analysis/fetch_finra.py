from __future__ import annotations

from datetime import date
import io
import json
import logging
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


def _fetch_finra_token(config: Config) -> str:
    if not config.finra_token_url:
        raise ValueError("FINRA_TOKEN_URL is required to fetch an auth token.")
    if not config.finra_api_key or not config.finra_api_secret:
        raise ValueError("FINRA_API_KEY and FINRA_API_SECRET are required for token auth.")
    response = requests.post(
        config.finra_token_url,
        auth=(config.finra_api_key, config.finra_api_secret),
        data={"grant_type": "client_credentials"},
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
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "data" in payload:
            return pd.DataFrame(payload["data"])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        raise ValueError("Unsupported FINRA JSON file structure.")
    return pd.read_csv(path)


def _load_finra_from_api(config: Config, url: str) -> pd.DataFrame:
    headers = _build_finra_headers(config)
    method = config.finra_request_method
    params = config.finra_request_params or {}
    payload = config.finra_request_json
    if method == "POST":
        response = requests.post(
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=60,
        )
    else:
        response = requests.get(
            url,
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


def _normalize_finra_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    symbol_col = _resolve_column(
        df,
        config.finra_symbol_field,
        ["symbol", "issueSymbolIdentifier", "issueSymbol", "issue_symbol", "Symbol"],
    )
    date_col = _resolve_column(
        df,
        config.finra_date_field,
        ["week_start_date", "weekStartDate", "week_start", "weekStart", "tradeDate", "trade_date", "date"],
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
    if trade_count_col:
        normalized["trade_count"] = pd.to_numeric(df[trade_count_col], errors="coerce")
    else:
        normalized["trade_count"] = pd.NA

    # Preserve source column if present
    if "source" in df.columns:
        normalized["source"] = df["source"].values

    normalized = normalized.dropna(subset=["symbol", "week_start_date", "off_exchange_volume"])
    return normalized


def fetch_finra_otc_volume(
    config: Config, target_date: date
) -> Tuple[pd.DataFrame, pd.DataFrame, date]:
    if config.finra_otc_file:
        logger.info("Loading FINRA OTC data from file: %s", config.finra_otc_file)
        raw_df = _load_finra_from_file(config.finra_otc_file)
        raw_df["source"] = "file"
    else:
        # Fetch from all configured FINRA sources
        all_frames = []
        for source_name in config.finra_sources:
            if source_name not in config.finra_endpoints:
                logger.warning("Unknown FINRA source: %s, skipping", source_name)
                continue
            url = config.finra_endpoints[source_name]
            logger.info("Fetching FINRA data from %s: %s", source_name, url)
            try:
                df = _load_finra_from_api(config, url)
                df["source"] = source_name
                all_frames.append(df)
                logger.info("Fetched %d rows from %s", len(df), source_name)
            except Exception as exc:
                logger.error("Failed to fetch from %s: %s", source_name, exc)
                continue

        if not all_frames:
            raise RuntimeError("No FINRA data fetched from any source.")
        raw_df = pd.concat(all_frames, ignore_index=True)
        logger.info("Combined %d total rows from %d sources", len(raw_df), len(all_frames))

    normalized = _normalize_finra_columns(raw_df, config)
    normalized = normalized[normalized["symbol"].isin(config.finra_tickers)].copy()
    if normalized.empty:
        raise RuntimeError("FINRA OTC data returned no rows for configured tickers.")

    available_weeks = sorted(normalized["week_start_date"].unique())
    eligible_weeks = [week for week in available_weeks if week <= target_date]
    if not eligible_weeks:
        raise RuntimeError("No FINRA week_start_date available on or before target_date.")
    latest_week = max(eligible_weeks)
    week_df = normalized[normalized["week_start_date"] == latest_week].copy()
    if week_df.empty:
        raise RuntimeError("Latest FINRA week filter returned no rows.")

    return normalized, week_df, latest_week
