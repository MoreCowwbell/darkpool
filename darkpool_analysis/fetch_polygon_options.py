"""
Polygon Options Premium Fetcher.

Fetches options contracts and daily aggregates to compute call/put premium
for the ATM premium panel visualization.
"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Optional

import duckdb
import pandas as pd
import requests

try:
    from .config import Config, DAILY_EXPIRATION_TICKERS
except ImportError:
    from config import Config, DAILY_EXPIRATION_TICKERS

logger = logging.getLogger(__name__)

# Rate limiting: max concurrent requests
MAX_WORKERS = 8


def get_upcoming_friday(trade_date: date) -> date:
    """Calculate the upcoming Friday expiration for a given trade date."""
    days_until_friday = (4 - trade_date.weekday()) % 7
    if days_until_friday == 0:
        # If trade_date is Friday, return same day
        return trade_date
    return trade_date + timedelta(days=days_until_friday)


def get_0dte_expiration(trade_date: date) -> date:
    """Return same-day expiration (0DTE)."""
    return trade_date


def _build_option_ticker(
    underlying: str,
    expiration: date,
    option_type: str,
    strike: float,
) -> str:
    """
    Build Polygon option ticker symbol.
    Format: O:{UNDERLYING}{YYMMDD}{C|P}{STRIKE*1000 padded to 8 digits}
    Example: O:SPY251213C00600000
    """
    exp_str = expiration.strftime("%y%m%d")
    strike_int = int(strike * 1000)
    strike_str = f"{strike_int:08d}"
    return f"O:{underlying}{exp_str}{option_type}{strike_str}"


def fetch_option_contracts(
    underlying: str,
    expiration: date,
    config: Config,
    as_of: Optional[date] = None,
) -> list[dict]:
    """
    Fetch available option contracts for an underlying and expiration.
    Returns list of contract info dicts with strike and type.
    """
    url = f"{config.polygon_base_url}/v3/reference/options/contracts"
    params = {
        "underlying_ticker": underlying,
        "expiration_date": expiration.isoformat(),
        "limit": 250,
        "apiKey": config.polygon_api_key,
    }
    if as_of:
        params["as_of"] = as_of.isoformat()

    contracts = []
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 403:
            logger.warning("Options API 403 for %s - subscription may not include options", underlying)
            return []
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        for contract in results:
            contracts.append({
                "ticker": contract.get("ticker"),
                "strike": contract.get("strike_price"),
                "option_type": contract.get("contract_type", "").upper()[:1],  # "call" -> "C"
                "expiration": contract.get("expiration_date"),
            })
        # Handle pagination
        next_url = data.get("next_url")
        while next_url:
            resp = requests.get(f"{next_url}&apiKey={config.polygon_api_key}", timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for contract in data.get("results", []):
                contracts.append({
                    "ticker": contract.get("ticker"),
                    "strike": contract.get("strike_price"),
                    "option_type": contract.get("contract_type", "").upper()[:1],
                    "expiration": contract.get("expiration_date"),
                })
            next_url = data.get("next_url")
    except requests.RequestException as e:
        logger.error("Failed to fetch option contracts for %s exp %s: %s", underlying, expiration, e)
        return []

    return contracts


def get_contracts_around_atm(
    contracts: list[dict],
    atm_price: float,
    count: int = 30,
) -> list[dict]:
    """
    Filter contracts to the nearest ±count strikes from ATM price.
    Returns both calls and puts for each strike level.
    """
    if not contracts:
        return []

    # Get unique strikes and sort by distance from ATM
    strikes = sorted(set(c["strike"] for c in contracts if c["strike"]))
    if not strikes:
        return []

    # Find strikes closest to ATM
    strikes_with_distance = [(s, abs(s - atm_price)) for s in strikes]
    strikes_with_distance.sort(key=lambda x: x[1])
    selected_strikes = set(s for s, _ in strikes_with_distance[:count * 2])  # Get more to ensure coverage

    # Filter contracts to selected strikes
    filtered = [c for c in contracts if c["strike"] in selected_strikes]
    return filtered


def fetch_option_daily_agg(
    option_ticker: str,
    trade_date: date,
    config: Config,
) -> Optional[dict]:
    """
    Fetch daily aggregate (OHLCV) for a single option contract.
    Returns dict with close and volume, or None on failure.
    """
    url = f"{config.polygon_base_url}/v2/aggs/ticker/{option_ticker}/range/1/day/{trade_date.isoformat()}/{trade_date.isoformat()}"
    params = {"apiKey": config.polygon_api_key}

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 404:
            # No data for this contract on this date
            return None
        if resp.status_code == 403:
            logger.debug("Options agg 403 for %s", option_ticker)
            return None
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        bar = results[0]
        return {
            "ticker": option_ticker,
            "close": bar.get("c", 0),
            "volume": bar.get("v", 0),
            "open": bar.get("o", 0),
            "high": bar.get("h", 0),
            "low": bar.get("l", 0),
        }
    except requests.RequestException as e:
        logger.debug("Failed to fetch option agg for %s: %s", option_ticker, e)
        return None


def fetch_options_premium_for_date(
    symbol: str,
    trade_date: date,
    expiration: date,
    expiration_type: str,
    atm_price: float,
    config: Config,
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Fetch options premium data for a symbol on a given date.

    Returns DataFrame with columns:
    - symbol, trade_date, expiration_date, expiration_type
    - strike, option_type, premium, volume, close_price
    """
    # Check if already cached
    cached = conn.execute(
        """
        SELECT COUNT(*) FROM options_premium_summary
        WHERE symbol = ? AND trade_date = ? AND expiration_type = ?
        """,
        [symbol, trade_date, expiration_type],
    ).fetchone()
    if cached and cached[0] > 0:
        logger.debug("Options premium cached for %s %s %s", symbol, trade_date, expiration_type)
        return pd.DataFrame()

    logger.info("Fetching options premium: %s %s exp=%s (%s)", symbol, trade_date, expiration, expiration_type)

    # Fetch available contracts
    contracts = fetch_option_contracts(symbol, expiration, config, as_of=trade_date)
    if not contracts:
        logger.warning("No contracts found for %s exp %s", symbol, expiration)
        return pd.DataFrame()

    # Filter to ATM ± N strikes
    filtered = get_contracts_around_atm(contracts, atm_price, config.options_strike_count)
    if not filtered:
        logger.warning("No contracts near ATM for %s (ATM=%.2f)", symbol, atm_price)
        return pd.DataFrame()

    logger.info("Fetching %d option contracts for %s", len(filtered), symbol)

    # Fetch daily aggregates in parallel
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_option_daily_agg, c["ticker"], trade_date, config): c
            for c in filtered
        }
        for future in as_completed(futures):
            contract = futures[future]
            try:
                agg = future.result()
                if agg and agg.get("volume", 0) > 0:
                    # Premium = close * volume * 100 / 1,000,000 (in $M)
                    premium = agg["close"] * agg["volume"] * 100 / 1_000_000
                    results.append({
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "expiration_date": expiration,
                        "expiration_type": expiration_type,
                        "strike": contract["strike"],
                        "option_type": contract["option_type"],
                        "premium": premium,
                        "volume": agg["volume"],
                        "close_price": agg["close"],
                        "fetch_timestamp": datetime.utcnow(),
                    })
            except Exception as e:
                logger.debug("Error fetching %s: %s", contract["ticker"], e)

    if not results:
        logger.warning("No option data with volume for %s on %s", symbol, trade_date)
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def compute_premium_summary(df: pd.DataFrame, atm_price: float) -> pd.DataFrame:
    """
    Aggregate individual contract premiums into summary metrics with ITM/OTM breakdown.

    ITM/OTM determination (per WTD's methodology):
    - For calls: OTM = strike > atm_price, ITM = strike < atm_price
    - For puts: OTM = strike < atm_price, ITM = strike > atm_price

    Returns DataFrame with:
    - symbol, trade_date, expiration_type
    - total_call_premium, total_put_premium, net_premium, log_ratio
    - otm_call_premium, itm_call_premium, otm_put_premium, itm_put_premium
    - directional_score
    """
    if df.empty:
        return pd.DataFrame()

    def compute_group_summary(g: pd.DataFrame) -> pd.Series:
        calls = g[g["option_type"] == "C"]
        puts = g[g["option_type"] == "P"]

        # Total premiums (backwards compatible)
        total_call_premium = calls["premium"].sum()
        total_put_premium = puts["premium"].sum()

        # ITM/OTM breakdown using atm_price
        # Calls: OTM when strike > ATM, ITM when strike < ATM
        otm_call_premium = calls[calls["strike"] > atm_price]["premium"].sum()
        itm_call_premium = calls[calls["strike"] < atm_price]["premium"].sum()

        # Puts: OTM when strike < ATM, ITM when strike > ATM
        otm_put_premium = puts[puts["strike"] < atm_price]["premium"].sum()
        itm_put_premium = puts[puts["strike"] > atm_price]["premium"].sum()

        # Directional score (per WTD's insight):
        # OTM calls are bullish, OTM puts are bearish, ITM calls sold are bearish (hedge)
        directional_score = otm_call_premium - otm_put_premium - (itm_call_premium * 0.5)

        return pd.Series({
            "total_call_premium": total_call_premium,
            "total_put_premium": total_put_premium,
            "strikes_count": g["strike"].nunique(),
            "atm_strike": atm_price,  # Use actual ATM price, not median
            "otm_call_premium": otm_call_premium,
            "itm_call_premium": itm_call_premium,
            "otm_put_premium": otm_put_premium,
            "itm_put_premium": itm_put_premium,
            "directional_score": directional_score,
        })

    grouped = df.groupby(["symbol", "trade_date", "expiration_type"]).apply(
        compute_group_summary
    ).reset_index()

    grouped["net_premium"] = grouped["total_call_premium"] - grouped["total_put_premium"]

    # Compute log ratio: ln(call/put)
    epsilon = 0.01
    grouped["log_ratio"] = grouped.apply(
        lambda row: math.log((row["total_call_premium"] + epsilon) / (row["total_put_premium"] + epsilon))
        if row["total_call_premium"] > 0 or row["total_put_premium"] > 0
        else None,
        axis=1,
    )
    grouped["fetch_timestamp"] = datetime.utcnow()

    return grouped


def fetch_and_store_options_premium(
    symbol: str,
    trade_date: date,
    atm_price: float,
    config: Config,
    conn: duckdb.DuckDBPyConnection,
) -> dict:
    """
    Fetch options premium for all relevant expirations and store in database.

    For indices (SPY, QQQ, etc.): fetch both 0DTE and weekly
    For stocks: fetch weekly only

    Returns dict with fetch statistics.
    """
    stats = {"fetched": 0, "cached": 0, "errors": 0}

    expirations_to_fetch = []

    # Weekly expiration (all tickers)
    weekly_exp = get_upcoming_friday(trade_date)
    expirations_to_fetch.append((weekly_exp, "WEEKLY"))

    # 0DTE expiration (indices only)
    if symbol.upper() in DAILY_EXPIRATION_TICKERS:
        dte_exp = get_0dte_expiration(trade_date)
        # Only add if different from weekly (avoid duplicate on Fridays)
        if dte_exp != weekly_exp:
            expirations_to_fetch.append((dte_exp, "0DTE"))

    all_detail_rows = []
    all_summary_rows = []

    for expiration, exp_type in expirations_to_fetch:
        df = fetch_options_premium_for_date(
            symbol=symbol,
            trade_date=trade_date,
            expiration=expiration,
            expiration_type=exp_type,
            atm_price=atm_price,
            config=config,
            conn=conn,
        )

        if df.empty:
            # Check if it was cached
            cached = conn.execute(
                """
                SELECT COUNT(*) FROM options_premium_summary
                WHERE symbol = ? AND trade_date = ? AND expiration_type = ?
                """,
                [symbol, trade_date, exp_type],
            ).fetchone()
            if cached and cached[0] > 0:
                stats["cached"] += 1
            else:
                stats["errors"] += 1
            continue

        stats["fetched"] += 1
        all_detail_rows.append(df)

        # Compute summary with ITM/OTM breakdown
        summary = compute_premium_summary(df, atm_price)
        if not summary.empty:
            all_summary_rows.append(summary)

    # Store in database
    if all_detail_rows:
        detail_df = pd.concat(all_detail_rows, ignore_index=True)
        conn.register("detail_view", detail_df)
        conn.execute(
            """
            INSERT INTO options_premium_daily
            SELECT * FROM detail_view
            ON CONFLICT (symbol, trade_date, expiration_date, strike, option_type)
            DO UPDATE SET premium = EXCLUDED.premium, volume = EXCLUDED.volume,
                          close_price = EXCLUDED.close_price, fetch_timestamp = EXCLUDED.fetch_timestamp
            """
        )
        conn.unregister("detail_view")

    if all_summary_rows:
        summary_df = pd.concat(all_summary_rows, ignore_index=True)
        conn.register("summary_view", summary_df)
        conn.execute(
            """
            INSERT INTO options_premium_summary
            SELECT * FROM summary_view
            ON CONFLICT (symbol, trade_date, expiration_type)
            DO UPDATE SET total_call_premium = EXCLUDED.total_call_premium,
                          total_put_premium = EXCLUDED.total_put_premium,
                          net_premium = EXCLUDED.net_premium,
                          log_ratio = EXCLUDED.log_ratio,
                          strikes_count = EXCLUDED.strikes_count,
                          atm_strike = EXCLUDED.atm_strike,
                          otm_call_premium = EXCLUDED.otm_call_premium,
                          itm_call_premium = EXCLUDED.itm_call_premium,
                          otm_put_premium = EXCLUDED.otm_put_premium,
                          itm_put_premium = EXCLUDED.itm_put_premium,
                          directional_score = EXCLUDED.directional_score,
                          fetch_timestamp = EXCLUDED.fetch_timestamp
            """
        )
        conn.unregister("summary_view")

    return stats


def fetch_options_premium_batch(
    symbols: list[str],
    trade_date: date,
    atm_prices: dict[str, float],
    config: Config,
    conn: duckdb.DuckDBPyConnection,
) -> dict:
    """
    Fetch options premium for multiple symbols on a single date.

    Args:
        symbols: List of ticker symbols
        trade_date: The trading date
        atm_prices: Dict mapping symbol -> ATM price (usually close price)
        config: Config object
        conn: DuckDB connection

    Returns dict with aggregate statistics.
    """
    total_stats = {"fetched": 0, "cached": 0, "errors": 0, "skipped": 0}

    for symbol in symbols:
        atm = atm_prices.get(symbol)
        if not atm:
            logger.warning("No ATM price for %s on %s, skipping options", symbol, trade_date)
            total_stats["skipped"] += 1
            continue

        try:
            stats = fetch_and_store_options_premium(symbol, trade_date, atm, config, conn)
            total_stats["fetched"] += stats["fetched"]
            total_stats["cached"] += stats["cached"]
            total_stats["errors"] += stats["errors"]
        except Exception as e:
            logger.error("Error fetching options for %s: %s", symbol, e)
            total_stats["errors"] += 1

    return total_stats
