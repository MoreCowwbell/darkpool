from __future__ import annotations

from datetime import date
from io import StringIO
import logging

import pandas as pd
import requests

try:
    from .config import Config
    from .fetch_finra_short import normalize_short_sale_df
except ImportError:
    from config import Config
    from fetch_finra_short import normalize_short_sale_df

logger = logging.getLogger(__name__)


def _build_cdn_url(config: Config, target_date: date) -> str:
    date_tag = target_date.strftime("%Y%m%d")
    base = config.finra_cdn_url.rstrip("/")
    return f"{base}/CNMSshvol{date_tag}.txt"


def fetch_finra_short_cdn_daily(config: Config, target_date: date) -> pd.DataFrame:
    url = _build_cdn_url(config, target_date)
    logger.info("Downloading FINRA short sale CDN file: %s", url)
    response = requests.get(url, timeout=60)

    if response.status_code in {403, 404}:
        logger.warning(
            "FINRA CDN file not available for %s (status %s)",
            target_date.isoformat(),
            response.status_code,
        )
        return pd.DataFrame()

    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text), sep="|")
    source_file = f"CNMSshvol{target_date.strftime('%Y%m%d')}.txt"
    normalized = normalize_short_sale_df(df, source_file)
    normalized["market"] = normalized["market"].fillna("ALL")
    normalized["source"] = normalized["source"].fillna("CNMS")
    normalized = normalized[normalized["trade_date"] == target_date].copy()
    return normalized
