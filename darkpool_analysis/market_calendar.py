from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

# US stock market holidays (NYSE/NASDAQ closed)
US_MARKET_HOLIDAYS = {
    # 2024
    date(2024, 1, 1),    # New Year's Day
    date(2024, 1, 15),   # MLK Day
    date(2024, 2, 19),   # Presidents' Day
    date(2024, 3, 29),   # Good Friday
    date(2024, 5, 27),   # Memorial Day
    date(2024, 6, 19),   # Juneteenth
    date(2024, 7, 4),    # Independence Day
    date(2024, 9, 2),    # Labor Day
    date(2024, 11, 28),  # Thanksgiving
    date(2024, 12, 25),  # Christmas
    # 2025
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 20),   # MLK Day
    date(2025, 2, 17),   # Presidents' Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
}


def is_trading_day(day: date) -> bool:
    """Return True if day is a market trading day (weekday and not holiday)."""
    return day.weekday() < 5 and day not in US_MARKET_HOLIDAYS


def filter_trading_days(days: Iterable[date]) -> list[date]:
    """Return only trading days from an iterable of dates."""
    return [d for d in days if is_trading_day(d)]


def get_past_trading_days(count: int, from_date: date) -> list[date]:
    """Return the last N trading days (excludes weekends and holidays)."""
    days = []
    current = from_date
    while len(days) < count:
        if is_trading_day(current):
            days.append(current)
        current -= timedelta(days=1)
    return days  # Most recent first
