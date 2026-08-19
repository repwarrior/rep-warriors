"""When the loop is allowed to be awake.

An unattended loop with no session check ticks at 03:00 on a stale quote, and
whatever it decides there is decided on numbers that stopped moving hours ago.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Protocol


class SessionCalendar(Protocol):
    def is_open(self, now: datetime) -> bool: ...


class AlwaysOpen:
    """Crypto, FX, and tests."""

    def is_open(self, now: datetime) -> bool:
        return True


class USEquityRTH:
    """Regular US equity hours, 09:30-16:00 America/New_York, weekdays.

    Holiday- and half-day-blind. It will happily call Thanksgiving open. Before
    running this unattended, swap in `exchange_calendars` or
    `pandas_market_calendars` — this class exists so the loop has a real gate
    from the first run, not so you can ship it.
    """

    OPEN = time(9, 30)
    CLOSE = time(16, 0)

    def __init__(self, tz_name: str = "America/New_York") -> None:
        from zoneinfo import ZoneInfo

        self.tz = ZoneInfo(tz_name)

    def is_open(self, now: datetime) -> bool:
        local = now.astimezone(self.tz)
        if local.weekday() >= 5:
            return False
        return self.OPEN <= local.time() < self.CLOSE


def seconds_to_next_tick(now: datetime, tick_seconds: float) -> float:
    """Sleep to the next wall-clock boundary rather than for a fixed interval,
    so API latency doesn't accumulate into drift."""
    remainder = now.timestamp() % tick_seconds
    return tick_seconds - remainder if remainder else tick_seconds


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
