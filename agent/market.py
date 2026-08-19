"""The market snapshot, and the trust boundary that runs through it.

`features` holds numbers your own data layer computed. `context` holds text
that came off a feed — headlines, earnings blurbs, anything written by someone
else. The split is the whole point:

  * rules read `features` only, so nothing anybody writes can produce a signal
  * the veto prompt sees `context`, clearly fenced as untrusted

Under this arrangement the worst a hostile headline can do is talk the veto
into blocking a trade you would otherwise have taken. It can never originate
one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Snapshot:
    symbol: str
    as_of: datetime
    """Exchange timestamp of the quote, tz-aware. NOT the time you fetched it."""
    last: float
    bid: float
    ask: float
    features: dict = field(default_factory=dict)
    """Numbers you computed. Rules may read these."""
    context: dict = field(default_factory=dict)
    """Untrusted text from feeds. Rules must never read these."""

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("snapshot.as_of must be timezone-aware")
        if self.last <= 0 or self.bid <= 0 or self.ask <= 0:
            raise ValueError("snapshot prices must be positive")
        if self.ask < self.bid:
            raise ValueError(f"crossed quote: bid {self.bid} > ask {self.ask}")

    def age_s(self, now: datetime) -> float:
        return (now - self.as_of).total_seconds()

    def is_stale(self, now: datetime, max_age_s: float) -> bool:
        age = self.age_s(now)
        return age > max_age_s or age < -1.0  # future-dated quotes are broken too

    def spread_pp(self) -> float:
        mid = (self.bid + self.ask) / 2
        return (self.ask - self.bid) / mid * 100

    def sha256(self) -> str:
        """Content hash for the journal, so a decision can be replayed against
        the exact inputs that produced it."""
        blob = json.dumps(
            {
                "symbol": self.symbol,
                "as_of": self.as_of.astimezone(timezone.utc).isoformat(),
                "last": self.last,
                "bid": self.bid,
                "ask": self.ask,
                "features": self.features,
                "context": self.context,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()
