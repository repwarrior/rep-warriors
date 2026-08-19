"""Account state as the gates need to see it.

Everything here should be reconciled against the broker at startup and after
any restart. A daily trade count that lives only in memory is a duplicate-order
bug waiting for the first crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AccountState:
    equity: float
    day_start_equity: float
    positions: dict[str, int] = field(default_factory=dict)
    """Signed share counts. Positive is long, negative is short."""
    daily_trades: int = 0
    group_exposure: dict[str, float] = field(default_factory=dict)
    """Notional currently on, by correlation group."""
    symbol_group: dict[str, str] = field(default_factory=dict)
    """Symbol -> correlation group. Unmapped symbols get their own group."""

    def __post_init__(self) -> None:
        if self.equity <= 0:
            raise ValueError("equity must be positive")
        if self.day_start_equity <= 0:
            raise ValueError("day_start_equity must be positive")

    def position(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)

    def group_of(self, symbol: str) -> str:
        return self.symbol_group.get(symbol, symbol)

    def exposure_of(self, symbol: str) -> float:
        return self.group_exposure.get(self.group_of(symbol), 0.0)

    def day_pnl_pct(self) -> float:
        """Signed fraction. -0.02 means the account is down 2% on the day."""
        return (self.equity - self.day_start_equity) / self.day_start_equity
