"""An intent is a fully-specified order the gates can reason about.

The `kind` field is what makes the gates asymmetric. OPEN increases exposure
and gets the full treatment. REDUCE decreases it and is never blocked by an
entry filter — a system that can refuse to let you out is worse than no system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from account import AccountState
from config import Limits
from rules import LONG, Signal
from sizing import shares_for

BUY = "buy"
SELL = "sell"


class Kind(str, Enum):
    OPEN = "open"
    REDUCE = "reduce"


@dataclass(frozen=True)
class Intent:
    symbol: str
    side: str
    qty: int
    kind: Kind
    stop_price: float | None
    reference_price: float
    signal: Signal | None = None

    def __post_init__(self) -> None:
        if self.side not in (BUY, SELL):
            raise ValueError(f"side must be {BUY!r} or {SELL!r}")
        if self.qty < 0:
            raise ValueError("qty must be >= 0")

    @property
    def notional(self) -> float:
        return self.qty * self.reference_price

    @property
    def signed_qty(self) -> int:
        return self.qty if self.side == BUY else -self.qty

    def as_log(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "kind": self.kind.value,
            "stop_price": self.stop_price,
            "reference_price": self.reference_price,
            "rule": self.signal.rule if self.signal else None,
            "edge_pp": round(self.signal.edge_pp, 4) if self.signal else None,
        }


def build(signal: Signal, account: AccountState, limits: Limits) -> Intent | None:
    """Turn a signal into an intent, classifying it against the open position.

    A signal that opposes an existing position is an exit, sized to the position
    rather than by risk, and clamped so it can never cross through zero into a
    fresh position on the other side.
    """
    held = account.position(signal.symbol)
    wants_long = signal.direction == LONG

    opposes = (held > 0 and not wants_long) or (held < 0 and wants_long)
    if opposes:
        return Intent(
            symbol=signal.symbol,
            side=SELL if held > 0 else BUY,
            qty=abs(held),  # exactly flat, never through zero
            kind=Kind.REDUCE,
            stop_price=None,
            reference_price=signal.reference_price,
            signal=signal,
        )

    qty = shares_for(
        equity=account.equity,
        reference_price=signal.reference_price,
        stop_price=signal.stop_price,
        limits=limits,
    )
    if qty == 0:
        return None

    return Intent(
        symbol=signal.symbol,
        side=BUY if wants_long else SELL,
        qty=qty,
        kind=Kind.OPEN,
        stop_price=signal.stop_price,
        reference_price=signal.reference_price,
        signal=signal,
    )


def flatten(symbol: str, account: AccountState, reference_price: float) -> Intent | None:
    """Unconditional exit intent for an open position. Used by the kill switch."""
    held = account.position(symbol)
    if held == 0:
        return None
    return Intent(
        symbol=symbol,
        side=SELL if held > 0 else BUY,
        qty=abs(held),
        kind=Kind.REDUCE,
        stop_price=None,
        reference_price=reference_price,
    )
