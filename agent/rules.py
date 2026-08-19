"""The signal layer. Deterministic, and the only thing that may say "trade".

Two invariants hold everything else up:

  1. An edge estimate is computed from a recorded backtest, not asserted.
     `BacktestRecord.edge_pp()` is arithmetic over numbers you had to write
     down. There is no code path where an edge is a free-form number.
  2. A rule with too small a sample cannot fire, no matter what it computes.

Rules see `snapshot.features` only. Nothing from `snapshot.context` reaches
here — see market.py for why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from market import Snapshot

LONG = "long"
SHORT = "short"


@dataclass(frozen=True)
class BacktestRecord:
    """Provenance for a rule's edge. A rule cannot fire without one.

    The numbers here are the ones you would have to defend to somebody else.
    Keep the period string honest, including any out-of-sample split.
    """

    period: str
    samples: int
    win_rate: float
    avg_win_pp: float
    avg_loss_pp: float

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples must be > 0")
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError("win_rate must be in [0, 1]")
        if self.avg_win_pp <= 0:
            raise ValueError("avg_win_pp must be > 0")
        if self.avg_loss_pp < 0:
            raise ValueError("avg_loss_pp must be >= 0")

    def edge_pp(self) -> float:
        """Expected value per trade in percentage points."""
        return self.win_rate * self.avg_win_pp - (1.0 - self.win_rate) * self.avg_loss_pp


@dataclass(frozen=True)
class Signal:
    symbol: str
    rule: str
    rules_version: str
    direction: str
    reference_price: float
    stop_price: float
    backtest: BacktestRecord

    def __post_init__(self) -> None:
        if self.direction not in (LONG, SHORT):
            raise ValueError(f"direction must be {LONG!r} or {SHORT!r}")
        if self.reference_price <= 0 or self.stop_price <= 0:
            raise ValueError("prices must be positive")
        # An inverted stop is the classic way to turn a small loser into an
        # unbounded one. Refuse to build the signal at all.
        if self.direction == LONG and self.stop_price >= self.reference_price:
            raise ValueError("long stop must sit below the reference price")
        if self.direction == SHORT and self.stop_price <= self.reference_price:
            raise ValueError("short stop must sit above the reference price")

    @property
    def edge_pp(self) -> float:
        return self.backtest.edge_pp()

    @property
    def risk_per_share(self) -> float:
        return abs(self.reference_price - self.stop_price)


class Rule(Protocol):
    name: str
    backtest: BacktestRecord

    def evaluate(self, snapshot: Snapshot) -> Signal | None: ...


@dataclass(frozen=True)
class RuleSetResult:
    signal: Signal | None
    fired: list[str]
    skipped: list[str]
    note: str

    def as_log(self) -> dict:
        return {
            "fired": self.fired,
            "skipped": self.skipped,
            "note": self.note,
            "edge_pp": round(self.signal.edge_pp, 4) if self.signal else None,
        }


class RuleSet:
    """Runs every rule and reconciles them conservatively.

    Disagreement means stand aside. Agreement takes the *lowest* edge of the
    rules that fired, so adding a rule can never inflate an edge estimate.
    """

    def __init__(self, rules: Sequence[Rule], version: str, min_samples: int) -> None:
        self.rules = list(rules)
        self.version = version
        self.min_samples = min_samples

    def evaluate(self, snapshot: Snapshot) -> RuleSetResult:
        fired: list[Signal] = []
        skipped: list[str] = []

        for rule in self.rules:
            if rule.backtest.samples < self.min_samples:
                skipped.append(f"{rule.name}: {rule.backtest.samples} samples")
                continue
            signal = rule.evaluate(snapshot)
            if signal is not None:
                fired.append(signal)

        names = [s.rule for s in fired]
        if not fired:
            return RuleSetResult(None, names, skipped, "no rule fired")

        directions = {s.direction for s in fired}
        if len(directions) > 1:
            return RuleSetResult(None, names, skipped, "rules disagree, standing aside")

        weakest = min(fired, key=lambda s: s.edge_pp)
        return RuleSetResult(weakest, names, skipped, f"agreed on {weakest.direction}")


# ---------------------------------------------------------------------------
# Example rule. Delete it.
# ---------------------------------------------------------------------------


class PullbackToSupport:
    """EXAMPLE ONLY — this rule is here to show the shape, not to be traded.

    The BacktestRecord below is a PLACEHOLDER. It is not the output of any
    backtest; the numbers are invented. `RuleSet` cannot tell the difference,
    which is exactly why you have to replace it with numbers you measured
    before this rule is allowed anywhere near an order.
    """

    name = "pullback_to_support"

    def __init__(self, backtest: BacktestRecord, rules_version: str) -> None:
        self.backtest = backtest
        self.rules_version = rules_version

    def evaluate(self, snapshot: Snapshot) -> Signal | None:
        f = snapshot.features
        support = f.get("support")
        width = f.get("support_width")
        if support is None or width is None or not f.get("trend_up"):
            return None
        if not support <= snapshot.last <= support + width:
            return None
        return Signal(
            symbol=snapshot.symbol,
            rule=self.name,
            rules_version=self.rules_version,
            direction=LONG,
            reference_price=snapshot.last,
            stop_price=support - width,
            backtest=self.backtest,
        )
