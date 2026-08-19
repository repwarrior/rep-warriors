"""The signal layer. Deterministic, and the only thing that may say "trade".

Two invariants hold everything else up:

  1. An edge is a measured number carrying its provenance. `BacktestRecord`
     records where it came from, over what period, on how many trades, with
     whatever interval and null-comparison the grader produced. There is no
     code path where an edge is asserted.
  2. A rule fires on the record for the regime it is actually in, not on an
     average across regimes. An edge of +0.32% that is -0.20% in calm markets
     and +0.87% in stressed ones is two strategies wearing one number.

Rules see `snapshot.features` only. Nothing from `snapshot.context` reaches
here — see market.py for why.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Protocol, Sequence

from market import Snapshot

LONG = "long"
SHORT = "short"

UNGRADED = "UNGRADED"


@dataclass(frozen=True)
class BacktestRecord:
    """A measured edge and everything needed to judge whether to believe it.

    `net_pp` is the mean net return per trade in percentage points, after
    costs. It is the number, not a number derived from three other numbers —
    if your grader measured the mean directly, pass it directly.

    The optional statistics are what turn a point estimate into evidence. A
    mean of +0.3% whose bootstrap interval spans zero is not the same claim as
    one whose interval does not, and gates.py can tell them apart only if you
    carry them through.
    """

    period: str
    samples: int
    net_pp: float
    source: str
    """Where this came from. 'edgelab:rsi_oversold_30@2026-08-18', not 'me'."""
    verdict: str = UNGRADED
    regime: str | None = None
    win_rate: float | None = None
    avg_win_pp: float | None = None
    avg_loss_pp: float | None = None
    ci_low_pp: float | None = None
    ci_high_pp: float | None = None
    beats_random_pct: float | None = None
    gross_pp: float | None = None
    cost_pp: float | None = None

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples must be > 0")
        if not self.source:
            raise ValueError("a record without a source is not evidence")
        if self.win_rate is not None and not 0.0 <= self.win_rate <= 1.0:
            raise ValueError("win_rate must be in [0, 1]")
        if (
            self.ci_low_pp is not None
            and self.ci_high_pp is not None
            and self.ci_low_pp > self.ci_high_pp
        ):
            raise ValueError("ci_low_pp must be <= ci_high_pp")

    @classmethod
    def from_win_loss(
        cls,
        *,
        period: str,
        samples: int,
        win_rate: float,
        avg_win_pp: float,
        avg_loss_pp: float,
        source: str,
        **kw,
    ) -> "BacktestRecord":
        """Derive the mean from win rate and average win/loss.

        Use only when the grader did not report a mean directly — deriving
        loses whatever the real distribution did at the tails.
        """
        if avg_win_pp <= 0:
            raise ValueError("avg_win_pp must be > 0")
        if avg_loss_pp < 0:
            raise ValueError("avg_loss_pp must be >= 0")
        net = win_rate * avg_win_pp - (1.0 - win_rate) * avg_loss_pp
        return cls(
            period=period, samples=samples, net_pp=net, source=source,
            win_rate=win_rate, avg_win_pp=avg_win_pp, avg_loss_pp=avg_loss_pp,
            **kw,
        )

    def edge_pp(self) -> float:
        return self.net_pp

    @property
    def ci_excludes_zero(self) -> bool | None:
        """None when no interval was supplied — which is not the same as False
        and gates.py treats it differently."""
        if self.ci_low_pp is None or self.ci_high_pp is None:
            return None
        return self.ci_low_pp > 0 or self.ci_high_pp < 0

    def as_log(self) -> dict:
        return {
            "source": self.source,
            "verdict": self.verdict,
            "regime": self.regime,
            "period": self.period,
            "samples": self.samples,
            "net_pp": round(self.net_pp, 4),
            "ci_pp": None if self.ci_low_pp is None
            else [round(self.ci_low_pp, 4), round(self.ci_high_pp or 0.0, 4)],
            "beats_random_pct": self.beats_random_pct,
        }


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


class RegimeConditional:
    """Wraps a rule so it fires on the record for the current regime.

    A regime with no record produces no signal: absence of evidence here means
    no trade, not the average of the regimes you did measure. This is the
    difference between a rule that made money in stressed markets and a rule
    you believe works everywhere because its mean was positive.
    """

    def __init__(
        self,
        rule: Rule,
        records: Mapping[str, BacktestRecord],
        regime_key: str = "regime",
    ) -> None:
        if not records:
            raise ValueError("RegimeConditional needs at least one regime record")
        self.rule = rule
        self.records = dict(records)
        self.regime_key = regime_key
        self.name = rule.name

    @property
    def backtest(self) -> BacktestRecord:
        """Weakest regime, for introspection. Gating uses the record actually
        selected at evaluation time, not this one."""
        return min(self.records.values(), key=lambda r: r.samples)

    def evaluate(self, snapshot: Snapshot) -> Signal | None:
        regime = snapshot.features.get(self.regime_key)
        record = self.records.get(regime) if regime is not None else None
        if record is None:
            return None
        signal = self.rule.evaluate(snapshot)
        if signal is None:
            return None
        return replace(
            signal,
            rule=f"{signal.rule}[{regime}]",
            backtest=replace(record, regime=regime),
        )


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
            "backtest": self.signal.backtest.as_log() if self.signal else None,
        }


class RuleSet:
    """Runs every rule and reconciles them conservatively.

    Disagreement means stand aside. Agreement takes the *lowest* edge of the
    rules that fired, so adding a rule can never inflate an edge estimate.

    The sample-size floor applies to the record the signal actually carries,
    which for a regime-conditional rule is the record for the regime it fired
    in — not some blended count across regimes.
    """

    def __init__(self, rules: Sequence[Rule], version: str, min_samples: int) -> None:
        self.rules = list(rules)
        self.version = version
        self.min_samples = min_samples

    def evaluate(self, snapshot: Snapshot) -> RuleSetResult:
        fired: list[Signal] = []
        skipped: list[str] = []

        for rule in self.rules:
            signal = rule.evaluate(snapshot)
            if signal is None:
                continue
            if signal.backtest.samples < self.min_samples:
                skipped.append(
                    f"{signal.rule}: {signal.backtest.samples} samples "
                    f"< {self.min_samples}"
                )
                continue
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
    """EXAMPLE ONLY — here to show the shape, not to be traded.

    Whatever BacktestRecord you hand this has to come from a grader. See
    edgelab.py for the adapter that builds one from a graded rule table; a
    record you typed by hand is an opinion with a dataclass around it.
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
