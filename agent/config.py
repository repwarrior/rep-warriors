"""Every number that can stop a trade, in one place.

No other module in this package hard-codes a limit. If you want to know what
this system will and will not do, this file is the entire answer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    """Risk limits. Frozen on purpose — nothing mutates these at runtime."""

    # --- entry filters -------------------------------------------------
    min_edge_pp: float = 2.0
    """Minimum measured net edge, in percentage points, for an opening trade.

    Carried over from the original skeleton's 2-4pp threshold. Be aware that
    this is very large for short-hold mean reversion in liquid names: a grader
    measuring that strategy class over 12 years put its best rule at 0.315pp
    net per trade, six times below this bar. Either the threshold belongs to a
    different strategy class or nothing in that class is tradeable — worth
    settling deliberately rather than inheriting. The statistical checks below
    are the principled version of the same question."""

    min_backtest_samples: int = 100
    """A signal whose record has fewer trades than this cannot fire."""

    accepted_verdicts: frozenset[str] = frozenset({"PROVEN"})
    """Grader verdicts allowed to trade. Widen to include weaker verdicts only
    deliberately — e.g. frozenset({"PROVEN", "LIKELY"}) when paper trading to
    collect execution data, where the gate blocking every entry is fine."""

    require_ci_excludes_zero: bool = True
    """Refuse a rule whose confidence interval on the mean spans zero, and
    refuse one that reports no interval at all. A positive point estimate whose
    interval includes zero is a hypothesis, not an edge."""

    min_beats_random_pct: float | None = 95.0
    """Percentile the rule must reach against a random-entry null. None skips
    the check; a rule reporting no null comparison fails while it is set."""

    # --- sizing --------------------------------------------------------
    risk_fraction: float = 0.005
    """Fraction of equity risked between entry and stop. 0.5% per trade."""

    max_position_pct: float = 0.02
    """Hard cap on notional per position as a fraction of equity."""

    # --- caps ----------------------------------------------------------
    max_daily_trades: int = 5
    max_correlated_pct: float = 0.05
    """Cap on combined notional within one correlation group."""

    # --- halts ---------------------------------------------------------
    daily_loss_halt_pct: float = 0.02
    """Stop opening anything once the day is down this much. Exits stay open."""

    max_consecutive_errors: int = 3
    """Consecutive failed ticks before the loop stops rather than grinding on."""

    # --- data hygiene --------------------------------------------------
    max_snapshot_age_s: float = 60.0
    """Refuse to act on a quote older than this."""

    def __post_init__(self) -> None:
        if not 0 < self.risk_fraction <= 0.05:
            raise ValueError("risk_fraction must be in (0, 0.05]")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        if self.min_edge_pp < 0:
            raise ValueError("min_edge_pp must be >= 0")
        if self.max_daily_trades < 0:
            raise ValueError("max_daily_trades must be >= 0")
        if self.daily_loss_halt_pct <= 0:
            raise ValueError("daily_loss_halt_pct must be > 0")
        if self.max_snapshot_age_s <= 0:
            raise ValueError("max_snapshot_age_s must be > 0")
