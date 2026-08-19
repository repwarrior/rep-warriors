"""Hard risk gates, enforced in code, asymmetric by design.

The asymmetry is the point. An opening intent has to clear every check below.
A reducing intent clears only the sanity checks, because a risk system that
can block an exit has stopped being a risk system: a full daily trade count,
a thin edge and a bad day are all reasons to get *out*, and a gate that reads
them as reasons to stay in will eventually hold a position it cannot close.

Every check is recorded whether it passed or not, so the journal shows what the
gate saw rather than just its verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from account import AccountState
from config import Limits
from intents import Intent, Kind


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: list[Check]

    @property
    def reason(self) -> str:
        failed = [c for c in self.checks if not c.passed]
        return failed[0].detail if failed else "ok"

    def as_log(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def _result(checks: list[Check]) -> GateResult:
    return GateResult(passed=all(c.passed for c in checks), checks=checks)


def evaluate(intent: Intent, account: AccountState, limits: Limits) -> GateResult:
    if intent.kind is Kind.REDUCE:
        return _evaluate_reduce(intent, account)
    return _evaluate_open(intent, account, limits)


def _evaluate_reduce(intent: Intent, account: AccountState) -> GateResult:
    """Sanity only. No edge test, no trade cap, no loss halt — by design."""
    held = account.position(intent.symbol)
    checks = [
        Check("qty_positive", intent.qty > 0, f"qty {intent.qty}"),
        Check(
            "reduces_exposure",
            held != 0 and (intent.signed_qty * held) < 0,
            f"holding {held}, intent {intent.signed_qty:+d}",
        ),
        Check(
            "does_not_cross_zero",
            intent.qty <= abs(held),
            f"qty {intent.qty} vs position {abs(held)}",
        ),
    ]
    return _result(checks)


def _evaluate_open(intent: Intent, account: AccountState, limits: Limits) -> GateResult:
    signal = intent.signal
    edge = signal.edge_pp if signal else 0.0
    samples = signal.backtest.samples if signal else 0

    day_pnl = account.day_pnl_pct()
    # Caps are on the resulting position, not on the increment. Checking only
    # the new order lets a rule that keeps firing pyramid past the cap one
    # compliant slice at a time.
    held_notional = abs(account.position(intent.symbol)) * intent.reference_price
    resulting_pct = (held_notional + intent.notional) / account.equity
    correlated_pct = (account.exposure_of(intent.symbol) + intent.notional) / account.equity

    checks = [
        Check(
            "daily_loss_halt",
            day_pnl > -limits.daily_loss_halt_pct,
            f"day P&L {day_pnl:.2%} vs halt at -{limits.daily_loss_halt_pct:.2%}",
        ),
        Check(
            "has_signal",
            signal is not None,
            f"rule {signal.rule}" if signal else "no signal on an opening intent",
        ),
        Check(
            "backtest_samples",
            samples >= limits.min_backtest_samples,
            f"{samples} backtest samples, need {limits.min_backtest_samples}",
        ),
        Check(
            "edge_threshold",
            edge >= limits.min_edge_pp,
            f"edge {edge:.2f}pp vs minimum {limits.min_edge_pp:.2f}pp",
        ),
        Check(
            "stop_present",
            intent.stop_price is not None,
            f"stop {intent.stop_price}" if intent.stop_price is not None
            else "opening intent must carry a stop",
        ),
        Check("qty_positive", intent.qty > 0, f"qty {intent.qty}"),
        Check(
            "daily_trade_cap",
            account.daily_trades < limits.max_daily_trades,
            f"{account.daily_trades} trades today, cap {limits.max_daily_trades}",
        ),
        Check(
            "position_cap",
            resulting_pct <= limits.max_position_pct,
            f"resulting position {resulting_pct:.2%} "
            f"(holding {account.position(intent.symbol)}) "
            f"vs cap {limits.max_position_pct:.2%}",
        ),
        Check(
            "correlated_cap",
            correlated_pct <= limits.max_correlated_pct,
            f"group {account.group_of(intent.symbol)} at {correlated_pct:.2%} "
            f"vs cap {limits.max_correlated_pct:.2%}",
        ),
    ]
    return _result(checks)
