"""The tick.

Order of operations, and why:

    session open?          an unattended loop should sleep when the market does
    fresh quote?           stale data is worse than no data
    rules -> signal        deterministic; the only thing that may say "trade"
    classify + size        exit or entry, and how big, computed here not passed in
    gates                  hard limits, asymmetric: entries filtered, exits not
    veto                   Claude, last, and only able to remove a trade
    journal the intent     fsync'd BEFORE the order exists
    place the order        client_order_id = decision_id, so replays don't double
    journal the outcome    what actually came back

The veto sits after the gates so a blocked trade never costs an API call, and
so the model is never in a position to be the reason something happened — only
ever the reason something didn't.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

import gates
import intents
from account import AccountState
from broker import Broker, assert_paper_account
from clock import SessionCalendar, seconds_to_next_tick, utcnow
from config import Limits
from intents import Intent, Kind
from journal import Journal, new_decision_id
from market import Snapshot
from rules import RuleSet
from veto import Veto, VetoMode

log = logging.getLogger("agent.loop")

MARKET_CLOSED = "market_closed"
STALE = "stale_data"
NO_SIGNAL = "no_signal"
NO_SIZE = "no_size"
BLOCKED = "blocked"
VETOED = "vetoed"
ORDERED = "ordered"
ORDER_FAILED = "order_failed"


class DataSource(Protocol):
    def snapshot(self, symbol: str) -> Snapshot: ...


@dataclass
class Runtime:
    ruleset: RuleSet
    data: DataSource
    veto: Veto
    broker: Broker
    journal: Journal
    account: AccountState
    calendar: SessionCalendar
    limits: Limits
    veto_mode: VetoMode = VetoMode.ENFORCE


@dataclass(frozen=True)
class TickResult:
    outcome: str
    decision_id: str | None = None
    detail: str = ""


def preflight(rt: Runtime) -> None:
    """Startup checks. Both of these refuse to continue rather than warn.

    An unreconciled intent means a previous run wrote an intent and never
    recorded what happened to it — there may be a live order out there. Sorting
    that out is a human's job, and doing it before the loop places anything new
    is the whole reason the intent record is written first.
    """
    assert_paper_account(rt.broker)

    pending = rt.journal.unreconciled()
    if pending:
        ids = ", ".join(r["decision_id"][:8] for r in pending[:5])
        raise RuntimeError(
            f"{len(pending)} unreconciled intent(s) in {rt.journal.path} ({ids}). "
            "Check these against the broker and append an outcome record for "
            "each before restarting."
        )


def run_tick(rt: Runtime, symbol: str, now: datetime | None = None) -> TickResult:
    now = now or utcnow()

    if not rt.calendar.is_open(now):
        return TickResult(MARKET_CLOSED)

    snapshot = rt.data.snapshot(symbol)
    decision_id = new_decision_id()

    def record(record_type: str, **payload) -> None:
        rt.journal.write(
            record_type,
            decision_id,
            symbol=symbol,
            snapshot_sha256=snapshot.sha256(),
            snapshot_as_of=snapshot.as_of,
            rules_version=rt.ruleset.version,
            **payload,
        )

    if snapshot.is_stale(now, rt.limits.max_snapshot_age_s):
        age = snapshot.age_s(now)
        record(STALE, age_s=round(age, 2), max_age_s=rt.limits.max_snapshot_age_s)
        return TickResult(STALE, decision_id, f"quote {age:.1f}s old")

    result = rt.ruleset.evaluate(snapshot)
    if result.signal is None:
        record(NO_SIGNAL, rules=result.as_log())
        return TickResult(NO_SIGNAL, decision_id, result.note)

    intent = intents.build(result.signal, rt.account, rt.limits)
    if intent is None:
        # Sized to zero — below one share at this risk budget. Not an error, and
        # emphatically not an order for nothing.
        record(NO_SIZE, rules=result.as_log(), equity=rt.account.equity)
        return TickResult(NO_SIZE, decision_id, "sized to zero shares")

    gate = gates.evaluate(intent, rt.account, rt.limits)
    if not gate.passed:
        record(BLOCKED, intent=intent.as_log(), gate=gate.as_log(),
               rules=result.as_log())
        return TickResult(BLOCKED, decision_id, gate.reason)

    # Exits skip the veto entirely. Nothing gets to talk you out of an exit.
    veto_log = None
    if intent.kind is Kind.OPEN and rt.veto_mode is not VetoMode.OFF:
        verdict = rt.veto.consult(intent, snapshot)
        veto_log = {**verdict.as_log(), "mode": rt.veto_mode.value}
        if not verdict.allowed and rt.veto_mode is VetoMode.ENFORCE:
            record(VETOED, intent=intent.as_log(), gate=gate.as_log(),
                   veto=veto_log, rules=result.as_log())
            return TickResult(VETOED, decision_id, verdict.reason)
        if not verdict.allowed:
            # Shadow mode: the objection is on the record, the trade proceeds,
            # and whether the objection was worth anything is a later question
            # for the data rather than a judgement made here.
            log.info("shadow veto on %s: %s", symbol, verdict.reason)

    # Phase one: the intent is durable before the order can exist.
    record("intent", intent=intent.as_log(), gate=gate.as_log(),
           veto=veto_log, rules=result.as_log())

    try:
        ack = rt.broker.place_bracket(
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            stop_price=intent.stop_price,
            client_order_id=decision_id,
        )
    except Exception as exc:  # noqa: BLE001 - the outcome record must be written
        record("outcome", status="error", error=f"{type(exc).__name__}: {exc}")
        raise

    record("outcome", status=ack.status, order=ack.as_log())
    _apply_fill(rt.account, intent)

    log.info(
        "%s %s %d @ ~%.2f (%s) order=%s",
        intent.side, symbol, intent.qty, intent.reference_price,
        intent.kind.value, ack.order_id,
    )
    return TickResult(ORDERED, decision_id, ack.order_id)


def _apply_fill(account: AccountState, intent: Intent) -> None:
    """Optimistic local update. Reconcile against the broker on the next tick —
    this is a cache, not the truth."""
    account.positions[intent.symbol] = account.position(intent.symbol) + intent.signed_qty
    if intent.kind is Kind.OPEN:
        account.daily_trades += 1
        group = account.group_of(intent.symbol)
        account.group_exposure[group] = (
            account.group_exposure.get(group, 0.0) + intent.notional
        )


def run_forever(
    rt: Runtime,
    symbols: Sequence[str],
    tick_seconds: float = 300.0,
    sleeper=None,
) -> None:
    """Runs until the error kill switch trips or it is interrupted.

    A loop that logs an exception and carries on regardless will happily run all
    night on a broken data feed. After `max_consecutive_errors` clean-failing
    ticks it stops and leaves the account alone.
    """
    import time as _time

    sleep = sleeper or _time.sleep
    preflight(rt)
    consecutive_errors = 0

    while True:
        for symbol in symbols:
            try:
                result = run_tick(rt, symbol)
                consecutive_errors = 0
                if result.outcome not in (MARKET_CLOSED,):
                    log.debug("%s: %s (%s)", symbol, result.outcome, result.detail)
            except Exception:
                consecutive_errors += 1
                log.exception(
                    "tick failed for %s (%d/%d consecutive)",
                    symbol, consecutive_errors, rt.limits.max_consecutive_errors,
                )
                if consecutive_errors >= rt.limits.max_consecutive_errors:
                    log.error("error kill switch tripped; stopping")
                    return
        sleep(seconds_to_next_tick(utcnow(), tick_seconds))
