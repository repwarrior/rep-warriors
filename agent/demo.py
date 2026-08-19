"""Run the whole loop against a fake feed and a dry-run broker.

    cd agent && python3 demo.py

No credentials, no broker, no network. It walks a price into a support zone,
takes the entry, then walks it back out and takes the exit — and prints the
journal, which is the thing actually worth looking at.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import loop
import veto
from account import AccountState
from broker import DryRunBroker
from clock import AlwaysOpen
from config import Limits
from journal import Journal
from market import Snapshot
from rules import SHORT, BacktestRecord, PullbackToSupport, RuleSet, Signal

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Placeholder, and labelled as one. In anything real this comes out of
# edgelab.from_rows(), which is the point of that module existing.
BACKTEST = BacktestRecord(
    period="2019-01..2025-12", samples=240, net_pp=3.15,
    source="PLACEHOLDER - not a real backtest", verdict="PROVEN",
    ci_low_pp=1.10, ci_high_pp=5.20, beats_random_pct=99.4,
)

SUPPORT, WIDTH = 99.5, 1.0


class BreakdownExit:
    """Fires short when price loses the zone. Holding long, that classifies as
    a REDUCE and leaves by the path that no gate and no model can block."""

    name = "breakdown"

    def __init__(self, backtest):
        self.backtest = backtest

    def evaluate(self, snapshot):
        support = snapshot.features["support"]
        if snapshot.last >= support - snapshot.features["support_width"]:
            return None
        return Signal(
            symbol=snapshot.symbol, rule=self.name, rules_version="demo-1",
            direction=SHORT, reference_price=snapshot.last,
            stop_price=support + 0.5, backtest=self.backtest,
        )


class ScriptedFeed:
    """Prices that walk into the support zone and then out the other side."""

    def __init__(self, prices):
        self.prices = list(prices)
        self.i = 0

    def snapshot(self, symbol):
        price = self.prices[min(self.i, len(self.prices) - 1)]
        self.i += 1
        # Below the zone the trend filter gives up, which flips the rule short
        # and turns the next tick into an exit rather than an entry.
        return Snapshot(
            symbol=symbol,
            as_of=datetime.now(timezone.utc) - timedelta(seconds=2),
            last=price, bid=price - 0.01, ask=price + 0.01,
            features={"support": SUPPORT, "support_width": WIDTH,
                      "trend_up": price >= SUPPORT},
            context={"headline": "Analyst calls it a generational buy"},
        )


def main() -> None:
    tmp = Path(tempfile.mkdtemp()) / "decisions.jsonl"
    broker = DryRunBroker(starting_equity=100_000)
    journal = Journal(tmp)

    rt = loop.Runtime(
        ruleset=RuleSet(
            [PullbackToSupport(BACKTEST, "demo-1"), BreakdownExit(BACKTEST)],
            "demo-1", min_samples=100,
        ),
        data=ScriptedFeed([101.2, 100.0, 99.7, 98.9, 98.4]),
        # AlwaysAllow stands in for Claude so the demo runs offline. In anything
        # real this is veto.default_veto(), which blocks when it cannot reach
        # the API.
        veto=veto.AlwaysAllow(),
        broker=broker,
        journal=journal,
        account=AccountState(equity=100_000, day_start_equity=100_000),
        calendar=AlwaysOpen(),
        limits=Limits(),
    )

    loop.preflight(rt)
    for tick in range(5):
        result = loop.run_tick(rt, "AAPL")
        print(f"tick {tick}: {result.outcome:<12} {result.detail}")

    print(f"\norders placed: {broker.orders}")
    print(f"position: {broker.position('AAPL')}")
    print(f"\njournal ({tmp}):")
    for record in journal.read_all():
        keep = {k: record[k] for k in ("type", "decision_id", "intent", "gate")
                if k in record}
        keep["decision_id"] = keep["decision_id"][:8]
        print("  " + json.dumps(keep, default=str))

    graded_rules_report()


def graded_rules_report() -> None:
    """The other half: what a real graded rule table does at the gate.

    These are the published edgelab figures. Every one is blocked, which is the
    system agreeing with the grader rather than a second opinion about it.
    """
    import edgelab

    rows = [
        {"rule": "rsi_oversold_30", "verdict": "LIKELY", "n": 524,
         "mean_net": "+0.315%", "ci_low": -0.088, "ci_high": 0.707,
         "beats_random_pct": "100.0%", "period": "2014-08-15..2026-08-12"},
        {"rule": "ultosc_oversold_30", "verdict": "LIKELY", "n": 373,
         "mean_net": "+0.177%", "beats_random_pct": "99.2%",
         "period": "2014-08-15..2026-08-12"},
        {"rule": "macd_bull_cross", "verdict": "NO_EDGE", "n": 1935,
         "mean_net": "-0.503%", "beats_random_pct": "20.0%",
         "period": "2014-08-15..2026-08-12"},
    ]
    loaded = edgelab.from_rows(rows, source="edgelab@2026-08-18",
                               scale=edgelab.PP)
    print("\ngraded rules at the gate:")
    print(edgelab.gate_report(loaded.records, Limits()))


if __name__ == "__main__":
    main()
