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

# Placeholder, as the class docstring says at length. Real numbers or no trade.
BACKTEST = BacktestRecord(
    period="2019-01..2025-12 (PLACEHOLDER - not a real backtest)",
    samples=240, win_rate=0.55, avg_win_pp=9.0, avg_loss_pp=4.0,
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


if __name__ == "__main__":
    main()
