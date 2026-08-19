"""Adapter: a grader's output becomes the only source of edge in this package.

`rules.BacktestRecord` is the object the gates trust. This module builds those
records from a graded rule table — the point being that after wiring this up
there is no way to introduce an edge estimate except by grading a rule.

Column names are yours, not mine. Supply a `FieldMap` and this reads whatever
your grader emits; nothing here assumes a schema it has not been told about,
and a missing required column raises rather than defaulting to something
convenient.

Units are the one thing worth being paranoid about. A grader that stores
0.00315 and a grader that stores 0.315 are both describing the same 0.315%
edge, and getting it wrong by 100x silently sails through every gate. Set
`scale` explicitly and check the loaded output against your own table.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from config import Limits
from rules import BacktestRecord

PP = "pp"          # values are already percentage points: 0.315 means 0.315%
FRACTION = "fraction"  # values are fractions: 0.00315 means 0.315%


@dataclass(frozen=True)
class FieldMap:
    """Logical name -> the column your grader actually calls it.

    Only `rule`, `samples` and `net` are required. Everything else is optional,
    but a record missing its interval or its random-entry null will fail the
    corresponding gate — absent evidence is not favourable evidence.
    """

    rule: str = "rule"
    samples: str = "n"
    net: str = "mean_net"
    verdict: str | None = "verdict"
    period: str | None = "period"
    regime: str | None = None
    ci_low: str | None = "ci_low"
    ci_high: str | None = "ci_high"
    beats_random: str | None = "beats_random_pct"
    gross: str | None = None
    cost: str | None = None
    win_rate: str | None = None

    def required(self) -> tuple[str, ...]:
        return (self.rule, self.samples, self.net)


@dataclass
class LoadReport:
    records: dict[str, BacktestRecord] = field(default_factory=dict)
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)


def _num(row: Mapping, key: str | None, scale: float) -> float | None:
    if key is None or key not in row:
        return None
    value = row[key]
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "").replace("+", "").replace(",", "")
        if not value:
            return None
    return float(value) * scale


def from_rows(
    rows: Iterable[Mapping],
    *,
    source: str,
    fields: FieldMap | None = None,
    scale: str = PP,
    require_verdicts: Sequence[str] | None = None,
) -> LoadReport:
    """Build records from graded rows.

    `source` should identify the grading run, not just the tool — the string
    ends up in every journal entry for every trade the rule produces, and
    "which grading run said this was tradeable" is a question you will ask.

    `require_verdicts` refuses to build records outside that set at load time.
    That is separate from `Limits.accepted_verdicts`, which decides at trade
    time; use the former to keep ungraded rules out of a live rule set
    entirely, and the latter to widen what may trade in a particular run.
    """
    fields = fields or FieldMap()
    if scale not in (PP, FRACTION):
        raise ValueError(f"scale must be {PP!r} or {FRACTION!r}")
    factor = 100.0 if scale == FRACTION else 1.0
    allowed = {v.strip().upper() for v in require_verdicts} if require_verdicts else None

    report = LoadReport()
    for i, row in enumerate(rows):
        missing = [c for c in fields.required() if c not in row]
        if missing:
            raise KeyError(
                f"row {i} is missing required column(s) {missing}. "
                f"Available: {sorted(row)}. Adjust the FieldMap."
            )

        name = str(row[fields.rule]).strip()
        verdict = str(row.get(fields.verdict, "")).strip().upper() if fields.verdict else ""
        verdict = verdict or "UNGRADED"

        if allowed is not None and verdict not in allowed:
            report.rejected.append((name, f"verdict {verdict} not in {sorted(allowed)}"))
            continue

        net = _num(row, fields.net, factor)
        if net is None:
            report.rejected.append((name, "no net return reported"))
            continue

        samples = int(float(row[fields.samples]))
        if samples <= 0:
            report.rejected.append((name, f"{samples} samples"))
            continue

        report.records[name] = BacktestRecord(
            period=str(row.get(fields.period, "unspecified")) if fields.period else "unspecified",
            samples=samples,
            net_pp=net,
            source=f"{source}:{name}",
            verdict=verdict,
            regime=str(row[fields.regime]) if fields.regime and row.get(fields.regime) else None,
            win_rate=_num(row, fields.win_rate, 1.0),
            ci_low_pp=_num(row, fields.ci_low, factor),
            ci_high_pp=_num(row, fields.ci_high, factor),
            beats_random_pct=_num(row, fields.beats_random, 1.0),
            gross_pp=_num(row, fields.gross, factor),
            cost_pp=_num(row, fields.cost, factor),
        )
    return report


def from_json(path: str | Path, *, source: str, **kw) -> LoadReport:
    data = json.loads(Path(path).read_text())
    rows = data if isinstance(data, list) else data.get("rules", [])
    return from_rows(rows, source=source, **kw)


def from_csv(path: str | Path, *, source: str, **kw) -> LoadReport:
    with Path(path).open(newline="") as f:
        return from_rows(list(csv.DictReader(f)), source=source, **kw)


def gate_report(records: Mapping[str, BacktestRecord], limits: Limits) -> str:
    """Which loaded rules would clear the entry gate, and what stops the rest.

    Run this before wiring anything to a broker. If the answer is "none", that
    is the system working — and the itemised reasons are more useful than the
    verdict, because they say what would have to change.
    """
    from account import AccountState
    from intents import Intent, Kind
    from rules import LONG, Signal
    import gates

    account = AccountState(equity=100_000.0, day_start_equity=100_000.0)
    lines = []
    for name in sorted(records):
        record = records[name]
        signal = Signal(
            symbol="PROBE", rule=name, rules_version=record.source,
            direction=LONG, reference_price=100.0, stop_price=97.0,
            backtest=record,
        )
        intent = Intent(
            symbol="PROBE", side="buy", qty=1, kind=Kind.OPEN,
            stop_price=97.0, reference_price=100.0, signal=signal,
        )
        result = gates.evaluate(intent, account, limits)
        if result.passed:
            lines.append(f"  PASS  {name}  ({record.net_pp:+.3f}pp, {record.verdict})")
        else:
            failed = [c for c in result.checks if not c.passed]
            lines.append(f"  BLOCK {name}  ({record.net_pp:+.3f}pp, {record.verdict})")
            lines += [f"          {c.name}: {c.detail}" for c in failed]

    passed = sum(
        1 for line in lines if line.startswith("  PASS")
    )
    header = f"{len(records)} rule(s) loaded, {passed} would trade under these limits"
    return "\n".join([header, *lines])
