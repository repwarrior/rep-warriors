"""Append-only decision journal, written in two phases.

Phase one records the intent *before* the order goes out. Phase two records
what came back. Both carry the same decision_id, so a crash between them leaves
an intent with no outcome — which is exactly the state you want to find on
restart, because it is the one that tells you to go reconcile with the broker.

Writing the log after the order, as one record, loses precisely that case: the
process dies mid-order and the journal says nothing ever happened.

Every record carries the snapshot hash, the rules version, and the model id, so
a decision can be replayed against the inputs and the code that produced it.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2


def new_decision_id() -> str:
    return uuid.uuid4().hex


class Journal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record_type: str, decision_id: str, **payload) -> dict:
        record = {
            "schema_version": SCHEMA_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": record_type,
            "decision_id": decision_id,
            **payload,
        }
        line = json.dumps(record, sort_keys=True, default=str)
        # Opened per write and fsync'd: the intent record has to survive a
        # power cut between phases, which a buffered handle does not guarantee.
        with self.path.open("a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def unreconciled(self) -> list[dict]:
        """Intents with no matching outcome — orders that may have gone out
        while the process was dying. Check these against the broker on startup
        before placing anything new."""
        records = self.read_all()
        done = {r["decision_id"] for r in records if r["type"] == "outcome"}
        return [
            r for r in records
            if r["type"] == "intent" and r["decision_id"] not in done
        ]
