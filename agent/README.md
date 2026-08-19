# agent

A paper-trading loop where the model cannot cause a trade — only prevent one.

```
grader ──► BacktestRecord ──► rules ──► size ──► gates ──► veto ──► order
(edgelab)   the only source    signal    code      code    Claude    +journal
            of an edge                                       │
                                                    can only say no
```

Deterministic rules produce the signal and the edge estimate. Hard-coded gates
filter it. Claude sees the result last, and its response schema has exactly two
fields — a boolean and a sentence — so there is no representable answer that
raises size, flips direction or relaxes a limit. Every model failure mode
(a bad call, drift between versions, a hostile headline in the data) costs at
most a trade not taken.

## Run it

```bash
cd agent
python3 demo.py                                    # full loop, no credentials
python3 -m unittest discover -s . -t . -v          # 89 tests, no network
```

The demo walks a price into a support zone and prints the journal — an entry, a
second entry blocked at the position cap, then an exit — followed by a real
graded rule table run through the gate, where nothing clears.

## The five decisions worth knowing about

**Gates are asymmetric.** Opening intents clear nine checks. Reducing intents
clear three sanity checks and nothing else — no edge test, no trade cap, no
loss halt. A risk system that can block an exit will eventually hold a position
it cannot close.

**Edges come from a grader, and carry their statistics with them.** A
`BacktestRecord` holds the measured mean *and* its source, verdict, confidence
interval and random-entry percentile, so the gate can distinguish "+0.3% with
an interval clear of zero" from "+0.3% with an interval spanning it". A record
that reports no interval **fails** the check rather than passing it — absent
evidence is not favourable evidence. `edgelab.py` builds these from a graded
rule table via a `FieldMap`, so your grader's column names are yours.

**Rules fire on the record for the regime they are in.** `RegimeConditional`
selects the per-regime record and produces no signal in a regime with no
evidence. A rule measured at −0.20% in calm markets and +0.87% in stressed ones
is two strategies; trading its +0.32% average is trading neither.

**The journal is written in two phases.** The intent record is fsync'd *before*
the order exists; the outcome record follows. A crash between them leaves an
intent with no outcome, which `preflight()` refuses to start on — that is the
state that means "go check the broker", and writing one record after the order
loses it entirely.

**Paper mode is enforced at the connection.** `assert_paper_account` reads the
account id back from the broker and refuses anything that isn't a `DU`/`DF`
IBKR paper account. A `paper=True` field on a log line stops nothing.

**Untrusted text is quarantined.** `Snapshot.features` holds numbers you
computed and is all the rules ever see. `Snapshot.context` holds text off a
feed and only ever reaches the veto prompt, fenced. Nothing anybody else writes
can originate a trade.

**The veto can be graded before it is trusted.** `VetoMode.SHADOW` records the
model's verdict and places the trade anyway. `journal.shadow_rows()` exports
the verdicts for joining to your scored returns on `decision_id`, and
`veto_scorecard()` compares what the veto objected to against what it allowed.
Promote to `ENFORCE` when it has earned it; delete it when it has not. This is
the only honest way to add a language model to a system whose whole premise is
measurement.

## Wiring it up

Four things are yours to supply:

| Thing | Where | Notes |
|---|---|---|
| Rules | `rules.py` | `PullbackToSupport` is an example with a **placeholder** record. Delete it. |
| Graded records | `edgelab.py` | `from_rows`/`from_csv`/`from_json` + a `FieldMap`. Set `scale` explicitly — 0.00315 and 0.315 are the same edge and confusing them is silent. |
| Data source | anything with `.snapshot(symbol)` | Split numbers into `features`, feed text into `context`. |
| Broker | `broker.IBKRPaperBroker` | Only the socket is missing; ports and the account assertion are there. |
| Calendar | `clock.USEquityRTH` | Holiday-blind. Swap in `exchange_calendars` before running unattended. |

Then:

```python
graded = edgelab.from_csv("graded_rules.csv", source="edgelab@2026-08-18",
                          scale=edgelab.PP)
print(edgelab.gate_report(graded.records, Limits()))   # do this first

rt = loop.Runtime(ruleset=..., data=..., veto=veto.default_veto(), broker=...,
                  journal=Journal("decisions.jsonl"), account=...,
                  calendar=..., limits=Limits(),
                  veto_mode=veto.VetoMode.SHADOW)
loop.run_forever(rt, ["AAPL"], tick_seconds=300)
```

Run `gate_report` before wiring anything to a broker. If the answer is "0 would
trade", that is the system working, and the itemised reasons say what would
have to change.

If you already have a scheduler, call `loop.run_tick` from it rather than using
`run_forever` — one clock is better than two, and a job timed deliberately
around the exchange date rollover beats a generic interval.

`veto.default_veto()` returns the blocking stub when no Anthropic credentials
are present, so a misconfigured deployment trades nothing rather than trading
unsupervised.

## Notes on the API call

Pinned to `claude-opus-5` and logged on every decision, so a change in
behaviour is attributable. Uses structured outputs (`output_config.format`)
rather than parsing prose. **No `temperature`** — it is rejected with a 400 on
current models; consistency comes from the pinned model and fixed prompt.

## What this does not do

Not a grader. It consumes verdicts; it does not produce them, and it has no
opinion about whether a rule is any good beyond the statistics it was handed.

No live trading — there is no code path to a live account, and adding one
should take more than deleting an assertion. No P&L attribution, no slippage
model, no reconciliation loop against broker fills (`_apply_fill` is an
optimistic local cache and says so). Position sizing assumes whole shares and
one currency. The daily loss halt reads `AccountState.equity`, which you have
to keep current from the broker — it is only as good as that number.

And the strategy is still yours. This is the part that runs a strategy safely;
it is not one.
