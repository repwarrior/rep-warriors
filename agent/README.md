# agent

A paper-trading loop where the model cannot cause a trade — only prevent one.

```
rules ──► signal ──► size ──► gates ──► veto ──► journal ──► order ──► journal
 code      code      code      code    Claude    (intent)             (outcome)
                                         │
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
python3 -m unittest discover -s . -t . -v          # 60 tests, no network
```

The demo walks a price into a support zone and prints the journal: an entry, a
second entry blocked for exceeding the position cap, then an exit.

## The five decisions worth knowing about

**Gates are asymmetric.** Opening intents clear nine checks. Reducing intents
clear three sanity checks and nothing else — no edge test, no trade cap, no
loss halt. A risk system that can block an exit will eventually hold a position
it cannot close.

**Edges come from a `BacktestRecord`, not from a number someone typed.**
`edge_pp()` is arithmetic over a win rate and average win/loss you had to write
down, and a rule with fewer than `min_backtest_samples` trades cannot fire at
all.

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

## Wiring it up

Four things are yours to supply:

| Thing | Where | Notes |
|---|---|---|
| Rules | `rules.py` | `PullbackToSupport` is an example with a **placeholder** backtest. Delete it. |
| Data source | anything with `.snapshot(symbol)` | Split numbers into `features`, feed text into `context`. |
| Broker | `broker.IBKRPaperBroker` | Only the socket is missing; ports and the account assertion are there. |
| Calendar | `clock.USEquityRTH` | Holiday-blind. Swap in `exchange_calendars` before running unattended. |

Then:

```python
rt = loop.Runtime(ruleset=..., data=..., veto=veto.default_veto(), broker=...,
                  journal=Journal("decisions.jsonl"), account=...,
                  calendar=..., limits=Limits())
loop.run_forever(rt, ["AAPL"], tick_seconds=300)
```

`veto.default_veto()` returns the blocking stub when no Anthropic credentials
are present, so a misconfigured deployment trades nothing rather than trading
unsupervised.

## Notes on the API call

Pinned to `claude-opus-5` and logged on every decision, so a change in
behaviour is attributable. Uses structured outputs (`output_config.format`)
rather than parsing prose. **No `temperature`** — it is rejected with a 400 on
current models; consistency comes from the pinned model and fixed prompt.

## What this does not do

No live trading — there is no code path to a live account, and adding one
should take more than deleting an assertion. No P&L attribution, no slippage
model, no reconciliation loop against broker fills (`_apply_fill` is an
optimistic local cache and says so). Position sizing assumes whole shares and
one currency. The daily loss halt reads `AccountState.equity`, which you have
to keep current from the broker — it is only as good as that number.

And the strategy is still yours. This is the part that runs a strategy safely;
it is not one.
