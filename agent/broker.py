"""Broker interface, and the one assertion that actually keeps this on paper.

A `paper=True` field on a log record is documentation. It stops nothing. The
enforcement that matters is here: the account id is read back from the broker
after connecting and checked against the paper prefixes, so a wrong port, a
stale config or a copied launch script fails at startup instead of at the fill.

IBKR account ids: DU/DF are paper, U/F are live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger("agent.broker")

PAPER_PREFIXES = ("DU", "DF")


class LiveAccountRefused(RuntimeError):
    """Raised when something that is not a paper account answers the connection."""


@dataclass(frozen=True)
class OrderAck:
    order_id: str
    status: str
    detail: str = ""

    def as_log(self) -> dict:
        return {"order_id": self.order_id, "status": self.status, "detail": self.detail}


class Broker(Protocol):
    def account_id(self) -> str: ...
    def equity(self) -> float: ...
    def position(self, symbol: str) -> int: ...
    def place_bracket(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        stop_price: float | None,
        client_order_id: str,
    ) -> OrderAck: ...


def assert_paper_account(broker: Broker) -> str:
    """Call this once at startup, before the loop. Refuses to continue on a
    live account regardless of what any config file says."""
    account = broker.account_id()
    if not account.startswith(PAPER_PREFIXES):
        raise LiveAccountRefused(
            f"account {account!r} is not an IBKR paper account "
            f"(expected one of {PAPER_PREFIXES}). Refusing to start."
        )
    log.info("paper account confirmed: %s", account)
    return account


@dataclass
class DryRunBroker:
    """Fully functional stand-in that records orders instead of sending them.

    The loop runs end to end against this — no credentials, no sockets — so the
    journal, the gates and the veto can all be exercised before anything is
    wired to a real account.
    """

    starting_equity: float = 100_000.0
    account: str = "DU0000000"
    orders: list[dict] = field(default_factory=list)
    positions: dict[str, int] = field(default_factory=dict)
    _seen: dict[str, OrderAck] = field(default_factory=dict)

    def account_id(self) -> str:
        return self.account

    def equity(self) -> float:
        return self.starting_equity

    def position(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)

    def place_bracket(
        self,
        *,
        symbol: str,
        side: str,
        qty: int,
        stop_price: float | None,
        client_order_id: str,
    ) -> OrderAck:
        # Idempotency: the client_order_id is the decision_id, so replaying a
        # decision after a crash returns the original ack instead of doubling up.
        if client_order_id in self._seen:
            return self._seen[client_order_id]

        order = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "stop_price": stop_price,
            "client_order_id": client_order_id,
        }
        self.orders.append(order)
        signed = qty if side == "buy" else -qty
        self.positions[symbol] = self.positions.get(symbol, 0) + signed

        ack = OrderAck(order_id=f"dry-{len(self.orders)}", status="accepted")
        self._seen[client_order_id] = ack
        return ack


class IBKRPaperBroker:
    """Sketch of the real connector. Everything except the socket is above.

    Ports, so a typo cannot quietly become a live session:
        TWS      paper 7497   live 7496
        Gateway  paper 4002   live 4001

    Whatever client library you use (ib_insync, ibapi), connect here, then let
    `assert_paper_account` read the account id back. Two independent things then
    have to be wrong before a live order is possible: the port and the account
    prefix. Send the entry and its stop as one bracket — an entry that reaches
    the book without its stop attached is the failure mode this whole package
    exists to avoid.
    """

    PAPER_PORTS = (7497, 4002)

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        if port not in self.PAPER_PORTS:
            raise LiveAccountRefused(
                f"port {port} is not a paper port {self.PAPER_PORTS}"
            )
        self.host, self.port, self.client_id = host, port, client_id
        raise NotImplementedError(
            "Wire to ib_insync/ibapi here, then run assert_paper_account(self)"
        )
