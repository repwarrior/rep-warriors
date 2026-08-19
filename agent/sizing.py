"""Position sizing, computed from the decision — not handed to it.

Size follows from two things the strategy already committed to: how much of
the account you are willing to lose on the trade, and where the stop sits.
Nothing else gets a vote. A missing or nonsensical stop sizes to zero, and
zero shares means no order rather than an order for nothing.
"""

from __future__ import annotations

import math

from config import Limits


def shares_for(
    *,
    equity: float,
    reference_price: float,
    stop_price: float,
    limits: Limits,
) -> int:
    """Whole shares to trade. 0 means "do not trade" and is not an error."""
    if equity <= 0 or reference_price <= 0:
        return 0

    risk_per_share = abs(reference_price - stop_price)
    if risk_per_share <= 0:
        return 0  # no stop distance means unbounded risk; refuse to size it

    by_risk = (equity * limits.risk_fraction) / risk_per_share
    by_notional = (equity * limits.max_position_pct) / reference_price

    return max(0, math.floor(min(by_risk, by_notional)))
