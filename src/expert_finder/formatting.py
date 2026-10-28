"""Small formatting helpers shared across modules."""

from __future__ import annotations

import re


def usd(amount: float | None) -> str:
    """Money for human readers: no cents above $10, 'n/a' when unknown."""
    if amount is None:
        return "n/a"
    return f"${amount:,.0f}" if abs(amount) >= 10 else f"${amount:,.2f}"


def short_name(name: str) -> str:
    """A business name without its legal suffix, as people actually say it."""
    return re.sub(r"\s+(LLC|Inc\.?|Co\.)$", "", name).strip()
