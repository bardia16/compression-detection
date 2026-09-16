"""Shared data models."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candle:
    ts: int          # open time (ms)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Pivot:
    """A confirmed ZigZag pivot. Type: +1 = high, -1 = low."""
    type: int
    price: float
    bar_index: int    # index into the candle array
    ts: int           # open time of the pivot candle (ms)

    @property
    def is_high(self) -> bool:
        return self.type == 1

    @property
    def is_low(self) -> bool:
        return self.type == -1

    def __repr__(self) -> str:
        label = "H" if self.is_high else "L"
        return f"Pivot({label}@{self.price:.6g} bar={self.bar_index})"
