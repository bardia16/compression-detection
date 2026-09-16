"""ATR (Average True Range) — two flavors.

- close_only: TR = |close_t - close_{t-1}|. TR at bar 0 does not exist
  (no previous close). Matches the close-only philosophy of the ZigZag.
- true_range: TR = max(high-low, |high-prevClose|, |low-prevClose|).
  TR at bar 0 = high - low, but for consistent alignment between both
  flavors the RMA is seeded from bars 1..length in both cases, so both
  are first available at index >= length (i.e. length+1 candles needed).

Smoothing is Wilder's RMA (same as Pine's ta.atr):
  rma_t = (prev * (length - 1) + x_t) / length
  seeded with the SMA of the first `length` TR values (bars 1..length).
"""
from __future__ import annotations

from typing import List, Optional

from .models import Candle


def true_ranges_close_only(closes: List[float]) -> List[Optional[float]]:
    """TR[i] = |close[i] - close[i-1]|; TR[0] is None."""
    return [None if i == 0 else abs(closes[i] - closes[i - 1]) for i in range(len(closes))]


def true_ranges_standard(
    highs: List[float], lows: List[float], closes: List[float]
) -> List[Optional[float]]:
    """Standard TR. TR[0] = high-low; bars >= 1 use prev close."""
    out: List[Optional[float]] = []
    for i in range(len(closes)):
        if i == 0:
            out.append(highs[0] - lows[0])
        else:
            out.append(
                max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
            )
    return out


def _wilder_rma(values: List[float], length: int) -> List[float]:
    """Wilder RMA over a list of TR values (no Nones). Returns same-length list."""
    out: List[float] = []
    prev: Optional[float] = None
    for i, x in enumerate(values):
        if i < length - 1:
            out.append(float("nan"))
            continue
        if i == length - 1:
            prev = sum(values[:length]) / length  # SMA seed
        else:
            prev = (prev * (length - 1) + x) / length
        out.append(prev)
    return out


def atr_close_only(candles: List[Candle], length: int) -> List[Optional[float]]:
    """Close-only ATR. None until index >= length (needs length+1 candles)."""
    closes = [c.close for c in candles]
    trs = true_ranges_close_only(closes)
    # bars 1..len: TR[0] is None — RMA consumes TR[1..]
    usable = [t for t in trs if t is not None]
    rma = _wilder_rma(usable, length)
    out: List[Optional[float]] = [None] * len(candles)
    for j, v in enumerate(rma):
        idx = j + 1  # usable[j] corresponds to bar j+1
        if idx >= length and v == v:  # v == v filters NaN
            out[idx] = v
    return out


def atr_true_range(candles: List[Candle], length: int) -> List[Optional[float]]:
    """Standard true-range ATR. None until index >= length (alignment choice,
    see module docstring)."""
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    trs = true_ranges_standard(highs, lows, closes)
    usable = trs[1:]  # seed from bars 1..length for cross-flavor alignment
    rma = _wilder_rma(usable, length)
    out: List[Optional[float]] = [None] * len(candles)
    for j, v in enumerate(rma):
        idx = j + 1
        if idx >= length and v == v:
            out[idx] = v
    return out


def atr_series(candles: List[Candle], length: int, method: str = "close_only") -> List[Optional[float]]:
    """ATR by method: 'close_only' (default) or 'true_range'."""
    if method == "close_only":
        return atr_close_only(candles, length)
    if method == "true_range":
        return atr_true_range(candles, length)
    raise ValueError(f"unknown ATR method: {method!r}")
