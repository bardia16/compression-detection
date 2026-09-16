"""Compression structure math: boundary fitting, slope classification,
convergence measurement, ATR-normalized tolerance metrics.

Design rules (spec §13-16):
- boundaries are derived from the relevant confirmed swing pivots per side
- slope classes are measured in ATR units (never absolute prices)
- convergence needs multiple observations (>=3 width samples), not 2 points
- all tolerances ATR-normalized and explicit (no hardcoded constants)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

FLAT = "flat"
RISING = "rising"
FALLING = "falling"


@dataclass(frozen=True)
class Line:
    """A fitted boundary line: price = slope * (bar - base_bar) + intercept."""

    slope: float
    intercept: float
    base_bar: float

    def at(self, bar: float) -> float:
        return self.slope * (bar - self.base_bar) + self.intercept


def fit_line(points: Sequence[Tuple[float, float]]) -> Optional[Line]:
    """OLS fit over (bar, price) points. None when fewer than 2 points or
    all bars identical (degenerate denominator)."""
    if len(points) < 2:
        return None
    xs = [float(b) for b, _ in points]
    ys = [float(p) for _, p in points]
    n = len(xs)
    base = xs[0]
    xb = [x - base for x in xs]
    sx = sum(xb)
    sy = sum(ys)
    sxx = sum(x * x for x in xb)
    sxy = sum(x * y for x, y in zip(xb, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return Line(slope=slope, intercept=intercept, base_bar=base)


def slope_class(
    line: Line, first_bar: float, last_bar: float, atr: float, flat_tol_atr: float
) -> str:
    """Classify a boundary as flat/rising/falling by its total change over
    the structure span, measured in ATR units.

    |delta| <= flat_tol_atr * atr  -> flat   (inclusive: exact tol is flat)
    delta  >  flat_tol_atr * atr   -> rising
    delta  < -flat_tol_atr * atr   -> falling
    """
    if atr <= 0:
        return FLAT
    delta = line.at(last_bar) - line.at(first_bar)
    tol = flat_tol_atr * atr
    if delta > tol:
        return RISING
    if delta < -tol:
        return FALLING
    return FLAT


def fit_error(points: Sequence[Tuple[float, float]], line: Line, atr: float) -> float:
    """Mean absolute deviation of pivots from the line, in ATR units."""
    if not points or atr <= 0:
        return 0.0
    dev = sum(abs(p - line.at(b)) for b, p in points) / len(points)
    return dev / atr


def respects(
    points: Sequence[Tuple[float, float]], line: Line, atr: float, boundary_tol_atr: float
) -> List[bool]:
    """Per-pivot boundary respect flags (inclusive tolerance)."""
    if atr <= 0:
        return [False] * len(points)
    tol = boundary_tol_atr * atr
    return [abs(p - line.at(b)) <= tol for b, p in points]


def convergence_samples(
    upper: Line, lower: Line, bars: Sequence[float]
) -> List[float]:
    """Vertical distance between the boundaries sampled at the given bars."""
    return [upper.at(b) - lower.at(b) for b in bars]


def evaluate_convergence(
    widths: Sequence[float], min_shrink_frac: float, bounce_frac: float
) -> Tuple[bool, float, float]:
    """Multi-observation convergence check (spec §16).

    Returns (converging, shrink_fraction, max_bounce_fraction).

    Requirements:
    - at least 3 width samples (never define convergence from 2 points)
    - all widths positive (boundaries must not cross)
    - overall shrink (first -> last) >= min_shrink_frac
    - consecutive samples never WIDEN by more than bounce_frac (noise slack)
    """
    if len(widths) < 3:
        return False, 0.0, 0.0
    w0 = widths[0]
    if w0 <= 0 or any(w <= 0 for w in widths):
        return False, 0.0, 0.0
    shrink = (w0 - widths[-1]) / w0
    max_bounce = 0.0
    ribbon_ok = True
    for a, b in zip(widths, widths[1:]):
        bounce = (b - a) / a
        if bounce > max_bounce:
            max_bounce = bounce
        if bounce > bounce_frac:
            ribbon_ok = False
    converging = ribbon_ok and shrink >= min_shrink_frac
    return converging, shrink, max_bounce
