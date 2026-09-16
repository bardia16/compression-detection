"""ATR-based ZigZag — CLOSE for reversal detection, TRUE-RANGE for the
threshold ATR (option A, 2026-09-13).

Reference: TradingView 'Zig Zag (ATR-based, Close) (2, 7)' — the user's
eyeball reference. Key behaviors:
- threshold = coef * ATR(atr_length), evaluated bar by bar; the ATR is
  Wilder RMA of TRUE RANGE (max(high-low, |high-prevC|, |low-prevC|)) —
  matches the TV indicator. (Before 2026-09-13 it was close-to-close
  only; that deflated during tight ranges and produced much finer
  pivots than the reference.)
- ONLY the bar's close is used for reversal detection and pivot extremes
- pivots strictly alternate high/low
- pivot price = the close at the extreme bar (not the wick)
- ATR is the Wilder RMA of close-to-close changes (see atr.py) — same
  family as Pine's ta.atr but computed on closes only, consistent with
  the close-only design. No ATR before bar `atr_length` (needs
  atr_length + 1 closes) — same as Pine, where the na threshold blocks
  any transition during warmup.

The Pine `cumVol` accumulation is display-only in the reference and is
not needed for structure classification — omitted.
"""
from __future__ import annotations

from typing import List, Optional

from .models import Candle, Pivot

_DIR_NONE = 0
_DIR_UP = 1
_DIR_DOWN = -1


class _RunningTrueRangeAtr:
    """Incremental Wilder RMA of TRUE RANGE:
    TR = max(high-low, |high-prevClose|, |low-prevClose|), seeded with the
    SMA of the first `length` TRs. First non-None value at bar length-1
    (same seed alignment as compression_detection.atr._wilder_rma).

    2026-09-13, user decision: option A — the zigzag's threshold ATR
    switches from close-only to true-range so pivots match the
    TradingView reference indicator 'Zig Zag (ATR-based, Close) (2, 7)'
    (close PRICE is still used for the reversal check; only the
    threshold unit uses the wick range).
    """

    def __init__(self, length: int):
        self.length = length
        self.prev_close: Optional[float] = None
        self._seed_sum = 0.0
        self._tr_count = 0
        self.value: Optional[float] = None

    def update(self, close: float, high: float = 0.0,
               low: float = 0.0) -> Optional[float]:
        if self.prev_close is None:
            # bar 0: no previous close — TR = high - low (like TV's ATR)
            if high is not None and low is not None and high > low:
                tr = high - low
                self._tr_count += 1
                self._seed_sum += tr
                if self._tr_count == self.length:
                    self.value = self._seed_sum / self.length
            self.prev_close = close
            return self.value
        tr = max(high - low if high and low else 0.0,
                 abs(high - self.prev_close) if high else 0.0,
                 abs(low - self.prev_close) if low else 0.0)
        self._tr_count += 1
        if self._tr_count <= self.length:
            self._seed_sum += tr
            if self._tr_count == self.length:
                self.value = self._seed_sum / self.length
        else:
            # _tr_count == length seeds self.value in the branch above,
            # so by the time we're in the else, value is never None
            assert self.value is not None  # noqa: S101
            self.value = (self.value * (self.length - 1) + tr) / self.length
        self.prev_close = close
        return self.value


class ZigZag:
    """Stateful true-range ATR ZigZag (option A, 2026-09-13).

    Feed candles in order (feed() per bar or feed_all() for a list).
    Confirmed pivots accumulate in `pivots`. The live/unconfirmed extreme
    is exposed as `provisional` for display only — never used for state
    transitions.
    """

    def __init__(self, coef: float = 2.0, atr_length: int = 7,
             shadow_reversal: bool = False):
        if atr_length < 1:
            raise ValueError("atr_length must be >= 1")
        # shadow_reversal (2026-09-13, SR-T only): reversal detected when a
        # SHADOW (low in an up-leg, high in a down-leg) pokes past the
        # running extreme by the threshold — not the close. Pivot price
        # is still the close-based extreme. Default False = classic
        # close-only rule used by state/leg classification.
        self.shadow_reversal = shadow_reversal
        self.coef = coef
        self.atr_length = atr_length

        self.last_pivot_price: Optional[float] = None
        self.last_pivot_bar: Optional[int] = None
        self.dir: int = _DIR_NONE
        self.extreme_price: Optional[float] = None
        self.extreme_bar: Optional[int] = None

        self.pivots: List[Pivot] = []
        self.provisional: Optional[Pivot] = None

        self._atr = _RunningTrueRangeAtr(atr_length)
        self._ts_list: List[int] = []

    # ── public API ──────────────────────────────────────────────────────
    def feed_all(self, candles: List[Candle]) -> List[Pivot]:
        """Process a full candle list; returns confirmed pivots."""
        for candle in candles:
            self.feed(candle)
        return self.pivots

    def feed(self, candle: Candle) -> Optional[Pivot]:
        """Process one candle (streaming). Returns a pivot if one confirmed
        on this bar."""
        bar = len(self._ts_list)
        self._ts_list.append(candle.ts)
        atr = self._atr.update(candle.close, candle.high, candle.low)
        threshold = None if atr is None else self.coef * atr
        return self._feed_bar(bar, candle, threshold)

    # ── internals ───────────────────────────────────────────────────────
    def _feed_bar(self, bar: int, candle: Candle, threshold: Optional[float]) -> Optional[Pivot]:
        close = candle.close

        # Bar 0 init (Pine: bar_index == 0)
        if bar == 0:
            self.last_pivot_price = close
            self.last_pivot_bar = 0
            self.extreme_price = close
            self.extreme_bar = 0
            self.provisional = Pivot(type=1, price=close, bar_index=0, ts=candle.ts)
            return None

        if threshold is None:
            # ATR warmup: Pine's na threshold makes every comparison false.
            self._update_provisional(candle)
            return None

        confirmed: Optional[Pivot] = None

        if self.dir == _DIR_NONE:
            if close - self.last_pivot_price > threshold:
                self.dir = _DIR_UP
                self.extreme_price = close
                self.extreme_bar = bar
            elif self.last_pivot_price - close > threshold:
                self.dir = _DIR_DOWN
                self.extreme_price = close
                self.extreme_bar = bar
        elif self.dir == _DIR_UP:
            if close > self.extreme_price:
                self.extreme_price = close
                self.extreme_bar = bar
            # SR-T shadow rule: the bar's LOW may confirm the high pivot
            probe = candle.low if self.shadow_reversal else close
            if self.extreme_price - probe > threshold:
                confirmed = Pivot(
                    type=1, price=self.extreme_price,
                    bar_index=self.extreme_bar, ts=self._ts_of(self.extreme_bar),
                )
                self.pivots.append(confirmed)
                self.last_pivot_price = self.extreme_price
                self.last_pivot_bar = self.extreme_bar
                self.dir = _DIR_DOWN
                self.extreme_price = close
                self.extreme_bar = bar
        else:  # dir == _DIR_DOWN
            if close < self.extreme_price:
                self.extreme_price = close
                self.extreme_bar = bar
            # SR-T shadow rule: the bar's HIGH (shadow) may confirm the low
            # pivot; classic rule uses the close only.
            probe = candle.high if self.shadow_reversal else close
            if probe - self.extreme_price > threshold:
                confirmed = Pivot(
                    type=-1, price=self.extreme_price,
                    bar_index=self.extreme_bar, ts=self._ts_of(self.extreme_bar),
                )
                self.pivots.append(confirmed)
                self.last_pivot_price = self.extreme_price
                self.last_pivot_bar = self.extreme_bar
                self.dir = _DIR_UP
                self.extreme_price = close
                self.extreme_bar = bar

        self._update_provisional(candle)
        return confirmed

    def _ts_of(self, bar: int) -> int:
        return self._ts_list[bar]

    def _update_provisional(self, candle: Candle) -> None:
        """Display-only live extreme. Never drives state."""
        if self.extreme_price is None:
            return
        self.provisional = Pivot(
            type=1 if self.dir >= 0 else -1,
            price=self.extreme_price,
            bar_index=self.extreme_bar,
            ts=self._ts_of(self.extreme_bar),
        )
