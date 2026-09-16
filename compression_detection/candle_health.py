"""Candle-problem gate — ported from daw-breakouts (2026-09-15), itself from
the setup-candle scanner's CANDLE_PROBLEM_FILTER (same constants).

Consecutive-bar gap detection: |prev.close − next.open| ≥ 12% × ATR7 in the
last 30 candles counts as a "problem". More than 4 problems → the feed is
untrustworthy (halt / gap / fresh-listing artifact) → the notification is
suppressed (audited as `candle_problems`).

Applied before EVERY post (user rule 2026-09-16):
- compression notices      (scan candles)
- close-verdict breakouts  (scan candles)
- T−3:00 probe heads-ups   (fresh 31-kline fetch, daw-breakouts parity)
"""
from __future__ import annotations

from typing import List

from .models import Candle

CANDLE_PROBLEM_LOOKBACK = 30
CANDLE_PROBLEM_ATR_MULT = 12          # 12% of ATR7
CANDLE_PROBLEM_MAX_COUNT = 4          # ≤4 fine, 5+ → problem


def atr7_wilder(candles: List[Candle]) -> float:
    """Wilder ATR(7) via TR series over the given candles; 0.0 when
    insufficient data (needs ≥8 candles)."""
    if len(candles) < 8:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        prev_close = candles[i - 1].close
        h, l = candles[i].high, candles[i].low
        trs.append(max(h - l, abs(h - prev_close), abs(l - prev_close)))
    if not trs:
        return 0.0
    atr = sum(trs[:7]) / 7.0
    for t in trs[7:]:
        atr = (atr * 6 + t) / 7.0
    return atr


def has_candle_problems(candles: List[Candle]) -> bool:
    """Gap check over the last `CANDLE_PROBLEM_LOOKBACK` candles (newest
    last). Mirrors daw-breakouts._has_candle_problems exactly:

    - < 3 candles → False (can't judge, don't block)
    - threshold = 12% × ATR7(lookback); when ATR7 is 0 fall back to the
      smallest non-zero gap so any real jump counts
    - ≥ `CANDLE_PROBLEM_MAX_COUNT + 1` gaps → True (feed untrustworthy)
    """
    if len(candles) < 3:
        return False
    lookback = candles[-CANDLE_PROBLEM_LOOKBACK:]
    if len(lookback) < 2:
        return False
    atr7 = atr7_wilder(lookback)
    if atr7 <= 0:
        diffs = [abs(lookback[i].close - lookback[i + 1].open)
                 for i in range(len(lookback) - 1)
                 if abs(lookback[i].close - lookback[i + 1].open) > 0]
        if not diffs:
            return False
        threshold = min(diffs)
    else:
        threshold = atr7 * CANDLE_PROBLEM_ATR_MULT / 100.0
    problems = 0
    for i in range(len(lookback) - 1):
        if abs(lookback[i].close - lookback[i + 1].open) >= threshold:
            problems += 1
            if problems > CANDLE_PROBLEM_MAX_COUNT:
                return True
    return False
