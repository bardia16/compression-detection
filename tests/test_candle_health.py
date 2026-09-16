"""Candle-problem gate tests — ported from daw-theory's gate suite,
adapted to Candle objects."""
from compression_detection.candle_health import (
    CANDLE_PROBLEM_MAX_COUNT,
    atr7_wilder,
    has_candle_problems,
)
from compression_detection.models import Candle


def _mk_candles(n, start, step, gap_at=None, gap_size=10.0):
    """Smooth uptrend candles; optional single index where open gaps up."""
    candles = []
    o = start
    for i in range(n):
        if gap_at is not None and i == gap_at:
            o = candles[-1].close + gap_size  # open far above prev close
        c = o + step
        h = max(o, c) + 0.1
        l = min(o, c) - 0.1
        candles.append(Candle(ts=i * 1000, open=o, high=h, low=l, close=c,
                              volume=1.0))
        o = c
    return candles


def _with_gap(candles, i):
    """Replace candle i's open so it gaps above previous close by 1.0."""
    c = candles[i]
    candles[i] = Candle(ts=c.ts, open=candles[i - 1].close + 1.0,
                        high=c.high, low=c.low, close=c.close, volume=c.volume)


def test_clean_series_no_problems():
    candles = _mk_candles(40, 100.0, 0.05)  # tiny steps, no gaps
    assert has_candle_problems(candles) is False


def test_more_than_four_gaps_detected():
    candles = _mk_candles(40, 100.0, 0.01)
    for i in range(6):
        _with_gap(candles, i * 6 + 5)
    assert has_candle_problems(candles) is True


def test_four_gaps_ok():
    candles = _mk_candles(40, 100.0, 0.01)
    for i in range(4):
        _with_gap(candles, i * 8 + 5)
    assert has_candle_problems(candles) is False


def test_short_series_safe():
    assert has_candle_problems(_mk_candles(2, 100.0, 0.01)) is False
    assert has_candle_problems([]) is False


def test_atr7_wilder_sane():
    candles = _mk_candles(40, 100.0, 0.5)
    atr = atr7_wilder(candles)
    assert 0 < atr < 2.0  # ranges ≈0.6-1.1 → ATR in that ballpark


def test_max_count_constant_matches_daw():
    assert CANDLE_PROBLEM_MAX_COUNT == 4
