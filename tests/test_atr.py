"""Tests for the ATR module (close-only and true-range flavors)."""
import math

from compression_detection.atr import (
    atr_close_only,
    atr_series,
    atr_true_range,
    true_ranges_close_only,
    true_ranges_standard,
)
from compression_detection.models import Candle


def mk_candles(closes, highs=None, lows=None):
    highs = highs or closes
    lows = lows or closes
    return [
        Candle(ts=i * 60_000, open=c, high=h, low=l, close=c, volume=1.0)
        for i, (c, h, l) in enumerate(zip(closes, highs, lows))
    ]


class TestTrueRanges:
    def test_close_only_first_is_none(self):
        trs = true_ranges_close_only([100.0, 102.0, 99.0])
        assert trs[0] is None
        assert trs[1] == 2.0
        assert trs[2] == 3.0

    def test_standard_uses_high_low_and_gaps(self):
        # bar1: H-L = 4; |H - prevC| = 8; |L - prevC| = 2 -> TR = 8
        trs = true_ranges_standard(
            highs=[100, 108], lows=[96, 104], closes=[100, 106]
        )
        assert trs[0] == 4.0
        assert trs[1] == 8.0

    def test_standard_no_gap_is_high_minus_low(self):
        trs = true_ranges_standard(highs=[100, 101], lows=[90, 95], closes=[100, 99])
        # TR1 = max(6, |101-100|=1, |95-100|=5) = 6
        assert trs[1] == 6.0


class TestAtrCloseOnly:
    def test_warmup_returns_none_until_length(self):
        candles = mk_candles([100, 101, 102, 103, 105, 104, 106, 108])
        series = atr_close_only(candles, length=3)
        # TRs exist at bars 1..; RMA seed = SMA(TR[1..3]) -> first value at bar 3
        assert series[0] is None
        assert series[1] is None
        assert series[2] is None
        assert series[3] is not None  # first value at index == length
        assert all(v is not None for v in series[3:])

    def test_constant_closes_give_zero_atr(self):
        candles = mk_candles([100.0] * 30)
        series = atr_close_only(candles, length=7)
        assert series[-1] == 0.0

    def test_hand_computed_value(self):
        # closes: TRs = 1,1,1,1,1,... constant -> RMA = 1
        candles = mk_candles([100, 101, 102, 103, 104, 105, 106, 107])
        series = atr_close_only(candles, length=4)
        assert series[4] == 1.0
        assert series[5] == 1.0

    def test_wilder_smoothing_weighting(self):
        # TRs: 1,1,1,1 then a big 10 -> RMA should sit between 1 and 10,
        # closer to 1 with length=4: seed=1, rma=(1*3+10)/4=3.25
        closes = [100, 101, 102, 103, 104, 114]
        candles = mk_candles(closes)
        series = atr_close_only(candles, length=4)
        assert math.isclose(series[5], 3.25)

    def test_needs_length_plus_one_candles(self):
        candles = mk_candles([100, 101])
        series = atr_close_only(candles, length=7)
        assert all(v is None for v in series)


class TestAtrTrueRange:
    def test_warmup_alignment_matches_close_only(self):
        closes = [100, 102, 99, 103, 105, 104, 106, 108, 107, 110]
        highs = [c + 1.5 for c in closes]
        lows = [c - 1.5 for c in closes]
        co = atr_close_only(mk_candles(closes), length=5)
        tr = atr_true_range(mk_candles(closes, highs, lows), length=5)
        co_first = next(i for i, v in enumerate(co) if v is not None)
        tr_first = next(i for i, v in enumerate(tr) if v is not None)
        assert co_first == tr_first  # aligned warmup by design

    def test_hand_computed(self):
        # bars 1..5 TRs (no gaps): H-L = 3,2,2,2,2 ; seed SMA(2,2,2,2,3)... use len=5
        closes = [100, 101, 102, 103, 104, 105]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        candles = mk_candles(closes, highs, lows)
        series = atr_true_range(candles, length=5)
        # TRs bars1..5 = all 2.0 -> seed = 2.0, RMA stays 2.0
        assert math.isclose(series[5], 2.0)


class TestAtrSeriesDispatch:
    def test_dispatch(self):
        candles = mk_candles([100, 101, 102, 103, 104, 105, 106, 107, 108])
        co = atr_series(candles, 4, method="close_only")
        tr = atr_series(candles, 4, method="true_range")
        assert co[4] is not None and tr[4] is not None

    def test_unknown_method_raises(self):
        candles = mk_candles([100, 101])
        try:
            atr_series(candles, 4, method="bogus")
            assert False, "should have raised"
        except ValueError:
            pass
