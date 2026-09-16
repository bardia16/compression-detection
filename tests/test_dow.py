"""Tests for Dow pivot labelling (HH/LH/EH, HL/LL/EL)."""
import pytest

from compression_detection.dow import (
    EL,
    EH,
    HH,
    HL,
    LH,
    LL,
    PivotLabel,
    label_high,
    label_low,
    label_pivot,
    label_pivots,
)
from compression_detection.models import Pivot


def hp(price, bar=0):
    return Pivot(type=1, price=price, bar_index=bar, ts=bar * 60_000)


def lp(price, bar=0):
    return Pivot(type=-1, price=price, bar_index=bar, ts=bar * 60_000)


class TestLabelHigh:
    def test_hh_clears_by_more_than_atr(self):
        assert label_high(110.0, 100.0, 5.0) is PivotLabel.HH

    def test_lh_below_by_more_than_atr(self):
        assert label_high(94.0, 100.0, 5.0) is PivotLabel.LH

    def test_eh_exactly_atr_above(self):
        # exactly 1*ATR14 above is NOT a HH (must clear by MORE than)
        assert label_high(105.0, 100.0, 5.0) is PivotLabel.EH

    def test_eh_exactly_atr_below(self):
        assert label_high(95.0, 100.0, 5.0) is PivotLabel.EH

    def test_eh_inside_band(self):
        assert label_high(102.0, 100.0, 5.0) is PivotLabel.EH

    def test_hh_epsilon_just_over(self):
        assert label_high(105.0001, 100.0, 5.0) is PivotLabel.HH

    def test_lh_epsilon_just_under(self):
        assert label_high(94.9999, 100.0, 5.0) is PivotLabel.LH


class TestLabelLow:
    def test_ll_below_by_more_than_atr(self):
        assert label_low(90.0, 100.0, 5.0) is PivotLabel.LL

    def test_hl_above_by_more_than_atr(self):
        assert label_low(106.0, 100.0, 5.0) is PivotLabel.HL

    def test_el_exactly_atr_below(self):
        assert label_low(95.0, 100.0, 5.0) is PivotLabel.EL

    def test_el_inside_band(self):
        assert label_low(99.0, 100.0, 5.0) is PivotLabel.EL

    def test_ll_epsilon_just_over(self):
        assert label_low(94.9999, 100.0, 5.0) is PivotLabel.LL


class TestLabelPivot:
    def test_no_prev_returns_none(self):
        assert label_pivot(hp(100.0), None, 5.0) is None

    def test_zero_atr_returns_none(self):
        assert label_pivot(hp(100.0), hp(100.0), 0.0) is None

    def test_none_atr_returns_none(self):
        assert label_pivot(hp(100.0), hp(100.0), None) is None

    def test_high_vs_high(self):
        assert label_pivot(hp(110.0), hp(100.0), 5.0) is PivotLabel.HH

    def test_low_vs_low(self):
        assert label_pivot(lp(106.0), lp(100.0), 5.0) is PivotLabel.HL

    def test_high_ignores_lows(self):
        # prev same-type only: a high is compared to previous HIGH, not low
        assert label_pivot(hp(102.0), hp(100.0), 5.0) is PivotLabel.EH


class TestLabelPivots:
    def test_first_of_each_type_skipped(self):
        pivots = [lp(100.0, 0), hp(110.0, 1), lp(108.0, 2), hp(114.0, 3)]
        atr = [2.0] * 10
        labeled = label_pivots(pivots, atr)
        # first high and first low can't be labeled -> 2 labeled pivots
        assert len(labeled) == 2
        assert labeled[0].label is PivotLabel.HL  # low 108 vs low 100
        assert labeled[1].label is PivotLabel.HH  # high 114 vs high 110 (delta 4 > 2)

    def test_uses_atr_at_pivot_bar(self):
        pivots = [lp(100.0, 0), lp(90.0, 5)]
        atr = [None] * 5 + [3.0] + [2.0] * 5
        labeled = label_pivots(pivots, atr)
        # ATR at bar 5 is 3.0: 90 vs 100 is -10 > -3 -> LL
        assert labeled[0].label is PivotLabel.LL

    def test_atr_missing_at_pivot_bar_skips(self):
        pivots = [lp(100.0, 0), lp(90.0, 1)]
        atr = [None, None, None]
        assert label_pivots(pivots, atr) == []
