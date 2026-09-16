"""Tests for compression structure math (fit, slope class, convergence)."""
import pytest

from compression_detection.structure import (
    FALLING,
    FLAT,
    RISING,
    Line,
    convergence_samples,
    evaluate_convergence,
    fit_error,
    fit_line,
    respects,
    slope_class,
)


# ── fit_line ───────────────────────────────────────────────────────────

def test_fit_line_exact_for_collinear_points():
    line = fit_line([(10.0, 100.0), (11.0, 102.0), (12.0, 104.0)])
    assert line is not None
    assert line.slope == pytest.approx(2.0)
    assert line.at(10.0) == pytest.approx(100.0)
    assert line.at(12.0) == pytest.approx(104.0)


def test_fit_line_flat_for_equal_prices():
    line = fit_line([(5.0, 50.0), (9.0, 50.0)])
    assert line is not None
    assert line.slope == pytest.approx(0.0)
    assert line.at(7.0) == pytest.approx(50.0)


def test_fit_line_none_with_less_than_two_points():
    assert fit_line([(1.0, 2.0)]) is None
    assert fit_line([]) is None


def test_fit_line_none_for_degenerate_bars():
    assert fit_line([(3.0, 10.0), (3.0, 12.0)]) is None


def test_fit_line_least_squares_midpoint():
    # noisy but symmetric around a rising line
    line = fit_line([(0.0, 10.0), (1.0, 12.5), (2.0, 14.0), (3.0, 16.5)])
    assert line is not None
    assert line.slope == pytest.approx(2.1)


# ── slope_class ────────────────────────────────────────────────────────

def _line(slope, intercept=0.0, base=0.0):
    return Line(slope=slope, intercept=intercept, base_bar=base)


def test_slope_class_flat_within_tolerance_inclusive():
    # delta over span = 1.0 * atr -> exactly at tolerance -> flat
    line = _line(0.1)  # 10 bars * 0.1 = 1.0 delta
    assert slope_class(line, 0.0, 10.0, atr=1.0, flat_tol_atr=1.0) == FLAT


def test_slope_class_rising_beyond_tolerance():
    line = _line(0.15)  # delta 1.5 > 1.0
    assert slope_class(line, 0.0, 10.0, atr=1.0, flat_tol_atr=1.0) == RISING


def test_slope_class_falling_beyond_tolerance():
    line = _line(-0.15)
    assert slope_class(line, 0.0, 10.0, atr=1.0, flat_tol_atr=1.0) == FALLING


def test_slope_class_flat_when_atr_zero():
    assert slope_class(_line(5.0), 0.0, 10.0, atr=0.0, flat_tol_atr=1.0) == FLAT


# ── fit_error / respects ───────────────────────────────────────────────

def test_fit_error_mean_deviation_in_atr_units():
    line = _line(0.0, intercept=100.0)
    err = fit_error([(0.0, 101.0), (1.0, 99.0)], line, atr=2.0)
    assert err == pytest.approx(0.5)  # mean dev 1.0 / atr 2.0


def test_respects_inclusive_tolerance():
    line = _line(0.0, intercept=100.0)
    flags = respects([(0.0, 102.0), (1.0, 102.1)], line, atr=2.0, boundary_tol_atr=1.0)
    assert flags == [True, False]


def test_respects_all_false_when_atr_zero():
    line = _line(0.0, intercept=100.0)
    assert respects([(0.0, 100.0)], line, atr=0.0, boundary_tol_atr=1.0) == [False]


# ── convergence ────────────────────────────────────────────────────────

def test_convergence_samples_distance_between_lines():
    up, lo = _line(0.0, intercept=110.0), _line(0.0, intercept=100.0)
    assert convergence_samples(up, lo, [0.0, 1.0, 2.0]) == [10.0, 10.0, 10.0]


def test_converging_triangle_passes():
    widths = [10.0, 8.0, 6.0, 4.0]
    ok, shrink, bounce = evaluate_convergence(widths, 0.25, 0.10)
    assert ok is True
    assert shrink == pytest.approx(0.6)
    assert bounce == pytest.approx(0.0)


def test_parallel_channel_not_converging():
    ok, shrink, _ = evaluate_convergence([10.0, 10.0, 10.0, 10.0], 0.25, 0.10)
    assert ok is False
    assert shrink == pytest.approx(0.0)


def test_diverging_not_converging():
    ok, shrink, _ = evaluate_convergence([5.0, 6.0, 7.0, 8.0], 0.25, 0.10)
    assert ok is False
    assert shrink < 0


def test_bounce_beyond_slack_fails():
    widths = [10.0, 6.0, 8.0, 5.0]  # step 2 widens 33% > 10% slack
    ok, shrink, bounce = evaluate_convergence(widths, 0.25, 0.10)
    assert ok is False
    assert bounce == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_bounce_within_slack_passes():
    widths = [10.0, 9.0, 9.3, 6.0]  # 3.3% widen <= 10%
    ok, shrink, _ = evaluate_convergence(widths, 0.25, 0.10)
    assert ok is True
    assert shrink == pytest.approx(0.4)


def test_two_samples_never_converge():
    ok, _, _ = evaluate_convergence([10.0, 5.0], 0.25, 0.10)
    assert ok is False


def test_crossing_boundaries_fail():
    ok, _, _ = evaluate_convergence([10.0, 5.0, -1.0], 0.25, 0.10)
    assert ok is False


def test_insufficient_shrink_fails():
    widths = [10.0, 9.0, 8.5, 8.0]  # 20% shrink < 25%
    ok, shrink, _ = evaluate_convergence(widths, 0.25, 0.10)
    assert ok is False
    assert shrink == pytest.approx(0.2)
