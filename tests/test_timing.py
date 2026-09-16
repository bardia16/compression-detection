"""Timing helper tests — grid math with absolute epoch values pinned."""
from compression_detection.timing import (
    GRID_MS,
    candle_close_ms,
    forming_candle_open_ms,
    next_scan_ms,
    probe_due,
)

# 2026-09-16 08:00:00 UTC
BASE = 1789545600000 - 1789545600000  # placeholder replaced below


def _ts(h, m, s=0):
    """UTC ms for 2026-09-16 hh:mm:ss (epoch anchor via brute-force check)."""
    import datetime as _dt
    d = _dt.datetime(2026, 9, 16, h, m, s, tzinfo=_dt.timezone.utc)
    return int(d.timestamp() * 1000)


def test_next_scan_mid_candle():
    # at 08:05 -> next scan is 08:01:30 of the NEXT boundary? No: 08:16:30
    now = _ts(8, 5)
    assert next_scan_ms(now, 90) == _ts(8, 16, 30)


def test_next_scan_before_offset_same_grid():
    # at 08:00:05 the 08:01:30 slot is still ahead
    now = _ts(8, 0, 5)
    assert next_scan_ms(now, 90) == _ts(8, 1, 30)


def test_next_scan_exactly_at_offset_rolls_forward():
    now = _ts(8, 1, 30)
    assert next_scan_ms(now, 90) == _ts(8, 16, 30)


def test_next_scan_just_after_offset_rolls_forward():
    now = _ts(8, 1, 31)
    assert next_scan_ms(now, 90) == _ts(8, 16, 30)


def test_forming_candle_open():
    assert forming_candle_open_ms(_ts(8, 7), 15 * 60 * 1000) == _ts(8, 0)
    assert forming_candle_open_ms(_ts(8, 7), 60 * 60 * 1000) == _ts(8, 0)
    assert forming_candle_open_ms(_ts(9, 59), 4 * 60 * 60 * 1000) == _ts(8, 0)
    assert forming_candle_open_ms(_ts(12, 0), 4 * 60 * 60 * 1000) == _ts(12, 0)


def test_candle_close():
    assert candle_close_ms(_ts(8, 0), 15 * 60 * 1000) == _ts(8, 15)
    assert candle_close_ms(_ts(8, 0), 4 * 60 * 60 * 1000) == _ts(12, 0)


def test_probe_due_within_window():
    # 15m candle closing 08:15; at 08:12 exactly 180s left -> due
    assert probe_due(0, _ts(8, 12), 15 * 60 * 1000, 180) == _ts(8, 0)


def test_probe_not_due_too_early():
    # at 08:11:59 -> 181s left -> not due
    assert probe_due(0, _ts(8, 11, 59), 15 * 60 * 1000, 180) is None


def test_probe_once_per_candle():
    open_ms = _ts(8, 0)
    assert probe_due(open_ms, _ts(8, 13), 15 * 60 * 1000, 180) is None


def test_probe_not_due_after_close():
    # at 08:16 the 08:00 candle already closed; next candle 14min out
    assert probe_due(0, _ts(8, 16), 15 * 60 * 1000, 180) is None


def test_probe_due_for_4h_candle():
    # 4h candle closing 12:00; at 11:57 -> due, open 08:00
    assert probe_due(0, _ts(11, 57), 4 * 60 * 60 * 1000, 180) == _ts(8, 0)
