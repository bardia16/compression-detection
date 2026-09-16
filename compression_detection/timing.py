"""Clock helpers: 15m-grid scan scheduling + forming-candle probe timing.

All Binance candle boundaries for 15m/1h/4h are epoch-aligned (UTC), so a
single 15m grid drives both the scan cadence and every probe window:

    probe windows open at  boundary − 3min   (e.g. :12, :27, :42, :57)
    scans run at          boundary + offset  (e.g. :16:30 → default +90s)

No websocket needed for timing — the grid is derived from the clock.
"""
from __future__ import annotations

from typing import Optional

GRID_MS = 15 * 60 * 1000


def next_scan_ms(now_ms: int, offset_s: int, grid_ms: int = GRID_MS) -> int:
    """Next scan time (ms): the next grid boundary shifted by `offset_s`.

    A boundary already passed (or exactly now) rolls to the next grid slot.
    """
    off = (offset_s * 1000) % grid_ms
    base = now_ms - (now_ms % grid_ms) + off
    if base <= now_ms:
        base += grid_ms
    return base


def forming_candle_open_ms(now_ms: int, tf_ms: int) -> int:
    """Open time (ms) of the currently-forming candle for `tf_ms`."""
    return (now_ms // tf_ms) * tf_ms


def candle_close_ms(open_ms: int, tf_ms: int) -> int:
    """Close time (ms) of the candle that opened at `open_ms`."""
    return open_ms + tf_ms


def probe_due(
    probe_last_ms: int, now_ms: int, tf_ms: int, warn_s: int
) -> Optional[int]:
    """Forming-candle open (ms) if a probe should fire now, else None.

    Fires once per candle per instance: within `warn_s` seconds of the
    candle close, and only if this candle hasn't been probed yet.
    """
    open_ms = forming_candle_open_ms(now_ms, tf_ms)
    if open_ms == probe_last_ms:
        return None
    close_ms = candle_close_ms(open_ms, tf_ms)
    if 0 <= close_ms - now_ms <= warn_s * 1000:
        return open_ms
    return None
