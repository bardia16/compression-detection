"""Potential-break detection (user rule 2026-09-28 — trigger rewrite).

A pattern needs a BOX first. From there the alert is a per-tf TRIGGER:

  trigger tfs   = the box's LOWER tfs, nearest first, PLUS the box's own
                  tf (engine scan list: 1h -> [15m, 1h], 4h -> [1h, 15m,
                  4h], 1d -> [4h, 1h, 15m, 1d]).

  close zone    = price within PROX_BAND_ATR (1.5) x ATR14 of the boundary,
                  evaluated PER TF with that tf's own ATR ("their 1.5 atr
                  range") — and never past the line (a close beyond it
                  belongs to the breakout verdict).

  trigger       = on that tf, the LAST TWO pivots are
                    LONG : HIGH labeled EH at the boundary, then LOW  HL
                    SHORT: LOW  labeled EL at the boundary, then HIGH LH
                  the swing needs NO confirmation — the LIVE provisional
                  counts as long as it carries the HL/LH label (a
                  confirmed swing is fine too). Adjacency is positional:
                  nothing may sit between the two pivots.

Any trigger tf passing sends ONE alert (freshest swing wins) — the caption
states the box tf and the tf the break was found on ("4H box · break at
1H").

Kept guards: box lines required (fail closed), one alert per touch (the
EH/EL armed via since_touch_ts), anchor_ts, POTENTIAL_FRESH_S freshness
on the swing.

`touch_atr` stays in the signature for compatibility; the band is the
per-tf PROX_BAND_ATR x ATR14 now. `live_price` lets the fast proximity
pass (60s poll) override the last closed close.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .atr import atr_series
from .dow import label_all_pivots
from .models import Candle
from .zigzag import ZigZag

# box TF -> the LOWER tfs scanned for touches / confirming swings,
# nearest first (user rule 2026-09-28):
#   1h box -> 15m only
#   4h box -> 1h and 15m
#   1d box -> 4h, 1h and 15m
# 15m has no key: it is never a pattern tf anymore.
LOWER_TF: Dict[str, List[str]] = {
    "1h": ["15m"],
    "4h": ["1h", "15m"],
    "1d": ["4h", "1h", "15m"],
}

# Close-zone / boundary-touch band per trigger tf (user rule
# 2026-09-28): price AND the EH/EL touch sit within 1.5 x ATR14 of
# that tf's boundary line.
PROX_BAND_ATR = 1.5

# Freshness window (user choice 2026-09-27): a swing older than this
# is old news — the pass would otherwise backfill patterns that completed
# days before the box was even detected (PIEVERSE/BANK case: 3 of the 11
# first-scan hits were 4-5 days stale).
POTENTIAL_FRESH_S = 12 * 3600


def _zigzag(candles: List[Candle], cfg):
    return ZigZag(coef=cfg.zigzag_coef,
                  atr_length=cfg.zigzag_atr_length).feed_all(candles)


def _provisional(candles: List[Candle], cfg):
    """The UNCONFIRMED zigzag extreme (live swing) of a candle series."""
    zz = ZigZag(coef=cfg.zigzag_coef, atr_length=cfg.zigzag_atr_length)
    zz.feed_all(candles)
    return zz.provisional


def find_potential(side: str,
                   main_candles: List[Candle],
                   scan: Sequence[Tuple[str, List[Candle]]],
                   cfg,
                   *,
                   main_tf: str,
                   anchor_ts: Optional[int] = None,
                   box_at=None,
                   touch_atr: Optional[float] = None,
                   since_touch_ts: int = 0,
                   live_price: Optional[float] = None) -> Optional[dict]:
    """New trigger logic (user rule 2026-09-28, see module docstring).

    Checks every trigger tf (lower tfs nearest-first + the box's own tf)
    independently and returns the FRESHEST passing hit, or None.
    touch_atr is accepted but unused — the band is per-tf
    PROX_BAND_ATR x ATR14 now.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be long|short, got {side!r}")
    # fail closed on BOTH contract params — no lines, no pattern (the
    # touch_atr value itself is unused since the per-tf band took over,
    # but a caller missing it is not running the contract)
    if box_at is None or touch_atr is None:
        return None
    if not main_candles:
        return None

    want = "EH" if side == "long" else "EL"      # the touch at the boundary
    need = "HL" if side == "long" else "LH"      # the swing after it

    def lines(ts):
        try:
            up, lo = box_at(ts)
        except Exception:
            return None, None
        return (up, lo) if up is not None and lo is not None else (None, None)

    # trigger tfs: engine scan order (lower nearest-first, own tf last);
    # own tf always participates even if a caller forgot it (tests).
    srcs: List[Tuple[str, List[Candle]]] = []
    seen = set()
    for tf, candles in scan:
        if (tf and candles and tf not in seen
                and len(candles) >= cfg.atr14_length + 6):
            srcs.append((tf, candles))
            seen.add(tf)
    if main_tf not in seen and len(main_candles) >= cfg.atr14_length + 6:
        srcs.append((main_tf, main_candles))

    hits: List[dict] = []
    for tf, candles in srcs:
        a14 = atr_series(candles, cfg.atr14_length, cfg.atr14_method)
        if not a14 or not a14[-1]:
            continue
        band = PROX_BAND_ATR * a14[-1]       # this tf's own close zone

        pivots = _zigzag(candles, cfg)
        confirmed = list(label_all_pivots(pivots, a14))
        # TWO candidate tails: the confirmed last-two (a live bounce-high
        # printed after the swing must not hide a valid EH+HL pair) and
        # the confirmed+provisional tail (the SWING itself still live).
        tails: List[tuple] = []
        if len(confirmed) >= 2:
            tails.append((confirmed[-2:], None))
        prov = _provisional(candles, cfg)
        if prov is not None and pivots \
                and prov.ts not in {p.ts for p in pivots}:
            full = label_all_pivots(list(pivots) + [prov], a14)
            if full and full[-1][0].ts == prov.ts and full[-1][1] is not None:
                tails.append((full[-2:], prov.ts))

        for tail, live_ts in tails:
            # adjacency: the tail's LAST TWO pivots are the trigger
            (p_a, l_a), (p_b, l_b) = tail
            if side == "long":
                if not (p_a.is_high and not p_b.is_high):
                    continue
            else:
                if not (not p_a.is_high and p_b.is_high):
                    continue
            if l_a is None or l_a.name != want:
                continue
            if l_b is None or l_b.name != need:
                continue

            # the touch sits AT the boundary (same per-tf band)
            up, lo = lines(p_a.ts)
            line = up if side == "long" else lo
            if line is None:
                continue
            if abs(p_a.price - line) > band:
                continue

            # close zone: price in [line - band, line], never past the line
            price = live_price if live_price is not None else candles[-1].close
            if side == "long":
                if price > line or price < line - band:
                    continue
            else:
                if price < line or price > line + band:
                    continue

            # one alert per touch + box anchor
            if p_a.ts <= since_touch_ts:
                continue
            if anchor_ts is not None and p_b.ts < anchor_ts:
                continue

            hits.append({
                "side": side,
                "pattern_tf": tf,
                "touch_ts": p_a.ts,
                "touch_price": p_a.price,
                "touch_label": want,
                "touch_tf": tf,
                "first_ts": p_a.ts,       # kept for chart/caption compat
                "first_price": p_a.price,
                "mid_ts": p_a.ts,         # the touch IS the mid pivot
                "mid_price": p_a.price,
                "mid_label": want,
                "second_ts": p_b.ts,
                "second_price": p_b.price,
                "second_label": need,
                "confirm_ts": p_b.ts,     # swing recency (freshness/order)
                "live": bool(live_ts and p_b.ts == live_ts),
            })

    if not hits:
        return None
    return max(hits, key=lambda h: h["confirm_ts"])   # freshest wins


def potential_lines(hit: dict, box_lo: Optional[float],
                    box_hi: Optional[float]) -> list:
    """Chart lines for the LOW-TF chart (user rule 2026-09-27): the box
    boundary on the non-breaking side, the TOUCH pivot on the side being
    broken."""
    if hit["side"] == "long":
        return [box_lo, hit["mid_price"]]
    return [hit["mid_price"], box_hi]
