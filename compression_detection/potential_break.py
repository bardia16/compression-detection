"""Potential-break confirmation pattern (user rule 2026-09-27, revised).

Three swings, and EVERY judgment is made against the BOX's boundary lines —
never against the neighbouring pivot on the scanned timeframe (user rule
2026-09-27, XPL case: "this high should be a lower high to the box high!,
not the prev low"):

  LONG : touch the box HIGH with the middle pivot, then the confirmed
         low must sit on/above the box LOW line
  SHORT: touch the box LOW  with the middle pivot, then the confirmed
         high must sit on/below the box HIGH line

"touch" = |pivot − line| ≤ the box's own ATR. The old rule compared the
middle pivot to the previous pivot of the same type (EH/EL label) and the
third swing to the first swing (HL/LH label), which let a 15m mid 7% below
the box top count as "EH" (XPL: 0.11262 vs box 0.121 — a mere middle
pivot, no alert) and blocked a third swing that sat right on the box line.

The third swing only exists once the zigzag CONFIRMS it (close beyond
coef × ATR7), so the alert lands on the confirming candle.

Scanned on the box's own TF (main) and one TF down; `LOWER_TF` maps a
box TF to the timeframe it is checked against. `box_at` is supplied by the
engine (it owns the Instance and its Line objects) — no box lines, no
pattern (fail closed).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .atr import atr_series
from .models import Candle
from .zigzag import ZigZag

# box TF -> the TF scanned below it (None = nothing under 15m)
LOWER_TF: Dict[str, Optional[str]] = {"4h": "1h", "1h": "15m", "15m": None}

# Freshness window (user choice 2026-09-27): a confirmation older than this
# is old news — the scan would otherwise backfill patterns that completed
# days before the box was even detected (PIEVERSE/BANK case: 3 of the 11
# first-scan hits were 4-5 days stale). Generous enough to cover the box's
# own detection lag (observed up to ~3.5h) plus a scan gap; short enough
# that nothing retroactively surfaces.
POTENTIAL_FRESH_S = 12 * 3600

# float slack for the "sits on the box line" side test: a pivot that IS the
# line (the line is fitted through it) differs from line(ts) by ~1e-17 and
# must not be read as "below the box".
LINE_EPS_REL = 1e-9


def _confirm_map(candles: List[Candle], cfg) -> Dict[int, int]:
    """pivot ts -> ts of the candle that CONFIRMED it.

    `ZigZag.feed_all` only yields confirmed pivots and stamps them with the
    pivot bar; replaying `feed` one candle at a time surfaces the confirming
    candle, which is what the alert fires on.
    """
    zz = ZigZag(coef=cfg.zigzag_coef, atr_length=cfg.zigzag_atr_length)
    out: Dict[int, int] = {}
    for c in candles:
        p = zz.feed(c)
        if p is not None:
            out[p.ts] = c.ts
    return out


def find_potential(candles: List[Candle], side: str, cfg,
                   anchor_ts: Optional[int] = None,
                   box_at=None, touch_atr: Optional[float] = None
                   ) -> Optional[dict]:
    """Latest matching pattern on ONE timeframe, or None.

    side      "long" | "short" — direction of the expected break
    anchor_ts the compression's anchor (first pivot ts): the pattern must
              CONFIRM at or after it — a confirmation that predates the box
              is not this box's signal.
    box_at    callable ts(ms) -> (upper, lower): the box's boundary LINES
              evaluated at that time (the engine owns the Instance).
    touch_atr the box's own ATR — tolerance of the "middle pivot touches
              its box side" test.

    Every judgment is made against the BOX lines (see module docstring);
    with no lines there is no pattern (fail closed).
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be long|short, got {side!r}")
    if box_at is None or touch_atr is None:
        return None
    if len(candles) < cfg.atr14_length + cfg.zigzag_atr_length + 6:
        return None

    zz = ZigZag(coef=cfg.zigzag_coef, atr_length=cfg.zigzag_atr_length)
    pivots = zz.feed_all(candles)
    if len(pivots) < 3:
        return None
    confirms = _confirm_map(candles, cfg)

    def box_lines(ts):
        """(upper, lower) at ts, or (None, None) when unavailable."""
        try:
            up, lo = box_at(ts)
        except Exception:
            return None, None
        return (up, lo) if up is not None and lo is not None else (None, None)

    hit: Optional[dict] = None
    for i in range(len(pivots) - 2):
        first, mid, second = pivots[i], pivots[i + 1], pivots[i + 2]
        # alternation only — no neighbour labels are consulted any more
        if side == "long":
            if first.is_high or not mid.is_high or second.is_high:
                continue
        else:
            if not first.is_high or mid.is_high or not second.is_high:
                continue

        up, lo = box_lines(mid.ts)
        if up is None:
            continue
        if side == "long":
            # 1) the middle HIGH touches the box HIGH (± the box's ATR) —
            #    a mid that is merely an equal-high to the previous pivot of
            #    this timeframe is a middle pivot, not the box top (XPL:
            #    15m mid 0.11262 vs box 0.121 -> 7% away -> no touch -> no
            #    alert), while the 1H's own 0.121 sits on the line.
            if abs(mid.price - up) > touch_atr:
                continue
            # 2) the confirmed LOW sits on/above the box LOW line — judged
            #    against the box, not against the previous low (XPL 1H:
            #    0.10976 == the line at that bar -> higher low, whereas the
            #    neighbour label said EL and blocked it).
            _, lo2 = box_lines(second.ts)
            if lo2 is None or second.price + abs(lo2) * LINE_EPS_REL < lo2:
                continue
            mid_label = "EH"
        else:
            # 1) the middle LOW touches the box LOW
            if abs(mid.price - lo) > touch_atr:
                continue
            # 2) the confirmed HIGH sits on/below the box HIGH line
            up2, _ = box_lines(second.ts)
            if up2 is None or second.price - abs(up2) * LINE_EPS_REL > up2:
                continue
            mid_label = "EL"

        conf = confirms.get(second.ts)
        if conf is None:      # unreachable — feed_all only yields confirmed
            continue
        # Anchor test is on the CONFIRMATION, not the first swing (QNT case):
        # the box's anchor sits between the pattern's swings, so gating on
        # first.ts dropped a pattern that both completed and confirmed inside
        # the box's life. The 12h freshness window bounds anything older.
        if anchor_ts is not None and conf < anchor_ts:
            continue
        hit = {
            "side": side,
            "first_ts": first.ts,
            "first_price": first.price,
            "mid_ts": mid.ts,
            "mid_price": mid.price,
            "mid_label": mid_label,
            "second_ts": second.ts,
            "second_price": second.price,
            "confirm_ts": conf,
        }
    return hit


def potential_lines(hit: dict, box_lo: Optional[float],
                    box_hi: Optional[float]) -> list:
    """Chart lines for the LOW-TF chart (user rule 2026-09-27): the box
    boundary on the non-breaking side, the pattern's mid pivot — the last
    low-TF high (LONG) / low (SHORT) — on the side being broken."""
    if hit["side"] == "long":
        return [box_lo, hit["mid_price"]]
    return [hit["mid_price"], box_hi]
