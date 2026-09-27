"""Potential-break confirmation pattern (user rule 2026-09-27).

A compression coiling for a break prints this three-swing sequence:

  LONG : low  → EH (equal high = the box high) → HIGHER low, confirmed
  SHORT: high → EL (equal low  = the box low)  → LOWER high, confirmed

Zigzag pivots alternate, so the middle pivot is by construction the LAST
high (LONG) / low (SHORT) on that timeframe before the second swing — the
low-TF chart draws its adjusted line exactly there (user rule 2026-09-27,
SUI case: the 1H box's own last high was 0.0061 below the 15m EH).

The third swing only exists once the zigzag CONFIRMS it (close beyond
coef × ATR7), so the alert lands on the confirming candle — on the parent
timeframe's close verdict that is minutes away, not after it.

Scanned on the box's own TF (main) and one TF down; `LOWER_TF` maps a
box TF to the timeframe it is checked against.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .atr import atr_series
from .dow import label_all_pivots
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
                   anchor_ts: Optional[int] = None) -> Optional[dict]:
    """Latest matching pattern on ONE timeframe, or None.

    side      "long" | "short" — direction of the expected break
    anchor_ts the compression's anchor (first pivot ts): the pattern must
              CONFIRM at or after it — a confirmation that predates the box
              is not this box's signal.

    Returned dict carries every swing plus the confirming candle, so the
    engine can draw the chart lines without recomputing the zigzag.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be long|short, got {side!r}")
    if len(candles) < cfg.atr14_length + cfg.zigzag_atr_length + 6:
        return None

    zz = ZigZag(coef=cfg.zigzag_coef, atr_length=cfg.zigzag_atr_length)
    pivots = zz.feed_all(candles)
    atr14 = atr_series(candles, cfg.atr14_length, cfg.atr14_method)
    # label_all_pivots keeps EVERY pivot (label may be None) so the first and
    # second swings survive even when they are not labelable — only the middle
    # one must carry an EH/EL label.
    labeled = label_all_pivots(pivots, atr14)
    if len(labeled) < 3:
        return None
    confirms = _confirm_map(candles, cfg)

    hit: Optional[dict] = None
    for i in range(len(labeled) - 2):
        first, first_lab = labeled[i]
        mid, mid_lab = labeled[i + 1]
        second, second_lab = labeled[i + 2]
        if mid_lab is None:
            continue
        if side == "long":
            # low -> EH -> higher low
            if first.is_high or not mid.is_high or second.is_high:
                continue
            if mid_lab.name != "EH":
                continue
            # the second swing must be LABELED a higher low — a price that is
            # merely numerically higher but inside the ±ATR14 equality band is
            # an EL (equal low), not this pattern (user 2026-09-27, QNT case:
            # 180.06 < 181.25 numerically but 0.32x ATR14 apart -> labeled EH,
            # so there was no lower high at all).
            if second_lab is None or second_lab.name != "HL":
                continue
            if not second.price > first.price:
                continue
        else:
            # high -> EL -> lower high
            if not first.is_high or mid.is_high or not second.is_high:
                continue
            if mid_lab.name != "EL":
                continue
            # ...and LABELED a lower high (EH inside the equality band is an
            # equal high, which is not a lower high)
            if second_lab is None or second_lab.name != "LH":
                continue
            if not second.price < first.price:
                continue
        conf = confirms.get(second.ts)
        if conf is None:      # unreachable — feed_all only yields confirmed
            continue
        # Anchor test is on the CONFIRMATION, not the first swing (QNT case):
        # the box's anchor sits between the pattern's swings (H 188.95 ->
        # [anchor = EL 170.57] -> LH 181.25), so gating on first.ts dropped a
        # pattern that both completed and confirmed inside the box's life.
        # What matters is that the confirmation landed while the box existed;
        # the 12h freshness window bounds anything older.
        if anchor_ts is not None and conf < anchor_ts:
            continue
        hit = {
            "side": side,
            "first_ts": first.ts,
            "first_price": first.price,
            "mid_ts": mid.ts,
            "mid_price": mid.price,
            "mid_label": mid_lab.name,
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
