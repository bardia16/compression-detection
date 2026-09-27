"""Potential-break confirmation pattern (user rules 2026-09-27, final).

Two steps, both anchored on the BOX (never on a neighbouring pivot of the
scanned timeframe):

STEP 1 — the TOUCH lives in the MAIN tf:
    the box side is touched by a CONFIRMED EH (long) / EL (short) on the
    box's own timeframe, sitting on the box line (± the box's ATR).
STEP 2 — the CONFIRMING swing may live in the MAIN tf or the LOWER tf and
    must clear the EQUALITY threshold against the OTHER box side:
    LONG : a confirmed low  > box_low  + eq    ("higher than the low of
           box, exceeding the equal threshold")
    SHORT: a confirmed high < box_high − eq
    eq = ATR14 of the scanned tf (the same band EL/EL labels are built
    from).

Positional freshness of the touch (user rule 2026-09-27):
    · third swing on the LOWER tf -> the touch must be the LAST confirmed
      pivot of the main tf (nothing has superseded it)
    · third swing on the MAIN  tf -> the pivot right BEFORE the third must
      be the touch (and the third is the last main pivot)

STEP 2 has two confirmation routes (loosened 2026-09-27, PENDLE case):
    · CONFIRMED third swing — the alert lands on the candle that confirmed
      it (`_confirm_map`), as before.
    · LIVE third swing — when the zigzag threshold (1.2 x ATR7) is
      inflated by earlier volatility it can exceed the whole box height,
      so a valid higher low only "confirms" ON the breakout candle and
      `_drop_moot_potential` eats the alert. Accept the UNCONFIRMED
      extreme instead once a CLOSE has displaced LIVE_CLEAR_ATR (1.75,
      user choice 2026-09-27 EIGEN/XLM: 1x fired on the first chop) x
      ATR14 of the swing bar beyond it in the potential direction;
      `confirm_ts` is that candle. Positional freshness: the touch must
      still be the LAST confirmed main-tf pivot. A confirmed hit always
      wins when both qualify.

Why the shape changed (XPL/PIEVERSE, 2026-09-27): the earlier
"mid touches the line, third sits on the other line" rule compared nothing
to the equality band, and a lower-tf candidate could fire while the main tf
had already moved on — PIEVERSE printed a short 5 seconds after its long.

`box_at` (ts -> (upper, lower)) and `touch_atr` come from the engine, which
owns the Instance and its Line objects — no box lines, no pattern (fail
closed).  `LOWER_TF` maps a box tf to the tf one step below it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .atr import atr_series
from .dow import label_all_pivots
from .models import Candle
from .zigzag import ZigZag

# box TF -> the TF scanned below it (None = nothing under 15m)
LOWER_TF: Dict[str, Optional[str]] = {"4h": "1h", "1h": "15m", "15m": None}

# Freshness window (user choice 2026-09-27): a confirmation older than this
# is old news — the scan would otherwise backfill patterns that completed
# days before the box was even detected (PIEVERSE/BANK case: 3 of the 11
# first-scan hits were 4-5 days stale).
POTENTIAL_FRESH_S = 12 * 3600

# float slack for comparisons against a line value (a line fitted through
# the pivot differs from the pivot price by ~1e-17).
LINE_EPS_REL = 1e-9

# Displacement the close must show beyond the LIVE third swing in the
# potential direction before the alert means anything (user choice
# 2026-09-27, EIGEN/XLM case: the first dip past 1x eq fired while price
# kept chopping — "the price should still move a bit in the potential
# direction"). 1.75 x ATR14 of the swing bar: blocks EIGEN (1.58x) and
# XLM (1.52x), keeps PENDLE (budget 1.92x) firing pre-breakout, and sits
# at zigzag-grade displacement so the confirmed route can't bypass it
# with a smaller move. Confirmed-route alerts keep the original spec.
LIVE_CLEAR_ATR = 1.75


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
                   touch_atr: Optional[float] = None) -> Optional[dict]:
    """Latest matching pattern for `side`, or None (see module docstring).

    main_candles  candles of the box's own timeframe (STEP 1 lives there)
    scan          [(tf, candles), ...] in preference order — the engine
                  passes the lower tf first, the main tf last
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be long|short, got {side!r}")
    if box_at is None or touch_atr is None:
        return None
    if not main_candles:
        return None

    # ── STEP 1: the touch = confirmed EH/EL in the MAIN tf, on the line ──
    want = "EH" if side == "long" else "EL"
    main_lab = label_all_pivots(_zigzag(main_candles, cfg),
                                atr_series(main_candles, cfg.atr14_length,
                                           cfg.atr14_method))
    main_pivots = [p for p, _ in main_lab]
    if len(main_pivots) < 2:
        return None

    def lines(ts):
        try:
            up, lo = box_at(ts)
        except Exception:
            return None, None
        return (up, lo) if up is not None and lo is not None else (None, None)

    touch = None
    for p, lab in main_lab:
        if lab is None or lab.name != want:
            continue
        up, lo = lines(p.ts)
        if up is None:
            continue
        side_line = up if side == "long" else lo
        if abs(p.price - side_line) <= touch_atr:
            touch = p          # keep the LAST qualifying touch
    if touch is None:
        return None

    # ── STEP 2: the confirming swing, main tf or lower tf ────────────────
    hit: Optional[dict] = None
    for tf, candles in scan:
        if not candles or len(candles) < cfg.atr14_length + 6:
            continue
        pivots = _zigzag(candles, cfg)
        a14 = atr_series(candles, cfg.atr14_length, cfg.atr14_method)
        confirms = _confirm_map(candles, cfg)
        idx = {c.ts: i for i, c in enumerate(candles)}
        main_seq = tf == main_tf
        last_main = main_pivots[-1] if main_pivots else None

        for i, p in enumerate(pivots):
            if p.ts <= touch.ts or p.ts not in idx:
                continue
            eq = a14[idx[p.ts]]
            if eq is None or eq <= 0:
                continue
            up, lo = lines(p.ts)
            if up is None or lo is None:
                continue
            if side == "long":
                if p.is_high:
                    continue
                # higher than the box low, CLEARING the equality band
                if not p.price > lo + eq:
                    continue
            else:
                if not p.is_high:
                    continue
                # lower than the box high, CLEARING the equality band
                if not p.price < up - eq:
                    continue

            # positional freshness of the touch (user rule 2026-09-27)
            if main_seq:
                # third must be the LAST main pivot, touch right before it
                if last_main is None or last_main.ts != p.ts or i == 0:
                    continue
                if main_pivots[-2].ts != touch.ts:
                    continue
            else:
                # the touch must still be the last main-tf pivot
                if last_main is None or last_main.ts != touch.ts:
                    continue

            conf = confirms.get(p.ts)
            if conf is None:
                continue
            if anchor_ts is not None and conf < anchor_ts:
                continue
            hit = {
                "side": side,
                "pattern_tf": tf,
                "touch_ts": touch.ts,
                "touch_price": touch.price,
                "touch_label": want,
                "first_ts": touch.ts,        # kept for chart/caption compat
                "first_price": touch.price,
                "mid_ts": touch.ts,          # the touch IS the mid pivot
                "mid_price": touch.price,
                "mid_label": want,
                "second_ts": p.ts,
                "second_price": p.price,
                "confirm_ts": conf,
            }

        # ── LIVE third swing (loosened 2026-09-27, PENDLE case) ─────────
        # The zigzag confirm (1.2 x ATR7) can be wider than the box itself
        # after a volatile leg: PENDLE's higher low needed close > 2.648
        # while the whole box was 2.58-2.655, so it only confirmed on the
        # breakout candle and the same-scan breakout dropped the alert.
        # Route: the UNCONFIRMED extreme, once a CLOSE clears it by eq
        # (ATR14). Touch must still be the last confirmed main pivot.
        if hit is None and last_main is not None \
                and last_main.ts == touch.ts:
            live = _provisional(candles, cfg)
            if live is not None and touch.ts < live.ts and live.ts in idx:
                eq_l = a14[idx[live.ts]]
                if eq_l is not None and eq_l > 0:
                    up_l, lo_l = lines(live.ts)
                    band_ok = up_l is not None and lo_l is not None
                    if band_ok:
                        if side == "long":
                            band_ok = (not live.is_high
                                       and live.price > lo_l + eq_l)
                        else:
                            band_ok = (live.is_high
                                       and live.price < up_l - eq_l)
                    if band_ok:
                        confirm_ts = None
                        clear = LIVE_CLEAR_ATR * eq_l
                        for c in candles[idx[live.ts]:]:
                            if side == "long" and c.close > live.price + clear:
                                confirm_ts = c.ts
                                break
                            if side == "short" and c.close < live.price - clear:
                                confirm_ts = c.ts
                                break
                        if confirm_ts is not None and (
                                anchor_ts is None
                                or confirm_ts >= anchor_ts):
                            hit = {
                                "side": side,
                                "pattern_tf": tf,
                                "touch_ts": touch.ts,
                                "touch_price": touch.price,
                                "touch_label": want,
                                "first_ts": touch.ts,
                                "first_price": touch.price,
                                "mid_ts": touch.ts,
                                "mid_price": touch.price,
                                "mid_label": want,
                                "second_ts": live.ts,
                                "second_price": live.price,
                                "confirm_ts": confirm_ts,
                                "live": True,
                            }
        if hit is not None:
            return hit     # preference order: lower tf first
    return None


def potential_lines(hit: dict, box_lo: Optional[float],
                    box_hi: Optional[float]) -> list:
    """Chart lines for the LOW-TF chart (user rule 2026-09-27): the box
    boundary on the non-breaking side, the TOUCH pivot on the side being
    broken."""
    if hit["side"] == "long":
        return [box_lo, hit["mid_price"]]
    return [hit["mid_price"], box_hi]
