"""Compression structure candidate detection over DAW-labeled pivot sequences.

Active types (user rule 2026-09-20: boxes + asc/desc triangles ONLY):
  box                  EH ↔ EL   upper flat,    lower flat        ≥4 pivots
  descending_triangle  LH ↔ EL   upper falling, lower flat        ≥4  starts at the first EL tap
  ascending_triangle   EH ↔ HL   upper flat,    lower rising      ≥4  starts at the first EH tap

Triangle anchoring (user rule 2026-09-20, CUSDT case): the flat side is
determined FIRST; the pattern's pivots start at the first flat-side tap
(asc: a high, desc: a low), and the sloped side's pivots ALL count — the
first pivot after the first flat tap onward (no first-pivot exemption on
the sloped side), so a pre-pattern pivot can never anchor a boundary line.

Disabled (kept for reference — do not re-enable without the user):
  symmetrical_triangle LH ↔ HL   upper falling, lower rising      ≥4  (off 2026-09-20)
  falling_wedge        LH ↔ LL   upper falling, lower falling     ≥5  (off 2026-09-16)
  rising_wedge         HH ↔ HL   upper rising,  lower rising      ≥5  (off 2026-09-16)

A candidate is valid only when the label gate AND the boundary geometry
(flat/rising/falling in ATR units) AND (for converging types) the
multi-observation convergence check all pass — never labels alone
(spec §4, §17, §29). Boxes keep the spec §5 first-pivot label exemption
on both sides; triangles exempt only the flat side's first tap (user
rule 2026-09-20 — see above).

All candidates for all types/window sizes are returned; ranking is
deterministic: (pivot_count desc, total fit error asc, selection order).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import structure as st
from .dow import PivotLabel
from .models import Pivot

TYPE_BOX = "box"
TYPE_DESC_TRI = "descending_triangle"
TYPE_ASC_TRI = "ascending_triangle"
TYPE_SYM_TRI = "symmetrical_triangle"
TYPE_FALLING_WEDGE = "falling_wedge"
TYPE_RISING_WEDGE = "rising_wedge"

# priority order (user rule 2026-09-19): when several compressions exist for
# one coin+tf, triangles (asc/desc) surface first, then box
ALL_TYPES = (
    TYPE_ASC_TRI, TYPE_DESC_TRI, TYPE_BOX,
    # symmetrical disabled 2026-09-20 (user rule: boxes + asc/desc only)
    # TYPE_SYM_TRI,
    # wedges disabled 2026-09-16 (buggy — will revisit)
    # TYPE_FALLING_WEDGE, TYPE_RISING_WEDGE,
)


@dataclass(frozen=True)
class TypeSpec:
    name: str
    # allowed labels for pivots of each side
    upper_labels: Tuple[PivotLabel, ...]
    lower_labels: Tuple[PivotLabel, ...]
    # allowed slope classes for each boundary
    upper_slopes: Tuple[str, ...]
    lower_slopes: Tuple[str, ...]
    converging: bool
    # user rule 2026-09-20 (CUSDT): triangles anchor at the flat side's
    # first tap — the window must START on that side ("H"/"L"; None =
    # either) and the sloped side has NO first-pivot label exemption (all
    # its pivots count, starting from the first pivot after the first
    # flat-side tap). Boxes keep both first-pivot exemptions.
    lead_side: Optional[str] = None
    exempt_first_upper: bool = True
    exempt_first_lower: bool = True


TYPE_SPECS: Dict[str, TypeSpec] = {
    TYPE_BOX: TypeSpec(
        TYPE_BOX,
        (PivotLabel.EH,), (PivotLabel.EL,),
        (st.FLAT,), (st.FLAT,), False,
    ),
    TYPE_DESC_TRI: TypeSpec(
        TYPE_DESC_TRI,
        (PivotLabel.LH,), (PivotLabel.EL,),
        (st.FALLING,), (st.FLAT,), False,
        lead_side="L", exempt_first_upper=False,
    ),
    TYPE_ASC_TRI: TypeSpec(
        TYPE_ASC_TRI,
        (PivotLabel.EH,), (PivotLabel.HL,),
        (st.FLAT,), (st.RISING,), False,
        lead_side="H", exempt_first_lower=False,
    ),
    # symmetrical disabled 2026-09-20 (user rule: boxes + asc/desc only)
    # TYPE_SYM_TRI: TypeSpec(
    #     TYPE_SYM_TRI,
    #     (PivotLabel.LH,), (PivotLabel.HL,),
    #     (st.FALLING,), (st.RISING,), True,
    # ),
    # wedges disabled 2026-09-16 (buggy — will revisit)
    # TYPE_FALLING_WEDGE: TypeSpec(
    #     TYPE_FALLING_WEDGE,
    #     (PivotLabel.LH,), (PivotLabel.LL,),
    #     (st.FALLING,), (st.FALLING,), True,
    # ),
    # TYPE_RISING_WEDGE: TypeSpec(
    #     TYPE_RISING_WEDGE,
    #     (PivotLabel.HH,), (PivotLabel.HL,),
    #     (st.RISING,), (st.RISING,), True,
    # ),
}


@dataclass(frozen=True)
class DetectConfig:
    min_pivots: Dict[str, int]
    max_window_pivots: int = 12
    max_pivot_age_bars: int = 96
    flat_tol_atr: float = 1.0
    boundary_tol_atr: float = 1.0
    min_convergence_pct: float = 25.0
    max_width_bounce_frac: float = 0.10
    confirm_extra_pivots: int = 1
    established_extra_pivots: int = 2
    min_boundary_hits: int = 2
    # Interior close integrity (user rule 2026-09-17, BTW case): a boundary
    # "broken several times" by candle CLOSES inside the window means the
    # structure was never a compression.  A close beyond a boundary by more
    # than close_breach_tol_atr × ATR counts as a breach; more than
    # max_close_breaches breaches on any side rejects the candidate.
    close_breach_tol_atr: float = 0.5
    max_close_breaches: int = 2
    # Range-mode box (user request 2026-09-17, JTO case): textbook ranges /
    # consolidation boxes with INTERNAL swings are detected via touch
    # clusters instead of requiring every non-first pivot to carry EH/EL
    # (the strict model rejected JTO 4H 0.40–0.466 — internal swings broke
    # the label chain: HH→EH→LH→HH / LL→HL→LL→HL→LL).
    box_range_mode: bool = False
    selection_order: Tuple[str, ...] = ALL_TYPES


@dataclass(frozen=True)
class PivotRef:
    ts: int
    bar_index: int
    abs_bar: int
    price: float
    is_high: bool
    label: Optional[str]  # "HH"/"LH"/"EH"/"HL"/"LL"/"EL" or None

    def to_dict(self) -> dict:
        return {
            "ts": self.ts, "bar": self.abs_bar, "price": self.price,
            "side": "H" if self.is_high else "L", "label": self.label,
        }


@dataclass
class Candidate:
    type: str
    refs: List[PivotRef]                # chronological window, includes latest pivot
    upper: st.Line
    lower: st.Line
    atr: float
    upper_class: str
    lower_class: str
    metrics: Dict[str, object]

    @property
    def pivot_count(self) -> int:
        return len(self.refs)

    @property
    def pivot_tss(self) -> set:
        return {r.ts for r in self.refs}

    @property
    def first_ts(self) -> int:
        return self.refs[0].ts

    @property
    def last_ts(self) -> int:
        return self.refs[-1].ts

    def to_dict(self) -> dict:
        d = {
            "type": self.type,
            "pivots": [r.to_dict() for r in self.refs],
            "upper": {"slope": self.upper.slope, "intercept": self.upper.intercept,
                      "base_bar": self.upper.base_bar},
            "lower": {"slope": self.lower.slope, "intercept": self.lower.intercept,
                      "base_bar": self.lower.base_bar},
            "upper_class": self.upper_class,
            "lower_class": self.lower_class,
            "atr": self.atr,
        }
        d.update(self.metrics)
        return d


def rank_key(c: Candidate, order: Sequence[str]) -> tuple:
    """Deterministic ranking: more pivots first, then lower fit error, then
    the configured selection order."""
    try:
        idx = list(order).index(c.type)
    except ValueError:
        idx = len(order)
    return (-c.pivot_count, c.metrics.get("fit_err_atr", 0.0), idx)


def rank_candidates(cands: Sequence[Candidate], order: Sequence[str]) -> List[Candidate]:
    return sorted(cands, key=lambda c: rank_key(c, order))


def _label_ok(
    refs: Sequence[PivotRef], spec: TypeSpec
) -> bool:
    """Label gate.  The first pivot of a side is exempt only when the spec
    says so (user rule 2026-09-20: for triangles the SLOPED side counts
    from the first pivot after the first flat-side tap — no exemption
    there), and the window must START on the spec's lead side (asc: a
    high, desc: a low) so a pre-pattern pivot never anchors a line."""
    if spec.lead_side == "H" and not refs[0].is_high:
        return False
    if spec.lead_side == "L" and refs[0].is_high:
        return False
    seen_high = False
    seen_low = False
    for r in refs:
        if r.is_high:
            first = not seen_high
            seen_high = True
            if first and spec.exempt_first_upper:
                continue
            if r.label is None or PivotLabel(r.label) not in spec.upper_labels:
                return False
        else:
            first = not seen_low
            seen_low = True
            if first and spec.exempt_first_lower:
                continue
            if r.label is None or PivotLabel(r.label) not in spec.lower_labels:
                return False
    return True


def _check_window(
    spec: TypeSpec,
    refs: Sequence[PivotRef],
    atr14_series: List[Optional[float]],
    cfg: DetectConfig,
    close_by_bar: Optional[Dict[int, float]] = None,
) -> Optional[Candidate]:
    highs = [r for r in refs if r.is_high]
    lows = [r for r in refs if not r.is_high]
    if len(highs) < 2 or len(lows) < 2:
        return None

    # ATR reference: last known ATR14 at/before the window's last pivot.
    last_vote = refs[-1].bar_index
    atr = None
    for a in reversed(atr14_series[: last_vote + 1]):
        if a is not None:
            atr = a
            break
    if atr is None or atr <= 0:
        return None

    if not _label_ok(refs, spec):
        return None

    upper_pts = [(r.abs_bar, r.price) for r in highs]
    lower_pts = [(r.abs_bar, r.price) for r in lows]
    upper = st.fit_line(upper_pts)
    lower = st.fit_line(lower_pts)
    if upper is None or lower is None:
        return None

    # Boundary respect for every pivot in the window (all pivots, both sides)
    up_flags = st.respects(upper_pts, upper, atr, cfg.boundary_tol_atr)
    lo_flags = st.respects(lower_pts, lower, atr, cfg.boundary_tol_atr)
    if not all(up_flags) or not all(lo_flags):
        return None

    # Ordering guard: boundaries must not cross anywhere inside the
    # structure span (first → last pivot bar; linear lines → checking the
    # window bars suffices). A vertex forming AFTER the last pivot is the
    # normal end-state of a converging structure, not invalidity — the
    # breakout check owns everything past the structure edge.
    bars = {r.abs_bar for r in refs}
    if any(upper.at(b) <= lower.at(b) for b in bars):
        return None

    # Slope classes, measured over each side's own pivot span. The class
    # threshold uses the volatility ACROSS that span (max of the endpoint
    # ATR14 readings) — a compression whose volatility cools must not have
    # its flat boundary re-classified as rising/falling by the shrunken
    # tail ATR (user rule 2026-09-16, LTC case).
    def _span_atr(side_pivots):
        vals = []
        for p in (side_pivots[0], side_pivots[-1]):
            if p.bar_index < len(atr14_series) and atr14_series[p.bar_index]:
                vals.append(atr14_series[p.bar_index])
        return max(vals) if vals else atr

    upper_class = st.slope_class(
        upper, highs[0].abs_bar, highs[-1].abs_bar, _span_atr(highs), cfg.flat_tol_atr
    )
    lower_class = st.slope_class(
        lower, lows[0].abs_bar, lows[-1].abs_bar, _span_atr(lows), cfg.flat_tol_atr
    )
    if upper_class not in spec.upper_slopes or lower_class not in spec.lower_slopes:
        return None

    metrics: Dict[str, object] = {
        "pivot_count": len(refs),
        "touches": int(sum(up_flags) + sum(lo_flags)),
        "fit_err_atr": st.fit_error(upper_pts, upper, atr) + st.fit_error(lower_pts, lower, atr),
        "upper_eq_err_atr": (max(p for _, p in upper_pts) - min(p for _, p in upper_pts)) / atr,
        "lower_eq_err_atr": (max(p for _, p in lower_pts) - min(p for _, p in lower_pts)) / atr,
        "span_bars": refs[-1].abs_bar - refs[0].abs_bar,
        "upper_at_last_bar": upper.at(refs[-1].abs_bar),
        "lower_at_last_bar": lower.at(refs[-1].abs_bar),
    }

    # Boundary-hit counts (confirmation rule, 2026-09-16 revised):
    # at least one side must have >= `min_boundary_hits` pivots (confirmed
    # or live).  Only ONE side needs the count — the last pivot completing
    # the pair can be live/unconfirmed (user rule 2026-09-16).
    upper_hits = len(highs)
    lower_hits = len(lows)
    metrics["upper_hits"] = upper_hits
    metrics["lower_hits"] = lower_hits
    metrics["hit_requirement"] = max(upper_hits, lower_hits) >= cfg.min_boundary_hits
    metrics["hit_boundary"] = ("upper" if upper_hits >= cfg.min_boundary_hits
                               else "lower" if lower_hits >= cfg.min_boundary_hits else "")

    # Interior close integrity (user rule 2026-09-17): candle CLOSES between
    # the window's first and last pivot must stay within the boundaries —
    # a line broken several times by closes was never a compression (BTW
    # case: 48 closes below the fitted lower line).  Tolerance absorbs
    # fit/noise: close beyond by > close_breach_tol_atr × ATR = a breach.
    if close_by_bar is not None:
        b0, b1 = refs[0].abs_bar, refs[-1].abs_bar
        tol = cfg.close_breach_tol_atr * atr
        up_breaches = lo_breaches = 0
        worst_breach = 0.0
        for b in range(b0, b1 + 1):
            cl = close_by_bar.get(b)
            if cl is None:
                continue
            u = upper.at(b)
            l = lower.at(b)
            if cl > u + tol:
                up_breaches += 1
                worst_breach = max(worst_breach, (cl - u) / atr)
            elif cl < l - tol:
                lo_breaches += 1
                worst_breach = max(worst_breach, (l - cl) / atr)
        metrics["close_breaches_upper"] = up_breaches
        metrics["close_breaches_lower"] = lo_breaches
        metrics["worst_close_breach_atr"] = worst_breach
        if max(up_breaches, lo_breaches) > cfg.max_close_breaches:
            return None

    if spec.converging:
        sample_from = max(highs[0].abs_bar, lows[0].abs_bar)
        sample_bars = sorted({r.abs_bar for r in refs if r.abs_bar >= sample_from})
        widths = st.convergence_samples(upper, lower, sample_bars)
        ok, shrink, bounce = st.evaluate_convergence(
            widths, cfg.min_convergence_pct / 100.0, cfg.max_width_bounce_frac
        )
        if not ok:
            return None
        metrics["convergence_rate"] = shrink
        metrics["width_reduction_pct"] = shrink * 100.0
        metrics["max_bounce_frac"] = bounce
        metrics["width_first"] = widths[0]
        metrics["width_last"] = widths[-1]
    else:
        metrics["convergence_rate"] = 0.0
        metrics["width_reduction_pct"] = 0.0

    return Candidate(
        type=spec.name, refs=list(refs), upper=upper, lower=lower, atr=atr,
        upper_class=upper_class, lower_class=lower_class, metrics=metrics,
    )


def _check_box_range(
    refs: Sequence[PivotRef],
    atr14_series: List[Optional[float]],
    cfg: DetectConfig,
    close_by_bar: Optional[Dict[int, float]] = None,
) -> Optional[Candidate]:
    """Range-mode box (user request 2026-09-17, JTO case).

    A consolidation range is defined by its boundary TOUCH CLUSTERS, not by
    every pivot: swings strictly inside the range are allowed (the strict
    model required every non-first pivot to be EH/EL and so rejected
    textbook ranges whose internal swings interleave between the taps).

    Rules:
    - >=2 high touches within flat_tol_atr × ATR (side span ATR) of the
      highest high; same for lows around the lowest low;
    - the two sides' touch spans must overlap in time (both boundaries
      tested during a common stretch — rejects break + dead-cat windows
      where one side's taps are stale);
    - closes between first and last pivot stay within the envelope
      (close_breach_tol_atr / max_close_breaches, same as strict path).
    """
    highs = [r for r in refs if r.is_high]
    lows = [r for r in refs if not r.is_high]
    if len(highs) < 2 or len(lows) < 2:
        return None

    last_vote = refs[-1].bar_index
    atr = None
    for a in reversed(atr14_series[: last_vote + 1]):
        if a is not None:
            atr = a
            break
    if atr is None or atr <= 0:
        return None

    def span_atr(side_pivots):
        vals = []
        for p in (side_pivots[0], side_pivots[-1]):
            if p.bar_index < len(atr14_series) and atr14_series[p.bar_index]:
                vals.append(atr14_series[p.bar_index])
        return max(vals) if vals else atr

    u_ref = max(r.price for r in highs)
    l_ref = min(r.price for r in lows)
    u_tol = cfg.flat_tol_atr * span_atr(highs)
    l_tol = cfg.flat_tol_atr * span_atr(lows)
    u_touch = [r for r in highs if u_ref - r.price <= u_tol]
    l_touch = [r for r in lows if r.price - l_ref <= l_tol]
    if len(u_touch) < 2 or len(l_touch) < 2:
        return None

    # Both boundaries must have been tested during a common stretch.
    if max(u_touch[0].abs_bar, l_touch[0].abs_bar) >= \
            min(u_touch[-1].abs_bar, l_touch[-1].abs_bar):
        return None

    metrics: Dict[str, object] = {
        "pivot_count": len(refs),
        "touches": len(u_touch) + len(l_touch),
        "upper_touches": len(u_touch),
        "lower_touches": len(l_touch),
        "span_bars": refs[-1].abs_bar - refs[0].abs_bar,
        "upper_eq_err_atr": (max(r.price for r in u_touch)
                             - min(r.price for r in u_touch)) / atr,
        "lower_eq_err_atr": (max(r.price for r in l_touch)
                             - min(r.price for r in l_touch)) / atr,
        "upper_hits": len(highs),
        "lower_hits": len(lows),
        "hit_requirement": True,
        "hit_boundary": "upper",
        "convergence_rate": 0.0,
        "width_reduction_pct": 0.0,
        "mode": "range",
    }

    # Boundaries: strictly horizontal at the LAST touch of each side (the
    # same price the breakout level and chart line use, per family rules).
    u_level = u_touch[-1].price
    l_level = l_touch[-1].price
    metrics["upper_at_last_bar"] = u_level
    metrics["lower_at_last_bar"] = l_level

    # Interior close integrity against the range envelope.
    if close_by_bar is not None:
        b0, b1 = refs[0].abs_bar, refs[-1].abs_bar
        tol = cfg.close_breach_tol_atr * atr
        up_breaches = lo_breaches = 0
        worst_breach = 0.0
        for b in range(b0, b1 + 1):
            cl = close_by_bar.get(b)
            if cl is None:
                continue
            if cl > u_ref + tol:
                up_breaches += 1
                worst_breach = max(worst_breach, (cl - u_ref) / atr)
            elif cl < l_ref - tol:
                lo_breaches += 1
                worst_breach = max(worst_breach, (l_ref - cl) / atr)
        metrics["close_breaches_upper"] = up_breaches
        metrics["close_breaches_lower"] = lo_breaches
        metrics["worst_close_breach_atr"] = worst_breach
        if max(up_breaches, lo_breaches) > cfg.max_close_breaches:
            return None

    upper = st.Line(slope=0.0, intercept=u_level, base_bar=0.0)
    lower = st.Line(slope=0.0, intercept=l_level, base_bar=0.0)
    return Candidate(
        type=TYPE_BOX, refs=list(refs), upper=upper, lower=lower, atr=atr,
        upper_class=st.FLAT, lower_class=st.FLAT, metrics=metrics,
    )


def detect_candidates(
    labeled: Sequence[Tuple[Pivot, Optional[PivotLabel]]],
    atr14_series: List[Optional[float]],
    tf_ms: int,
    last_closed_abs_bar: int,
    cfg: DetectConfig,
    candles: Optional[Sequence] = None,
) -> List[Candidate]:
    """All valid candidates across all types and trailing window sizes.

    Windows are end-anchored at the latest confirmed pivot; pivots older
    than cfg.max_pivot_age_bars are ignored. Every window size from the
    type minimum up to max_window_pivots is tried, so the most established
    (largest) valid windows rank first.

    `candles` (optional) enables the interior close-integrity check.
    """
    refs: List[PivotRef] = []
    for p, label in labeled:
        abs_bar = p.ts // tf_ms
        if abs_bar < last_closed_abs_bar - cfg.max_pivot_age_bars:
            continue
        refs.append(PivotRef(
            ts=p.ts, bar_index=p.bar_index, abs_bar=abs_bar,
            price=p.price, is_high=p.is_high,
            label=label.value if label is not None else None,
        ))
    if not refs:
        return []

    close_by_bar: Optional[Dict[int, float]] = None
    if candles is not None:
        close_by_bar = {int(c.ts // tf_ms): float(c.close) for c in candles}

    out: List[Candidate] = []
    for type_name in ALL_TYPES:
        spec = TYPE_SPECS[type_name]
        min_k = cfg.min_pivots[type_name]
        top_k = min(cfg.max_window_pivots, len(refs))
        for k in range(min_k, top_k + 1):
            window = refs[-k:]
            cand = _check_window(spec, window, atr14_series, cfg, close_by_bar)
            if cand is None and type_name == TYPE_BOX and cfg.box_range_mode:
                cand = _check_box_range(window, atr14_series, cfg, close_by_bar)
            if cand is not None:
                out.append(cand)
    return rank_candidates(out, cfg.selection_order)
