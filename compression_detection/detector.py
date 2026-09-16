"""Compression structure candidate detection over DAW-labeled pivot sequences.

Six canonical types (spec §28):
  box                  EH ↔ EL   upper flat,    lower flat        ≥4 pivots
  descending_triangle  LH ↔ EL   upper falling, lower flat        ≥4
  ascending_triangle   EH ↔ HL   upper flat,    lower rising      ≥4
  symmetrical_triangle LH ↔ HL   upper falling, lower rising      ≥4  + converging
  falling_wedge        LH ↔ LL   upper falling, lower falling     ≥5  + converging
  rising_wedge         HH ↔ HL   upper rising,  lower rising      ≥5  + converging

A candidate is valid only when the label gate AND the boundary geometry
(flat/rising/falling in ATR units) AND (for converging types) the
multi-observation convergence check all pass — never labels alone
(spec §4, §17, §29). The first pivot of each side is label-exempt
(spec §5: initial reference pivots don't need literal labels when the
history already establishes the boundary).

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

ALL_TYPES = (
    TYPE_BOX, TYPE_DESC_TRI, TYPE_ASC_TRI, TYPE_SYM_TRI,
    TYPE_FALLING_WEDGE, TYPE_RISING_WEDGE,
)


@dataclass(frozen=True)
class TypeSpec:
    name: str
    # allowed labels for NON-first pivots of each side
    upper_labels: Tuple[PivotLabel, ...]
    lower_labels: Tuple[PivotLabel, ...]
    # allowed slope classes for each boundary
    upper_slopes: Tuple[str, ...]
    lower_slopes: Tuple[str, ...]
    converging: bool


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
    ),
    TYPE_ASC_TRI: TypeSpec(
        TYPE_ASC_TRI,
        (PivotLabel.EH,), (PivotLabel.HL,),
        (st.FLAT,), (st.RISING,), False,
    ),
    TYPE_SYM_TRI: TypeSpec(
        TYPE_SYM_TRI,
        (PivotLabel.LH,), (PivotLabel.HL,),
        (st.FALLING,), (st.RISING,), True,
    ),
    TYPE_FALLING_WEDGE: TypeSpec(
        TYPE_FALLING_WEDGE,
        (PivotLabel.LH,), (PivotLabel.LL,),
        (st.FALLING,), (st.FALLING,), True,
    ),
    TYPE_RISING_WEDGE: TypeSpec(
        TYPE_RISING_WEDGE,
        (PivotLabel.HH,), (PivotLabel.HL,),
        (st.RISING,), (st.RISING,), True,
    ),
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
    metrics: Dict[str, float]

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
    """Label gate: every NON-first pivot of each side must carry one of the
    spec's allowed labels; the first pivot of each side is exempt."""
    seen_high = False
    seen_low = False
    for r in refs:
        if r.is_high:
            if not seen_high:
                seen_high = True
                continue
            if r.label is None or PivotLabel(r.label) not in spec.upper_labels:
                return False
        else:
            if not seen_low:
                seen_low = True
                continue
            if r.label is None or PivotLabel(r.label) not in spec.lower_labels:
                return False
    return True


def _check_window(
    spec: TypeSpec,
    refs: Sequence[PivotRef],
    atr14_series: List[Optional[float]],
    cfg: DetectConfig,
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

    # Slope classes, measured over each side's own pivot span
    upper_class = st.slope_class(
        upper, highs[0].abs_bar, highs[-1].abs_bar, atr, cfg.flat_tol_atr
    )
    lower_class = st.slope_class(
        lower, lows[0].abs_bar, lows[-1].abs_bar, atr, cfg.flat_tol_atr
    )
    if upper_class not in spec.upper_slopes or lower_class not in spec.lower_slopes:
        return None

    metrics: Dict[str, float] = {
        "pivot_count": len(refs),
        "touches": int(sum(up_flags) + sum(lo_flags)),
        "fit_err_atr": st.fit_error(upper_pts, upper, atr) + st.fit_error(lower_pts, lower, atr),
        "upper_eq_err_atr": (max(p for _, p in upper_pts) - min(p for _, p in upper_pts)) / atr,
        "lower_eq_err_atr": (max(p for _, p in lower_pts) - min(p for _, p in lower_pts)) / atr,
        "span_bars": refs[-1].abs_bar - refs[0].abs_bar,
        "upper_at_last_bar": upper.at(refs[-1].abs_bar),
        "lower_at_last_bar": lower.at(refs[-1].abs_bar),
    }

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


def detect_candidates(
    labeled: Sequence[Tuple[Pivot, Optional[PivotLabel]]],
    atr14_series: List[Optional[float]],
    tf_ms: int,
    last_closed_abs_bar: int,
    cfg: DetectConfig,
) -> List[Candidate]:
    """All valid candidates across all types and trailing window sizes.

    Windows are end-anchored at the latest confirmed pivot; pivots older
    than cfg.max_pivot_age_bars are ignored. Every window size from the
    type minimum up to max_window_pivots is tried, so the most established
    (largest) valid windows rank first.
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

    out: List[Candidate] = []
    for type_name in ALL_TYPES:
        spec = TYPE_SPECS[type_name]
        min_k = cfg.min_pivots[type_name]
        top_k = min(cfg.max_window_pivots, len(refs))
        for k in range(min_k, top_k + 1):
            window = refs[-k:]
            cand = _check_window(spec, window, atr14_series, cfg)
            if cand is not None:
                out.append(cand)
    return rank_candidates(out, cfg.selection_order)
