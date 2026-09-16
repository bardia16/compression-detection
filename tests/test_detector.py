"""Detector tests — synthetic labeled-pivot sequences with hand-computed
geometry. Positive detections, negative rejections (trends, parallel
channels, label-gate violations), first-of-side exemption, ranking.
"""
import pytest

from compression_detection import structure as st
from compression_detection.detector import (
    ALL_TYPES,
    TYPE_ASC_TRI,
    TYPE_BOX,
    TYPE_DESC_TRI,
    TYPE_FALLING_WEDGE,
    TYPE_RISING_WEDGE,
    TYPE_SYM_TRI,
    Candidate,
    DetectConfig,
    PivotRef,
    detect_candidates,
    rank_candidates,
)
from compression_detection.dow import PivotLabel
from compression_detection.models import Pivot

TF_MS = 60_000
MIN_PIVOTS = {
    TYPE_BOX: 4, TYPE_DESC_TRI: 4, TYPE_ASC_TRI: 4,
    TYPE_SYM_TRI: 4, TYPE_FALLING_WEDGE: 5, TYPE_RISING_WEDGE: 5,
}


def entries_to_labeled(entries, tf_ms=TF_MS):
    """entries: [(side 'H'/'L', price, abs_bar, label|None)] -> labeled tuples."""
    out = []
    for side, price, bar, label in entries:
        p = Pivot(type=1 if side == "H" else -1, price=price,
                  bar_index=bar, ts=bar * tf_ms)
        out.append((p, PivotLabel(label) if label else None))
    return out


def atr_series(val=2.0, n=800):
    return [val if i >= 15 else None for i in range(n)]


def detect(entries, last_bar, **cfg_over):
    cfg = DetectConfig(min_pivots=MIN_PIVOTS, **cfg_over)
    return detect_candidates(
        entries_to_labeled(entries), atr_series(), TF_MS, last_bar, cfg
    )


def types_of(cands):
    return [c.type for c in cands]


# ── positive detections ────────────────────────────────────────────────

def test_box_min_4_pivots():
    cands = detect([
        ("H", 100.0, 10, None),
        ("L", 90.0, 14, None),
        ("H", 100.5, 18, "EH"),
        ("L", 90.5, 22, "EL"),
    ], last_bar=24)
    assert TYPE_BOX in types_of(cands)
    box = [c for c in cands if c.type == TYPE_BOX][0]
    assert box.pivot_count == 4
    assert box.upper_class == st.FLAT and box.lower_class == st.FLAT
    # 1 EH + 1 EL -> boundary-hit requirement not yet satisfied (4 pivots)
    assert box.metrics["upper_hits"] == 1 and box.metrics["lower_hits"] == 1
    assert box.metrics["hit_requirement"] is False
    assert box.metrics["hit_boundary"] == ""


def test_box_larger_window_ranked_first():
    cands = detect([
        ("H", 100.0, 10, None),
        ("L", 90.0, 14, None),
        ("H", 100.5, 18, "EH"),
        ("L", 90.5, 22, "EL"),
        ("H", 100.3, 26, "EH"),
        ("L", 90.7, 30, "EL"),
    ], last_bar=32)
    assert cands[0].type == TYPE_BOX
    assert cands[0].pivot_count == 6
    # second EH + second EL -> requirement satisfied
    assert cands[0].metrics["upper_hits"] == 2
    assert cands[0].metrics["lower_hits"] == 2
    assert cands[0].metrics["hit_requirement"] is True
    assert cands[0].metrics["hit_boundary"] == "upper"


def test_descending_triangle():
    cands = detect([
        ("H", 100.0, 10, None),
        ("L", 90.0, 14, None),
        ("H", 96.0, 18, "LH"),
        ("L", 90.2, 22, "EL"),
    ], last_bar=24)
    assert TYPE_DESC_TRI in types_of(cands)
    d = [c for c in cands if c.type == TYPE_DESC_TRI][0]
    assert d.upper_class == st.FALLING and d.lower_class == st.FLAT


def test_ascending_triangle():
    cands = detect([
        ("L", 90.0, 10, None),
        ("H", 100.0, 14, None),
        ("L", 94.0, 18, "HL"),
        ("H", 100.4, 22, "EH"),
    ], last_bar=24)
    assert TYPE_ASC_TRI in types_of(cands)
    a = [c for c in cands if c.type == TYPE_ASC_TRI][0]
    assert a.upper_class == st.FLAT and a.lower_class == st.RISING


def test_symmetrical_triangle_converging():
    cands = detect([
        ("H", 100.0, 10, None),
        ("L", 90.0, 14, None),
        ("H", 97.0, 18, "LH"),
        ("L", 94.0, 22, "HL"),
    ], last_bar=24)
    assert TYPE_SYM_TRI in types_of(cands)
    s = [c for c in cands if c.type == TYPE_SYM_TRI][0]
    assert s.upper_class == st.FALLING and s.lower_class == st.RISING
    assert s.metrics["convergence_rate"] == pytest.approx(7.0 / 8.5)


def test_falling_wedge_converging():
    cands = detect([
        ("H", 100.0, 10, None),
        ("L", 90.0, 14, None),
        ("H", 92.0, 18, "LH"),
        ("L", 82.0, 26, "LL"),
        ("H", 84.0, 30, "LH"),
    ], last_bar=32)
    assert TYPE_FALLING_WEDGE in types_of(cands)
    fw = [c for c in cands if c.type == TYPE_FALLING_WEDGE][0]
    assert fw.upper_class == st.FALLING and fw.lower_class == st.FALLING
    assert fw.metrics["convergence_rate"] > 0.25
    # 2 LH on the upper boundary -> requirement satisfied at min pivots
    assert fw.metrics["upper_hits"] == 2 and fw.metrics["lower_hits"] == 1
    assert fw.metrics["hit_requirement"] is True
    assert fw.metrics["hit_boundary"] == "upper"


def test_rising_wedge_converging():
    cands = detect([
        ("L", 90.0, 10, None),
        ("H", 100.0, 14, None),
        ("L", 96.0, 18, "HL"),
        ("H", 104.0, 26, "HH"),
        ("L", 105.0, 30, "HL"),
    ], last_bar=32)
    assert TYPE_RISING_WEDGE in types_of(cands)
    rw = [c for c in cands if c.type == TYPE_RISING_WEDGE][0]
    assert rw.upper_class == st.RISING and rw.lower_class == st.RISING
    assert rw.metrics["convergence_rate"] > 0.25
    # 2 HL on the lower boundary -> requirement satisfied at min pivots
    assert rw.metrics["lower_hits"] == 2 and rw.metrics["upper_hits"] == 1
    assert rw.metrics["hit_requirement"] is True
    assert rw.metrics["hit_boundary"] == "lower"


def test_first_of_side_label_exempt():
    """Window's first high carries a non-canonical label — still valid
    (initial reference pivot exemption, spec §5)."""
    cands = detect([
        ("H", 100.0, 10, "LH"),
        ("L", 90.0, 14, "EL"),
        ("H", 100.5, 18, "EH"),
        ("L", 90.3, 22, "EL"),
    ], last_bar=24)
    assert TYPE_BOX in types_of(cands)


# ── negative rejections ────────────────────────────────────────────────

def test_box_rejected_when_upper_drifts():
    cands = detect([
        ("H", 100.0, 10, None),
        ("L", 90.0, 14, None),
        ("H", 104.0, 18, "EH"),   # geometry falling, 4 > flat tol 2
        ("L", 90.5, 22, "EL"),
    ], last_bar=24)
    assert TYPE_BOX not in types_of(cands)


def test_box_label_gate_rejects_wrong_low_label():
    """Geometry flat but a non-first low labeled HL — not a box."""
    cands = detect([
        ("H", 100.0, 10, None),
        ("L", 90.0, 14, None),
        ("H", 100.5, 18, "EH"),
        ("L", 91.9, 22, "HL"),
    ], last_bar=24)
    assert TYPE_BOX not in types_of(cands)


def test_sym_triangle_rejected_without_convergence():
    """Falling upper + rising lower classes but only ~22% width shrink."""
    cands = detect([
        ("H", 100.0, 10, None),
        ("L", 80.0, 14, None),
        ("H", 97.9, 18, "LH"),
        ("L", 82.1, 22, "HL"),
    ], last_bar=24)
    assert TYPE_SYM_TRI not in types_of(cands)


def test_parallel_falling_channel_not_a_wedge():
    """LH+LL labels, both boundaries falling — but no convergence."""
    cands = detect([
        ("H", 100.0, 10, None),
        ("L", 90.0, 14, None),
        ("H", 96.0, 18, "LH"),
        ("L", 84.0, 26, "LL"),
        ("H", 88.0, 30, "LH"),
    ], last_bar=32)
    assert TYPE_FALLING_WEDGE not in types_of(cands)


def test_rising_trend_hh_hl_not_a_wedge():
    """Classic bullish HH/HL sequence — parallel boundaries, no wedge."""
    cands = detect([
        ("L", 90.0, 10, None),
        ("H", 100.0, 14, None),
        ("L", 96.0, 18, "HL"),
        ("H", 106.0, 26, "HH"),
        ("L", 100.0, 30, "HL"),
    ], last_bar=32)
    assert TYPE_RISING_WEDGE not in types_of(cands)
    assert TYPE_SYM_TRI not in types_of(cands)
    assert TYPE_FALLING_WEDGE not in types_of(cands)


def test_old_pivots_filtered_by_age():
    cands = detect([
        ("H", 100.0, 10, None),       # way older than age window
        ("L", 90.0, 190, None),
        ("H", 100.5, 194, "EH"),
        ("L", 90.5, 198, "EL"),
    ], last_bar=200, max_pivot_age_bars=50)
    assert cands == []


def test_boundary_crossing_rejected():
    """Upper and lower lines cross inside the window."""
    cands = detect([
        ("H", 100.0, 10, None),
        ("L", 90.0, 14, None),
        ("H", 90.5, 18, "LH"),   # upper now below lower-region
        ("L", 90.2, 22, "EL"),
    ], last_bar=24)
    # upper fit through (10,100)-(18,90.5), lower through (14,90)-(22,90.2):
    # width goes negative inside the span -> everything must be rejected.
    assert cands == []


def test_missing_atr_blocks_detection():
    entries = [
        ("H", 100.0, 10, None), ("L", 90.0, 14, None),
        ("H", 100.5, 18, "EH"), ("L", 90.5, 22, "EL"),
    ]
    cands = detect_candidates(
        entries_to_labeled(entries), [None] * 800, TF_MS, 24,
        DetectConfig(min_pivots=MIN_PIVOTS),
    )
    assert cands == []


# ── ranking ────────────────────────────────────────────────────────────

def mk_cand(type_name, count, fit_err):
    line = st.Line(slope=0.0, intercept=0.0, base_bar=0.0)
    refs = [
        PivotRef(ts=i, bar_index=i, abs_bar=i, price=1.0,
                 is_high=(i % 2 == 0), label=None)
        for i in range(count)
    ]
    return Candidate(
        type=type_name, refs=refs, upper=line, lower=line, atr=1.0,
        upper_class=st.FLAT, lower_class=st.FLAT,
        metrics={"fit_err_atr": fit_err},
    )


def test_rank_more_pivots_first():
    a = mk_cand(TYPE_BOX, 4, 0.1)
    b = mk_cand(TYPE_BOX, 6, 0.9)
    ranked = rank_candidates([a, b], ALL_TYPES)
    assert ranked[0] is b


def test_rank_tiebreak_by_fit_error_then_order():
    a = mk_cand(TYPE_ASC_TRI, 4, 0.2)
    b = mk_cand(TYPE_BOX, 4, 0.2)
    c = mk_cand(TYPE_BOX, 4, 0.1)
    ranked = rank_candidates([a, b, c], ALL_TYPES)
    # same count: lower fit error wins; then selection order (box before asc)
    assert ranked[0] is c
    assert ranked[1] is b
    assert ranked[2] is a
