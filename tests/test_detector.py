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


def candles_for(bars_closes):
    from compression_detection.models import Candle
    return [Candle(ts=b * TF_MS, open=c, high=c, low=c, close=c, volume=1.0)
            for b, c in bars_closes]


def detect_with_candles(entries, last_bar, candles, **cfg_over):
    cfg = DetectConfig(min_pivots=MIN_PIVOTS, **cfg_over)
    return detect_candidates(
        entries_to_labeled(entries), atr_series(), TF_MS, last_bar, cfg,
        candles=candles,
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
    # 4 pivots -> 2 highs + 2 lows: boundary-hit gate satisfied
    assert box.metrics["upper_hits"] == 2 and box.metrics["lower_hits"] == 2
    assert box.metrics["hit_requirement"] is True
    assert box.metrics["hit_boundary"] == "upper"


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
    # 3 highs + 3 lows -> requirement satisfied
    assert cands[0].metrics["upper_hits"] == 3
    assert cands[0].metrics["lower_hits"] == 3
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
    """DISABLED: symmetrical triangles removed from detection 2026-09-20
    (user rule: boxes + ascending/descending triangles only)."""
    pass


def test_falling_wedge_converging():
    """DISABLED: wedges removed from detection 2026-09-16 (buggy)."""
    pass


def test_rising_wedge_converging():
    """DISABLED: wedges removed from detection 2026-09-16 (buggy)."""
    pass


# ── interior close integrity (2026-09-17, BTW case) ────────────────────

ASC_ENTRIES = [
    ("L", 90.0, 10, None),
    ("H", 100.0, 30, None),
    ("L", 94.0, 40, "HL"),
    ("H", 100.5, 44, "EH"),
]


def test_interior_close_breaches_reject_candidate():
    """A boundary 'broken several times' by closes inside the window means
    the structure was never a compression (BTW case: 37 closes below the
    fitted lower line)."""
    lows = [(b, 91.0) for b in range(11, 40)]
    cands = detect_with_candles(ASC_ENTRIES, 46, candles_for(lows))
    assert TYPE_ASC_TRI not in types_of(cands)


def test_interior_close_breaches_within_limit_accepted():
    """Up to 2 fit-noise breaches are tolerated; metrics expose counts."""
    noisy = [(b, 91.5) for b in (30, 35)]
    clean = [(b, 96.0) for b in range(11, 40) if b not in (30, 35)]
    cands = detect_with_candles(ASC_ENTRIES, 46, candles_for(noisy + clean))
    assert TYPE_ASC_TRI in types_of(cands)
    c = [x for x in cands if x.type == TYPE_ASC_TRI][0]
    assert c.metrics["close_breaches_lower"] == 2
    assert c.metrics["close_breaches_upper"] == 0


def test_no_candles_no_close_check():
    """Without candle data the interior check is skipped (pure-pivot mode)."""
    cands = detect(ASC_ENTRIES, 46)
    assert TYPE_ASC_TRI in types_of(cands)


# ── range-mode box (2026-09-17, JTO case) ──────────────────────────────

RANGE_ENTRIES = [
    ("H", 102.0, 10, None),
    ("L", 88.0, 14, None),
    ("H", 97.0, 18, "LH"),      # internal swing — never touches a boundary
    ("L", 95.0, 22, "HL"),      # internal swing
    ("H", 101.5, 26, "EH"),
    ("L", 88.5, 30, "EL"),
]


def test_range_box_rejected_when_flag_off():
    """Strict model: interleaved internal swings break the EH/EL chain."""
    cands = detect_with_candles(RANGE_ENTRIES, 32,
                                candles_for([(b, 95.0) for b in range(10, 31)]))
    assert TYPE_BOX not in types_of(cands)


def test_range_box_detected_when_flag_on():
    """Touch clusters + overlap → textbook range with internal swings."""
    cands = detect_with_candles(
        RANGE_ENTRIES, 32,
        candles_for([(b, 95.0) for b in range(10, 31)]),
        box_range_mode=True,
    )
    boxes = [c for c in cands if c.type == TYPE_BOX]
    assert boxes
    b = boxes[0]
    assert b.metrics["mode"] == "range"
    assert b.metrics["upper_touches"] == 2 and b.metrics["lower_touches"] == 2
    assert b.metrics["upper_at_last_bar"] == 101.5
    assert b.metrics["lower_at_last_bar"] == 88.5


def test_range_box_needs_both_sides_tested_in_common_stretch():
    """Ceiling taps long before floor taps (break + dead-cat) → reject."""
    entries = [
        ("H", 102.0, 10, None), ("L", 96.0, 20, None),
        ("H", 101.0, 24, "LH"), ("L", 88.0, 60, "LL"),
        ("H", 90.0, 64, "LH"), ("L", 88.5, 70, "EL"),
    ]
    cands = detect_with_candles(
        entries, 72, candles_for([(b, 95.0) for b in range(10, 71)]),
        box_range_mode=True,
    )
    assert TYPE_BOX not in types_of(cands)


def test_range_box_needs_two_touches_per_side():
    entries = [
        ("H", 102.0, 10, None), ("H", 97.0, 14, "LH"),
        ("L", 88.0, 20, None), ("L", 95.0, 24, "HL"),
    ]
    cands = detect_with_candles(
        entries, 26, candles_for([(b, 95.0) for b in range(10, 25)]),
        box_range_mode=True,
    )
    assert TYPE_BOX not in types_of(cands)


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
    # same count: lower fit error wins; then selection order (triangles
    # before box — user rule 2026-09-19)
    assert ranked[0] is c
    assert ranked[1] is a
    assert ranked[2] is b
