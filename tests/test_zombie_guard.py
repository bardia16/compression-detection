"""Already-broken (zombie) creation guard — post-last-pivot closes that
already breach a boundary must prevent instance creation entirely.
Regression class: QNT/ETH 2026-09-16 (detected from stale pivots after
price had left; the break was back-attributed to the first post-creation
candle).
"""
from types import SimpleNamespace

from compression_detection import structure as st
from compression_detection.detector import (
    TYPE_BOX, TYPE_DESC_TRI, Candidate, PivotRef,
)
from compression_detection.lifecycle import (
    STATE_DETECTED, update_for_scan,
)
from compression_detection.models import Candle

TF_MS = 60_000
CFG = SimpleNamespace(
    min_pivots={TYPE_BOX: 4, TYPE_DESC_TRI: 4},
    confirm_extra_pivots=1,
    established_extra_pivots=2,
    catchup_max_candles=8,
    breakout_buffer_atr=0.0,
    notify_min_state="confirmed",
)


def ref(bar, price, side, label=None):
    return PivotRef(ts=bar * TF_MS, bar_index=bar, abs_bar=bar,
                    price=price, is_high=(side == "H"), label=label)


BOX = Candidate(
    type=TYPE_BOX,
    refs=[ref(10, 100.0, "H"), ref(14, 90.0, "L"),
          ref(18, 100.5, "H", "EH"), ref(22, 90.5, "L", "EL")],
    upper=st.Line(slope=0.0, intercept=100.0, base_bar=0.0),
    lower=st.Line(slope=0.0, intercept=90.0, base_bar=0.0),
    atr=2.0, upper_class=st.FLAT, lower_class=st.FLAT,
    metrics={"fit_err_atr": 0.0},
)

# descending triangle with a strongly falling upper line: at bar 24 the line
# sits at 93 — far below the last LH level (96) — the BOME class
DESC = Candidate(
    type=TYPE_DESC_TRI,
    refs=[ref(10, 100.0, "H"), ref(14, 90.0, "L"),
          ref(18, 96.0, "H", "LH"), ref(22, 90.2, "L", "EL")],
    upper=st.Line(slope=-0.5, intercept=105.0, base_bar=0.0),
    lower=st.Line(slope=0.0, intercept=90.0, base_bar=0.0),
    atr=2.0, upper_class=st.FALLING, lower_class=st.FLAT,
    metrics={"fit_err_atr": 0.0},
)


def candle(bar, close):
    return Candle(ts=bar * TF_MS, open=close, high=close, low=close,
                  close=close, volume=1.0)


def test_post_pivot_close_inside_creates_instance():
    instances = {}
    candles = [candle(24, 95.0), candle(25, 99.0)]     # inside boundaries
    actions = update_for_scan(instances, "AAA", "15m", [BOX], candles,
                              TF_MS, 1000, CFG)
    assert len(instances) == 1
    assert not any(a.kind == "skip_create" for a in actions)


def test_post_pivot_close_beyond_upper_skips_creation():
    instances = {}
    candles = [candle(24, 101.0)]                      # close above upper
    actions = update_for_scan(instances, "AAA", "15m", [BOX], candles,
                              TF_MS, 1000, CFG)
    assert len(instances) == 0
    skips = [a for a in actions if a.kind == "skip_create"]
    assert len(skips) == 1
    assert skips[0].detail["reason"] == "already_broken"
    assert skips[0].detail["side"] == "up"
    assert skips[0].instance is None


def test_earlier_post_pivot_break_also_skips_even_if_reclaimed():
    """Price breached after the last pivot and came back — the structure
    still had a live breach event; do not create it."""
    instances = {}
    candles = [candle(24, 101.0), candle(25, 95.0)]    # breached then inside
    actions = update_for_scan(instances, "AAA", "15m", [BOX], candles,
                              TF_MS, 1000, CFG)
    assert len(instances) == 0
    assert any(a.kind == "skip_create" for a in actions)


def test_breach_between_pivots_does_not_block_creation():
    """A close beyond the boundary BETWEEN window pivots (pre last-pivot)
    is not a creation blocker — only post-last-pivot closes are checked."""
    instances = {}
    candles = [candle(16, 101.0), candle(24, 95.0)]    # bar16: mid-window
    actions = update_for_scan(instances, "AAA", "15m", [BOX], candles,
                              TF_MS, 1000, CFG)
    assert len(instances) == 1


def test_no_candles_after_last_pivot_creates():
    instances = {}
    actions = update_for_scan(instances, "AAA", "15m", [BOX], [], TF_MS, 1000, CFG)
    assert len(instances) == 1


def test_triangle_line_already_broken_skips_creation():
    """BOME class (user rule 2026-09-16): post-pivot closes through the
    fitted LINE (without touching the pivot levels) mean the triangle is
    already dead — a range from that bar; do not create it."""
    instances = {}
    # close 95: beyond the upper line (93 at bar 24) but below the LH level (96)
    actions = update_for_scan(instances, "AAA", "15m", [DESC], [candle(24, 95.0)],
                              TF_MS, 1000, CFG)
    assert len(instances) == 0
    skips = [a for a in actions if a.kind == "skip_create"]
    assert len(skips) == 1
    assert skips[0].detail["side"] == "up"


def test_triangle_inside_creates():
    instances = {}
    update_for_scan(instances, "AAA", "15m", [DESC], [candle(24, 92.0)],
                    TF_MS, 1000, CFG)
    assert len(instances) == 1


def test_stale_triangle_line_break_dies_on_next_scan():
    """BOME class at instance level (user rule 2026-09-16): a live triangle
    whose post-pivot closes already crossed the line (no level touch) is
    dead on the next scan — retract any pending probe + invalidate."""
    refs = [ref(10, 100.0, "H"), ref(14, 90.0, "L"),
            ref(18, 96.0, "H", "LH"), ref(22, 90.2, "L", "EL")]
    cand = Candidate(
        type=TYPE_DESC_TRI, refs=refs,
        upper=st.Line(slope=-0.5, intercept=105.0, base_bar=0.0),
        lower=st.Line(slope=0.0, intercept=90.0, base_bar=0.0),
        atr=2.0, upper_class=st.FALLING, lower_class=st.FLAT,
        metrics={"fit_err_atr": 0.0},
    )
    instances = {}
    update_for_scan(instances, "AAA", "15m", [cand], [], TF_MS, 1000, CFG)
    inst = list(instances.values())[0]
    inst.probe_msg_id = 555

    actions = update_for_scan(instances, "AAA", "15m", [cand],
                              [candle(24, 95.0)], TF_MS, 1060, CFG)
    assert [a.kind for a in actions] == ["retract", "invalidate"]
    assert actions[1].detail["reason"] == "line_break"


def test_stale_triangle_inside_stays_alive():
    refs = [ref(10, 100.0, "H"), ref(14, 90.0, "L"),
            ref(18, 96.0, "H", "LH"), ref(22, 90.2, "L", "EL")]
    cand = Candidate(
        type=TYPE_DESC_TRI, refs=refs,
        upper=st.Line(slope=-0.5, intercept=105.0, base_bar=0.0),
        lower=st.Line(slope=0.0, intercept=90.0, base_bar=0.0),
        atr=2.0, upper_class=st.FALLING, lower_class=st.FLAT,
        metrics={"fit_err_atr": 0.0},
    )
    instances = {}
    update_for_scan(instances, "AAA", "15m", [cand], [], TF_MS, 1000, CFG)
    inst = list(instances.values())[0]
    actions = update_for_scan(instances, "AAA", "15m", [cand],
                              [candle(24, 91.0)], TF_MS, 1060, CFG)
    assert [a.kind for a in actions] == []
    assert inst.state not in ("invalidated",)


def test_agent_skip_emitted_once_per_type():
    """Multiple zombie window candidates of one type -> a single skip event."""

    big = Candidate(
        type=TYPE_BOX,
        refs=[ref(10, 100.0, "H"), ref(14, 90.0, "L"),
              ref(18, 100.5, "H", "EH"), ref(22, 90.5, "L", "EL"),
              ref(26, 100.2, "H", "EH"), ref(30, 90.3, "L", "EL")],
        upper=st.Line(slope=0.0, intercept=100.0, base_bar=0.0),
        lower=st.Line(slope=0.0, intercept=90.0, base_bar=0.0),
        atr=2.0, upper_class=st.FLAT, lower_class=st.FLAT,
        metrics={"fit_err_atr": 0.0},
    )
    instances = {}
    candles = [candle(32, 101.0)]
    actions = update_for_scan(instances, "AAA", "15m", [big, BOX], candles,
                              TF_MS, 1000, CFG)
    skips = [a for a in actions if a.kind == "skip_create"]
    assert len(skips) == 1


def test_terminal_instance_consumes_its_pivots():
    """After a breakout, sub-windows of the SAME structure are never
    re-created — and no skip noise is emitted either."""
    from compression_detection.lifecycle import STATE_BREAKOUT

    instances = {}
    # scan 1: create
    update_for_scan(instances, "AAA", "15m", [BOX], [candle(24, 95.0)],
                    TF_MS, 1000, CFG)
    inst = list(instances.values())[0]
    inst.state = STATE_BREAKOUT                      # simulate a break

    # sub-window (drops the oldest pivot) still shares 3 pivots
    sub = Candidate(
        type=TYPE_BOX,
        refs=[ref(14, 90.0, "L"), ref(18, 100.5, "H", "EH"),
              ref(22, 90.5, "L", "EL"), ref(26, 100.3, "H", "EH")],
        upper=st.Line(slope=0.0, intercept=100.4, base_bar=0.0),
        lower=st.Line(slope=0.0, intercept=90.4, base_bar=0.0),
        atr=2.0, upper_class=st.FLAT, lower_class=st.FLAT,
        metrics={"fit_err_atr": 0.0},
    )
    actions = update_for_scan(instances, "AAA", "15m", [sub], [candle(28, 96.0)],
                              TF_MS, 2000, CFG)
    assert len(instances) == 1                       # nothing new created
    assert not any(a.kind == "skip_create" for a in actions)
