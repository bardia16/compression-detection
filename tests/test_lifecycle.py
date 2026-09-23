"""Lifecycle tests — state machine, continuation, verdicts, invalidation,
breakout posts, anti-respawn, idempotency, best-only notify dedupe.
"""
from types import SimpleNamespace

import pytest

from compression_detection import structure as st
from compression_detection.detector import (
    TYPE_ASC_TRI, TYPE_BOX, TYPE_DESC_TRI, TYPE_FALLING_WEDGE, TYPE_SYM_TRI,
    TYPE_SPECS, Candidate, PivotRef,
)
from compression_detection.lifecycle import (
    STATE_BREAKOUT, STATE_COMPRESSING, STATE_CONFIRMED, STATE_DETECTED,
    STATE_INVALIDATED, Instance, best_per_coin_tf, create_instance,
    evaluate_closed_candles, find_continuation, level_for,
    retire_out_of_universe, state_for_level, update_for_scan,
)
from compression_detection.models import Candle

TF_MS = 60_000
MIN_PIVOTS = {
    TYPE_BOX: 4, TYPE_ASC_TRI: 4, TYPE_DESC_TRI: 4, TYPE_SYM_TRI: 4,
    TYPE_FALLING_WEDGE: 5,
}
CFG = SimpleNamespace(
    min_pivots=MIN_PIVOTS,
    confirm_extra_pivots=1,
    established_extra_pivots=2,
    min_boundary_hits=2,
    catchup_max_candles=8,
    breakout_buffer_atr=0.0,
    notify_min_state="confirmed",
    det=SimpleNamespace(selection_order=(
        TYPE_ASC_TRI, TYPE_DESC_TRI, TYPE_BOX)),
)


def ref(bar, price, side, label=None):
    return PivotRef(ts=bar * TF_MS, bar_index=bar, abs_bar=bar,
                    price=price, is_high=(side == "H"), label=label)


def mk_cand(type_name, refs, up=(0.0, 100.0), lo=(0.0, 90.0), atr=2.0,
            up_cls=st.FLAT, lo_cls=st.FLAT):
    uh = sum(1 for r in refs if r.is_high)
    lh = sum(1 for r in refs if not r.is_high)
    meta = {"fit_err_atr": 0.1, "pivot_count": len(refs),
            "upper_hits": uh, "lower_hits": lh}
    return Candidate(
        type=type_name, refs=refs,
        upper=st.Line(slope=up[0], intercept=up[1], base_bar=0.0),
        lower=st.Line(slope=lo[0], intercept=lo[1], base_bar=0.0),
        atr=atr, upper_class=up_cls, lower_class=lo_cls, metrics=meta,
    )


def box_refs(n=4, end=30):
    """n box pivots ending at bar `end`; returns refs + candidate."""
    base = [
        ref(10, 100.0, "H"), ref(14, 90.0, "L"),
        ref(18, 100.5, "H", "EH"), ref(22, 90.5, "L", "EL"),
        ref(26, 100.2, "H", "EH"), ref(30, 90.3, "L", "EL"),
    ]
    refs = base[:n]
    return refs, mk_cand(TYPE_BOX, refs)


def candle(bar, close, tf_ms=TF_MS):
    return Candle(ts=bar * tf_ms, open=close, high=close, low=close,
                  close=close, volume=1.0)


def mk_instance(cand, symbol="AAA", tf="15m", now_s=1000):
    return create_instance(cand, symbol, tf, now_s, CFG)


def mark_notified(insts, actions):
    """Simulate engine: sent compression notifications get flagged."""
    for a in actions:
        if a.kind == "compression_notify":
            a.instance.notified["compression"] = True


# ── levels ─────────────────────────────────────────────────────────────

def test_level_for_thresholds():
    assert level_for(4, TYPE_BOX, CFG) == "detected"
    assert level_for(5, TYPE_BOX, CFG) == "confirmed"
    assert level_for(6, TYPE_BOX, CFG) == "established"
    assert level_for(5, TYPE_FALLING_WEDGE, CFG) == "detected"
    assert level_for(6, TYPE_FALLING_WEDGE, CFG) == "confirmed"
    assert level_for(7, TYPE_FALLING_WEDGE, CFG) == "established"
    assert state_for_level("established") == STATE_COMPRESSING


# ── creation + notify gate ─────────────────────────────────────────────

def test_confirmed_creation_notifies_once():
    refs, cand = box_refs(n=5)
    instances = {}
    actions = update_for_scan(instances, "AAA", "15m", [cand], [], TF_MS, 1000, CFG)
    kinds = [a.kind for a in actions]
    assert "compression_notify" in kinds
    inst = list(instances.values())[0]
    assert inst.state == STATE_CONFIRMED
    mark_notified(instances, actions)

    # same candidate again -> no duplicate notify (flag), no duplicate instance
    actions2 = update_for_scan(instances, "AAA", "15m", [cand], [], TF_MS, 1060, CFG)
    assert [a.kind for a in actions2] == []
    assert len(instances) == 1


def test_detected_only_then_upgrade_notifies():
    """4-pivot box: gate passes (2 highs + 2 lows) → straight to
    CONFIRMED + notification.  Upgrade to COMPRESSING at 6 pivots."""
    refs4, cand4 = box_refs(n=4)
    instances = {}
    actions = update_for_scan(instances, "AAA", "15m", [cand4], [], TF_MS, 1000, CFG)
    kinds = [a.kind for a in actions]
    assert "compression_notify" in kinds
    mark_notified(instances, actions)
    inst = list(instances.values())[0]
    assert inst.state == STATE_CONFIRMED
    assert inst.metrics["upper_hits"] == 2 and inst.metrics["lower_hits"] == 2

    # 5-pivot scan: still confirmed, already notified — no new actions
    refs5, cand5 = box_refs(n=5)
    actions2 = update_for_scan(instances, "AAA", "15m", [cand5], [], TF_MS, 1060, CFG)
    assert [a.kind for a in actions2 if a.kind != "skip_create"] == []

    # 6-pivot scan: compressing → upgrade
    refs6, cand6 = box_refs(n=6)
    actions3 = update_for_scan(instances, "AAA", "15m", [cand6], [], TF_MS, 1120, CFG)
    assert "upgrade" in [a.kind for a in actions3]
    assert inst.state == STATE_COMPRESSING
    assert inst.metrics["upper_hits"] == 3


def test_established_upgrades_to_compressing_single_notify():
    refs5, cand5 = box_refs(n=5)
    instances = {}
    actions = update_for_scan(instances, "AAA", "15m", [cand5], [], TF_MS, 1000, CFG)
    mark_notified(instances, actions)

    refs6, cand6 = box_refs(n=6)
    actions2 = update_for_scan(instances, "AAA", "15m", [cand6], [], TF_MS, 1060, CFG)
    kinds = [a.kind for a in actions2]
    assert "upgrade" in kinds
    assert "compression_notify" not in kinds   # already notified once
    inst = list(instances.values())[0]
    assert inst.state == STATE_COMPRESSING


# ── continuation matching ──────────────────────────────────────────────

def test_find_continuation_needs_two_shared_timestamps():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)

    shared = [ref(18, 100.5, "H", "EH"), ref(22, 90.5, "L", "EL"),
              ref(26, 100.2, "H", "EH"), ref(30, 90.3, "L", "EL")]
    c_ok = mk_cand(TYPE_BOX, shared)
    assert find_continuation(inst, [c_ok]) is c_ok

    only_one = [ref(22, 90.5, "L", "EL"), ref(26, 100.2, "H", "EH"),
                ref(30, 90.3, "L", "EL"), ref(34, 100.1, "H", "EH")]
    # shares exactly one timestamp (bar 22) -> not a continuation
    c_far = mk_cand(TYPE_BOX, only_one)
    assert find_continuation(inst, [c_far]) is None

    other_type = mk_cand(TYPE_SYM_TRI, shared)
    assert find_continuation(inst, [other_type]) is None


def test_continuation_extends_instance():
    refs4, cand4 = box_refs(n=4)
    instances = {}
    update_for_scan(instances, "AAA", "15m", [cand4], [], TF_MS, 1000, CFG)
    inst = list(instances.values())[0]
    ts_before = inst.pivot_tss

    refs5, cand5 = box_refs(n=5)
    update_for_scan(instances, "AAA", "15m", [cand5], [], TF_MS, 1060, CFG)
    assert len(instances) == 1
    assert inst.pivot_count == 5
    assert ts_before - inst.pivot_tss  == set()   # superset keeps all old tss


def test_invalidation_when_candidate_disappears():
    refs, cand = box_refs(n=4)
    instances = {}
    update_for_scan(instances, "AAA", "15m", [cand], [], TF_MS, 1000, CFG)
    inst = list(instances.values())[0]

    actions = update_for_scan(instances, "AAA", "15m", [], [], TF_MS, 1060, CFG)
    assert [a.kind for a in actions] == ["invalidate"]
    assert inst.state == STATE_INVALIDATED


# ── close verdicts / breakouts ─────────────────────────────────────────

def test_close_beyond_upper_posts_breakout():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    actions = evaluate_closed_candles(
        inst, [candle(31, 101.0)], TF_MS, 1000, CFG)
    assert [a.kind for a in actions] == ["breakout_post"]
    assert actions[0].detail["side"] == "up"
    assert inst.state == STATE_BREAKOUT
    assert inst.notified["breakout"] is True


def test_close_beyond_lower_posts_breakout_down():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    actions = evaluate_closed_candles(
        inst, [candle(31, 89.0)], TF_MS, 1000, CFG)
    assert [a.kind for a in actions] == ["breakout_post"]
    assert actions[0].detail["side"] == "down"


def test_wick_only_poke_no_breakout():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    # candle pokes above by wick but CLOSES inside (close 99.5)
    c = Candle(ts=31 * TF_MS, open=99.0, high=102.0, low=98.0,
               close=99.5, volume=1.0)
    actions = evaluate_closed_candles(inst, [c], TF_MS, 1000, CFG)
    assert actions == []
    assert inst.state != STATE_BREAKOUT


def test_probe_verdict_keep_on_confirmed_close():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    inst.probe_msg_id = 555
    inst.probe_candle_ms = 31 * TF_MS
    inst.probe_side = "up"

    actions = evaluate_closed_candles(inst, [candle(31, 101.0)], TF_MS, 1000, CFG)
    assert [a.kind for a in actions] == ["breakout_keep"]
    assert actions[0].detail["msg_id"] == 555
    assert inst.state == STATE_BREAKOUT
    assert inst.probe_msg_id is None


def test_probe_verdict_retract_on_failed_close():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    inst.probe_msg_id = 555
    inst.probe_candle_ms = 31 * TF_MS
    inst.probe_side = "up"

    actions = evaluate_closed_candles(inst, [candle(31, 99.0)], TF_MS, 1000, CFG)
    assert [a.kind for a in actions] == ["retract"]
    assert actions[0].detail["msg_id"] == 555
    assert inst.state == STATE_CONFIRMED           # still active (4-pivot instance)
    assert inst.probe_msg_id is None


def test_probe_failed_close_but_opposite_side_broke():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    inst.probe_msg_id = 555
    inst.probe_candle_ms = 31 * TF_MS
    inst.probe_side = "up"

    actions = evaluate_closed_candles(inst, [candle(31, 89.0)], TF_MS, 1000, CFG)
    kinds = [a.kind for a in actions]
    assert kinds == ["retract", "breakout_post"]
    assert actions[1].detail["side"] == "down"
    assert inst.state == STATE_BREAKOUT


def test_stale_probe_gets_retracted():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    inst.probe_msg_id = 555
    inst.probe_candle_ms = 5 * TF_MS       # way older than the processed candle

    actions = evaluate_closed_candles(inst, [candle(31, 99.0)], TF_MS, 1000, CFG)
    assert [a.kind for a in actions] == ["retract"]
    assert actions[0].detail.get("stale") is True


def test_verdict_uses_pre_update_boundary():
    """Close beyond the OLD pivot levels fires even when the same scan brings
    a candidate that would shift the boundary (real-time semantics)."""
    refs, cand = box_refs(n=4)
    instances = {}
    update_for_scan(instances, "AAA", "15m", [cand], [], TF_MS, 1000, CFG)
    inst = list(instances.values())[0]

    # fresh candidate with a shifted upper boundary (105) — would NOT contain 101
    shifted = mk_cand(TYPE_BOX, [
        ref(26, 100.2, "H", "EH"), ref(30, 90.3, "L", "EL"),
        ref(34, 105.0, "H", "EH"), ref(38, 90.6, "L", "EL"),
    ], up=(0.0, 105.0))
    actions = update_for_scan(
        instances, "AAA", "15m", [shifted], [candle(31, 101.0)], TF_MS, 1060, CFG)
    assert any(a.kind == "breakout_post" and a.detail["side"] == "up" for a in actions)
    assert inst.state == STATE_BREAKOUT


def test_catchup_cap_skips_old_breakouts():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    candles = [candle(31 + i, 101.0 if i == 0 else 99.0) for i in range(10)]
    actions = evaluate_closed_candles(inst, candles, TF_MS, 1000, CFG)
    # breakout candle (first) fell outside the 8 newest -> skipped
    assert actions == []
    assert inst.state != STATE_BREAKOUT
    assert inst.last_eval_ts == candles[-1].ts


def test_idempotent_double_evaluation():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    c = candle(31, 101.0)
    a1 = evaluate_closed_candles(inst, [c], TF_MS, 1000, CFG)
    assert [a.kind for a in a1] == ["breakout_post"]
    # same call again — candle already evaluated (and instance terminal)
    a2 = evaluate_closed_candles(inst, [c], TF_MS, 1060, CFG)
    assert a2 == []


# ── breakout levels = last pivot in that direction (2026-09-16) ────────

def test_breakout_level_is_last_pivot_price():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    actions = evaluate_closed_candles(inst, [candle(31, 101.0)], TF_MS, 1000, CFG)
    assert actions[0].detail["level"] == 100.5        # last EH, not line (100.0)

    refs2, cand2 = box_refs(n=4)
    inst2 = mk_instance(cand2)
    actions2 = evaluate_closed_candles(inst2, [candle(31, 89.0)], TF_MS, 1000, CFG)
    assert actions2[0].detail["level"] == 90.5        # last EL


def test_triangle_line_break_kills_pattern_before_level():
    """Descending upper boundary: a close between the falling LINE and the
    last LH level is NOT a breakout (level = pivot price, user rule) — it
    kills the triangle: line broken → from that bar it is a range, and the
    pattern doesn't hold anymore (user rule 2026-09-16)."""
    refs = [ref(10, 100.0, "H"), ref(14, 90.0, "L"),
            ref(18, 96.0, "H", "LH"), ref(22, 90.2, "L", "EL")]
    cand = mk_cand(TYPE_DESC_TRI, refs, up=(-0.5, 105.0), lo=(0.0, 90.0))
    # line at bar 31 = 105 - 0.5*31 = 89.5; last LH level = 96.0
    inst = mk_instance(cand)
    actions = evaluate_closed_candles(inst, [candle(31, 95.0)], TF_MS, 1000, CFG)
    assert [a.kind for a in actions] == ["invalidate"]
    assert actions[0].detail["reason"] == "line_break"
    assert actions[0].detail["side"] == "up"
    assert inst.state == STATE_INVALIDATED

    # a close above BOTH the level (96) and the line = the breakout, level 96
    inst2 = mk_instance(mk_cand(TYPE_DESC_TRI, refs, up=(-0.5, 105.0),
                                lo=(0.0, 90.0)))
    actions2 = evaluate_closed_candles(inst2, [candle(31, 97.0)], TF_MS, 1000, CFG)
    assert [a.kind for a in actions2] == ["breakout_post"]
    assert actions2[0].detail["level"] == 96.0        # last LH pivot price


def test_wedge_breakout_level_is_the_line():
    """DISABLED: wedges removed from detection 2026-09-16."""
    pass


def test_probe_level_is_last_pivot_price():
    refs, cand = box_refs(n=4)
    inst = mk_instance(cand)
    assert inst.beyond_side(100.4, 0.0, 24) is None   # between line (100) and EH (100.5)
    assert inst.beyond_side(100.6, 0.0, 24) == "up"
    assert inst.beyond_side(90.6, 0.0, 24) is None    # between EL (90.5) and line (90)
    assert inst.beyond_side(90.3, 0.0, 24) == "down"


def test_probe_level_wedge_uses_line():
    """DISABLED: wedges removed from detection 2026-09-16."""
    pass


# ── boundary-hit requirement (2026-09-16) ──────────────────────────────

def test_state_for_candidate_hit_gate():
    from compression_detection.lifecycle import state_for_candidate
    # hits satisfied on either boundary -> level decides
    assert state_for_candidate("confirmed", {"upper_hits": 2, "lower_hits": 0}, CFG) == STATE_CONFIRMED
    assert state_for_candidate("established", {"upper_hits": 0, "lower_hits": 3}, CFG) == STATE_COMPRESSING
    # below requirement -> stays DETECTED even at confirmed/established levels
    assert state_for_candidate("confirmed", {"upper_hits": 1, "lower_hits": 1}, CFG) == STATE_DETECTED
    assert state_for_candidate("established", {"upper_hits": 1, "lower_hits": 0}, CFG) == STATE_DETECTED
    # detected level with gate passing -> promoted to CONFIRMED
    assert state_for_candidate("detected", {"upper_hits": 5, "lower_hits": 5}, CFG) == STATE_CONFIRMED
    # detected level with gate failing -> stays DETECTED
    assert state_for_candidate("detected", {"upper_hits": 0, "lower_hits": 0}, CFG) == STATE_DETECTED


def test_hit_requirement_exposed_on_instance():
    refs, cand = box_refs(n=6)
    inst = mk_instance(cand)
    assert inst.metrics["upper_hits"] == 3 and inst.metrics["lower_hits"] == 3
    assert inst.state == STATE_COMPRESSING            # 6 pivots + gate satisfied


def test_trend_segments_start_at_each_sides_first_pivot():
    """Chart segments: each boundary drawn from its OWN first pivot (no left
    overhang); the FLAT side of a descending triangle is a strictly
    HORIZONTAL line at the last low pivot — no sloped bases (user rule
    2026-09-16, LTC case)."""
    refs = [ref(10, 100.0, "H"), ref(14, 90.0, "L"),
            ref(18, 96.0, "H", "LH"), ref(22, 90.2, "L", "EL")]
    cand = mk_cand(TYPE_DESC_TRI, refs, up=(-0.5, 105.0), lo=(0.0, 90.0),
                   up_cls=st.FALLING, lo_cls=st.FLAT)
    inst = mk_instance(cand)
    segs = inst.trend_segments(TF_MS, 31 * TF_MS)
    assert len(segs) == 2
    # upper (falling, first high at bar 10): 105 - 0.5*10 = 100.0 -> 89.5
    assert segs[0] == (10 * TF_MS, 100.0, 31 * TF_MS, 89.5)
    # lower (FLAT): horizontal at the LAST low pivot (90.2), from bar 14
    assert segs[1] == (14 * TF_MS, 90.2, 31 * TF_MS, 90.2)
    assert segs[1][1] == segs[1][3]                     # strictly horizontal


# ── creation guards ────────────────────────────────────────────────────

def test_anti_respawn_same_anchor():
    refs, cand = box_refs(n=4)
    instances = {}
    update_for_scan(instances, "AAA", "15m", [cand], [], TF_MS, 1000, CFG)
    inst = list(instances.values())[0]
    inst.state = STATE_BREAKOUT

    # same anchor candidate -> no new instance
    update_for_scan(instances, "AAA", "15m", [cand], [], TF_MS, 1060, CFG)
    assert len(instances) == 1


def test_new_structure_on_new_anchor_allowed():
    refs, cand = box_refs(n=4)
    instances = {}
    update_for_scan(instances, "AAA", "15m", [cand], [], TF_MS, 1000, CFG)
    inst = list(instances.values())[0]
    inst.state = STATE_INVALIDATED

    refs2, cand2 = box_refs(n=4, end=60)
    cand2 = mk_cand(TYPE_BOX, [
        ref(50, 100.0, "H"), ref(54, 90.0, "L"),
        ref(58, 100.4, "H", "EH"), ref(60, 90.4, "L", "EL"),
    ])
    update_for_scan(instances, "AAA", "15m", [cand2], [], TF_MS, 1200, CFG)
    assert len(instances) == 2


def test_best_per_coin_tf_keeps_best_notify():
    refs6, cand6 = box_refs(n=6)
    asc = mk_cand(TYPE_ASC_TRI, [
        ref(10, 90.0, "L"), ref(14, 100.0, "H"),
        ref(18, 94.0, "L", "HL"), ref(22, 100.5, "H", "EH"),
    ])
    i1 = mk_instance(cand6, symbol="AAA")          # box, 6 pivots
    i2 = mk_instance(asc, symbol="AAA")            # triangle, 4 pivots
    i3 = mk_instance(asc, symbol="BBB")
    from compression_detection.lifecycle import Action
    acts = [Action("compression_notify", i1), Action("compression_notify", i2),
            Action("compression_notify", i3), Action("retract", i1)]
    out = best_per_coin_tf(acts, CFG.det.selection_order)
    notifies = [a for a in out if a.kind == "compression_notify"]
    assert len(notifies) == 2                     # AAA best + BBB
    assert i2 in [a.instance for a in notifies]   # triangle beats bigger box
    assert i1 not in [a.instance for a in notifies]
    assert any(a.kind == "retract" for a in out)


def test_best_per_coin_tf_within_type_prefers_more_pivots():
    refs6, cand6 = box_refs(n=6)
    refs4, cand4 = box_refs(n=4)
    from compression_detection.lifecycle import Action
    i6 = mk_instance(cand6, symbol="AAA")
    i4 = mk_instance(cand4, symbol="AAA")
    out = best_per_coin_tf([Action("compression_notify", i4),
                            Action("compression_notify", i6)],
                           CFG.det.selection_order)
    notifies = [a for a in out if a.kind == "compression_notify"]
    assert len(notifies) == 1 and notifies[0].instance is i6


def test_notify_holds_lower_priority_while_higher_surfaces():
    from compression_detection.lifecycle import _maybe_notify
    _, boxc = box_refs(n=4)
    ascc = mk_cand(TYPE_ASC_TRI, [
        ref(10, 90.0, "L"), ref(14, 100.0, "H"),
        ref(18, 94.0, "L", "HL"), ref(22, 100.5, "H", "EH"),
    ])
    box = mk_instance(boxc)
    tri = mk_instance(ascc)
    box.state = STATE_CONFIRMED
    tri.state = STATE_CONFIRMED

    # higher type not yet notified → box may notify
    kinds = [a.kind for a in _maybe_notify(box, 2000, CFG, [box, tri])]
    assert kinds == ["compression_notify"]

    # higher type notified & active → box is held
    tri.notified["compression"] = True
    assert _maybe_notify(box, 2000, CFG, [box, tri]) == []

    # higher type ended → box may notify again
    tri.state = STATE_INVALIDATED
    assert [a.kind for a in _maybe_notify(box, 2000, CFG, [box, tri])] == \
        ["compression_notify"]

    # the higher type itself is never held
    tri.state = STATE_CONFIRMED
    tri.notified["compression"] = False
    assert [a.kind for a in _maybe_notify(tri, 2000, CFG, [box, tri])] == \
        ["compression_notify"]


# ── serialization ──────────────────────────────────────────────────────

def test_instance_roundtrip():
    refs, cand = box_refs(n=5)
    inst = mk_instance(cand)
    inst.probe_msg_id = 42
    inst.probe_candle_ms = 31 * TF_MS
    inst.probe_side = "up"
    inst.dm_msg_ids = [101]
    inst.probe_msg_id_dm = 55
    d = inst.to_dict()
    inst2 = Instance.from_dict(d)
    assert inst2.id == inst.id
    assert inst2.pivot_tss == inst.pivot_tss
    assert inst2.upper_at(20) == pytest.approx(inst.upper_at(20))
    assert inst2.probe_msg_id == 42
    assert inst2.dm_msg_ids == [101]
    assert inst2.probe_msg_id_dm == 55
    assert inst2.to_dict() == d


# ── out-of-universe TTL (user rule 2026-09-23, CC case) ────────────────

def test_out_of_universe_ttl_retires_after_grace():
    refs, cand = box_refs(n=5)
    inst = mk_instance(cand, symbol="AAA", tf="15m", now_s=1000)
    instances = {inst.id: inst}
    # first absent scan: timer starts, instance stays active
    out = retire_out_of_universe(instances, {"BBBUSDT"}, 10_000, 8 * 3600)
    assert out == [] and inst.state in (STATE_DETECTED, STATE_CONFIRMED, STATE_COMPRESSING)
    assert inst.out_of_universe_since == 10_000
    # still inside the grace period
    out = retire_out_of_universe(instances, {"BBBUSDT"}, 10_000 + 8 * 3600 - 1, 8 * 3600)
    assert out == [] and inst.state != STATE_INVALIDATED
    # TTL elapsed -> retired
    out = retire_out_of_universe(instances, {"BBBUSDT"}, 10_000 + 8 * 3600, 8 * 3600)
    assert out == [inst] and inst.state == STATE_INVALIDATED


def test_out_of_universe_timer_clears_when_coin_returns():
    refs, cand = box_refs(n=5)
    inst = mk_instance(cand, symbol="AAA", tf="15m", now_s=1000)
    instances = {inst.id: inst}
    retire_out_of_universe(instances, set(), 10_000, 8 * 3600)
    assert inst.out_of_universe_since == 10_000
    retire_out_of_universe(instances, {"AAA"}, 10_600, 8 * 3600)
    assert inst.out_of_universe_since is None
    assert inst.state != STATE_INVALIDATED


def test_out_of_universe_terminal_instances_untouched():
    refs, cand = box_refs(n=5)
    inst = mk_instance(cand, symbol="AAA", tf="15m", now_s=1000)
    inst.state = STATE_INVALIDATED
    instances = {inst.id: inst}
    out = retire_out_of_universe(instances, set(), 10_000, 8 * 3600)
    assert out == [] and inst.out_of_universe_since is None
