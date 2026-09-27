"""Potential-break confirmation pattern (user rule 2026-09-27).

  LONG : low  → EH (box high)  → higher low, confirmed
  SHORT: high → EL (box low)   → lower high, confirmed

Detection is exercised on synthetic zigzag-friendly paths (each leg ≈ 10
ATR so coef × ATR7 confirms unambiguously), the engine pass on real
Config, and the dispatch with a fake Telegram + chart stub.
"""
import asyncio
import sys

import pytest

sys.path.insert(0, "/root/compression-detection")

import compression_detection.engine as eng_mod  # noqa: E402
from compression_detection import notifier as nt  # noqa: E402
from compression_detection.config import Config  # noqa: E402
from compression_detection import structure as st  # noqa: E402
from compression_detection.lifecycle import (  # noqa: E402
    STATE_BREAKOUT, STATE_CONFIRMED, Instance,
)
from compression_detection.models import Candle  # noqa: E402
from compression_detection.potential_break import (  # noqa: E402
    LOWER_TF, find_potential, potential_lines,
)

CFG = Config.load()
T0 = 1_700_000_000_000
STEP = 60_000
NOW_S = T0 // 1000 + 3600          # inside the 12h freshness window
STALE_S = NOW_S + 13 * 3600         # beyond it (pattern is old news)


# ── fixtures ───────────────────────────────────────────────────────────
def _line(a: float, b: float, n: int):
    return [a + (b - a) * i / n for i in range(1, n + 1)]


def mk(path, spread=0.02, t0=T0):
    return [Candle(ts=t0 + i * STEP, open=c, high=c + spread, low=c - spread,
                   close=c, volume=1.0) for i, c in enumerate(path)]


def long_path(mid_is_eh: bool = True, second_higher: bool = True):
    """H0 → L1 → H1 → L2 → rally. H1 is EH when it sits back on H0; L2 is
    the higher low when it stays above L1."""
    hi0 = 110.1
    hi1 = 110.0 if mid_is_eh else 115.0      # HH (not EH) when False
    lo2 = 105.0 if second_higher else 95.0   # LL (not HL) when False
    return (_line(100, hi0, 12) + _line(hi0, 100, 12) +
            _line(100, hi1, 12) + _line(hi1, lo2, 8) +
            _line(lo2, 120, 12))


def short_path(second_lower: bool = True):
    """L0 → H1 → L1(EL) → H2 → decline."""
    hi2 = 108.0 if second_lower else 118.0   # HH (not LH) when False
    return (_line(110, 100, 12) + _line(100, 110, 12) +
            _line(110, 100.1, 12) + _line(100.1, hi2, 8) +
            _line(hi2, 95, 12))


def _inst(symbol="SUIUSDT", tf="1h", type_="box", state=STATE_CONFIRMED,
          anchor=T0, msg_ids=None, notified=None):
    line = st.Line(slope=0.0, intercept=100.0, base_bar=0)
    return Instance(
        id=f"{symbol}|{tf}|{type_}|{anchor}", symbol=symbol, tf=tf,
        type=type_, state=state, level="confirmed", created_ts=T0,
        anchor_ts=anchor, last_scan_ts=T0, last_eval_ts=T0,
        pivots=[{"ts": T0, "side": "L", "price": 100.0, "label": "EL"},
                {"ts": T0 + STEP, "side": "H", "price": 110.0, "label": "EH"}],
        upper_line=line, lower_line=line, atr=1.0,
        upper_class="flat", lower_class="flat", metrics={},
        notified=notified or {}, msg_ids=msg_ids or [],
    )


class FakeTG:
    def __init__(self):
        self.photos = []      # (caption, img, reply_to, chat_id)
        self.groups = []      # (caption, [imgs], reply_to, chat_id)
        self.texts = []       # (text, reply_to, chat_id)
        self._n = 100

    async def post(self, text, reply_to_id=None, chat_id=None):
        self._n += 1
        self.texts.append((text, reply_to_id, chat_id))
        return self._n

    async def post_photo(self, caption, img, reply_to_id=None, chat_id=None):
        self._n += 1
        self.photos.append((caption, img, reply_to_id, chat_id))
        return self._n

    async def post_media_group(self, caption, images, reply_to_id=None,
                               chat_id=None):
        self._n += 1
        self.groups.append((caption, list(images), reply_to_id, chat_id))
        return self._n

    async def delete(self, mid, why=""):
        return True


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = Config.load()
    c.state_path = tmp_path / "structure_state.json"
    monkeypatch.setattr(eng_mod, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(eng_mod, "AUDIT_DIR", tmp_path / "audit")
    return c


# ── detection: positive ────────────────────────────────────────────────
def test_long_pattern_detected():
    hit = find_potential(mk(long_path()), "long", CFG)
    assert hit is not None
    assert hit["first_price"] == pytest.approx(100.0)   # the low
    assert hit["mid_price"] == pytest.approx(110.0)     # EH (box high)
    assert hit["mid_label"] == "EH"
    assert hit["second_price"] == pytest.approx(105.0)  # higher low
    assert hit["second_price"] > hit["first_price"]
    assert hit["confirm_ts"] >= hit["second_ts"]        # confirmed on a later bar


def test_short_pattern_detected():
    hit = find_potential(mk(short_path()), "short", CFG)
    assert hit is not None
    assert hit["mid_price"] == pytest.approx(100.1)     # EL (box low)
    assert hit["mid_label"] == "EL"
    assert hit["second_price"] < hit["first_price"]     # lower high
    assert hit["confirm_ts"] >= hit["second_ts"]


def test_confirm_ts_is_the_replay_bar():
    """The alert fires on the candle that closes beyond coef × ATR7."""
    from compression_detection.potential_break import _confirm_map
    candles = mk(long_path())
    cmap = _confirm_map(candles, CFG)
    hit = find_potential(candles, "long", CFG)
    assert cmap[hit["second_ts"]] == hit["confirm_ts"]
    # ...and that bar is strictly after the pivot bar
    assert hit["confirm_ts"] > hit["second_ts"]


# ── detection: negative ────────────────────────────────────────────────
def test_rejects_mid_that_is_a_higher_high():
    """mid must be EH — a fresh HH is not 'to the box high'."""
    assert find_potential(mk(long_path(mid_is_eh=False)), "long", CFG) is None


def test_rejects_second_swing_that_is_lower():
    """the second swing must be a HIGHER low."""
    assert find_potential(mk(long_path(second_higher=False)), "long", CFG) is None


def test_rejects_short_when_second_higher():
    assert find_potential(mk(short_path(second_lower=False)), "short", CFG) is None


def test_anchor_filter_drops_pre_box_pattern():
    """A pattern that finished before the compression existed is not this
    box's signal."""
    candles = mk(long_path())
    hit = find_potential(candles, "long", CFG, anchor_ts=T0)
    assert hit is not None
    # anchor pushed past the whole pattern → nothing qualifies
    assert find_potential(candles, "long", CFG,
                          anchor_ts=hit["second_ts"] + 10 * STEP) is None


def test_too_few_candles_returns_none():
    assert find_potential(mk(_line(100, 110, 4)), "long", CFG) is None


def test_bad_side_raises():
    with pytest.raises(ValueError):
        find_potential(mk(long_path()), "sideways", CFG)


# ── chart lines ────────────────────────────────────────────────────────
def test_potential_lines_long_adjusts_the_high():
    hit = find_potential(mk(long_path()), "long", CFG)
    # box low stays, the high becomes the last low-TF high pivot (the mid)
    assert potential_lines(hit, 100.0, 110.0) == [100.0, hit["mid_price"]]


def test_potential_lines_short_adjusts_the_low():
    hit = find_potential(mk(short_path()), "short", CFG)
    # box high stays, the low becomes the last low-TF low pivot (the mid)
    assert potential_lines(hit, 100.0, 110.0) == [hit["mid_price"], 110.0]


def test_lower_tf_map():
    assert LOWER_TF == {"4h": "1h", "1h": "15m", "15m": None}


# ── caption ────────────────────────────────────────────────────────────
def test_fmt_potential_long():
    inst = _inst()
    hit = find_potential(mk(long_path()), "long", CFG)
    cap = nt.fmt_potential(inst, "long", "15m", hit)
    lines = cap.split("\n")
    assert lines[0] == ("🟢 <b>SUIUSDT</b> — 1H Compression  ·  "
                        "Long Potential Break")
    assert lines[1] == "🎯 15m pattern · 100 → EH 110 → higher low 105 ✓"


def test_fmt_potential_short():
    inst = _inst()
    hit = find_potential(mk(short_path()), "short", CFG)
    cap = nt.fmt_potential(inst, "short", "1h", hit)
    lines = cap.split("\n")
    assert lines[0] == ("🔴 <b>SUIUSDT</b> — 1H Compression  ·  "
                        "Short Potential Break")
    assert lines[1] == "🎯 1H pattern · 110 → EL 100.1 → lower high 108 ✓"


# ── engine: detection pass ─────────────────────────────────────────────
def _engine(cfg, monkeypatch, dm=None):
    if dm:
        monkeypatch.setenv("TELEGRAM_DM_CHAT_ID", dm)
    return eng_mod.Engine(cfg, dry_run=True, tg=FakeTG())


def test_actions_prefer_low_tf_when_both_have_it(cfg, monkeypatch):
    """SUI case: 1H and 15m both carry the pattern → the low-TF one wins
    (its alert ships both charts)."""
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    long_c = mk(long_path())
    acts = e._potential_actions({inst.id: inst},
                                {(inst.symbol, "1h"): long_c,
                                 (inst.symbol, "15m"): long_c},
                                now_s=NOW_S)
    longs = [a for a in acts if a.detail["side"] == "long"]
    assert len(longs) == 1
    assert longs[0].detail["pattern_tf"] == "15m"
    assert inst.notified["potential_long"] is True


def test_actions_fall_back_to_main_tf(cfg, monkeypatch):
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    long_c = mk(long_path())
    # 15m has no pattern (flat path), 1h does
    acts = e._potential_actions({inst.id: inst},
                                {(inst.symbol, "1h"): long_c,
                                 (inst.symbol, "15m"): mk(_line(100, 100, 40))},
                                now_s=NOW_S)
    longs = [a for a in acts if a.detail["side"] == "long"]
    assert len(longs) == 1
    assert longs[0].detail["pattern_tf"] == "1h"


def test_actions_fire_once_per_side(cfg, monkeypatch):
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    long_c = mk(long_path())
    key = {(inst.symbol, "15m"): long_c}
    first = e._potential_actions({inst.id: inst}, key, now_s=NOW_S)
    assert len(first) == 1
    second = e._potential_actions({inst.id: inst}, key, now_s=NOW_S + 600)
    assert second == []
    assert len(inst.events) == 1
    assert inst.events[0]["kind"] == "potential"


def test_actions_both_sides_can_fire(cfg, monkeypatch):
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    acts = e._potential_actions(
        {inst.id: inst},
        {(inst.symbol, "15m"): mk(long_path())},
        now_s=NOW_S)
    sides = {a.detail["side"] for a in acts}
    assert "long" in sides
    # short needs its own path — with the long one there is no short hit
    assert "short" not in sides


def test_actions_skip_non_box(cfg, monkeypatch):
    e = _engine(cfg, monkeypatch)
    inst = _inst(type_="ascending_triangle")
    acts = e._potential_actions({inst.id: inst},
                                {(inst.symbol, "1h"): mk(long_path())}, now_s=NOW_S)
    assert acts == []


def test_actions_skip_terminal_state(cfg, monkeypatch):
    e = _engine(cfg, monkeypatch)
    inst = _inst(state=STATE_BREAKOUT)
    acts = e._potential_actions({inst.id: inst},
                                {(inst.symbol, "1h"): mk(long_path())}, now_s=NOW_S)
    assert acts == []


def test_actions_skip_already_flagged_side(cfg, monkeypatch):
    e = _engine(cfg, monkeypatch)
    inst = _inst(notified={"potential_long": True})
    acts = e._potential_actions({inst.id: inst},
                                {(inst.symbol, "1h"): mk(long_path())}, now_s=NOW_S)
    assert acts == []


# ── dispatch ───────────────────────────────────────────────────────────
def _dispatch(cfg, monkeypatch, detail, inst, pattern_on_low_tf=True,
              dm="5659605264"):
    monkeypatch.setenv("TELEGRAM_DM_CHAT_ID", dm)
    monkeypatch.setattr(eng_mod, "has_candle_problems", lambda c: False)
    calls = []

    async def fake_chart(self, ses, i, lines, trend_lines=None, tf=None):
        calls.append({"tf": tf or i.tf, "lines": lines})
        return f"PNG-{tf or i.tf}".encode()

    monkeypatch.setattr(eng_mod.Engine, "_chart", fake_chart)
    tg = FakeTG()
    e = eng_mod.Engine(cfg, dry_run=False, tg=tg)
    e.instances[inst.id] = inst
    inst.msg_ids = [11]
    inst.dm_msg_ids = [22]
    candles_by_key = {(inst.symbol, inst.tf): mk(long_path())}
    if pattern_on_low_tf:
        candles_by_key[(inst.symbol, "15m")] = mk(long_path())
    actions = [eng_mod.Action("potential_break", inst, detail)]
    sends = asyncio.run(e._dispatch(actions, None, candles_by_key))
    return tg, calls, sends


def test_dispatch_sends_main_and_low_tf_charts(cfg, monkeypatch):
    inst = _inst()
    hit = find_potential(mk(long_path()), "long", CFG)
    detail = {"pattern_tf": "15m", **hit}
    tg, calls, sends = _dispatch(cfg, monkeypatch, detail, inst)

    assert [c["tf"] for c in calls] == ["1h", "15m"]     # main then low
    # main chart = the box's own last-pivot boundaries
    assert calls[0]["lines"] == [100.0, 110.0]
    # low chart = box low + the pattern's mid (last low-TF high)
    assert calls[1]["lines"] == [100.0, hit["mid_price"]]
    # BOTH charts in ONE album message (channel), caption on the album
    chan_groups = [g for g in tg.groups if g[3] is None]
    assert len(chan_groups) == 1
    assert len(chan_groups[0][1]) == 2
    assert "Long Potential Break" in chan_groups[0][0]
    # no stray separate photos for the two-chart case
    assert not [p for p in tg.photos if p[3] is None]
    assert sends[0]["msg_id"] is not None


def test_dispatch_replies_to_box_confirmation(cfg, monkeypatch):
    inst = _inst(msg_ids=[11])
    hit = find_potential(mk(long_path()), "long", CFG)
    tg, calls, _ = _dispatch(cfg, monkeypatch, {"pattern_tf": "15m", **hit}, inst)
    # the album replies onto msg 11 (the compression confirmation)
    chan = [g for g in tg.groups if g[3] is None]
    assert chan[0][2] == 11


def test_dispatch_mirrors_both_charts_to_dm(cfg, monkeypatch):
    inst = _inst(msg_ids=[11])
    inst.dm_msg_ids = [22]
    hit = find_potential(mk(long_path()), "long", CFG)
    tg, _, _ = _dispatch(cfg, monkeypatch, {"pattern_tf": "15m", **hit}, inst)
    dm_groups = [g for g in tg.groups if g[3] == "5659605264"]
    assert len(dm_groups) == 1            # one album, not two photos
    assert len(dm_groups[0][1]) == 2
    assert dm_groups[0][2] == 22          # replies onto the mirrored compression


def test_dispatch_main_only_when_pattern_is_on_main_tf(cfg, monkeypatch):
    inst = _inst()
    hit = find_potential(mk(long_path()), "long", CFG)
    tg, calls, _ = _dispatch(cfg, monkeypatch, {"pattern_tf": "1h", **hit}, inst,
                             pattern_on_low_tf=False)
    assert [c["tf"] for c in calls] == ["1h"]
    # exactly one message per destination, carrying exactly one chart
    chan = [g for g in tg.groups if g[3] is None] + \
           [p for p in tg.photos if p[3] is None]
    dm = [g for g in tg.groups if g[3]] + [p for p in tg.photos if p[3]]
    assert len(chan) == 1 and len(dm) == 1
    assert len(chan[0][1]) == 1 if isinstance(chan[0], tuple) and len(chan[0]) == 4 and isinstance(chan[0][1], list) else True


# ── freshness window (user choice 2026-09-27: 12h) ────────────────────
def test_stale_confirmation_is_skipped(cfg, monkeypatch):
    """A confirmation older than POTENTIAL_FRESH_S is old news: no action,
    and the side stays UNflagged (a later fresh pattern may still fire)."""
    from compression_detection import engine as eng_mod_local
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    acts = e._potential_actions({inst.id: inst},
                                {(inst.symbol, "1h"): mk(long_path())},
                                now_s=STALE_S)
    assert acts == []
    assert not inst.notified.get("potential_long")
    assert inst.events == []


def test_fresh_window_boundary(cfg, monkeypatch):
    """Fires at the last instant of the window; one second later it is
    stale (now_s <= confirm + POTENTIAL_FRESH_S)."""
    from compression_detection import potential_break as pb
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    key = {(inst.symbol, "1h"): mk(long_path())}
    hit = find_potential(key[(inst.symbol, "1h")], "long", CFG)
    last_fresh = hit["confirm_ts"] / 1000 + pb.POTENTIAL_FRESH_S

    e._potential_actions({inst.id: inst}, key, now_s=int(last_fresh))
    assert inst.notified.get("potential_long") is True

    inst2 = _inst()
    e._potential_actions({inst2.id: inst2}, key, now_s=int(last_fresh) + 1)
    assert not inst2.notified.get("potential_long")


def test_stale_low_tf_falls_through_to_fresh_main(cfg, monkeypatch):
    """The freshness check only skips that TF — the box's own TF keeps
    being probed (low stale, main fresh -> main fires)."""
    from compression_detection import engine as eng_mod_local
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    # build a low-TF hit far in the past and a main-TF hit that is fresh
    fresh = mk(long_path(), t0=(STALE_S - 3600) * 1000)     # confirms now
    stale = mk(long_path(), t0=(STALE_S - 90 * 3600) * 1000)  # confirms 3d ago
    acts = e._potential_actions({inst.id: inst},
                                {(inst.symbol, "1h"): fresh,
                                 (inst.symbol, "15m"): stale},
                                now_s=STALE_S)
    longs = [a for a in acts if a.detail["side"] == "long"]
    assert len(longs) == 1
    assert longs[0].detail["pattern_tf"] == "1h"


# ── state round-trip (the restart-duplicate bug) ───────────────────────
def test_all_notified_flags_survive_state_round_trip():
    """to_dict/from_dict must keep every flag.

    Regression: from_dict rebuilt `notified` from only compression/breakout,
    so each engine restart dropped potential_long/potential_short and
    re-fired alerts it had already posted (5 duplicates on 2026-09-27)."""
    inst = _inst(notified={"compression": True, "breakout": False,
                           "potential_long": True, "potential_short": False})
    back = type(inst).from_dict(inst.to_dict())
    assert back.notified["potential_long"] is True
    assert back.notified["potential_short"] is False
    assert back.notified["compression"] is True
    assert back.notified["breakout"] is False


def test_notified_defaults_when_state_has_no_flags():
    inst = _inst()
    d = inst.to_dict()
    d["notified"] = {}
    back = type(inst).from_dict(d)
    assert back.notified == {"compression": False, "breakout": False}


# ── direction label on the SECOND swing (user 2026-09-27, QNT case) ───
def test_rejects_second_low_that_is_only_equal():
    """A second low that is numerically higher but inside the ±ATR14
    equality band is labeled EL, NOT HL — that is not a higher low.
    (QNT: 180.06 < 181.25 at 0.32x ATR14 -> labeled EH, so the 'lower
    high' the alert claimed did not exist.)"""
    # L1=100, L2=100.5 -> |Δ|=0.5 <= ATR14 (~0.84) -> EL
    path = (_line(100, 110.1, 12) + _line(110.1, 100, 12) +
            _line(100, 110, 12) + _line(110, 100.5, 8) +
            _line(100.5, 120, 12))
    from compression_detection.zigzag import ZigZag
    from compression_detection.atr import atr_series
    from compression_detection.dow import label_all_pivots
    candles = mk(path)
    lab = label_all_pivots(
        ZigZag(coef=1.2, atr_length=7).feed_all(candles),
        atr_series(candles, 14, "close_only"))
    lows = [(p.price, l.name if l else None) for p, l in lab if not p.is_high]
    assert lows[-1][1] == "EL", f"fixture must end on an EL, got {lows}"
    assert find_potential(candles, "long", CFG) is None


def test_rejects_second_high_that_is_only_equal():
    """Mirror: second high numerically lower but within ATR14 -> EH."""
    path = (_line(110, 100, 12) + _line(100, 110, 12) +
            _line(110, 100.1, 12) + _line(100.1, 109.5, 8) +
            _line(109.5, 90, 12))
    from compression_detection.zigzag import ZigZag
    from compression_detection.atr import atr_series
    from compression_detection.dow import label_all_pivots
    candles = mk(path)
    lab = label_all_pivots(
        ZigZag(coef=1.2, atr_length=7).feed_all(candles),
        atr_series(candles, 14, "close_only"))
    highs = [(p.price, l.name if l else None) for p, l in lab if p.is_high]
    assert highs[-1][1] == "EH", f"fixture must end on an EH, got {highs}"
    assert find_potential(candles, "short", CFG) is None


# ── dispatch order: the compression anchor goes first (QNT 2026-09-27) ─
def test_compression_dispatches_before_potential_break():
    """The box confirmation is the message every other alert threads onto;
    best_per_coin_tf must emit it BEFORE passthrough actions, otherwise a
    potential break can post first with no reply target (QNT: potential at
    10:01:46, its own compression at 10:01:48)."""
    from compression_detection.lifecycle import Action, best_per_coin_tf
    inst = _inst()
    acts = [Action("potential_break", inst, {"side": "short"}),
            Action("compression_notify", inst, {}),
            Action("breakout_post", inst, {})]
    out = best_per_coin_tf(acts, cfg_det_order())
    kinds = [a.kind for a in out]
    assert kinds[0] == "compression_notify", kinds
    assert "potential_break" in kinds and "breakout_post" in kinds


def cfg_det_order():
    from compression_detection.config import Config
    return Config.load().det.selection_order


# ── ordering around the box -> triangle handover (user rule 2026-09-27) ─
def test_order_potential_puts_own_box_compression_first():
    """QNT case: the box's compression and its potential break land in the
    same scan — the compression (the reply anchor) must go out first."""
    from compression_detection.lifecycle import Action, order_potential
    box = _inst()
    tri = _inst(type_="ascending_triangle", anchor=T0 + 60_000)
    acts = [Action("potential_break", box, {"side": "short"}),
            Action("compression_notify", box, {})]
    out = order_potential(acts)
    assert [a.kind for a in out] == ["compression_notify", "potential_break"]


def test_order_potential_precedes_the_replacing_triangle():
    """Handover case: the box's potential break must go out BEFORE the
    ascending/descending triangle that replaces it, not after."""
    from compression_detection.lifecycle import Action, order_potential
    box = _inst()
    tri = _inst(type_="ascending_triangle", anchor=T0 + 60_000)
    acts = [Action("compression_notify", tri, {}),
            Action("invalidate", box, {}),
            Action("potential_break", box, {"side": "long"})]
    out = order_potential(acts)
    kinds = [a.kind for a in out]
    assert kinds.index("potential_break") < kinds.index("compression_notify")
    assert kinds.index("invalidate") > kinds.index("potential_break")


def test_order_potential_noop_without_potential():
    from compression_detection.lifecycle import Action, order_potential
    a1 = Action("compression_notify", _inst(), {})
    a2 = Action("breakout_post", _inst(), {})
    assert order_potential([a1, a2]) == [a1, a2]


def test_drop_moot_on_breakout_kept_on_invalidate():
    """A box that broke out in the same scan has resolved — its potential
    break is dropped. An invalidated box (the triangle handover) keeps it."""
    import compression_detection.engine as eng_mod
    broke = _inst(state=STATE_BREAKOUT)
    replaced = _inst(state="invalidated")
    acts = [eng_mod.Action("potential_break", broke, {"side": "long"}),
            eng_mod.Action("potential_break", replaced, {"side": "long"})]
    out = eng_mod.Engine._drop_moot_potential(acts)
    assert [a.instance.state for a in out] == ["invalidated"]
