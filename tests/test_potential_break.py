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


# The pattern is judged ONLY against the box lines (user rule 2026-09-27):
# flat box [lower=100, upper=110], touch tolerance = box ATR = 1.0.
BOX_UPPER, BOX_LOWER, BOX_ATR = 109.5, 100.0, 1.0
BOX_AT = lambda ts: (BOX_UPPER, BOX_LOWER)  # noqa: E731


def fp(candles, side, cfg=CFG, main_candles=None, scan=None, **kw):
    """find_potential with the fixture box lines injected.

    Default: one timeframe — it is both the MAIN tf (STEP 1 touch) and the
    only scan entry (STEP 2 confirming swing)."""
    kw.setdefault("box_at", BOX_AT)
    kw.setdefault("touch_atr", BOX_ATR)
    kw.setdefault("main_tf", "1h")
    main = candles if main_candles is None else main_candles
    entries = scan if scan is not None else [("1h", candles)]
    return find_potential(side, main, entries, cfg, **kw)


def long_path(mid_is_eh: bool = True, second_higher: bool = True,
              lo2=None):
    """H0 → L1 → H1 → L2 → rally. H1 touches when it sits back on H0; L2 is
    the confirming low (override it with `lo2` to place it inside/outside
    the equality band)."""
    hi0 = 110.1
    hi1 = 110.0 if mid_is_eh else 115.0      # no touch when False
    if lo2 is None:
        lo2 = 105.0 if second_higher else 95.0   # below the box when False
    return (_line(100, hi0, 12) + _line(hi0, 100, 12) +
            _line(100, hi1, 12) + _line(hi1, lo2, 8) +
            _line(lo2, 120, 12))


def long_main():
    """Main tf that ends with the touch CONFIRMED as the last pivot — a
    small unconfirmed pullback after it, so a lower-tf third can satisfy
    'the touch is the last main pivot'."""
    hi0, hi1 = 110.1, 110.0
    return (_line(100, hi0, 12) + _line(hi0, 100, 12) +
            _line(100, hi1, 12) + _line(hi1, 106, 4))   # ends mid-pullback


def short_path(second_lower: bool = True):
    """L0 → H1 → L1(EL) → H2 → decline."""
    hi2 = 108.0 if second_lower else 118.0   # HH (not LH) when False
    return (_line(110, 100, 12) + _line(100, 110, 12) +
            _line(110, 100.1, 12) + _line(100.1, hi2, 8) +
            _line(hi2, 95, 12))


def _inst(symbol="SUIUSDT", tf="1h", type_="box", state=STATE_CONFIRMED,
          anchor=T0, msg_ids=None, notified=None):
    upper = st.Line(slope=0.0, intercept=BOX_UPPER, base_bar=0)
    lower = st.Line(slope=0.0, intercept=BOX_LOWER, base_bar=0)
    return Instance(
        id=f"{symbol}|{tf}|{type_}|{anchor}", symbol=symbol, tf=tf,
        type=type_, state=state, level="confirmed", created_ts=T0,
        anchor_ts=anchor, last_scan_ts=T0, last_eval_ts=T0,
        pivots=[{"ts": T0, "bar": 0, "side": "L", "price": BOX_LOWER,
                 "label": "EL"},
                {"ts": T0 + STEP, "bar": 1, "side": "H", "price": BOX_UPPER,
                 "label": "EH"}],
        upper_line=upper, lower_line=lower, atr=BOX_ATR,
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
    hit = fp(mk(long_path()), "long", CFG)
    assert hit is not None
    assert hit["mid_label"] == "EH"                     # main-tf touch
    assert hit["mid_price"] == pytest.approx(110.0)     # on the box high
    assert hit["first_price"] == hit["mid_price"]       # the touch IS first
    assert hit["second_price"] == pytest.approx(105.0)  # higher low
    assert hit["pattern_tf"] == "1h"
    assert hit["confirm_ts"] >= hit["second_ts"]        # confirmed later


def test_short_pattern_detected():
    hit = fp(mk(short_path()), "short", CFG)
    assert hit is not None
    assert hit["mid_label"] == "EL"                     # main-tf touch
    assert hit["mid_price"] == pytest.approx(100.1)     # on the box low
    assert hit["second_price"] == pytest.approx(108.0)  # lower high
    assert hit["confirm_ts"] >= hit["second_ts"]


def test_confirm_ts_is_the_replay_bar():
    """The alert fires on the candle that closes beyond coef × ATR7."""
    from compression_detection.potential_break import _confirm_map
    candles = mk(long_path())
    cmap = _confirm_map(candles, CFG)
    hit = fp(candles, "long", CFG)
    assert cmap[hit["second_ts"]] == hit["confirm_ts"]
    # ...and that bar is strictly after the pivot bar
    assert hit["confirm_ts"] > hit["second_ts"]


# ── detection: negative ────────────────────────────────────────────────
def test_rejects_mid_that_does_not_touch_the_box_high():
    """mid 115 vs box upper 110 = 5 ATR away — a middle pivot, not the box
    top (XPL: the 15m's 0.11262 was 7% below the box 0.121)."""
    assert fp(mk(long_path(mid_is_eh=False)), "long", CFG) is None


def test_rejects_second_swing_below_the_box_low():
    """the third low must sit on/above the box LOW line — judged against
    the box, never against the previous low."""
    assert fp(mk(long_path(second_higher=False)), "long", CFG) is None


def test_touch_tolerance_is_the_box_atr():
    """|mid − upper| ≤ box ATR touches, just past it does not."""
    import compression_detection.potential_break as pb
    mid_px = 110.0                      # long_path()'s middle high
    for offset, expect in ((BOX_ATR, True), (BOX_ATR * 1.01, False)):
        hit = fp(mk(long_path()), "long",
                 box_at=lambda ts, o=offset: (mid_px + o, BOX_LOWER),
                 touch_atr=BOX_ATR)
        assert (hit is not None) is expect, f"offset={offset}"


def test_no_box_lines_fails_closed():
    """No box_at / touch_atr -> no pattern, never a guess."""
    assert find_potential("long", mk(long_path()), [("1h", mk(long_path()))],
                          CFG, main_tf="1h") is None
    assert find_potential("long", mk(long_path()), [("1h", mk(long_path()))],
                          CFG, main_tf="1h", box_at=BOX_AT) is None


def test_third_must_clear_the_equality_band():
    """'higher than the low of box, EXCEEDING the equal threshold': sitting
    on the line is not enough — the swing must clear box_low + ATR14."""
    from compression_detection.atr import atr_series

    def eq_at(candles, ts):
        a14 = atr_series(candles, 14, "close_only")
        return a14[{c.ts: i for i, c in enumerate(candles)}[ts]]

    # 105 clears the band comfortably -> fires
    hit = fp(mk(long_path()), "long")
    assert hit is not None
    candles = mk(long_path())
    eq = eq_at(candles, hit["second_ts"])
    assert hit["second_price"] > BOX_LOWER + eq
    # a third that clears the LINE but not the band -> rejected
    in_band = BOX_LOWER + 0.5 * eq
    assert fp(mk(long_path(lo2=in_band)), "long") is None
    # ...and a swing clearly above the band -> fires (each path carries its
    # own ATR14, so the clearance has to be generous)
    just_over = BOX_LOWER + 2.5 * eq
    assert fp(mk(long_path(lo2=just_over)), "long") is not None



def test_rejects_short_when_second_higher():
    assert fp(mk(short_path(second_lower=False)), "short", CFG) is None


def test_anchor_filter_drops_pre_box_pattern():
    """A pattern that finished before the compression existed is not this
    box's signal."""
    candles = mk(long_path())
    hit = fp(candles, "long", CFG, anchor_ts=T0)
    assert hit is not None
    # anchor pushed past the whole pattern → nothing qualifies
    assert fp(candles, "long", CFG,
                          anchor_ts=hit["second_ts"] + 10 * STEP) is None


def test_too_few_candles_returns_none():
    assert fp(mk(_line(100, 110, 4)), "long", CFG) is None


def test_bad_side_raises():
    with pytest.raises(ValueError):
        fp(mk(long_path()), "sideways", CFG)


# ── chart lines ────────────────────────────────────────────────────────
def test_potential_lines_long_adjusts_the_high():
    hit = fp(mk(long_path()), "long", CFG)
    # box low stays, the high becomes the last low-TF high pivot (the mid)
    assert potential_lines(hit, 100.0, 110.0) == [100.0, hit["mid_price"]]


def test_potential_lines_short_adjusts_the_low():
    hit = fp(mk(short_path()), "short", CFG)
    # box high stays, the low becomes the last low-TF low pivot (the mid)
    assert potential_lines(hit, 100.0, 110.0) == [hit["mid_price"], 110.0]


def test_lower_tf_map():
    assert LOWER_TF == {"4h": "1h", "1h": "15m", "15m": None}


# ── caption ────────────────────────────────────────────────────────────
def test_fmt_potential_long():
    inst = _inst()
    hit = fp(mk(long_path()), "long", CFG)
    cap = nt.fmt_potential(inst, "long", "15m", hit)
    lines = cap.split("\n")
    assert lines[0] == ("🟢 <b>SUIUSDT</b> — 1H Compression  ·  "
                        "Long Potential Break")
    assert lines[1] == "🎯 15m pattern · touch EH 110 → higher low 105 ✓"


def test_fmt_potential_short():
    inst = _inst()
    hit = fp(mk(short_path()), "short", CFG)
    cap = nt.fmt_potential(inst, "short", "1h", hit)
    lines = cap.split("\n")
    assert lines[0] == ("🔴 <b>SUIUSDT</b> — 1H Compression  ·  "
                        "Short Potential Break")
    assert lines[1] == "🎯 1H pattern · touch EL 100.1 → lower high 108 ✓"


# ── engine: detection pass ─────────────────────────────────────────────
def _engine(cfg, monkeypatch, dm=None):
    if dm:
        monkeypatch.setenv("TELEGRAM_DM_CHAT_ID", dm)
    return eng_mod.Engine(cfg, dry_run=True, tg=FakeTG())


def test_actions_prefer_low_tf_when_both_have_it(cfg, monkeypatch):
    """SUI case: the touch sits in the main tf, the confirming swing is
    found one tf below → the low-TF alert wins (it ships both charts)."""
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    acts = e._potential_actions({inst.id: inst},
                                {(inst.symbol, "1h"): mk(long_main()),
                                 (inst.symbol, "15m"): mk(long_path())},
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


def test_actions_fire_once_per_box(cfg, monkeypatch):
    """ONE potential per box (user rule 2026-09-27): firing one side writes
    BOTH flags, so no second alert can follow."""
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    long_c = mk(long_path())
    key = {(inst.symbol, "1h"): long_c, (inst.symbol, "15m"): long_c}
    first = e._potential_actions({inst.id: inst}, key, now_s=NOW_S)
    assert len(first) == 1
    assert inst.notified["potential_long"] is True
    assert inst.notified["potential_short"] is True   # both suppressed
    second = e._potential_actions({inst.id: inst}, key, now_s=NOW_S + 600)
    assert second == []
    assert len(inst.events) == 1
    assert inst.events[0]["kind"] == "potential"


def test_actions_both_sides_can_fire(cfg, monkeypatch):
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    long_c = mk(long_path())
    acts = e._potential_actions(
        {inst.id: inst},
        {(inst.symbol, "1h"): long_c, (inst.symbol, "15m"): long_c},
        now_s=NOW_S)
    sides = {a.detail["side"] for a in acts}
    assert "long" in sides
    # the long fixture has no EL touch -> no short hit
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
    hit = fp(mk(long_path()), "long", CFG)
    detail = {**hit, "pattern_tf": "15m"}
    tg, calls, sends = _dispatch(cfg, monkeypatch, detail, inst)

    assert [c["tf"] for c in calls] == ["1h", "15m"]     # main then low
    # main chart = the box's own boundary lines (the fixture's box sides)
    assert calls[0]["lines"] == [BOX_LOWER, BOX_UPPER]
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
    hit = fp(mk(long_path()), "long", CFG)
    tg, calls, _ = _dispatch(cfg, monkeypatch, {"pattern_tf": "15m", **hit}, inst)
    # the album replies onto msg 11 (the compression confirmation)
    chan = [g for g in tg.groups if g[3] is None]
    assert chan[0][2] == 11


def test_dispatch_mirrors_both_charts_to_dm(cfg, monkeypatch):
    inst = _inst(msg_ids=[11])
    inst.dm_msg_ids = [22]
    hit = fp(mk(long_path()), "long", CFG)
    tg, _, _ = _dispatch(cfg, monkeypatch, {**hit, "pattern_tf": "15m"}, inst)
    dm_groups = [g for g in tg.groups if g[3] == "5659605264"]
    assert len(dm_groups) == 1            # one album, not two photos
    assert len(dm_groups[0][1]) == 2
    assert dm_groups[0][2] == 22          # replies onto the mirrored compression


def test_dispatch_main_only_when_pattern_is_on_main_tf(cfg, monkeypatch):
    inst = _inst()
    hit = fp(mk(long_path()), "long", CFG)
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
    hit = fp(key[(inst.symbol, "1h")], "long", CFG)
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
def test_third_inside_the_band_is_rejected_short():
    """Short mirror: a lower high that does not clear box_high − ATR14 is
    inside the equality band, whatever the neighbour label says."""
    from compression_detection.atr import atr_series
    path = (_line(110, 100, 12) + _line(100, 110, 12) +
            _line(110, 100.1, 12) + _line(100.1, 108.9, 8) +
            _line(108.9, 90, 12))
    candles = mk(path)
    a14 = atr_series(candles, 14, "close_only")
    idx = {c.ts: i for i, c in enumerate(candles)}
    # find the third high's eq first (reject case: inside the band)
    assert fp(candles, "short") is None
    # and the default 108.0 clears band 109.5 − eq -> fires
    assert fp(mk(short_path()), "short") is not None
    _ = (a14, idx)


def test_one_per_box_freshest_confirm_wins(cfg, monkeypatch):
    """Both sides pass -> exactly ONE action, the fresher confirmation, and
    both flags written (PIEVERSE case: long+short 5 seconds apart)."""
    import compression_detection.engine as em
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    long_c = mk(long_path())
    key = {(inst.symbol, "1h"): long_c, (inst.symbol, "15m"): long_c}
    real = em.find_potential

    def fake(side, main_candles, scan, c, **kw):
        out = {"side": side, "pattern_tf": "1h",
               "touch_ts": T0, "touch_price": 110.0, "touch_label": "EH",
               "first_ts": T0, "first_price": 110.0,
               "mid_ts": T0, "mid_price": 110.0, "mid_label": "EH",
               "second_ts": T0 + 10 * STEP, "second_price": 105.0,
               "confirm_ts": T0 + (20 * STEP if side == "long"
                                   else 40 * STEP)}
        return out if side in ("long", "short") else None

    monkeypatch.setattr(em, "find_potential", fake)
    acts = e._potential_actions({inst.id: inst}, key, now_s=NOW_S)
    assert len(acts) == 1
    assert acts[0].detail["side"] == "short"      # freshest confirm wins
    assert inst.notified["potential_long"] is True
    assert inst.notified["potential_short"] is True



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


def test_pattern_straddling_the_anchor_still_fires():
    """QNT case: the box's anchor sits BETWEEN the pattern's swings, so
    gating on first.ts would drop a pattern that both completed and
    confirmed inside the box's life. The gate is on the CONFIRMATION."""
    candles = mk(long_path())
    hit = fp(candles, "long", CFG)
    assert hit is not None
    # anchor placed between first and mid -> still fires (conf > anchor)
    midway = hit["first_ts"] + 1
    assert fp(candles, "long", CFG, anchor_ts=midway) is not None
    # anchor pushed past the confirmation -> suppressed
    assert fp(candles, "long", CFG,
                          anchor_ts=hit["confirm_ts"] + 1) is None


def test_dead_reply_target_does_not_swallow_the_alert(cfg, monkeypatch):
    """A reply target deleted since it was recorded (QNT DM twin 2429)
    makes Telegram 400 on every post. The alert must still go out — first
    try with the reply, then without it."""
    monkeypatch.setenv("TELEGRAM_DM_CHAT_ID", "5659605264")
    monkeypatch.setattr(eng_mod, "has_candle_problems", lambda c: False)

    async def fake_chart(self, ses, i, lines, trend_lines=None, tf=None):
        return b"PNG"

    monkeypatch.setattr(eng_mod.Engine, "_chart", fake_chart)

    class FlakyTG(FakeTG):
        """post_photo/post fail whenever a reply target is supplied."""
        async def post_media_group(self, caption, images, reply_to_id=None,
                                   chat_id=None):
            if reply_to_id:
                return None
            return await super().post_media_group(caption, images,
                                                  reply_to_id=None,
                                                  chat_id=chat_id)

        async def post_photo(self, caption, img, reply_to_id=None,
                             chat_id=None):
            if reply_to_id:
                return None
            return await super().post_photo(caption, img, reply_to_id=None,
                                            chat_id=chat_id)

        async def post(self, text, reply_to_id=None, chat_id=None):
            if reply_to_id:
                return None
            return await super().post(text, reply_to_id=None, chat_id=chat_id)

    tg = FlakyTG()
    e = eng_mod.Engine(cfg, dry_run=False, tg=tg)
    inst = _inst()
    inst.msg_ids = [11]
    hit = fp(mk(long_path()), "long", CFG)
    actions = [eng_mod.Action("potential_break", inst,
                              {"pattern_tf": "15m", **hit})]
    sends = asyncio.run(e._dispatch(actions, None,
                                    {(inst.symbol, inst.tf): mk(long_path()),
                                     (inst.symbol, "15m"): mk(long_path())}))
    # the alert went out (without the reply) instead of vanishing
    assert sends and sends[0]["msg_id"] is not None
    assert tg.groups or tg.photos or tg.texts


# ── the handover invariant (what makes the box -> triangle case work) ──
def test_potential_pass_runs_before_the_lifecycle_pass(cfg, monkeypatch):
    """The box must still be ALIVE when the pattern is checked, i.e. before
    update_for_scan can invalidate it in the box -> ascending/descending
    triangle handover. Otherwise the replacement case can never alert."""
    from test_engine import patch_env, scan, box_closes

    order = []
    real_pot = eng_mod.Engine._potential_actions
    real_upd = eng_mod.update_for_scan

    def spy_pot(self, instances, candles_by_key, now_s):
        order.append("potential")
        return real_pot(self, instances, candles_by_key, now_s)

    def spy_upd(instances, symbol, tf, candidates, candles, tf_ms, now_s, cfg_):
        order.append("lifecycle")
        return real_upd(instances, symbol, tf, candidates, candles, tf_ms,
                        now_s, cfg_)

    monkeypatch.setattr(eng_mod.Engine, "_potential_actions", spy_pot)
    monkeypatch.setattr(eng_mod, "update_for_scan", spy_upd)
    patch_env(monkeypatch, box_closes())
    scan(cfg)
    assert order[0] == "potential", order[:4]
    assert "lifecycle" in order


# ── LIVE third swing (loosened 2026-09-27, PENDLE case) ───────────────
# spread=1.0 makes ATR7(true-range) ~2x2 wide, so the zigzag confirm
# (1.2xATR7) exceeds the whole move, while ATR14(close-only) stays small:
# the third swing can only be taken on the LIVE route.

def _eq_at(closes, idx):
    """The REAL eq band at bar idx — Wilder-RMA ATR14, exactly what
    find_potential multiplies (a plain mean mismatches and the tail
    misses the gate)."""
    from compression_detection.atr import atr_series
    return atr_series(mk(closes), CFG.atr14_length, CFG.atr14_method)[idx]


def _live_long_path(tail: str = "clear"):
    """H0 -> L1 -> H1(touch) -> L2 106.5 (never zigzag-confirms) -> tail.

    spread=1.0 pins ATR7(true-range) >= 2, so the zigzag confirm needs
    close > lo2 + 2.4 — the tails always stay under that (live route only).
    """
    hi0, hi1, lo2 = 110.1, 110.0, 106.5
    base = (_line(100, hi0, 12) + _line(hi0, 100, 12) +
            _line(100, hi1, 12) + _line(hi1, lo2, 8))
    eq = _eq_at(base, len(base) - 1)
    if tail == "clear":
        # clears LIVE_CLEAR_ATR(1.75) x eq with margin, stays < lo2+2.0
        target = min(lo2 + 1.75 * eq + 0.1, lo2 + 2.0)
        path = base + _line(lo2, target, 6)
    elif tail == "dip":
        # past 1x eq but short of 1.75x — held by the clear gate alone
        path = base + _line(lo2, lo2 + 1.4 * eq, 6)
    elif tail == "flat":
        path = base + _line(lo2, lo2 + 0.05, 6)
    else:                                    # below the box band
        path = base + _line(99.0, 100.6, 6)
    return mk(path, spread=1.0)


def test_live_third_swing_fires_before_the_zigzag_confirm():
    from compression_detection.potential_break import _zigzag
    candles = _live_long_path()
    # the third low is NOT in the confirmed zigzag — only the live route
    # can produce this hit
    assert all(abs(p.price - 106.5) > 1e-9 for p in _zigzag(candles, CFG))
    hit = fp(candles, "long")
    assert hit is not None
    assert hit["live"] is True
    assert hit["second_price"] == 106.5
    assert hit["touch_price"] == 110.0
    # confirm_ts = the close that cleared lo2 + eq (inside the tail)
    assert hit["confirm_ts"] in {c.ts for c in candles[-6:]}


def test_live_third_swing_never_cleared_by_close_returns_none():
    assert fp(_live_long_path(tail="flat"), "long") is None


def test_live_third_below_the_box_band_rejected():
    assert fp(_live_long_path(tail="below"), "long") is None


def test_live_third_short_mirror_fires():
    from compression_detection.potential_break import _zigzag
    lo0, hi1, lo1, hi2 = 99.8, 105.0, 100.0, 104.0
    base = (_line(110, lo0, 12) + _line(lo0, hi1, 12) +
            _line(hi1, lo1, 12) + _line(lo1, hi2, 8))
    eq = _eq_at(base, len(base) - 1)
    # clears 1.75*eq + margin below hi2, stays above hi2 - 2.4 (no zigzag)
    path = base + _line(hi2, hi2 - min(1.75 * eq + 0.1, 2.0), 6)
    candles = mk(path, spread=1.0)
    assert all(abs(p.price - hi2) > 1e-9 for p in _zigzag(candles, CFG))
    hit = fp(candles, "short")
    assert hit is not None
    assert hit["live"] is True
    assert hit["second_price"] == hi2
    assert hit["touch_price"] == 100.0


def test_live_third_requires_the_touch_to_be_the_last_main_pivot():
    """A confirmed pivot AFTER the touch supersedes it: the live route must
    refuse even though the live low itself would clear the band."""
    hi0, hi1 = 110.1, 110.0
    path = (_line(100, hi0, 12) + _line(hi0, 100, 12) +
            _line(100, hi1, 12) + _line(hi1, 104, 10) +   # low 104 confirms
            _line(104, 107, 8) +                          # high 107 confirms
            _line(107, 104.5, 14) +                       # low 104.5 (live)
            _line(104.5, 105.6, 6))                       # clears, not confirm
    candles = mk(path, spread=1.0)
    assert fp(candles, "long") is None


def test_live_dip_past_1x_but_short_of_the_clear_gate_is_held(monkeypatch):
    """The clear gate (LIVE_CLEAR_ATR = 1.75, user choice 2026-09-27,
    EIGEN/XLM case) is the ONLY reason this 1.4x dip doesn't alert: at
    1x the same path fires."""
    from compression_detection import potential_break as pb
    candles = _live_long_path(tail="dip")
    assert fp(candles, "long") is None
    monkeypatch.setattr(pb, "LIVE_CLEAR_ATR", 1.0)
    assert fp(candles, "long") is not None
