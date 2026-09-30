"""Potential-break trigger (user rule 2026-09-28 — rewrite).

  LONG : [box exists] + on a trigger tf: EH at the boundary, then a
         higher low (live OR confirmed), price inside that tf's
         1.0×ATR14 close zone, never past the line
  SHORT: mirror — EL at the boundary, then a lower high, zone above

Trigger tfs = the box's lower tfs (nearest first) + the box's own tf;
ANY tf passing sends one alert (freshest swing wins), the caption names
both frames ("4H box · break at 1H"). Touch AND swing come from the
SAME tf — no cross-tf pairing.

Exercised on synthetic zigzag-friendly paths (each leg ≈ 10 ATR so
coef × ATR7 confirms unambiguously), the engine pass on real Config,
the fast 60s proximity pass with stubbed fetches, and the dispatch
with a fake Telegram + chart stub.
"""
import asyncio
import sys
import types

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
              lo2=None, end=109.0, tail_n=6):
    """H0 → L1 → H1 → L2 → settle INSIDE the close zone (user rule
    price must sit within 1.0×ATR of the boundary (tightened
    2026-09-29), never past it — a rally-through is a breakout, not a
    potential). H1 touches when it sits back on H0; `lo2` places the
    higher low inside/outside the equality band; `end` parks the close
    in/out of the zone."""
    hi0 = 110.1
    hi1 = 110.0 if mid_is_eh else 115.0      # no touch when False
    if lo2 is None:
        lo2 = 105.0 if second_higher else 95.0   # below the box when False
    return (_line(100, hi0, 12) + _line(hi0, 100, 12) +
            _line(100, hi1, 12) + _line(hi1, lo2, 8) +
            _line(lo2, end, tail_n))


def long_main():
    """Main tf that ends with the touch CONFIRMED as the last pivot — a
    small unconfirmed pullback after it, so a lower-tf third can satisfy
    'the touch is the last main pivot'."""
    hi0, hi1 = 110.1, 110.0
    return (_line(100, hi0, 12) + _line(hi0, 100, 12) +
            _line(100, hi1, 12) + _line(hi1, 106, 4))   # ends mid-pullback


def short_path(second_lower: bool = True, end=100.5, tail_n=8):
    """L0 → H1 → L1(EL) → H2 → settle INSIDE the 1.0×ATR close zone
    above the box low (mirror of long_path)."""
    hi2 = 108.0 if second_lower else 118.0   # HH (not LH) when False
    return (_line(110, 100, 12) + _line(100, 110, 12) +
            _line(110, 100.1, 12) + _line(100.1, hi2, 8) +
            _line(hi2, end, tail_n))


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


def test_confirm_ts_is_the_swing_bar():
    """New rule (2026-09-28): the trigger fires on the swing itself —
    confirm_ts IS the swing's pivot ts (the replay-confirm machinery was
    retired with the trigger rewrite; live swings carry their bar too)."""
    hit = fp(mk(long_path()), "long", CFG)
    assert hit is not None
    assert hit["confirm_ts"] == hit["second_ts"]


# ── detection: negative ────────────────────────────────────────────────
def test_rejects_mid_that_does_not_touch_the_box_high():
    """mid 115 vs box upper 110 = 5 ATR away — a middle pivot, not the box
    top (XPL: the 15m's 0.11262 was 7% below the box 0.121)."""
    assert fp(mk(long_path(mid_is_eh=False)), "long", CFG) is None


def test_rejects_second_swing_below_the_box_low():
    """the third low must sit on/above the box LOW line — judged against
    the box, never against the previous low."""
    assert fp(mk(long_path(second_higher=False)), "long", CFG) is None


def test_touch_and_zone_follow_the_per_tf_band():
    """Touch AND close zone both use PROX_BAND_ATR (1.0 since
    2026-09-29) ×
    ATR14 of the trigger tf — the box ATR is no longer the yardstick."""
    from compression_detection.atr import atr_series
    from compression_detection.potential_break import PROX_BAND_ATR
    candles = mk(long_path())
    band = PROX_BAND_ATR * atr_series(candles, 14, "close_only")[-1]
    # default box top 109.5: touch (110.0) inside the band, close in the
    # zone -> fires
    assert fp(candles, "long") is not None
    # top moved so the touch sits OUTSIDE the band while the close zone
    # still contains the price -> no touch, no pattern
    touch_off = 108.8          # |110.0 - 108.8| = 1.2 > band (~1.04)
    assert fp(candles, "long",
              box_at=lambda ts: (touch_off, BOX_LOWER)) is None


def test_close_outside_the_zone_rejected():
    """Structure perfect but the close parks short of the 1.5×ATR zone
    (line - band, line] -> not close to the boundary, no alert."""
    assert fp(mk(long_path(end=107.0)), "long") is None


def test_price_past_the_line_is_the_breakouts_job():
    """A close BEYOND the boundary belongs to the breakout verdict —
    the old rally-to-120 tail must never fire a potential (2026-09-28)."""
    assert fp(mk(long_path(end=118.0, tail_n=8)), "long") is None


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
    # user rule 2026-09-28: detection 1h/4h/1d; 15m is lower-tf only
    assert LOWER_TF == {
        "1h": ["15m"],
        "4h": ["1h", "15m"],
        "1d": ["4h", "1h", "15m"],
    }


def test_fetch_tfs_include_potential_lower_tfs():
    """15m carries no patterns but is always fetched — 1h boxes touch
    and confirm swings on it (user rule 2026-09-28)."""
    from compression_detection.config import Config
    eng = eng_mod.Engine.__new__(eng_mod.Engine)
    eng.cfg = Config.load()
    assert eng._fetch_tfs() == ["1h", "4h", "1d", "15m"]


# ── caption ────────────────────────────────────────────────────────────
def test_fmt_potential_long():
    inst = _inst()
    hit = fp(mk(long_path()), "long", CFG)
    cap = nt.fmt_potential(inst, "long", "15m", hit)
    lines = cap.split("\n")
    assert lines[0] == ("🟢 <b>SUIUSDT</b> — 1H Compression  ·  "
                        "Long Potential Break")
    assert lines[1] == ("🎯 1H box · break at 15m · "
                        "touch EH 110 → higher low 105 ✓")


def test_fmt_potential_short():
    inst = _inst()
    hit = fp(mk(short_path()), "short", CFG)
    cap = nt.fmt_potential(inst, "short", "1h", hit)
    lines = cap.split("\n")
    assert lines[0] == ("🔴 <b>SUIUSDT</b> — 1H Compression  ·  "
                        "Short Potential Break")
    assert lines[1] == ("🎯 1H box · break at 1H · "
                        "touch EL 100.1 → lower high 108 ✓")


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
    # rev. "show fired too" (2026-09-30): the pattern still DISPLAYS for
    # a flagged side, marked fired — it just never re-alerts
    hits = inst.potential_hits or {}
    assert hits.get("long") and hits["long"]["fired"] is True
    assert hits["long"]["pattern_tf"] == "1h"


def test_potential_hits_show_fired_and_clear(cfg, monkeypatch):
    """sticky-potential display (rev. 2026-09-30 'show fired too'): the
    unfiltered pattern is shown even after its touch alerted (fired=True);
    it clears only when the pattern itself leaves (<12h window / zone)."""
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    key = {(inst.symbol, "1h"): mk(long_path()),
           (inst.symbol, "15m"): mk(_line(100, 100, 40))}
    acts = e._potential_actions({inst.id: inst}, key, now_s=NOW_S)
    assert len(acts) == 1
    hits = inst.potential_hits or {}
    assert hits["long"]["pattern_tf"] == "1h"
    assert hits["long"]["fired"] is True      # alerted in THIS pass
    assert isinstance(hits["long"]["swing_ts"], int)
    assert hits.get("short") is None          # long fixture has no EL touch
    # same touch next pass: no re-fire, but the PATTERN still displays
    e._potential_actions({inst.id: inst}, key, now_s=NOW_S + 600)
    hits2 = inst.potential_hits or {}
    assert hits2["long"]["fired"] is True
    # pattern leaves the data (flat path) -> display clears
    e._potential_actions({inst.id: inst},
                         {(inst.symbol, "1h"): mk(_line(100, 100, 40)),
                          (inst.symbol, "15m"): mk(_line(100, 100, 40))},
                         now_s=NOW_S + 1200)
    assert (inst.potential_hits or {}).get("long") is None
    assert (inst.potential_hits or {}).get("short") is None


def test_potential_hits_round_trip(cfg):
    inst = _inst()
    inst.notified = {"compression": False, "breakout": False}
    inst.potential_hits = {"long": {"pattern_tf": "15m", "swing_ts": 111,
                                     "touch_ts": 222},
                           "short": None}
    d = inst.to_dict()
    inst2 = Instance.from_dict(d)
    assert inst2.potential_hits == {"long": {"pattern_tf": "15m",
                                              "swing_ts": 111,
                                              "touch_ts": 222},
                                     "short": None}
    assert inst2.to_dict() == d


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


def test_touch_ts_int_survives_restart_not_coerced_to_bool():
    """2026-09-28: _notified_from coerced every value with bool() — the
    touch epoch ms became True, since_touch_ts=1 made both per-touch
    filters no-op and `hit_ts <= spent` never tripped -> SAND posted
    twice (3162 + 3167) on the restart between them."""
    inst = _inst(notified={"compression": False, "breakout": False,
                           "potential_long": True,
                           "potential_long_touch_ts": 1790550000000})
    back = type(inst).from_dict(inst.to_dict())
    ts = back.notified.get("potential_long_touch_ts")
    assert ts == 1790550000000
    assert isinstance(ts, int) and not isinstance(ts, bool)


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


# ── fast proximity pass (60s, user rule 2026-09-28) ────────────────────
def _afut(value):
    async def _ret(*a, **k):
        return value
    return _ret()


def _stub_prox_env(monkeypatch, e, inst, price):
    e.instances[inst.id] = inst      # the pass iterates self.instances
    long_c = mk(long_path())
    e._last_candles = {(inst.symbol, "1h"): long_c,
                       (inst.symbol, "15m"): long_c}
    monkeypatch.setattr(eng_mod, "fetch_last_price",
                        lambda ses, sym: _afut(price))
    monkeypatch.setattr(eng_mod, "fetch_klines",
                        lambda ses, sym, tf, lim: _afut(long_c))
    monkeypatch.setattr(eng_mod, "time",
                        types.SimpleNamespace(time=lambda: NOW_S))


def test_proximity_pass_fires_when_price_in_zone(cfg, monkeypatch):
    """Price inside a trigger tf's close zone -> the trigger runs BETWEEN
    scans (the 15-minute scan cadence was parking formed triggers for a
    full cycle) and fires through the shared register path."""
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    _stub_prox_env(monkeypatch, e, inst, price=109.0)   # in zone
    sends = asyncio.run(e._proximity_pass(None))
    assert inst.notified.get("potential_long") is True
    assert any(ev["kind"] == "potential" for ev in inst.events)
    assert sends == []               # dry-run engine -> dispatch suppresses


def test_proximity_pass_silent_when_out_of_zone(cfg, monkeypatch):
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    _stub_prox_env(monkeypatch, e, inst, price=105.0)   # far from the line
    sends = asyncio.run(e._proximity_pass(None))
    assert sends == []
    assert not inst.notified.get("potential_long")
    assert inst.events == []


def test_proximity_pass_silent_when_price_past_the_line(cfg, monkeypatch):
    """A close beyond the boundary belongs to the breakout verdict."""
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    _stub_prox_env(monkeypatch, e, inst, price=110.2)   # past 109.5
    sends = asyncio.run(e._proximity_pass(None))
    assert sends == []
    assert not inst.notified.get("potential_long")


# ── display mode (user rule 2026-09-30, AKE case) ────────────────────
def test_display_mode_skips_zone_gate():
    """for_display shows the PATTERN even when live price has walked away
    from the boundary — AKE: the alert fired, price left the zone, the
    cell blanked. Firing keeps rejecting the same input."""
    c = mk(long_path())
    assert fp(c, "long", CFG, live_price=95.0, for_display=True) is not None
    assert fp(c, "long", CFG, live_price=95.0) is None


def test_display_mode_shows_price_past_the_line():
    c = mk(long_path())
    assert fp(c, "long", CFG, live_price=115.0, for_display=True) is not None
    assert fp(c, "long", CFG, live_price=115.0) is None


def test_scan_display_survives_candle_walk_away(cfg, monkeypatch):
    """Scan pass: the swing tf's own close walked out of the zone — no
    fire (zone gate), but the display cell stays populated."""
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    acts = e._potential_actions(
        {inst.id: inst},
        {(inst.symbol, "1h"): mk(short_path(end=105.0, tail_n=12))},
        now_s=NOW_S)
    assert acts == []                       # firing: zone gate rejects
    hits = inst.potential_hits or {}
    assert hits.get("short"), hits          # display: pattern still shown
    assert hits["short"]["pattern_tf"] == "1h"
    assert hits["short"]["fired"] is False


def test_proximity_display_refresh_out_of_zone(cfg, monkeypatch):
    """60s display refresh is UNGATED: live price far from the boundary
    still writes the cell (firing stays silent — old code `continue`d
    here before any display write)."""
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    _stub_prox_env(monkeypatch, e, inst, price=95.0)   # far from lines
    inst.potential_hits = None
    sends = asyncio.run(e._proximity_pass(None))
    assert sends == []
    assert not inst.notified.get("potential_long")     # no fire (unchanged)
    hits = inst.potential_hits or {}
    assert hits.get("long") is not None               # display refreshed
    assert hits["long"]["fired"] is False


def test_proximity_display_shows_fired_side_out_of_zone(cfg, monkeypatch):
    """A fired touch keeps displaying (fired=True) while the pattern is
    <12h old, even with price far away — the show-fired-too contract."""
    e = _engine(cfg, monkeypatch)
    inst = _inst(notified={"compression": True, "breakout": False,
                           "potential_short": True,
                           "potential_short_touch_ts": T0 + 10_000_000})
    e.instances[inst.id] = inst
    short_c = mk(short_path())
    e._last_candles = {(inst.symbol, "1h"): short_c,
                       (inst.symbol, "15m"): short_c}
    monkeypatch.setattr(eng_mod, "fetch_last_price",
                        lambda ses, sym: _afut(95.0))
    monkeypatch.setattr(eng_mod, "fetch_klines",
                        lambda ses, sym, tf, lim: _afut(short_c))
    monkeypatch.setattr(eng_mod, "time",
                        types.SimpleNamespace(time=lambda: NOW_S))
    asyncio.run(e._proximity_pass(None))
    hits = inst.potential_hits or {}
    assert hits.get("short") and hits["short"]["fired"] is True


# ── the same-scan invariant (user rule 2026-09-28, ONEUSDT case) ──────
def test_potential_pass_runs_after_the_lifecycle_pass(cfg, monkeypatch):
    """A box born in THIS scan must be visible to the potential pass:
    lifecycle (creation) first, potential check after — the old order
    (B before C) forced every new box to wait a full scan cycle (15 min)
    before its first check (ONEUSDT: box 10:01:30, potential 10:16:50)."""
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
    assert "lifecycle" in order
    assert order[-1] == "potential", order[-4:]
    # potential sees the post-lifecycle instance dict (born boxes included)
    assert order.index("lifecycle") < len(order) - 1


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
    close > lo2 + 2.4 — every tail stays under that (live route only).
    The LIVE_CLEAR_ATR gate is GONE (2026-09-28 trigger rewrite): what
    decides is the close zone, so "clear" parks the close inside
    1.5×ATR of the boundary and the others deliberately outside it.
    """
    hi0, hi1, lo2 = 110.1, 110.0, 106.7
    base = (_line(100, hi0, 12) + _line(hi0, 100, 12) +
            _line(100, hi1, 12) + _line(hi1, lo2, 8))
    if tail == "clear":
        path = base + _line(lo2, 109.0, 6)        # in zone, low unconfirmed
    elif tail == "dip":
        path = base + _line(lo2, lo2 + 0.9, 6)   # clear of 1x, short of zone
    elif tail == "flat":
        path = base + _line(lo2, lo2 + 0.05, 6)
    else:                                        # below the box band
        path = base + _line(99.0, 100.6, 6)
    return mk(path, spread=1.0)


def test_live_third_swing_fires_before_the_zigzag_confirm():
    from compression_detection.potential_break import _zigzag
    candles = _live_long_path()
    # the third low is NOT in the confirmed zigzag — only the live route
    # can produce this hit
    assert all(abs(p.price - 106.7) > 1e-9 for p in _zigzag(candles, CFG))
    hit = fp(candles, "long")
    assert hit is not None
    assert hit["live"] is True
    assert hit["second_price"] == 106.7
    assert hit["touch_price"] == 110.0
    # the alert carries the swing bar itself (no replay-confirm)
    assert hit["confirm_ts"] == hit["second_ts"]


def test_live_swing_below_the_zone_is_rejected():
    """The retired clear gate was replaced by the close zone: a live
    swing whose close parks short of 1.5×ATR from the line never fires."""
    assert fp(_live_long_path(tail="flat"), "long") is None
    assert fp(_live_long_path(tail="dip"), "long") is None


def test_live_third_below_the_box_band_rejected():
    assert fp(_live_long_path(tail="below"), "long") is None


def test_short_mirror_fires():
    """SHORT mirror of the trigger (2026-09-28): EL at the floor, LH
    after it, close settling in the zone above the boundary. Here the
    decline CONFIRMS the swing — confirmed or live are both fine per
    the spec ("if it's confirmed, it's OK too"); the live route itself
    is covered by the long test above."""
    path = (_line(110, 100, 6) + _line(100, 110, 6) +
            _line(110, 100.3, 6) + _line(100.3, 103, 4) +
            _line(103, 101.2, 3))
    hit = fp(mk(path), "short")
    assert hit is not None
    assert hit["second_price"] == 103.0
    assert hit["second_label"] == "LH"
    assert hit["touch_price"] == 100.3


def test_touch_must_be_the_last_high_on_the_line():
    """The trigger's touch is the LAST high of its tf — a newer, off-line
    high (107 here) supersedes any earlier EH, so nothing qualifies."""
    hi0, hi1 = 110.1, 110.0
    path = (_line(100, hi0, 12) + _line(hi0, 100, 12) +
            _line(100, hi1, 12) + _line(hi1, 104, 10) +   # low 104 confirms
            _line(104, 107, 8) +                          # high 107 confirms
            _line(107, 104.5, 14) +                       # low 104.5 (live)
            _line(104.5, 105.6, 6))                       # clears, not confirm
    candles = mk(path, spread=1.0)
    assert fp(candles, "long") is None


# ── one alert PER TOUCH + lower-tf touches (user rules 2026-09-27 eve) ──

def _rearm_round1():
    """Round 1 complete: EH 110.0 -> HL 105 -> settle in-zone (109.1)."""
    return (_line(100, 110.1, 12) + _line(110.1, 100, 12) +
            _line(100, 110.0, 12) + _line(110.0, 105, 8) +
            _line(105, 109.1, 6))


def _rearm_round2():
    """Round 1, then a NEWER touch (109.7, inside the 1.0 band) ->
    HL 106.2 -> in-zone."""
    return (_rearm_round1() + _line(109.1, 109.7, 8) +
            _line(109.7, 106.2, 6) + _line(106.2, 109.1, 6))


def test_one_potential_per_touch_rearm():
    """A fired touch is SPENT (since_touch_ts); a strictly newer touch
    with its own swing re-arms — touch -> one swing, no exceptions."""
    full = mk(_rearm_round2())
    r_full = fp(full, "long")
    assert r_full is not None
    # round1 alone fires with its own touch
    r_mid = fp(mk(_rearm_round1()), "long")
    assert r_mid is not None
    assert r_mid["touch_ts"] < r_full["touch_ts"]
    # round1's touch spent -> round2's newer touch re-arms on full candles
    r_again = fp(full, "long", since_touch_ts=r_mid["touch_ts"])
    assert r_again is not None
    assert r_again["touch_ts"] == r_full["touch_ts"]
    # ...and once THAT touch is spent, nothing remains
    assert fp(full, "long", since_touch_ts=r_full["touch_ts"]) is None


def test_trigger_is_per_tf_no_cross_tf_pairing():
    """2026-09-28 rewrite: touch AND swing must come from the SAME tf.
    A touch on the lower tf with the swing on the main tf no longer
    combines — each tf is judged whole, or not at all."""
    main = mk(_line(100, 110.1, 12) + _line(110.1, 100, 12) +
              _line(100, 107, 12) + _line(107, 105, 4) +
              _line(105, 118, 10))              # L 105 = last main pivot
    low = mk(_line(100, 110.1, 12) + _line(110.1, 100, 12) +
             _line(100, 110, 12) + _line(110, 99, 4))   # EH 110 touch only
    hit = fp(main, "long", main_candles=main,
             scan=[("15m", low), ("1h", main)])
    assert hit is None, "cross-tf pairs must never combine"

    # a COMPLETE trigger on the box's own tf still fires with the lower
    # tf present but empty of patterns
    full = mk(_line(100, 110.1, 12) + _line(110.1, 100, 12) +
              _line(100, 110, 12) + _line(110, 105, 8) +
              _line(105, 108.9, 6))
    hit2 = fp(full, "long", main_candles=full,
              scan=[("15m", low), ("1h", full)])
    assert hit2 is not None
    assert hit2["pattern_tf"] == "1h"


# ── LIVE main-tf touch (user choice 2026-09-28, option A / SAND) ─────────

def _live_touch_main():
    """Main tf whose final HIGH is still UNCONFIRMED (the tail max close
    never drops 1.2 x ATR7 below it): the touch exists structurally but
    the zigzag won't yield it for hours — the SAND chain."""
    hi0, top = 110.1, 110.2
    return (_line(100, hi0, 12) + _line(hi0, 100, 12) +
            _line(100, top, 6) + _line(top, 109.6, 4))   # ends mid-dip


def _live_touch_third_confirmed():
    """15m: rise, minor high (HH, off the line), pullback low 105 = the
    confirming third, then a rally that confirms it. The 110.2 high sits
    ON the box line; the second 15m high (108) is HH and off-line, so no
    confirmed touch exists anywhere — only the LIVE one can fire."""
    return (_line(100, 110.1, 48) + _line(110.1, 100, 48) +
            _line(100, 108, 24) + _line(108, 105, 8) +
            _line(105, 118, 12))


def test_unconfirmed_touch_does_not_fire():
    """2026-09-28 rewrite: the touch must be a CONFIRMED EH/EL (the
    no-confirm clause belongs to the HIGHER LOW only). The SAND-era
    live-touch route is retired — lower tfs exist precisely so the touch
    confirms fast somewhere."""
    main = mk(_live_touch_main())
    low = mk(_live_touch_third_confirmed())
    hit = fp(main, "long", main_candles=main,
             scan=[("15m", low), ("1h", main)])
    assert hit is None, "a provisional (unconfirmed) touch never fires"


def test_live_main_touch_rejected_when_the_label_is_not_eh():
    """A provisional high that is HH (not equal to the previous high) is
    a new extreme, not a touch on the box ceiling."""
    main = mk(_line(100, 107, 12) + _line(107, 100, 12) +
              _line(100, 110.2, 6) + _line(110.2, 109.6, 4))
    low = mk(_live_touch_third_confirmed())
    hit = fp(main, "long", main_candles=main,
             scan=[("15m", low), ("1h", main)])
    assert hit is None


def test_live_main_touch_rejected_when_off_the_line():
    """Label is EH but the provisional sits way off the box line — no
    touch (the confirmed H is off-line too, so nothing else qualifies)."""
    main = mk(_line(100, 111.3, 12) + _line(111.3, 100, 12) +
              _line(100, 111.5, 6) + _line(111.5, 111.0, 4))
    low = mk(_live_touch_third_confirmed())
    hit = fp(main, "long", main_candles=main,
             scan=[("15m", low), ("1h", main)])
    assert hit is None


def test_lower_tf_touch_blocked_by_intervening_low_tf_pivot():
    """Adjacency: a pivot on the touch's own tf between touch and third
    means the structure already moved — no alert."""
    main = mk(_line(100, 110.1, 12) + _line(110.1, 100, 12) +
              _line(100, 107, 12) + _line(107, 105, 4) +
              _line(105, 118, 10))               # same third as above
    low = mk(_line(100, 110.1, 12) + _line(110.1, 100, 12) +
             _line(100, 110, 12) + _line(110, 99, 2) +
             _line(99, 105, 4))                  # L 99 pivot between them
    hit = fp(main, "long", main_candles=main,
             scan=[("15m", low), ("1h", main)])
    assert hit is None


def test_rearm_fires_again_on_a_new_touch(cfg, monkeypatch):
    """Engine arming: fire #1 stores its touch_ts; the same touch never
    fires twice; a newer touch arms fire #2; then silence."""
    e = _engine(cfg, monkeypatch)
    inst = _inst()
    k1 = mk(_rearm_round1())
    k2 = mk(_rearm_round2())

    a1 = e._potential_actions({inst.id: inst},
                              {(inst.symbol, "1h"): k1,
                               (inst.symbol, "15m"): k1}, now_s=NOW_S)
    assert len(a1) == 1
    stored1 = inst.notified.get("potential_long_touch_ts")
    assert stored1 is not None

    a2 = e._potential_actions({inst.id: inst},
                              {(inst.symbol, "1h"): k2,
                               (inst.symbol, "15m"): k2},
                              now_s=NOW_S + 600)
    assert len(a2) == 1, "a NEW touch must re-arm the same side"
    stored2 = inst.notified.get("potential_long_touch_ts")
    assert stored2 is not None and stored2 > stored1

    a3 = e._potential_actions({inst.id: inst},
                              {(inst.symbol, "1h"): k2,
                               (inst.symbol, "15m"): k2},
                              now_s=NOW_S + 1200)
    assert a3 == [], "the same touch cannot fire a third time"
    assert len([ev for ev in inst.events if ev["kind"] == "potential"]) == 2


# ── label gate: the third must BE a higher/lower low (PLUME case) ──────

def test_confirmed_long_third_equal_low_is_rejected():
    """Third clears box_low+eq but is EQUAL to the previous low (EL) —
    that's a retest, not a higher low. Control fires with a real HL."""
    eq_retest = (_line(100, 110, 12) + _line(110, 105, 8) +
                 _line(105, 109.9, 8) + _line(109.9, 105.3, 6) +
                 _line(105.3, 109.1, 6))        # 105.3 vs 105 = EL (in zone)
    assert fp(mk(eq_retest), "long") is None

    real_hl = (_line(100, 110, 12) + _line(110, 105, 8) +
               _line(105, 109.9, 8) + _line(109.9, 106.8, 6) +
               _line(106.8, 109.1, 6))          # 106.8 vs 105 = HL (in zone)
    hit = fp(mk(real_hl), "long")
    assert hit is not None
    assert hit["second_label"] == "HL"


def test_confirmed_short_third_equal_high_is_rejected():
    """Third below box_top - eq but EQUAL to the previous high (EH) —
    price retested the top, not a lower high. Control fires with a real LH."""
    eq_retest = (_line(110, 100, 12) + _line(100, 105, 10) +
                 _line(105, 100.2, 8) + _line(100.2, 105.0, 8) +
                 _line(105.0, 100.3, 6))        # 105.0 vs H1 105 = EH (in zone)
    assert fp(mk(eq_retest), "short") is None

    real_lh = (_line(110, 100, 12) + _line(100, 105, 10) +
               _line(105, 100.2, 8) + _line(100.2, 103.0, 8) +
               _line(103.0, 100.3, 6))          # 103 vs 105 = LH (in zone)
    hit = fp(mk(real_lh), "short")
    assert hit is not None
    assert hit["second_label"] == "LH"


def test_live_short_third_equal_high_is_rejected():
    """The LIVE route gets the same label gate: a provisional high equal
    to the previous confirmed high (PLUME 0.01873 vs 0.01881, delta 0.5x
    ATR) must NOT fire as a 'lower high'. The clear-gate passes, the
    band passes (sloped line) — only the label blocks it."""
    lo0, hi1, lo1, hi2 = 99.8, 105.0, 100.0, 104.9   # 104.9 vs H1 105 = EH
    base = (_line(110, lo0, 12) + _line(lo0, hi1, 12) +
            _line(hi1, lo1, 12) + _line(lo1, hi2, 8))
    eq = _eq_at(base, len(base) - 1)
    path = base + _line(hi2, hi2 - min(1.75 * eq + 0.1, 2.0), 6)
    candles = mk(path, spread=1.0)
    # zigzag never confirms the high -> only the live route could fire
    from compression_detection.potential_break import _zigzag
    assert all(abs(p.price - hi2) > 1e-9 for p in _zigzag(candles, CFG))
    assert fp(candles, "short") is None
