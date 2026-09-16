"""Engine integration tests — full pipeline on synthetic candles through
the REAL zigzag/labels/detector, with HTTP layers monkeypatched.

A 72-bar sine wave (100 ± 3) produces a clean 11-pivot Box (EH/EL
alternating) through the actual ZigZag — used as the fixture.
"""
import asyncio
import json
import math

import pytest

import compression_detection.engine as eng_mod
from compression_detection import universe as uni
from compression_detection.config import Config
from compression_detection.models import Candle

STEP = 900_000
START = 1_700_000_000_000


def box_closes(n=72):
    return [100 + 3 * math.sin(2 * math.pi * k / 12) for k in range(0, n)]


def mk_candles(closes):
    return [Candle(ts=START + i * STEP, open=c, high=c + 0.05, low=c - 0.05,
                   close=c, volume=1.0)
            for i, c in enumerate(closes)]


class FakeTG:
    def __init__(self):
        self.posts = []
        self.deletes = []
        self._n = 100

    async def post(self, text):
        self.posts.append(text)
        self._n += 1
        return self._n

    async def post_photo(self, caption, img):
        return await self.post(caption)

    async def delete(self, mid, why=""):
        self.deletes.append(mid)
        return True


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = Config.load()
    c.timeframes = ["15m"]
    c.state_path = tmp_path / "structure_state.json"
    # isolate audit + report writers (tests must not pollute prod dirs)
    monkeypatch.setattr(eng_mod, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(eng_mod, "AUDIT_DIR", tmp_path / "audit")
    return c


def patch_env(monkeypatch, closes, last_price=None):
    monkeypatch.setattr(uni, "load_api_key", lambda root: "k")
    monkeypatch.setattr(uni, "get_btc_price", lambda: 2.0)
    monkeypatch.setattr(uni, "get_universe", lambda *a, **kw: ["AAA/USDT"])

    async def avail(ses):
        return {"AAAUSDT"}
    monkeypatch.setattr(uni, "load_available_symbols", avail)

    async def fake_klines(ses, pair, tf, limit):
        return mk_candles(closes)
    monkeypatch.setattr(eng_mod, "fetch_klines", fake_klines)

    if last_price is not None:
        async def fake_price(ses, symbol):
            return last_price
        monkeypatch.setattr(eng_mod, "fetch_last_price", fake_price)

    async def no_chart(self, ses, inst, lines, trend_lines=None):
        return None
    monkeypatch.setattr(eng_mod.Engine, "_chart", no_chart)

    # Synthetic sine candles are not realistic for the candle-problem gate
    # (open==close per candle reads as gaps); gate tests override this to True.
    monkeypatch.setattr(eng_mod, "has_candle_problems", lambda candles: False)


def scan(cfg, tg=None, dry=False):
    e = eng_mod.Engine(cfg, dry_run=dry, tg=tg)
    return e, asyncio.run(e.scan_once())


def test_dry_run_detects_box_without_writes(cfg, monkeypatch):
    patch_env(monkeypatch, box_closes())
    e, res = scan(cfg, dry=True)
    s = res["summary"]
    assert s["universe_count"] == 1
    assert s["instances_active"] == 1
    inst = s["active_instances"][0]
    assert inst["type"] == "box"
    assert inst["pivot_count"] >= 11
    assert inst["state"] == "compressing"          # 11 pivots >> min 4
    assert not cfg.state_path.exists()             # dry run never writes
    kinds = {a["kind"] for a in s["actions"]}
    assert "compression_notify" in kinds           # would-notify is visible
    assert s["sends"] == []                        # but nothing sent
    types = {c["type"] for d in s["detections"] for c in d["candidates"]}
    assert "box" in types


def test_full_scan_persists_state(cfg, monkeypatch):
    patch_env(monkeypatch, box_closes())
    e, res = scan(cfg)
    assert cfg.state_path.exists()
    d = json.loads(cfg.state_path.read_text())
    assert len(d["instances"]) == 1
    inst = list(d["instances"].values())[0]
    assert inst["type"] == "box"
    # new-instance hygiene: watches only candles after creation scan
    assert inst["last_eval_ts"] == mk_candles(box_closes())[-1].ts


def test_breakout_close_fires_end_to_end(cfg, monkeypatch):
    tg = FakeTG()
    cfg.notif_enabled = True
    # scan 1: detect + notify
    patch_env(monkeypatch, box_closes())
    e1, res1 = scan(cfg, tg=tg)
    assert len(tg.posts) == 1
    assert "Compression" in tg.posts[0]

    # scan 2: breakout candle closes above the upper boundary (~103)
    closes2 = box_closes() + [104.5]
    patch_env(monkeypatch, closes2)
    e2, res2 = scan(cfg, tg=tg)
    assert any("Breakout" in p and "above" in p for p in tg.posts)
    d = json.loads(cfg.state_path.read_text())
    inst = list(d["instances"].values())[0]
    assert inst["state"] == "breakout"
    assert inst["notified"]["breakout"] is True

    # scan 3: terminal — no further actions
    e3, res3 = scan(cfg, tg=tg)
    assert res3["summary"]["actions"] == []


def test_held_notifications_flush_on_enable(cfg, monkeypatch):
    """Warmup holds notifications (no flag consumed); enabling flushes the
    current compressions (user rule 2026-09-16)."""
    cfg.notif_enabled = False
    tg = FakeTG()
    patch_env(monkeypatch, box_closes())
    e1, res1 = scan(cfg, tg=tg)                        # notifications disabled
    assert tg.posts == []
    inst1 = list(e1.instances.values())[0]
    assert inst1.notified["compression"] is False      # held, not consumed

    cfg.notif_enabled = True
    e2, res2 = scan(cfg, tg=tg)
    assert len(tg.posts) == 1
    assert "Compression" in tg.posts[0] and "boundaries" in tg.posts[0]
    inst2 = list(e2.instances.values())[0]
    assert inst2.notified["compression"] is True


def test_symbols_stored_full_binance_form(cfg, monkeypatch):
    """Instances carry the full symbol (AAAUSDT) — bare AAA broke charts,
    ticker calls and probe fetches (2026-09-16)."""
    patch_env(monkeypatch, box_closes())
    e, res = scan(cfg)
    inst = list(e.instances.values())[0]
    assert inst.symbol == "AAAUSDT"


def test_candle_gate_holds_compression(cfg, monkeypatch):
    tg = FakeTG()
    patch_env(monkeypatch, box_closes())
    monkeypatch.setattr(eng_mod, "has_candle_problems", lambda candles: True)
    e, res = scan(cfg, tg=tg)
    assert tg.posts == []
    inst = list(e.instances.values())[0]
    assert inst.notified["compression"] is False       # held for retry
    assert any(a["kind"] == "compression_notify"
               for a in res["summary"]["actions"])     # action existed, gated


def test_candle_gate_holds_then_sends_when_clean(cfg, monkeypatch):
    tg = FakeTG()
    patch_env(monkeypatch, box_closes())
    monkeypatch.setattr(eng_mod, "has_candle_problems", lambda candles: True)
    scan(cfg, tg=tg)
    assert tg.posts == []                              # suppressed while dirty

    monkeypatch.setattr(eng_mod, "has_candle_problems", lambda candles: False)
    scan(cfg, tg=tg)
    assert len(tg.posts) == 1                          # flush once clean


def test_probe_candle_gate_suppresses(cfg, monkeypatch):
    tg = FakeTG()
    cfg.notif_enabled = True
    patch_env(monkeypatch, box_closes())
    e, _ = scan(cfg, tg=tg)

    open_ms = mk_candles(box_closes())[-1].ts + STEP
    monkeypatch.setattr(eng_mod, "probe_due", lambda *a, **kw: open_ms)
    patch_env(monkeypatch, box_closes(), last_price=103.6)
    monkeypatch.setattr(eng_mod, "has_candle_problems", lambda candles: True)
    n = asyncio.run(e.probe_once(None))
    assert n == 0
    assert not any("Breakout" in p for p in tg.posts)
    inst = list(e.instances.values())[0]
    assert inst.probe_msg_id is None                   # heads-up suppressed


def test_probe_posts_heads_up_and_retracts_on_failed_close(cfg, monkeypatch):
    tg = FakeTG()
    cfg.notif_enabled = True
    patch_env(monkeypatch, box_closes())
    e, _ = scan(cfg, tg=tg)

    # force a probe due on the next candle; live price beyond upper
    open_ms = mk_candles(box_closes())[-1].ts + STEP
    monkeypatch.setattr(eng_mod, "probe_due", lambda *a, **kw: open_ms)
    patch_env(monkeypatch, box_closes(), last_price=103.6)
    n = asyncio.run(e.probe_once(None))
    assert n == 1
    assert any("Breakout" in p for p in tg.posts)
    inst = list(e.instances.values())[0]
    assert inst.probe_msg_id is not None
    mid = inst.probe_msg_id

    # scan: candle closes back INSIDE -> heads-up retracted, structure alive
    closes2 = box_closes() + [101.0]
    patch_env(monkeypatch, closes2)
    e2, res2 = scan(cfg, tg=tg)
    assert mid in tg.deletes
    d = json.loads(cfg.state_path.read_text())
    inst2 = list(d["instances"].values())[0]
    assert inst2["state"] != "breakout"
    assert inst2["probe_msg_id"] is None


def test_probe_confirmed_close_keeps_message(cfg, monkeypatch):
    tg = FakeTG()
    cfg.notif_enabled = True
    patch_env(monkeypatch, box_closes())
    e, _ = scan(cfg, tg=tg)

    open_ms = mk_candles(box_closes())[-1].ts + STEP
    monkeypatch.setattr(eng_mod, "probe_due", lambda *a, **kw: open_ms)
    patch_env(monkeypatch, box_closes(), last_price=103.6)
    asyncio.run(e.probe_once(None))
    inst = list(e.instances.values())[0]
    mid = inst.probe_msg_id

    # scan: candle CLOSES beyond -> message kept, breakout recorded
    closes2 = box_closes() + [104.5]
    patch_env(monkeypatch, closes2)
    e2, res2 = scan(cfg, tg=tg)
    assert mid not in tg.deletes
    d = json.loads(cfg.state_path.read_text())
    inst2 = list(d["instances"].values())[0]
    assert inst2["state"] == "breakout"
