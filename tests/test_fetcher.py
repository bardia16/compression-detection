"""Fetcher tests — forming-candle drop with pinned values + endpoint
fallback via monkeypatched HTTP layer."""
import asyncio

import compression_detection.fetcher as f
from compression_detection.models import Candle


def mk(ts, close=100.0):
    return Candle(ts=ts, open=close, high=close, low=close, close=close, volume=1.0)


def test_drop_forming_pinned():
    tf = "15m"
    now = 1_500_000
    # last candle ts=1_000_000, closes at 1_900_000 > now -> dropped
    out = f.drop_forming([mk(0), mk(1_000_000)], tf, now_ms=now)
    assert [c.ts for c in out] == [0]
    # last candle already closed -> kept
    out2 = f.drop_forming([mk(0), mk(1_000_000)], tf, now_ms=2_000_000)
    assert [c.ts for c in out2] == [0, 1_000_000]
    # unknown tf -> unchanged
    out3 = f.drop_forming([mk(0)], "7m", now_ms=now)
    assert len(out3) == 1


def test_tf_ms_values():
    assert f.TF_MS["15m"] == 900_000
    assert f.TF_MS["1h"] == 3_600_000
    assert f.TF_MS["4h"] == 14_400_000


def test_fetch_klines_conversion_and_fallback(monkeypatch):
    calls = []

    async def fake_get_json(session, url, params):
        calls.append(url)
        if "fapi" in url:
            return [[1000, "10", "12", "9", "11", "5"]]
        raise RuntimeError("spot down")

    monkeypatch.setattr(f, "_get_json", fake_get_json)
    out = asyncio.run(f.fetch_klines(None, "ADA/USDT", "15m", 10))
    assert len(calls) == 2                       # spot tried, futures fell back
    assert calls[0] == f.SPOT_KLINES and calls[1] == f.FAPI_KLINES
    assert out and out[0].ts == 1000 and out[0].close == 11.0


def test_fetch_klines_all_fail(monkeypatch):
    async def boom(session, url, params):
        raise RuntimeError("down")

    monkeypatch.setattr(f, "_get_json", boom)
    out = asyncio.run(f.fetch_klines(None, "ADA/USDT", "15m", 10))
    assert out == []


def test_fetch_last_price_fallback(monkeypatch):
    async def fake(session, url, params):
        if "fapi" in url:
            return {"price": "1.23"}
        raise RuntimeError("spot down")

    monkeypatch.setattr(f, "_get_json", fake)
    assert asyncio.run(f.fetch_last_price(None, "ADA/USDT")) == 1.23
