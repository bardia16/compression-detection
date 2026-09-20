"""Notifier tests — format strings are PINNED here.

⚠️ When Bardia approves a different mock, update fmt_* and these pinned
strings together (one-line diffs).
"""
from types import SimpleNamespace

from compression_detection.notifier import (
    chart_url, fmt_breakout, fmt_compression, pretty_type, tf_label,
)


def fake_inst():
    return SimpleNamespace(
        symbol="BTCUSDT", tf="4h", type="falling_wedge",
        metrics={"lower_at_last_bar": 64.12, "upper_at_last_bar": 66.8},
        pivots=[{"bar": 100}],
        lower_at=lambda b: 64.12, upper_at=lambda b: 66.8,
    )


def box_inst():
    return SimpleNamespace(
        symbol="BTCUSDT", tf="4h", type="box",
        metrics={"lower_at_last_bar": 64.0, "upper_at_last_bar": 67.0},
        pivots=[{"bar": 100, "side": "H", "price": 66.8},
                {"bar": 96, "side": "L", "price": 64.12}],
        last_low_price=lambda: 64.12, last_high_price=lambda: 66.8,
        lower_at=lambda b: 64.0, upper_at=lambda b: 67.0,
    )


def test_tf_label():
    assert tf_label("15m") == "15m"
    assert tf_label("1h") == "1H"
    assert tf_label("4h") == "4H"


def test_pretty_type():
    assert pretty_type("box") == "Box"
    assert pretty_type("descending_triangle") == "Descending Triangle"
    assert pretty_type("symmetrical_triangle") == "Symmetrical Triangle"
    assert pretty_type("rising_wedge") == "Rising Wedge"


def test_fmt_compression_pinned():
    assert fmt_compression(fake_inst()) == (
        "🔷 <b>BTCUSDT</b> — 4H Compression  ·  Falling Wedge\n"
        "🎯 boundaries 64.12 – 66.8"
    )


def test_fmt_compression_box_uses_last_pivot_levels():
    """Box boundaries show the last pivot levels — same values its flat
    chart lines and breakout levels use (user rule 2026-09-16)."""
    assert fmt_compression(box_inst()) == (
        "🔷 <b>BTCUSDT</b> — 4H Compression  ·  Box\n"
        "🎯 boundaries 64.12 – 66.8"
    )


def test_fmt_breakout_pinned_up():
    assert fmt_breakout(fake_inst(), "up", 66.8) == (
        "🟢 <b>BTCUSDT</b> — 4H Compression  ·  Breakout\n"
        "🎯 closing <b>above</b> 66.8"
    )


def test_fmt_breakout_pinned_down():
    assert fmt_breakout(fake_inst(), "down", 64.12) == (
        "🔴 <b>BTCUSDT</b> — 4H Compression  ·  Breakout\n"
        "🎯 closing <b>below</b> 64.12"
    )


def test_chart_url():
    u = chart_url("http://localhost:8002", "BTC/USDT", "4h", [64.12, 66.8])
    assert u == ("http://localhost:8002/chart/alert?symbol=BTCUSDT"
                 "&timeframe=4h&cross_lines=64.12,66.8")


def test_chart_url_trend_lines():
    """Sloped boundary segments ride along as trend_lines (user rule
    2026-09-16: draw the real lines, extrapolated from the pivots)."""
    segs = [(1788523200000, 0.00105789, 1789531200000, 0.00077562),
            (1788523200000, 0.00082193, 1789531200000, 0.00083038)]
    u = chart_url("http://localhost:8002", "BOMEUSDT", "4h", None, segs)
    assert u == ("http://localhost:8002/chart/alert?symbol=BOMEUSDT"
                 "&timeframe=4h&trend_lines="
                 "1788523200000,0.00105789,1789531200000,0.00077562|"
                 "1788523200000,0.00082193,1789531200000,0.00083038")


def test_chart_url_lines_and_trend_lines():
    u = chart_url("http://localhost:8002", "BOMEUSDT", "4h",
                  [0.0008724], [(1788523200000, 0.001, 1789531200000, 0.0008)])
    assert "&cross_lines=0.0008724" in u
    assert "&trend_lines=1788523200000,0.001,1789531200000,0.0008" in u


def test_post_passes_reply_to_message_id(monkeypatch):
    """Breakout messages thread onto their compression confirmation
    (user rule 2026-09-16)."""
    import asyncio
    from compression_detection.notifier import Telegram

    tg = Telegram("tok", "chat")
    captured = {}

    async def fake_api(method, **params):
        captured.update(params)
        captured["method"] = method
        return {"ok": True, "result": {"message_id": 7}}

    monkeypatch.setattr(tg, "_api", fake_api)
    mid = asyncio.run(tg.post("hi", reply_to_id=42))
    assert mid == 7
    assert captured["method"] == "sendMessage"
    assert captured["reply_to_message_id"] == 42

    captured.clear()
    asyncio.run(tg.post("hi"))                      # standalone: no reply key
    assert "reply_to_message_id" not in captured


def test_post_chat_id_override(monkeypatch):
    """DM mirror (user rule 2026-09-20): the same content can target another
    chat (Bardia's DM) without touching the channel default."""
    import asyncio
    from compression_detection.notifier import Telegram

    tg = Telegram("tok", "chan")
    captured = {}

    async def fake_api(method, chat_id=None, **params):
        captured["method"] = method
        captured["chat_id"] = chat_id
        return {"ok": True, "result": {"message_id": 9}}

    monkeypatch.setattr(tg, "_api", fake_api)
    mid = asyncio.run(tg.post("hi", chat_id="999"))
    assert mid == 9
    assert captured["chat_id"] == "999"
    asyncio.run(tg.post("hi"))                      # default: channel chat
    assert captured["chat_id"] is None
    assert asyncio.run(tg.delete(5, chat_id="999"))
    assert captured["method"] == "deleteMessage"
