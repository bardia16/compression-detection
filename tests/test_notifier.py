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
