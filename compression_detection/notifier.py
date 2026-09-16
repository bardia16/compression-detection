"""Telegram notifications for compression + breakout events.

Format approved by Bardia 2026-09-16 (the "A" suggestions — mirrors the
existing S/R breakout channel format per the standing reuse rule, only
the label changes):

  compression: `🔷 SYM — TF Compression · Type` / `🎯 boundaries lo – up`
  breakout:    `🟢|🔴 SYM — TF Compression · Breakout` /
               `🎯 closing above|below X`

Same bot as the Trading-Alerts scanner ("Trading alerts bot") posting
into the Breakouts channel. Changing these strings means updating the
pinned equality tests in tests/test_notifier.py in the same edit.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import aiohttp

log = logging.getLogger("notify")

_TF_DISPLAY = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}

_TYPE_DISPLAY = {
    "box": "Box",
    "descending_triangle": "Descending Triangle",
    "ascending_triangle": "Ascending Triangle",
    "symmetrical_triangle": "Symmetrical Triangle",
    "falling_wedge": "Falling Wedge",
    "rising_wedge": "Rising Wedge",
}


def tf_label(tf: str) -> str:
    return _TF_DISPLAY.get(tf, tf.upper())


def pretty_type(type_name: str) -> str:
    return _TYPE_DISPLAY.get(type_name, type_name.replace("_", " ").title())


def _fmt_price(v: float) -> str:
    return f"{v:g}"


def fmt_compression(inst) -> str:
    """Compression detected (no direction yet — it is a compression)."""
    lo = inst.metrics.get("lower_at_last_bar", inst.lower_at(inst.pivots[-1]["bar"]))
    up = inst.metrics.get("upper_at_last_bar", inst.upper_at(inst.pivots[-1]["bar"]))
    return (
        f"🔷 <b>{inst.symbol}</b> — {tf_label(inst.tf)} Compression  ·  {pretty_type(inst.type)}\n"
        f"🎯 boundaries {_fmt_price(lo)} – {_fmt_price(up)}"
    )


def fmt_breakout(inst, side: str, boundary: float) -> str:
    """Breakout heads-up / confirmed breakout (direction = actual event)."""
    arrow = "🟢" if side == "up" else "🔴"
    word = "above" if side == "up" else "below"
    return (
        f"{arrow} <b>{inst.symbol}</b> — {tf_label(inst.tf)} Compression  ·  Breakout\n"
        f"🎯 closing <b>{word}</b> {_fmt_price(boundary)}"
    )


def chart_url(chart_base: str, symbol: str, tf: str, lines: List[float]) -> str:
    ls = ",".join(_fmt_price(x) for x in lines)
    return (f"{chart_base}/chart/alert?symbol={symbol.replace('/', '')}"
            f"&timeframe={tf}&cross_lines={ls}")


class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._ses: Optional[aiohttp.ClientSession] = None

    async def _session(self) -> aiohttp.ClientSession:
        if not self._ses or self._ses.closed:
            self._ses = aiohttp.ClientSession()
        return self._ses

    async def _api(self, method: str, **params) -> dict:
        ses = await self._session()
        params["chat_id"] = self.chat_id
        async with ses.post(
            f"https://api.telegram.org/bot{self.token}/{method}", json=params,
        ) as r:
            return await r.json(content_type=None)

    async def post(self, text: str) -> Optional[int]:
        d = await self._api("sendMessage", text=text, parse_mode="HTML")
        if d.get("ok"):
            return d["result"]["message_id"]
        log.warning("tg post failed: %s", d)
        return None

    async def post_photo(self, caption: str, image: bytes) -> Optional[int]:
        ses = await self._session()
        form = aiohttp.FormData()
        form.add_field("chat_id", str(self.chat_id))
        form.add_field("caption", caption)
        form.add_field("parse_mode", "HTML")
        form.add_field("photo", image, filename="alert.png",
                       content_type="image/png")
        try:
            async with ses.post(
                f"https://api.telegram.org/bot{self.token}/sendPhoto", data=form,
            ) as r:
                d = await r.json(content_type=None)
        except Exception as exc:
            log.warning("tg post_photo failed: %s", exc)
            return None
        if d.get("ok"):
            return d["result"]["message_id"]
        log.warning("tg post_photo failed: %s", d)
        return None

    async def delete(self, msg_id: int, why: str = "") -> bool:
        d = await self._api("deleteMessage", message_id=msg_id)
        if d.get("ok"):
            log.info("tg deleted msg %s (%s)", msg_id, why)
            return True
        log.warning("tg delete failed: %s", d)
        return False


async def fetch_chart(
    session: aiohttp.ClientSession, base_url: str, symbol: str, tf: str,
    lines: List[float],
) -> Optional[bytes]:
    """Alert chart PNG from the chart service (:8002). None on failure."""
    url = chart_url(base_url, symbol, tf, lines)
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status == 200:
                data = await r.read()
                if data and len(data) > 100:
                    return data
            log.warning("alert chart %s %s -> HTTP %s", symbol, tf, r.status)
    except Exception as exc:
        log.warning("alert chart fetch failed for %s: %s", symbol, exc)
    return None
