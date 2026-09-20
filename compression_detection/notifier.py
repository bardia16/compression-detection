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
from urllib.parse import quote

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
    """Compression detected (no direction yet — it is a compression).
    Boundaries: box → the last pivot levels (same values its flat chart
    lines and breakout levels use); triangles/wedges → the fitted boundary
    lines' current values (their chart draws the lines themselves)."""
    if inst.type == "box":
        lo, up = inst.last_low_price(), inst.last_high_price()
    else:
        last = inst.pivots[-1]["bar"] if inst.pivots else 0
        lo = inst.metrics.get("lower_at_last_bar", inst.lower_at(last))
        up = inst.metrics.get("upper_at_last_bar", inst.upper_at(last))
    return (
        f"🔷 <b>{inst.symbol}</b> — {tf_label(inst.tf)} Compression  ·  {pretty_type(inst.type)}\n"
        f"🎯 boundaries {_fmt_price(lo)} – {_fmt_price(up)}"
    )


def fmt_breakout(inst, side: str, level: float) -> str:
    """Breakout heads-up / confirmed breakout (direction = actual event).
    `level` = the last pivot in that direction (user rule 2026-09-16)."""
    arrow = "🟢" if side == "up" else "🔴"
    word = "above" if side == "up" else "below"
    return (
        f"{arrow} <b>{inst.symbol}</b> — {tf_label(inst.tf)} Compression  ·  Breakout\n"
        f"🎯 closing <b>{word}</b> {_fmt_price(level)}"
    )


def chart_url(chart_base: str, symbol: str, tf: str,
              lines: Optional[List[float]] = None,
              trend_lines: Optional[List[tuple]] = None) -> str:
    """Alert chart URL. `lines` = horizontal levels; `trend_lines` =
    [(x1_ms, y1, x2_ms, y2)] sloped boundary segments extrapolated from the
    structure's pivots (user rule 2026-09-16: draw the real boundary lines,
    not just horizontal values)."""
    url = (f"{chart_base}/chart/alert?symbol={symbol.replace('/', '')}"
           f"&timeframe={tf}")
    if lines:
        url += "&cross_lines=" + ",".join(_fmt_price(x) for x in lines)
    if trend_lines:
        segs = "|".join(
            f"{int(round(x1))},{y1:.10g},{int(round(x2))},{y2:.10g}"
            for (x1, y1, x2, y2) in trend_lines)
        url += "&trend_lines=" + quote(segs, safe=",|")
    return url


class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._ses: Optional[aiohttp.ClientSession] = None

    async def _session(self) -> aiohttp.ClientSession:
        if not self._ses or self._ses.closed:
            self._ses = aiohttp.ClientSession()
        return self._ses

    async def _api(self, method: str, chat_id: Optional[str] = None, **params) -> dict:
        ses = await self._session()
        params["chat_id"] = chat_id or self.chat_id
        async with ses.post(
            f"https://api.telegram.org/bot{self.token}/{method}", json=params,
        ) as r:
            return await r.json(content_type=None)

    async def post(self, text: str, reply_to_id: Optional[int] = None,
                   chat_id: Optional[str] = None) -> Optional[int]:
        params = {"text": text, "parse_mode": "HTML"}
        if reply_to_id:
            params["reply_to_message_id"] = reply_to_id
        d = await self._api("sendMessage", chat_id=chat_id, **params)
        if d.get("ok"):
            return d["result"]["message_id"]
        log.warning("tg post failed: %s", d)
        return None

    async def post_photo(self, caption: str, image: bytes,
                         reply_to_id: Optional[int] = None,
                         chat_id: Optional[str] = None) -> Optional[int]:
        ses = await self._session()
        form = aiohttp.FormData()
        form.add_field("chat_id", str(chat_id or self.chat_id))
        form.add_field("caption", caption)
        form.add_field("parse_mode", "HTML")
        if reply_to_id:
            form.add_field("reply_to_message_id", str(reply_to_id))
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

    async def delete(self, msg_id: int, why: str = "",
                     chat_id: Optional[str] = None) -> bool:
        d = await self._api("deleteMessage", chat_id=chat_id, message_id=msg_id)
        if d.get("ok"):
            log.info("tg deleted msg %s (%s)", msg_id, why)
            return True
        log.warning("tg delete failed: %s", d)
        return False


async def fetch_chart(
    session: aiohttp.ClientSession, base_url: str, symbol: str, tf: str,
    lines: Optional[List[float]] = None,
    trend_lines: Optional[List[tuple]] = None,
) -> Optional[bytes]:
    """Alert chart PNG from the chart service (:8002). None on failure."""
    url = chart_url(base_url, symbol, tf, lines, trend_lines)
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
