"""Async Binance REST fetchers: klines (spot → futures fallback, forming
candle dropped) and last price for probes.

The forming candle is dropped so every structural fact is a fact of
history (same rule as the daw engine — a mid-candle close must never
confirm a pivot or a structure).
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import aiohttp

from .models import Candle

log = logging.getLogger("fetch")

TF_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

SPOT_KLINES = "https://api.binance.com/api/v3/klines"
FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
SPOT_TICKER = "https://api.binance.com/api/v3/ticker/price"
FAPI_TICKER = "https://fapi.binance.com/fapi/v1/ticker/price"

_TIMEOUT = aiohttp.ClientTimeout(total=15)


def drop_forming(candles: List[Candle], tf: str, now_ms: Optional[int] = None) -> List[Candle]:
    """Remove the still-forming last candle (closed-candles-only rule)."""
    if not candles or tf not in TF_MS:
        return candles
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    closes_at = candles[-1].ts + TF_MS[tf]
    if closes_at > now:
        return candles[:-1]
    return candles


async def _get_json(session: aiohttp.ClientSession, url: str, params: dict):
    async with session.get(url, params=params, timeout=_TIMEOUT) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status} for {url}")
        return await r.json(content_type=None)


async def fetch_klines(
    session: aiohttp.ClientSession, symbol: str, tf: str, limit: int
) -> List[Candle]:
    """Closed candles for symbol/tf. Spot first, futures fallback.

    Returns [] when both endpoints fail — caller logs & skips this
    symbol for the scan (never crashes the loop).
    """
    sym = symbol.replace("/", "")
    for base in (SPOT_KLINES, FAPI_KLINES):
        try:
            arr = await _get_json(session, base, {"symbol": sym, "interval": tf, "limit": limit})
            if isinstance(arr, list) and arr:
                candles = [
                    Candle(ts=int(k[0]), open=float(k[1]), high=float(k[2]),
                           low=float(k[3]), close=float(k[4]), volume=float(k[5]))
                    for k in arr
                ]
                return drop_forming(candles, tf)
        except Exception as exc:
            log.debug("klines %s %s via %s failed: %s", sym, tf, base.split("/")[2], exc)
    log.warning("klines fetch failed for %s %s (spot+futures)", sym, tf)
    return []


async def fetch_last_price(session: aiohttp.ClientSession, symbol: str) -> float:
    """Last traded price (spot → futures). 0.0 on failure."""
    sym = symbol.replace("/", "")
    for base in (SPOT_TICKER, FAPI_TICKER):
        try:
            d = await _get_json(session, base, {"symbol": sym})
            if isinstance(d, dict) and d.get("price"):
                return float(d["price"])
        except Exception:
            continue
    return 0.0
