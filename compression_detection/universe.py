"""Coin universe: ALL LiveCoinWatch coins with 24h volume >= min_volume_btc.

Deliberately independent of every other watchlist project — it reuses only
the LCW fetch PATTERN (same endpoint/payload/filters), not any shared
state. Binance availability filter uses REST exchangeInfo (spot +
futures) so there is no ccxt dependency.

Failure policy (user rule: errors over silent fallback): LCW failure
raises; the engine logs and skips the scan instead of scanning the wrong
universe. Binance exchangeInfo failure degrades to "everything
available" so per-symbol fetch errors still surface per coin.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import aiohttp
import requests

log = logging.getLogger("universe")

LCW_URL = "https://api.livecoinwatch.com/coins/list"
SPOT_EXINFO = "https://api.binance.com/api/v3/exchangeInfo"
FAPI_EXINFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
SPOT_TICKER = "https://api.binance.com/api/v3/ticker/price"
FAPI_TICKER = "https://fapi.binance.com/fapi/v1/ticker/price"

STABLECATEGORIES = {"stablecoins", "asset_backed_tokens"}

_TIMEOUT = aiohttp.ClientTimeout(total=15)


def load_api_key(repo_root: str) -> Optional[str]:
    """LIVECOINWATCH_API_KEY from env, else parsed from <repo_root>/.env."""
    import os
    key = os.getenv("LIVECOINWATCH_API_KEY")
    if key:
        return key.strip()
    env_path = os.path.join(repo_root, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("LIVECOINWATCH_API_KEY"):
                    _, _, val = line.partition("=")
                    return val.strip().strip('"').strip("'")
    return None


def get_btc_price() -> Optional[float]:
    """BTC/USDT last price via REST (spot → futures). None on failure."""
    for base in (SPOT_TICKER, FAPI_TICKER):
        try:
            r = requests.get(base, params={"symbol": "BTCUSDT"}, timeout=10)
            r.raise_for_status()
            return float(r.json()["price"])
        except Exception:
            continue
    return None


def fetch_coins(api_key: str, limit: int = 300, min_volume_btc: float = 100.0,
                btc_price: Optional[float] = None) -> List[dict]:
    """LCW coins sorted by 24h volume, filtered (same pattern as the
    watchlist/daw projects): dedupe by symbol keeping the highest volume,
    drop stablecoins/asset-backed tokens, then apply the min_volume_btc
    threshold. BTC is INCLUDED (user rule 2026-09-21: BTC compressions are
    wanted here — the watchlist/daw BTC-drop does not apply). Raises on
    HTTP errors."""
    headers = {"content-type": "application/json", "x-api-key": api_key}
    payload = {
        "currency": "USD", "sort": "volume", "order": "descending",
        "offset": 0, "limit": limit, "meta": True,
    }
    import re
    resp = requests.post(LCW_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    seen: dict = {}
    for item in data:
        code = re.sub(r"[^A-Z]", "", (item.get("code") or "").upper())
        if not code:
            continue
        categories = {c.lower() for c in item.get("categories", []) or []}
        if categories & STABLECATEGORIES:
            continue
        vol = item.get("volume", 0) or 0
        if code not in seen or vol > seen[code]["volume_24h"]:
            seen[code] = {"symbol": code, "volume_24h": vol,
                          "rate": item.get("rate")}

    coins = list(seen.values())
    if min_volume_btc and btc_price:
        threshold = min_volume_btc * btc_price
        coins = [c for c in coins if c["volume_24h"] >= threshold]
    coins.sort(key=lambda c: c["volume_24h"], reverse=True)
    return coins


def get_universe(api_key: str, limit: int = 300, min_volume_btc: float = 100.0,
                 btc_price: Optional[float] = None) -> List[str]:
    """Full pipeline -> ["AAA/USDT", ...]. Raises on LCW failure."""
    coins = fetch_coins(api_key, limit=limit, min_volume_btc=min_volume_btc,
                        btc_price=btc_price)
    return [c["symbol"] + "/USDT" for c in coins]


def pair_to_binance_symbol(pair: str) -> str:
    return pair.replace("/", "").upper()


def filter_available(pairs: List[str], available: Optional[set]) -> Tuple[List[str], List[str]]:
    """Split pairs into (available, missing) using the Binance symbol set.

    available=None (exchangeInfo unavailable) -> everything considered
    present so per-coin fetch errors surface individually.
    """
    if available is None:
        return list(pairs), []
    avail, miss = [], []
    for p in pairs:
        (avail if pair_to_binance_symbol(p) in available else miss).append(p)
    return avail, miss


async def load_available_symbols(session: aiohttp.ClientSession) -> Optional[set]:
    """Union of spot + futures exchangeInfo symbols. None when both fail."""
    out: set = set()
    got = False
    for base in (SPOT_EXINFO, FAPI_EXINFO):
        try:
            async with session.get(base, timeout=_TIMEOUT) as r:
                if r.status != 200:
                    continue
                d = await r.json(content_type=None)
            syms = d.get("symbols") or []
            for s in syms:
                name = s.get("symbol")
                if name:
                    out.add(name)
            got = True
        except Exception as exc:
            log.debug("exchangeInfo %s failed: %s", base, exc)
    return out if got else None
