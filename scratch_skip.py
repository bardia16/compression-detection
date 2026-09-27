"""What-if: drop MIDDLE pivots, re-check patterns ("pivot skipping").

User mechanism (2026-09-23): take the last K pivots like always; in the
pattern search, REMOVE some middle pivots (interior of the window — keep
first & last); RE-LABEL the remaining pivots against their new same-side
neighbours; then run the STANDARD pattern gates on the reduced sequence.

- triangles only in this test (boxes left untouched)
- max 2 pivots dropped, window reach up to the last 8 pivots
- dedupe reduced windows against each other and against strict candidates
"""
import asyncio
import sys
import time
import json
import glob
import itertools
from collections import Counter

sys.path.insert(0, '.')

import aiohttp
from compression_detection.config import Config
from compression_detection.fetcher import fetch_klines
from compression_detection.zigzag import ZigZag
from compression_detection.atr import atr_series
from compression_detection.dow import label_all_pivots, label_pivot
from compression_detection import detector as det
import compression_detection.engine as eng_mod
from compression_detection.detector import PivotRef, rank_candidates, TYPE_SPECS, _check_window
from compression_detection.engine import Engine

MAX_DROP = 2
MAX_K = 8

_orig_detect = det.detect_candidates


def skip_candidates(labeled, atr14_series, tf_ms, last_bar, cfg, candles):
    lab = [(p, l) for (p, l) in labeled
           if p.ts // tf_ms >= last_bar - cfg.max_pivot_age_bars]
    close_by_bar = None
    if candles is not None:
        close_by_bar = {int(c.ts // tf_ms): float(c.close) for c in candles}
    out = []
    seen = set()
    for type_name in (det.TYPE_ASC_TRI, det.TYPE_DESC_TRI):
        spec = TYPE_SPECS[type_name]
        min_k = cfg.min_pivots[type_name]
        for K in range(min_k + 1, min(MAX_K, len(lab)) + 1):
            win = lab[-K:]
            interiors = list(range(1, K - 1))
            for m in range(1, MAX_DROP + 1):
                for drop in itertools.combinations(interiors, m):
                    keep = [win[i] for i in range(K) if i not in drop]
                    if len(keep) < min_k:
                        continue
                    # reduced sequences must still alternate sides (the zigzag
                    # invariant — otherwise adjacent same-side pivots slip through)
                    if any(keep[i][0].is_high == keep[i - 1][0].is_high
                           for i in range(1, len(keep))):
                        continue
                    refs = []
                    last_h = last_l = None
                    for p, _ in keep:
                        prev = last_h if p.is_high else last_l
                        a1 = atr14_series[p.bar_index] if p.bar_index < len(atr14_series) else None
                        a0 = None
                        if prev is not None and prev.bar_index < len(atr14_series):
                            a0 = atr14_series[prev.bar_index]
                        lbl = label_pivot(p, prev, a1, a0)
                        if p.is_high:
                            last_h = p
                        else:
                            last_l = p
                        refs.append(PivotRef(
                            ts=p.ts, bar_index=p.bar_index, abs_bar=p.ts // tf_ms,
                            price=p.price, is_high=p.is_high,
                            label=(lbl.value if lbl is not None else None)))
                    sig = tuple((r.ts, r.price) for r in refs)
                    if sig in seen:
                        continue
                    cand = _check_window(spec, refs, atr14_series, cfg, close_by_bar)
                    if cand is not None:
                        seen.add(sig)
                        out.append(cand)
    return out


def custom_detect(labeled, atr14_series, tf_ms, last_bar, cfg, candles=None):
    out = list(_orig_detect(labeled, atr14_series, tf_ms, last_bar, cfg, candles))
    sigs = set(tuple((r.ts, r.price) for r in c.refs) for c in out)
    for c in skip_candidates(labeled, atr14_series, tf_ms, last_bar, cfg, candles):
        sig = tuple((r.ts, r.price) for r in c.refs)
        if sig not in sigs:
            sigs.add(sig)
            out.append(c)
    return rank_candidates(out, cfg.selection_order)


async def ava_check():
    cfg = Config.load()
    async with aiohttp.ClientSession() as ses:
        cs = await fetch_klines(ses, "AVA/USDT", "15m", cfg.candle_limit)
    zz = ZigZag(coef=cfg.zigzag_coef, atr_length=cfg.zigzag_atr_length)
    piv = zz.feed_all(cs)
    atr14 = atr_series(cs, cfg.atr14_length, cfg.atr14_method)
    lab = label_all_pivots(piv, atr14)
    last_bar = cs[-1].ts // (15 * 60_000)
    cands = custom_detect(lab, atr14, 15 * 60_000, last_bar, cfg.det, candles=cs)
    print("AVA 15m with pivot-skip:", [(c.type, c.pivot_count) for c in cands])
    for c in cands:
        p2 = c.to_dict()["pivots"]
        print("   ", c.type, " ".join(p['side'] + (p['label'] or '-') for p in p2),
              [p['price'] for p in p2])


async def universe_scan():
    cfg = Config.load()
    t0 = time.time()
    eng = Engine(cfg, dry_run=True)
    res = await eng.scan_once()
    s = res["summary"]
    dets = {}
    for d in s.get("detections", []):
        dets[(d["symbol"], d["tf"])] = d["candidates"]
    print(f"scan: {time.time()-t0:.0f}s report={res['report']} detections={len(dets)}")
    print("counts:", dict(Counter(c['type'] for v in dets.values() for c in v)))
    return dets


async def main():
    det.detect_candidates = custom_detect
    eng_mod.detect_candidates = custom_detect
    await ava_check()
    skipd = await universe_scan()

    def load(f):
        r = json.load(open(f'scan_reports/{f}.json'))
        out = {}
        for x in r.get('detections', []):
            out[(x['symbol'], x['tf'])] = x['candidates']
        return out

    strict = load('20260923_123130')
    A = load('20260923_123333')
    B = load('20260923_123352')
    sm = load('20260923_123756')

    def diff(d, base):
        out = []
        for k in sorted(set(d) | set(base)):
            b = Counter(x['type'] for x in base.get(k, []))
            n = Counter(x['type'] for x in d.get(k, []))
            if b != n:
                out.append((k, dict(b), dict(n)))
        return out

    print("\n== skip vs strict ==")
    for k, b, n in diff(skipd, strict):
        print(f"  {k[0]}({k[1]}): {b} -> {n}")
    print("\n== skip vs smoothing ==")
    for k, b, n in diff(skipd, sm):
        print(f"  {k[0]}({k[1]}): {b} -> {n}")
    print("\n== skip new triangle sequences (vs strict) ==")
    for k, b, n in diff(skipd, strict):
        for c in skipd.get(k, []):
            if c['type'] != 'box' and c['type'] not in b:
                seq = " ".join(p['side'] + p['label'] for p in c['pivots'])
                print(f"  {k[0]}({k[1]}) {c['type']} x{len(c['pivots'])}: {seq}")


asyncio.run(main())
