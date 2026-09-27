"""What-if: forgiving MIDDLE pivots ("smoothing").

Design (test implementation, not committed):
- Anchors stay strict: the FIRST pivot of each side is exempt as today; the
  LAST pivot of each side keeps the strict label requirement.
- INTERIOR pivots (neither first nor last of their side) in triangles:
  * label: allowed to be one step looser (asc low: HL|EL, asc high: EH|HH;
    desc high: LH|EH, desc low: EL|LL) — at most ONE such pivot per side;
  * boundary respect: the single interior violator may deviate up to
    MIDDLE_TOL x ATR instead of boundary_tol_atr (1.0) x ATR;
  * close-integrity: closes within +/- SMOOTH_BARS of a forgiven pivot's bar
    are not counted as breaches (the dip is the smoothing residue).
- Boxes untouched (no interior pivots in 4-pivot windows anyway -> delegates
  to the original checks).
"""
import asyncio
import sys
import time
import json
import glob
from collections import Counter
from dataclasses import replace

sys.path.insert(0, '.')

import aiohttp
from compression_detection.config import Config
from compression_detection.fetcher import fetch_klines
from compression_detection.zigzag import ZigZag
from compression_detection.atr import atr_series
from compression_detection.dow import label_all_pivots
from compression_detection import detector as det, structure as st
from compression_detection.detector import PivotLabel
from compression_detection.engine import Engine

MIDDLE_TOL = 1.5
SMOOTH_BARS = 3
TRI = {det.TYPE_ASC_TRI, det.TYPE_DESC_TRI}
FORGIVE = {
    det.TYPE_ASC_TRI: (('HL', 'EL'), ('EH', 'HH')),   # (low side, high side)
    det.TYPE_DESC_TRI: (('EL', 'LL'), ('LH', 'EH')),
}

_orig_check_window = det._check_window


def custom_check_window(spec, refs, atr14_series, cfg, close_by_bar=None):
    highs = [r for r in refs if r.is_high]
    lows = [r for r in refs if not r.is_high]
    if len(highs) < 2 or len(lows) < 2:
        return None

    last_vote = refs[-1].bar_index
    atr = None
    for a in reversed(atr14_series[: last_vote + 1]):
        if a is not None:
            atr = a
            break
    if atr is None or atr <= 0:
        return None

    is_tri = spec.name in TRI
    if is_tri:
        if spec.lead_side == 'H' and not refs[0].is_high:
            return None
        if spec.lead_side == 'L' and refs[0].is_high:
            return None
        lowf, highf = FORGIVE[spec.name]
        for side, allowed_enum, forgiven in (
            (lows, spec.lower_labels, set(lowf)),
            (highs, spec.upper_labels, set(highf)),
        ):
            allowed = set(x.value for x in allowed_enum)
            bad = 0
            for i, r in enumerate(side):
                if i == 0:
                    continue                      # first of side: exempt (as today)
                lab = r.label or ''
                if i == len(side) - 1:            # anchor: strict
                    if lab not in allowed:
                        return None
                else:                             # interior
                    if lab in allowed:
                        continue
                    if lab in forgiven:
                        bad += 1
                        if bad > 1:
                            return None
                    else:
                        return None
    else:
        if not det._label_ok(refs, spec):
            return None

    upper_pts = [(r.abs_bar, r.price) for r in highs]
    lower_pts = [(r.abs_bar, r.price) for r in lows]
    upper = st.fit_line(upper_pts)
    lower = st.fit_line(lower_pts)
    if upper is None or lower is None:
        return None

    forgiven_bars = set()
    for side_pivots, line, pts in ((highs, upper, upper_pts), (lows, lower, lower_pts)):
        flags = st.respects(pts, line, atr, cfg.boundary_tol_atr)
        fails = [i for i, ok in enumerate(flags) if not ok]
        if not fails:
            continue
        if not is_tri or len(fails) > 1:
            return None
        i = fails[0]
        r = side_pivots[i]
        interior = 0 < i < len(side_pivots) - 1
        dev = abs(r.price - line.at(r.abs_bar)) / atr
        if interior and dev <= MIDDLE_TOL:
            forgiven_bars.add(r.abs_bar)
        else:
            return None

    bars = {r.abs_bar for r in refs}
    if any(upper.at(b) <= lower.at(b) for b in bars):
        return None

    def _span_atr(side_pivots):
        vals = []
        for p in (side_pivots[0], side_pivots[-1]):
            if p.bar_index < len(atr14_series) and atr14_series[p.bar_index]:
                vals.append(atr14_series[p.bar_index])
        return max(vals) if vals else atr

    upper_class = st.slope_class(upper, highs[0].abs_bar, highs[-1].abs_bar, _span_atr(highs), cfg.flat_tol_atr)
    lower_class = st.slope_class(lower, lows[0].abs_bar, lows[-1].abs_bar, _span_atr(lows), cfg.flat_tol_atr)
    if upper_class not in spec.upper_slopes or lower_class not in spec.lower_slopes:
        return None

    metrics = {
        "pivot_count": len(refs),
        "touches": int(sum(st.respects(upper_pts, upper, atr, cfg.boundary_tol_atr)) + sum(st.respects(lower_pts, lower, atr, cfg.boundary_tol_atr))),
        "fit_err_atr": st.fit_error(upper_pts, upper, atr) + st.fit_error(lower_pts, lower, atr),
        "upper_eq_err_atr": (max(p for _, p in upper_pts) - min(p for _, p in upper_pts)) / atr,
        "lower_eq_err_atr": (max(p for _, p in lower_pts) - min(p for _, p in lower_pts)) / atr,
        "span_bars": refs[-1].abs_bar - refs[0].abs_bar,
        "upper_at_last_bar": upper.at(refs[-1].abs_bar),
        "lower_at_last_bar": lower.at(refs[-1].abs_bar),
        "forgiven_middle_pivots": len(forgiven_bars),
    }
    upper_hits = len(highs)
    lower_hits = len(lows)
    metrics["upper_hits"] = upper_hits
    metrics["lower_hits"] = lower_hits
    metrics["hit_requirement"] = max(upper_hits, lower_hits) >= cfg.min_boundary_hits
    metrics["hit_boundary"] = ("upper" if upper_hits >= cfg.min_boundary_hits
                               else "lower" if lower_hits >= cfg.min_boundary_hits else "")

    if close_by_bar is not None:
        b0, b1 = refs[0].abs_bar, refs[-1].abs_bar
        tol = cfg.close_breach_tol_atr * atr
        up_breaches = lo_breaches = 0
        worst_breach = 0.0
        for b in range(b0, b1 + 1):
            if any(abs(b - fb) <= SMOOTH_BARS for fb in forgiven_bars):
                continue
            cl = close_by_bar.get(b)
            if cl is None:
                continue
            u = upper.at(b)
            l = lower.at(b)
            if cl > u + tol:
                up_breaches += 1
                worst_breach = max(worst_breach, (cl - u) / atr)
            elif cl < l - tol:
                lo_breaches += 1
                worst_breach = max(worst_breach, (l - cl) / atr)
        metrics["close_breaches_upper"] = up_breaches
        metrics["close_breaches_lower"] = lo_breaches
        metrics["worst_close_breach_atr"] = worst_breach
        if max(up_breaches, lo_breaches) > cfg.max_close_breaches:
            return None

    if spec.converging:
        sample_from = max(highs[0].abs_bar, lows[0].abs_bar)
        sample_bars = sorted({r.abs_bar for r in refs if r.abs_bar >= sample_from})
        widths = st.convergence_samples(upper, lower, sample_bars)
        ok, shrink, bounce = st.evaluate_convergence(widths, cfg.min_convergence_pct / 100.0, cfg.max_width_bounce_frac)
        if not ok:
            return None
        metrics["convergence_rate"] = shrink
        metrics["width_reduction_pct"] = shrink * 100.0
        metrics["max_bounce_frac"] = bounce
    else:
        metrics["convergence_rate"] = 0.0
        metrics["width_reduction_pct"] = 0.0

    return det.Candidate(type=spec.name, refs=list(refs), upper=upper, lower=lower, atr=atr,
                         upper_class=upper_class, lower_class=lower_class, metrics=metrics)


async def ava_check():
    cfg = Config.load()
    async with aiohttp.ClientSession() as ses:
        cs = await fetch_klines(ses, "AVA/USDT", "15m", cfg.candle_limit)
    zz = ZigZag(coef=cfg.zigzag_coef, atr_length=cfg.zigzag_atr_length)
    piv = zz.feed_all(cs)
    atr14 = atr_series(cs, cfg.atr14_length, cfg.atr14_method)
    lab = label_all_pivots(piv, atr14)
    last_bar = cs[-1].ts // (15 * 60_000)
    cands = det.detect_candidates(lab, atr14, 15 * 60_000, last_bar, cfg.det, candles=cs)
    print("AVA 15m with smoothing:", [(c.type, c.pivot_count, c.metrics.get("forgiven_middle_pivots"), c.metrics.get("upper_hits"), c.metrics.get("lower_hits")) for c in cands])
    for c in cands:
        p2 = c.to_dict()["pivots"]
        print("   ", c.type, " ".join(p['side'] + p['label'] for p in p2))


async def universe_scan():
    cfg = Config.load()
    t0 = time.time()
    eng = Engine(cfg, dry_run=True)
    s = await eng.scan_once()
    dets = {}
    for d in s.get("detections", []):
        dets[(d["symbol"], d["tf"])] = [c for c in d["candidates"]]
    print(f"scan: {time.time()-t0:.0f}s detections={len(dets)}")
    from collections import Counter
    print("counts:", dict(Counter(c['type'] for v in dets.values() for c in v)))
    return dets


async def main():
    det._check_window = custom_check_window
    await ava_check()
    smoothed = await universe_scan()

    def load(f):
        r = json.load(open(f'scan_reports/{f}.json'))
        out = {}
        for x in r.get('detections', []):
            out[(x['symbol'], x['tf'])] = x['candidates']
        return out

    strict = load('20260923_123130')
    A = load('20260923_123333')
    B = load('20260923_123352')

    def diff(d, base):
        out = []
        for k in sorted(set(d) | set(base)):
            b = Counter(x['type'] for x in base.get(k, []))
            n = Counter(x['type'] for x in d.get(k, []))
            if b != n:
                out.append((k, dict(b), dict(n)))
        return out

    print("\n== smoothed vs strict ==")
    for k, b, n in diff(smoothed, strict):
        print(f"  {k[0]}({k[1]}): {b} -> {n}")
    print("\n== smoothed vs B ==")
    for k, b, n in diff(smoothed, B):
        print(f"  {k[0]}({k[1]}): {b} -> {n}")
    print("\n== smoothed new asc/desc sequences ==")
    for k, b, n in diff(smoothed, strict):
        for c in smoothed.get(k, []):
            if c['type'] in ('ascending_triangle', 'descending_triangle') and c['type'] not in b:
                seq = " ".join(p['side'] + p['label'] for p in c['pivots'])
                print(f"  {k[0]}({k[1]}) {c['type']} x{len(c['pivots'])}: {seq}")


asyncio.run(main())
