"""Engine — scan loop (15m grid + offset) + probe loop (T−3:00 heads-up).

Scan (every 15m boundary + offset_s; 15m/1h/4h candles all close on the
grid): refresh universe → fetch closed candles per coin/TF → zigzag →
labels → candidates → lifecycle pass → notifications (best-only dedupe)
→ persist state + write scan report.

Probe (every probe_loop_s): for each active instance whose current
candle closes within probe_warn_s and not yet probed this candle, fetch
the live price; if beyond a boundary → post the heads-up + chart.

Notification formats are awaiting Bardia's mock approval; with
notifications.enabled=false the engine suppresses sends (records
suppressed actions in the audit + report), so the pipeline can be
reviewed dry before going live.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

from . import notifier as nt
from . import universe as uni
from .atr import atr_series
from .audit import Audit
from .candle_health import has_candle_problems
from .config import REPO, Config
from .detector import detect_candidates
from .dow import label_all_pivots
from .fetcher import TF_MS, fetch_klines, fetch_last_price
from .lifecycle import (
    ACTIVE_STATES, STATE_BREAKOUT, TERMINAL_STATES, Action, Instance,
    alert_level_for, best_per_coin_tf, order_potential,
    retire_out_of_universe, update_for_scan,
)
from .potential_break import (
    LOWER_TF, POTENTIAL_FRESH_S, find_potential, potential_lines,
)
from .report import build_summary, format_summary_text, write_report
from .timing import next_scan_ms, probe_due
from .zigzag import ZigZag

log = logging.getLogger("engine")

AUDIT_DIR = Path(os.environ.get("COMPRESSION_AUDIT_DIR", str(REPO / "audit")))
REPORTS_DIR = Path(os.environ.get("COMPRESSION_REPORTS_DIR", str(REPO / "scan_reports")))
CHART_SPACING_S = 1.5          # min gap between chart-server requests
CHART_RETRY_DELAY_S = 3.0      # one retry before the text-only fallback (chart
                               # service 503s transiently during scan bursts)
UNIVERSE_AVAIL_TTL = 6 * 3600  # exchangeInfo cache

# DM mirror (user rule 2026-09-20): ascending/descending triangle alerts are
# ALSO sent to Bardia's DM, in addition to the channel. Chat id comes from
# TELEGRAM_DM_CHAT_ID (.env); empty/unset = mirroring off (channel only).
DM_MIRROR_TYPES = ("ascending_triangle", "descending_triangle")


class Engine:
    def __init__(self, cfg: Config, dry_run: bool = False,
                 tg: Optional[nt.Telegram] = None):
        self.cfg = cfg
        self.dry_run = dry_run
        self.audit = Audit(AUDIT_DIR)
        self.instances: Dict[str, Instance] = {}
        self.tg = tg
        self.dm_chat = os.environ.get("TELEGRAM_DM_CHAT_ID", "").strip() or None
        self._available: Optional[set] = None
        self._available_ts = 0.0
        self._chart_sem = asyncio.Semaphore(2)
        self._chart_last = 0.0
        if not dry_run:
            self._load_state()

    # ── state io ───────────────────────────────────────────────────────
    def _load_state(self) -> None:
        try:
            d = json.loads(self.cfg.state_path.read_text())
            raw = d.get("instances") or {}
            keep = set(self.cfg.timeframes)
            # instances on a tf that is no longer a detection tf (15m,
            # user rule 2026-09-28) would freeze mid-state and clutter the
            # website — drop them on load.
            self.instances = {k: Instance.from_dict(v)
                              for k, v in raw.items()
                              if v.get("tf") in keep}
            dropped = len(raw) - len(self.instances)
            log.info("state loaded: %d instances%s", len(self.instances),
                     f" ({dropped} stale-tf dropped)" if dropped else "")
        except FileNotFoundError:
            self.instances = {}
        except Exception as exc:
            log.error("state load failed (%s) — starting empty", exc)
            self.instances = {}

    def _save_state(self) -> None:
        tmp = self.cfg.state_path.with_suffix(".tmp")
        doc = {"instances": {k: v.to_dict() for k, v in self.instances.items()}}
        tmp.write_text(json.dumps(doc, indent=1, default=str))
        os.replace(tmp, self.cfg.state_path)

    def _prune(self) -> None:
        cutoff = time.time() - self.cfg.prune_after_days * 86400
        dead = [k for k, v in self.instances.items()
                if v.state in TERMINAL_STATES and v.created_ts < cutoff]
        for k in dead:
            self.instances.pop(k, None)
        if dead:
            log.info("pruned %d terminal instances", len(dead))

    # ── universe ───────────────────────────────────────────────────────
    async def _universe(self, ses: aiohttp.ClientSession) -> List[str]:
        api_key = uni.load_api_key(str(REPO))
        if not api_key:
            raise RuntimeError("LIVECOINWATCH_API_KEY missing (.env)")
        btc = await asyncio.to_thread(uni.get_btc_price)
        pairs = await asyncio.to_thread(
            uni.get_universe, api_key, self.cfg.lcw_limit,
            self.cfg.min_volume_btc, btc,
        )
        now = time.time()
        if self._available is None or now - self._available_ts > UNIVERSE_AVAIL_TTL:
            self._available = await uni.load_available_symbols(ses)
            self._available_ts = now
        avail, missing = uni.filter_available(pairs, self._available)
        log.info("universe: %d pairs (%d not on Binance)", len(avail), len(missing))
        return avail

    # ── pipeline per coin ──────────────────────────────────────────────
    def _analyze_candles(self, candles, tf: str) -> Optional[dict]:
        """Closed candles → pivots → labels → candidates (pure)."""
        tf_ms = TF_MS[tf]
        if len(candles) < self.cfg.atr14_length + self.cfg.zigzag_atr_length + 6:
            return None
        zz = ZigZag(coef=self.cfg.zigzag_coef, atr_length=self.cfg.zigzag_atr_length)
        pivots = zz.feed_all(candles)
        atr14 = atr_series(candles, self.cfg.atr14_length, self.cfg.atr14_method)
        last_bar = candles[-1].ts // tf_ms
        # Live last pivot (user rule 2026-09-27, AR case): hand the
        # detector the UNCONFIRMED zigzag extreme as a candidate final
        # pivot so a triangle completes while the leg develops instead of
        # waiting out the confirm lag (AR: HL 04:00 -> confirm 07:00, its
        # confirmation landing on top of the breakout). Labels are computed
        # on the extended list — earlier labels are prefix-stable — and the
        # the detector restricts live-final windows to asc/desc triangles
        # and boxes (option A, 2026-09-28).
        full = pivots
        live = None
        prov = zz.provisional
        if prov is not None and pivots and prov.ts not in {p.ts for p in pivots}:
            full = pivots + [prov]
        full_labeled = label_all_pivots(full, atr14)
        if len(full_labeled) > len(pivots):
            live, labeled = full_labeled[-1], full_labeled[:-1]
        else:
            labeled = full_labeled
        cands = detect_candidates(labeled, atr14, tf_ms, last_bar, self.cfg.det,
                                  candles=candles, live=live)
        return {"candles": candles, "pivots": pivots, "labeled": labeled,
                "candidates": cands, "last_bar": last_bar}

    def _fetch_tfs(self) -> List[str]:
        """Detection tfs + every lower tf the potential pass reads.

        15m carries no patterns anymore (user rule 2026-09-28) but 1h
        boxes still touch on it, so its candles are fetched candle-only.
        """
        out = list(self.cfg.timeframes)
        for tf in self.cfg.timeframes:
            for t in LOWER_TF.get(tf, ()):
                if t not in out:
                    out.append(t)
        return out

    async def _fetch_coin(self, ses: aiohttp.ClientSession, sem: asyncio.Semaphore,
                          pair: str) -> Dict[str, Optional[dict]]:
        async with sem:
            out: Dict[str, Optional[dict]] = {}
            det_tfs = set(self.cfg.timeframes)
            for tf in self._fetch_tfs():
                try:
                    candles = await fetch_klines(ses, pair, tf, self.cfg.candle_limit)
                    if tf in det_tfs:
                        out[tf] = self._analyze_candles(candles, tf)
                    else:
                        # potential-only lower tf: candles, no pattern scan
                        out[tf] = {"candles": candles, "candidates": []}
                except Exception as exc:
                    log.warning("analyze %s %s failed: %s", pair, tf, exc)
                    out[tf] = None
            return out

    # ── scan ───────────────────────────────────────────────────────────
    async def scan_once(self, symbol_filter: Optional[str] = None) -> dict:
        now_ms = int(time.time() * 1000)
        now_s = int(now_ms / 1000)
        self.audit.write({"event": "scan_start", "dry": self.dry_run})
        errors: List[str] = []

        async with aiohttp.ClientSession() as ses:
            try:
                pairs = await self._universe(ses)
            except Exception as exc:
                log.error("universe refresh failed: %s", exc)
                self.audit.write({"event": "scan_abort", "reason": f"universe: {exc}"})
                return {"error": str(exc)}

            if symbol_filter:
                pairs = [p for p in pairs if p.split("/")[0] == symbol_filter.upper()]
                if not pairs:
                    pairs = [f"{symbol_filter.upper()}/USDT"]

            sem = asyncio.Semaphore(8)
            results = await asyncio.gather(
                *[self._fetch_coin(ses, sem, p) for p in pairs])

            instances_src = copy.deepcopy(self.instances) if self.dry_run else self.instances
            raw_actions: List = []
            detections: List[dict] = []
            candles_by_key: Dict[tuple, list] = {}

            # Out-of-universe TTL (user rule 2026-09-23): active instances of
            # coins absent from the universe retire after
            # cfg.out_of_universe_ttl_s, with their pending breakout heads-up
            # retracted (the close verdict can never run for an unscanned
            # coin). The timer clears whenever the coin is back in universe.
            if not symbol_filter:
                universe_syms = {p.split("/")[0] + "USDT" for p in pairs}
                for inst in retire_out_of_universe(instances_src, universe_syms,
                                                   now_s, self.cfg.out_of_universe_ttl_s):
                    if inst.probe_msg_id is not None:
                        raw_actions.append(Action("retract", inst, {
                            "msg_id": inst.probe_msg_id,
                            "reason": "coin left the universe — structure retired",
                        }))
                        inst.probe_msg_id = None
                        inst.probe_candle_ms = 0
                        inst.probe_side = ""
                    raw_actions.append(Action("invalidate", inst,
                                              {"reason": "out_of_universe"}))
            # Grace-fetch (user pick 2026-09-27, A): an ACTIVE instance whose
            # coin left the LCW universe keeps FULL evaluation for its
            # out-of-universe TTL window. Before this, pass A simply had no
            # candles for it — probes still ran (independent fetch) but
            # lifecycle/potential starved silently (HUMA/BANK/TLM case:
            # qualified potentials never evaluated while the instance sat
            # in grace). universe_syms above stays the TRUE universe, so
            # the retire timer keeps ticking and TTL expiry still applies.
            universe_n = len(pairs)
            if not symbol_filter:
                have = {p.split("/")[0] + "USDT" for p in pairs}
                grace = sorted({i.symbol for i in instances_src.values()
                                if i.state in ACTIVE_STATES
                                and i.symbol not in have})
                if grace:
                    grace_pairs = [f"{s[:-4]}/USDT" for s in grace]
                    pairs = pairs + grace_pairs
                    results = list(results) + list(await asyncio.gather(
                        *[self._fetch_coin(ses, sem, p) for p in grace_pairs]))
                    log.info("grace-fetch: %d out-of-universe actives: %s",
                             len(grace), ", ".join(grace))
            # pass A — materialize candles + detections. Detection tfs
            # produce candidates; potential-only lower tfs (15m) contribute
            # candles alone (user rule 2026-09-28: no 15m patterns).
            for pair, res in zip(pairs, results):
                # store the full Binance symbol (ORCAUSDT) — messages, charts,
                # probe fetches and ticker calls all need it (bare ORCA broke
                # chart fetches 503 and made every probe miss, 2026-09-16)
                sym = pair.split("/")[0] + "USDT"
                for tf in self._fetch_tfs():
                    r = res.get(tf)
                    if r is None:
                        errors.append(f"{sym}:{tf} no data")
                        continue
                    candles_by_key[(sym, tf)] = r["candles"]
                    if r.get("candidates"):
                        detections.append({
                            "symbol": sym, "tf": tf,
                            "candidates": [c.to_dict() for c in r["candidates"][:8]],
                        })

            # pass C — lifecycle: close verdicts, continuation/invalidation,
            # new instances (this is where a triangle replaces the box).
            for pair, res in zip(pairs, results):
                sym = pair.split("/")[0] + "USDT"
                for tf in self.cfg.timeframes:
                    r = res.get(tf)
                    if r is None:
                        continue
                    raw_actions.extend(update_for_scan(
                        instances_src, sym, tf, r["candidates"], r["candles"],
                        TF_MS[tf], now_s, self.cfg,
                    ))

            # pass B — potential break, AFTER creation (user rule
            # 2026-09-28): a box born in THIS scan is checked right away.
            # The old order (B before C) forced every new box to wait a
            # full scan cycle (15 min) before its first potential check
            # (ONEUSDT case: box 10:01:30, potential 10:16:50).
            # Low → EH → confirmed higher low / high → EL → confirmed
            # lower high on the box's own TF and one TF down.
            raw_actions.extend(self._potential_actions(instances_src,
                                                       candles_by_key, now_s))

            # a box that reached BREAKOUT in this very scan already resolved —
            # announcing its 'potential' break one action later is noise
            raw_actions = self._drop_moot_potential(raw_actions)

            if self.cfg.best_only_per_coin_tf:
                raw_actions = best_per_coin_tf(raw_actions,
                                               self.cfg.det.selection_order)
            # potential break AFTER its own box's compression, BEFORE the
            # triangle that replaces that box
            raw_actions = order_potential(raw_actions)

            sends = await self._dispatch(raw_actions, ses, candles_by_key)
            actions = [{"kind": a.kind,
                        "id": a.instance.id if a.instance else None,
                        "detail": a.detail}
                       for a in raw_actions]

        summary = build_summary(now_ms, self.dry_run, universe_n, errors,
                                actions, detections, instances_src, sends)
        path = write_report(REPORTS_DIR, now_ms, summary)
        log.info("scan done: %d instances, %d actions, %d sends -> %s",
                 summary["instances_active"], len(actions), len(sends), path.name)
        self.audit.write({"event": "scan_end", "active": summary["instances_active"],
                          "actions": len(actions), "sends": len(sends)})

        if not self.dry_run:
            self._prune()
            self._save_state()
        return {"summary": summary, "report": str(path)}

    # ── potential break ─────────────────────────────────────────────────
    def _potential_actions(self, instances, candles_by_key: Dict[tuple, list],
                           now_s: int) -> List[Action]:
        """Potential-break pattern pass (user rules 2026-09-27).

        Active BOXES only. The touch lives in the BOX tf OR the tf below
        it (confirmed EH/EL on the line); the confirming swing may sit on
        the box tf or one tf below (lower tf first — its alert carries
        both charts).

        ONE alert PER TOUCH (user rule 2026-09-27 evening): a touch
        authorizes exactly one higher low / lower high. The touch ts used
        is stored on the instance, `since_touch_ts` is passed back into
        find_potential, and a strictly NEWER touch re-arms the side — no
        HL-beats-HL comparison, the touch→swing pairing is the structure.
        Within one scan: both sides evaluated, the FRESHEST confirmation
        wins, firing one side writes both flags (PIVERSE case: a long and
        a short went out 5 seconds apart). Flags written at EMIT time, the
        same contract the breakout verdict uses, so a scan can never
        re-arm an alert already in flight.
        """
        out: List[Action] = []
        for inst in list(instances.values()):
            if inst.type != "box" or inst.state in TERMINAL_STATES:
                continue
            # The box's boundary lines are the ONLY reference the pattern is
            # judged against (user rule 2026-09-27, XPL case). Lines live in
            # the BOX tf's bar space; a stored pivot carries both ts and bar,
            # so bar(ts) = ref_bar + (ts - ref_ts) / tf_ms lets us evaluate
            # them for the scanned tf too. No lines -> no pattern.
            ref = inst.pivots[0] if inst.pivots else None
            tf_ms = TF_MS.get(inst.tf)
            if ref is None or not tf_ms or ref.get("bar") is None:
                continue

            def box_at(ts, _ref=ref, _ms=tf_ms, _u=inst.upper_line,
                       _l=inst.lower_line):
                bar = _ref["bar"] + (ts - _ref["ts"]) / _ms
                return (_u.at(bar), _l.at(bar))

            # scan preference: the box's lower tfs nearest-first, then
            # the box's own tf (user rule 2026-09-28: 1h -> [15m],
            # 4h -> [1h, 15m], 1d -> [4h, 1h, 15m]).
            main_candles = candles_by_key.get((inst.symbol, inst.tf))
            if not main_candles:
                continue
            low_tfs = LOWER_TF.get(inst.tf, [])
            scan = [(t, candles_by_key.get((inst.symbol, t)))
                    for t in [*low_tfs, inst.tf] if t]
            scan = [(t, c) for t, c in scan if c]

            fresh_s = (now_s - POTENTIAL_FRESH_S) * 1000
            hits = []
            for side in ("long", "short"):
                # one alert PER TOUCH (user rule 2026-09-27 evening): the
                # touch that fired is stored; the side re-arms only on a
                # strictly NEWER touch. Flags written before that rule have
                # no touch attached — those sides stay spent (one-shot).
                spent = inst.notified.get(f"potential_{side}_touch_ts")
                if spent is None and inst.notified.get(f"potential_{side}"):
                    continue
                hit = find_potential(side, main_candles, scan, self.cfg,
                                     main_tf=inst.tf,
                                     anchor_ts=inst.anchor_ts,
                                     box_at=box_at,
                                     touch_atr=inst.atr,
                                     since_touch_ts=spent or 0)
                if hit is None:
                    continue
                if spent is not None and hit["touch_ts"] <= spent:
                    continue   # the same touch can never fire twice
                # freshness window (user choice 2026-09-27): a confirmation
                # older than 12h is old news.
                if hit["confirm_ts"] < fresh_s:
                    continue
                hit["side"] = side
                hits.append(hit)

            if not hits:
                continue
            # ONE per box: the freshest confirmation wins (a tie keeps the
            # earlier list order — long first).
            hits.sort(key=lambda h: h["confirm_ts"], reverse=True)
            win = hits[0]
            side = win["side"]
            for s in ("long", "short"):
                inst.notified[f"potential_{s}"] = True
            inst.notified[f"potential_{side}_touch_ts"] = win["touch_ts"]
            inst.add_event("potential", now_s, side=side,
                           pattern_tf=win["pattern_tf"],
                           mid=win["mid_price"],
                           second=win["second_price"],
                           confirm_ts=win["confirm_ts"])
            out.append(Action("potential_break", inst, dict(win)))
        return out

    @staticmethod
    def _drop_moot_potential(raw_actions: List[Action]) -> List[Action]:
        """Drop potential breaks for structures that reached BREAKOUT in the
        same scan — the structure already resolved, so a 'potential break'
        one action later is stale noise. INVALIDATED instances are KEPT: that
        is the box -> triangle handover (user rule 2026-09-27), whose
        potential break must go out before the replacement pattern."""
        return [a for a in raw_actions
                if not (a.kind == "potential_break" and a.instance is not None
                        and a.instance.state == STATE_BREAKOUT)]

    # ── dispatch ───────────────────────────────────────────────────────
    async def _dispatch(self, raw_actions: List, ses,
                        candles_by_key: Dict[tuple, list]) -> List[dict]:
        sends: List[dict] = []
        for a in raw_actions:
            if a.kind == "skip_create":
                self.audit.write({"event": "skip_create", **a.detail})
                continue
            inst = a.instance
            kind = a.kind
            if kind == "compression_notify":
                if not self.cfg.notif_enabled or self.dry_run or self.tg is None:
                    # HOLD, never consume: warmup keeps the notification pending
                    # (no flag write) — when notifications go live, the current
                    # compressions flush out (user rule 2026-09-16).
                    self.audit.write({"event": "notify_held", "kind": kind,
                                      "id": inst.id, "dry": self.dry_run})
                    continue
                # candle-problem gate (user rule 2026-09-16): anomalous gaps in
                # the last 30 candles → no notification (held for retry)
                candles = candles_by_key.get((inst.symbol, inst.tf)) or []
                if has_candle_problems(candles):
                    self.audit.write({"event": "candle_problems", "kind": kind,
                                      "id": inst.id, "action": "suppress"})
                    continue
                caption = nt.fmt_compression(inst)
                # chart (user rules 2026-09-16): triangles/wedges draw the
                # extrapolated boundary LINES (from each side's first pivot);
                # a box draws plain horizontals at its last pivot levels —
                # no line extrapolation for ranges.
                x2 = (candles[-1].ts + TF_MS[inst.tf]) if candles \
                    else inst.pivots[-1]["ts"]
                if inst.type == "box":
                    img = await self._chart(ses, inst, [
                        inst.last_low_price(), inst.last_high_price()], None)
                else:
                    segs = inst.trend_segments(TF_MS[inst.tf], x2)
                    img = await self._chart(ses, inst, [], segs)
                mid = None
                if img is not None:
                    mid = await self.tg.post_photo(caption, img)
                if mid is None:
                    mid = await self.tg.post(caption)
                if mid is not None:
                    inst.notified["compression"] = True
                    inst.msg_ids.append(mid)
                    # DM mirror (user rule 2026-09-20): triangles also to DM
                    dmid = await self._dm_post(inst, kind, caption, img)
                    if dmid is not None:
                        inst.dm_msg_ids.append(dmid)
                sends.append({"kind": kind, "id": inst.id, "msg_id": mid})
                self.audit.write({"event": "post", "kind": kind, "id": inst.id,
                                  "msg_id": mid, "chart": img is not None})

            elif kind == "breakout_post":
                detail = a.detail
                if not self.cfg.notif_enabled or self.dry_run or self.tg is None:
                    self.audit.write({"event": "notify_suppressed", "kind": kind,
                                      "id": inst.id, "side": detail.get("side"),
                                      "dry": self.dry_run})
                    continue
                candles = candles_by_key.get((inst.symbol, inst.tf)) or []
                if has_candle_problems(candles):
                    self.audit.write({"event": "candle_problems", "kind": kind,
                                      "id": inst.id, "side": detail.get("side"),
                                      "action": "suppress"})
                    continue
                caption = nt.fmt_breakout(inst, detail["side"], detail["level"])
                x2 = (candles[-1].ts + TF_MS[inst.tf]) if candles \
                    else inst.pivots[-1]["ts"]
                # For triangles the flat boundary IS the breakout level —
                # sending both creates a duplicate horizontal line on the
                # chart (user rule 2026-09-16).  Only send the level
                # cross_line when the breakout is on the sloped side.
                segs = inst.trend_segments(TF_MS[inst.tf], x2) \
                    if inst.type != "box" else None
                _flat_side = ("up" if inst.type == "ascending_triangle"
                              else "down" if inst.type == "descending_triangle"
                              else None)
                chart_level = None if detail["side"] == _flat_side \
                    else [detail["level"]]
                # reply-thread onto this structure's compression confirmation
                # message (user rule 2026-09-16); standalone if none was sent
                reply_to = inst.msg_ids[0] if inst.msg_ids else None
                mid = None
                if img := await self._chart(ses, inst, chart_level, segs):
                    mid = await self.tg.post_photo(caption, img, reply_to)
                if mid is None:
                    mid = await self.tg.post(caption, reply_to)
                sends.append({"kind": kind, "id": inst.id, "msg_id": mid})
                if mid is not None:
                    inst.msg_ids.append(mid)
                    inst.notified["breakout"] = True
                self.audit.write({"event": "post", "kind": kind, "id": inst.id,
                                  "side": detail.get("side"), "msg_id": mid,
                                  "reply_to": reply_to, "chart": img is not None})

            elif kind == "potential_break":
                detail = a.detail
                if not self.cfg.notif_enabled or self.dry_run or self.tg is None:
                    self.audit.write({"event": "notify_suppressed", "kind": kind,
                                      "id": inst.id, "side": detail.get("side"),
                                      "dry": self.dry_run})
                    continue
                side = detail["side"]
                ptf = detail["pattern_tf"]
                # same candle-health gate as every other post (2026-09-16)
                candles = candles_by_key.get((inst.symbol, inst.tf)) or []
                if has_candle_problems(candles):
                    self.audit.write({"event": "candle_problems", "kind": kind,
                                      "id": inst.id, "side": side,
                                      "action": "suppress"})
                    continue
                caption = nt.fmt_potential(inst, side, ptf, detail)
                sub = f"🎯 {nt.tf_label(ptf)} pattern chart"
                # chart 1 — MAIN TF: the structure's own boundaries, exactly
                # what its compression alert drew (context for the pattern).
                if inst.type == "box":
                    main_lines = [inst.last_low_price(), inst.last_high_price()]
                    main_segs = None
                else:
                    x2 = ((candles[-1].ts + TF_MS[inst.tf]) if candles
                          else inst.pivots[-1]["ts"])
                    main_lines, main_segs = [], inst.trend_segments(
                        TF_MS[inst.tf], x2)
                imgs: List[bytes] = []
                if img := await self._chart(ses, inst, main_lines, main_segs):
                    imgs.append(img)
                # chart 2 — LOW TF (user rule 2026-09-27): only when the
                # pattern was found there; box boundaries drawn, with the
                # breaking side adjusted to the pattern's mid pivot — the
                # LAST low-TF high (long) / low (short).
                if ptf != inst.tf and inst.type == "box":
                    low_lines = potential_lines(detail, inst.last_low_price(),
                                                inst.last_high_price())
                    if img2 := await self._chart(ses, inst, low_lines, None,
                                                 tf=ptf):
                        imgs.append(img2)
                # replies onto the box confirmation message (2026-09-16 rule)
                reply_to = inst.msg_ids[0] if inst.msg_ids else None
                mid = await self._post_photos(caption, sub, imgs, reply_to)
                if mid is not None:
                    inst.msg_ids.append(mid)
                    dmid = await self._dm_post(
                        inst, kind, caption, imgs=imgs, sub_caption=sub,
                        reply_to_dm=inst.dm_msg_ids[0] if inst.dm_msg_ids else None,
                        force=True)     # every potential break also to the DM
                    # NOTE: deliberately NOT appended to dm_msg_ids — that
                    # list's [0] is the mirrored COMPRESSION (the anchor every
                    # heads-up threads onto); a box has no DM compression, so
                    # appending here would hijack future reply targets.
                sends.append({"kind": kind, "id": inst.id, "msg_id": mid,
                              "side": side, "pattern_tf": ptf})
                self.audit.write({"event": "post", "kind": kind, "id": inst.id,
                                  "msg_id": mid, "side": side,
                                  "pattern_tf": ptf, "charts": len(imgs),
                                  "reply_to": reply_to})

            elif kind == "breakout_keep":
                inst.probe_msg_id_dm = None   # mirrored heads-up is kept too
                self.audit.write({"event": "breakout_confirmed", "id": inst.id,
                                  "side": a.detail.get("side"),
                                  "msg_id": a.detail.get("msg_id")})

            elif kind == "retract":
                mid = a.detail.get("msg_id")
                ok = False
                if mid and not self.dry_run and self.tg is not None:
                    ok = await self.tg.delete(mid, a.detail.get("reason", ""))
                    if mid in inst.msg_ids:
                        inst.msg_ids.remove(mid)
                # mirrored heads-up (if any) is retracted with its channel twin
                dmid = inst.probe_msg_id_dm
                if dmid is not None:
                    if not self.dry_run and self.tg is not None and self.dm_chat:
                        await self.tg.delete(dmid, a.detail.get("reason", ""),
                                             chat_id=self.dm_chat)
                    inst.probe_msg_id_dm = None
                self.audit.write({"event": "retract", "id": inst.id, "msg_id": mid,
                                  "reason": a.detail.get("reason"), "deleted": ok})

            elif kind in ("upgrade", "invalidate"):
                self.audit.write({"event": kind, "id": inst.id, **a.detail})
        return sends

    async def _chart(self, ses, inst: Instance, lines: List[float],
                     trend_lines: Optional[List[tuple]] = None,
                     tf: Optional[str] = None) -> Optional[bytes]:
        """Alert chart. `tf` defaults to the instance's own TF — pass the
        low TF for the potential-break pattern chart (user rule 2026-09-27)."""
        if self.dry_run:
            return None
        tf = tf or inst.tf
        async with self._chart_sem:
            dt = time.time() - self._chart_last
            if dt < CHART_SPACING_S:
                await asyncio.sleep(CHART_SPACING_S - dt)
            img = await nt.fetch_chart(ses, self.cfg.chart_url, inst.symbol,
                                       tf, lines, trend_lines)
            if img is None:
                # one retry before the text-only fallback
                await asyncio.sleep(CHART_RETRY_DELAY_S)
                img = await nt.fetch_chart(ses, self.cfg.chart_url, inst.symbol,
                                           tf, lines, trend_lines)
            self._chart_last = time.time()
            return img

    async def _post_photos(self, caption: str, sub_caption: str,
                           imgs: List[bytes], reply_to: Optional[int],
                           chat_id: Optional[str] = None) -> Optional[int]:
        """Post the alert's chart(s): ONE message when there is a single
        chart, ONE album carrying both when the pattern was found on the
        low TF (user rule 2026-09-27 — both charts in the same message).
        Separate photos then text are only fallbacks when the album fails.
        Returns the id of the FIRST message (the alert's id / anchor)."""
        if self.tg is None:
            return None
        imgs = imgs or []
        # A stale reply target must never swallow the alert: a message
        # deleted since it was recorded (QNT DM twin 2429) makes Telegram
        # answer 400 "message to be replied not found" for EVERY form of the
        # post. Each form is therefore tried WITH the reply first, then
        # WITHOUT it.
        targets = [reply_to] if reply_to else [None]
        if reply_to:
            targets.append(None)
        for target in targets:
            if imgs:
                mid = await self.tg.post_media_group(caption, imgs,
                                                     reply_to_id=target,
                                                     chat_id=chat_id)
            else:
                mid = await self.tg.post(caption, reply_to_id=target,
                                         chat_id=chat_id)
            if mid is not None:
                return mid
        # last resort: separate photos, no reply
        first: Optional[int] = None
        for i, im in enumerate(imgs):
            m = await self.tg.post_photo(caption if i == 0 else sub_caption,
                                         im, reply_to_id=None,
                                         chat_id=chat_id)
            if i == 0:
                first = m
        return first

    def _dm_mirror(self, inst: Instance, force: bool = False) -> bool:
        """True when this instance's alerts are also mirrored to the DM
        (user rule 2026-09-20: ascending/descending triangles;
        2026-09-27: every potential-break alert, whatever the type)."""
        return bool(self.dm_chat) and (force or inst.type in DM_MIRROR_TYPES)

    async def _dm_post(self, inst: Instance, kind: str, caption: str,
                       img: Optional[bytes] = None,
                       reply_to_dm: Optional[int] = None,
                       imgs: Optional[List[bytes]] = None,
                       sub_caption: str = "",
                       force: bool = False) -> Optional[int]:
        """Mirror one alert into the DM chat (chart(s) included when
        fetched). `imgs` supersedes `img` for multi-chart alerts.
        Returns the DM message id, or None when mirroring is off/failed."""
        if not self._dm_mirror(inst, force=force) or self.tg is None \
                or self.dry_run:
            return None
        seq = imgs if imgs is not None else ([img] if img else [])
        dmid = await self._post_photos(caption, sub_caption or caption, seq,
                                       reply_to_dm, self.dm_chat)
        self.audit.write({"event": "dm_mirror", "kind": kind, "id": inst.id,
                          "msg_id": dmid})
        return dmid

    # ── probes ─────────────────────────────────────────────────────────
    async def probe_once(self, ses: aiohttp.ClientSession) -> int:
        """One probe pass. Returns number of heads-up messages posted."""
        if not self.cfg.notif_enabled or self.dry_run:
            return 0
        now_ms = int(time.time() * 1000)
        posted = 0
        dirty = False
        for inst in list(self.instances.values()):
            if inst.state not in ACTIVE_STATES:
                continue
            tf_ms = TF_MS[inst.tf]
            open_ms = probe_due(inst.probe_last_ms, now_ms, tf_ms, self.cfg.probe_warn_s)
            if open_ms is None:
                continue
            inst.probe_last_ms = open_ms
            dirty = True
            self.audit.write({"event": "probe_attempt", "id": inst.id})
            if inst.probe_msg_id is not None:
                continue  # heads-up already pending for an earlier candle
            price = await fetch_last_price(ses, inst.symbol)
            if price <= 0:
                self.audit.write({"event": "probe_miss", "id": inst.id})
                continue
            side = inst.beyond_side(price, self.cfg.breakout_buffer_atr * inst.atr,
                                    open_ms // tf_ms)
            self.audit.write({"event": "probe", "id": inst.id, "tf": inst.tf,
                              "live": price, "side": side or ""})
            if side is None:
                continue
            # candle-problem gate (daw-breakouts parity): fresh 31-kline check
            # before posting the heads-up — anomalous gaps suppress the alert
            rows = await fetch_klines(ses, inst.symbol, inst.tf, 31)
            if has_candle_problems(rows):
                self.audit.write({"event": "candle_problems", "kind": "probe",
                                  "id": inst.id, "action": "suppress"})
                continue
            # level = last pivot in that direction for box/triangles, the
            # boundary line for wedges (user rules 2026-09-16)
            level = alert_level_for(inst, side, open_ms // tf_ms)
            if level is None:      # defensive; side != None implies the level exists
                continue
            caption = nt.fmt_breakout(inst, side, level)
            # For triangles the flat boundary IS the breakout level —
            # sending both creates a duplicate horizontal line.
            segs = inst.trend_segments(tf_ms, open_ms) \
                if inst.type != "box" else None
            _flat_side = ("up" if inst.type == "ascending_triangle"
                          else "down" if inst.type == "descending_triangle"
                          else None)
            chart_level = None if side == _flat_side else [level]
            # heads-up threads onto this structure's compression confirmation
            # message (user rule 2026-09-16); standalone if none was sent
            reply_to = inst.msg_ids[0] if inst.msg_ids else None
            mid = None
            if img := await self._chart(ses, inst, chart_level, segs):
                mid = await self.tg.post_photo(caption, img, reply_to)
            if mid is None:
                mid = await self.tg.post(caption, reply_to)
            if mid is not None:
                inst.probe_msg_id = mid
                inst.probe_candle_ms = open_ms
                inst.probe_side = side
                # DM mirror (user rule 2026-09-20): mirrored heads-up threads
                # onto the mirrored compression message in the DM
                dm_reply = inst.dm_msg_ids[0] if inst.dm_msg_ids else None
                dmid = await self._dm_post(inst, "probe", caption, img, dm_reply)
                if dmid is not None:
                    inst.probe_msg_id_dm = dmid
                posted += 1
                self.audit.write({"event": "probe_post", "id": inst.id,
                                  "side": side, "level": level, "msg_id": mid,
                                  "reply_to": reply_to})
        if dirty:
            self._save_state()
        return posted

    # ── daemon ─────────────────────────────────────────────────────────
    async def _scan_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            now_ms = int(time.time() * 1000)
            nxt = next_scan_ms(now_ms, self.cfg.scan_offset_s)
            wait_s = max(1.0, (nxt - now_ms) / 1000.0)
            try:
                await asyncio.wait_for(stop.wait(), timeout=wait_s)
                return
            except asyncio.TimeoutError:
                pass
            try:
                await self.scan_once()
            except Exception as exc:
                log.error("scan failed: %s", exc, exc_info=True)

    async def _probe_loop(self, stop: asyncio.Event, ses: aiohttp.ClientSession) -> None:
        while not stop.is_set():
            try:
                await self.probe_once(ses)
            except Exception as exc:
                log.error("probe pass failed: %s", exc, exc_info=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.cfg.probe_loop_s)
                return
            except asyncio.TimeoutError:
                pass

    async def run(self) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat:
            log.error("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing — "
                      "notifications would fail; check .env")
        self.tg = nt.Telegram(token, chat)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass

        log.info("compression-detection engine starting (tfs=%s, notify=%s, "
                 "dm_mirror=%s)",
                 self.cfg.timeframes, self.cfg.notif_enabled, self.dm_chat or "-")
        async with aiohttp.ClientSession() as ses:
            await self.scan_once()
            scan_task = asyncio.create_task(self._scan_loop(stop))
            probe_task = asyncio.create_task(self._probe_loop(stop, ses))
            await stop.wait()
            scan_task.cancel()
            probe_task.cancel()
        log.info("engine stopped")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = Config.load()
    engine = Engine(cfg)
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
