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
    ACTIVE_STATES, TERMINAL_STATES, Instance, alert_level_for, best_per_coin_tf,
    update_for_scan,
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


class Engine:
    def __init__(self, cfg: Config, dry_run: bool = False,
                 tg: Optional[nt.Telegram] = None):
        self.cfg = cfg
        self.dry_run = dry_run
        self.audit = Audit(AUDIT_DIR)
        self.instances: Dict[str, Instance] = {}
        self.tg = tg
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
            self.instances = {k: Instance.from_dict(v)
                              for k, v in (d.get("instances") or {}).items()}
            log.info("state loaded: %d instances", len(self.instances))
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
        labeled = label_all_pivots(pivots, atr14)
        last_bar = candles[-1].ts // tf_ms
        cands = detect_candidates(labeled, atr14, tf_ms, last_bar, self.cfg.det)
        return {"candles": candles, "pivots": pivots, "labeled": labeled,
                "candidates": cands, "last_bar": last_bar}

    async def _fetch_coin(self, ses: aiohttp.ClientSession, sem: asyncio.Semaphore,
                          pair: str) -> Dict[str, Optional[dict]]:
        async with sem:
            out: Dict[str, Optional[dict]] = {}
            for tf in self.cfg.timeframes:
                try:
                    candles = await fetch_klines(ses, pair, tf, self.cfg.candle_limit)
                    out[tf] = self._analyze_candles(candles, tf)
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
            for pair, res in zip(pairs, results):
                # store the full Binance symbol (ORCAUSDT) — messages, charts,
                # probe fetches and ticker calls all need it (bare ORCA broke
                # chart fetches 503 and made every probe miss, 2026-09-16)
                sym = pair.split("/")[0] + "USDT"
                for tf in self.cfg.timeframes:
                    r = res.get(tf)
                    if r is None:
                        errors.append(f"{sym}:{tf} no data")
                        continue
                    candles_by_key[(sym, tf)] = r["candles"]
                    acts = update_for_scan(
                        instances_src, sym, tf, r["candidates"], r["candles"],
                        TF_MS[tf], now_s, self.cfg,
                    )
                    raw_actions.extend(acts)
                    if r["candidates"]:
                        detections.append({
                            "symbol": sym, "tf": tf,
                            "candidates": [c.to_dict() for c in r["candidates"][:8]],
                        })

            if self.cfg.best_only_per_coin_tf:
                raw_actions = best_per_coin_tf(raw_actions)

            sends = await self._dispatch(raw_actions, ses, candles_by_key)
            actions = [{"kind": a.kind,
                        "id": a.instance.id if a.instance else None,
                        "detail": a.detail}
                       for a in raw_actions]

        summary = build_summary(now_ms, self.dry_run, len(pairs), errors,
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
                if inst.type == "box":
                    img = await self._chart(ses, inst, [detail["level"]], None)
                else:
                    segs = inst.trend_segments(TF_MS[inst.tf], x2)
                    img = await self._chart(ses, inst, [detail["level"]], segs)
                mid = None
                if img is not None:
                    mid = await self.tg.post_photo(caption, img)
                if mid is None:
                    mid = await self.tg.post(caption)
                sends.append({"kind": kind, "id": inst.id, "msg_id": mid})
                if mid is not None:
                    inst.msg_ids.append(mid)
                    inst.notified["breakout"] = True
                self.audit.write({"event": "post", "kind": kind, "id": inst.id,
                                  "side": detail.get("side"), "msg_id": mid,
                                  "chart": img is not None})

            elif kind == "breakout_keep":
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
                self.audit.write({"event": "retract", "id": inst.id, "msg_id": mid,
                                  "reason": a.detail.get("reason"), "deleted": ok})

            elif kind in ("upgrade", "invalidate"):
                self.audit.write({"event": kind, "id": inst.id, **a.detail})
        return sends

    async def _chart(self, ses, inst: Instance, lines: List[float],
                     trend_lines: Optional[List[tuple]] = None) -> Optional[bytes]:
        if self.dry_run:
            return None
        async with self._chart_sem:
            dt = time.time() - self._chart_last
            if dt < CHART_SPACING_S:
                await asyncio.sleep(CHART_SPACING_S - dt)
            img = await nt.fetch_chart(ses, self.cfg.chart_url, inst.symbol,
                                       inst.tf, lines, trend_lines)
            if img is None:
                # one retry before the text-only fallback
                await asyncio.sleep(CHART_RETRY_DELAY_S)
                img = await nt.fetch_chart(ses, self.cfg.chart_url, inst.symbol,
                                           inst.tf, lines, trend_lines)
            self._chart_last = time.time()
            return img

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
            if inst.type == "box":
                img = await self._chart(ses, inst, [level], None)
            else:
                segs = inst.trend_segments(tf_ms, open_ms)
                img = await self._chart(ses, inst, [level], segs)
            mid = None
            if img is not None:
                mid = await self.tg.post_photo(caption, img)
            if mid is None:
                mid = await self.tg.post(caption)
            if mid is not None:
                inst.probe_msg_id = mid
                inst.probe_candle_ms = open_ms
                inst.probe_side = side
                posted += 1
                self.audit.write({"event": "probe_post", "id": inst.id,
                                  "side": side, "level": level, "msg_id": mid})
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

        log.info("compression-detection engine starting (tfs=%s, notify=%s)",
                 self.cfg.timeframes, self.cfg.notif_enabled)
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
