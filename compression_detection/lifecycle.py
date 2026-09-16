"""Structure lifecycle: instance registry, state machine, continuation
matching, candle-close verdicts, breakout & invalidation decisions.

States:  DETECTED → CONFIRMED → COMPRESSING → BREAKOUT | INVALIDATED
Levels:  detected (min pivots) / confirmed (+confirm_extra) /
         established (+established_extra)   — spec §27

Rules (spec §20-24, locked decisions 2026-09-16):
- only confirmed pivots / closed candles ever touch an instance
- continuation = a same-type candidate sharing >=2 pivot timestamps
- candle-close verdict for a pending probe: confirmed close beyond the
  probed side → KEEP message + BREAKOUT; failed → DELETE message + stay
  active (retract only when it didn't close through)
- no probe but close beyond → post the breakout at close (missed probe)
- geometry dissolved without a breakout → INVALIDATED (never confused
  with BREAKOUT)
- a structure is never locked: every scan re-evaluates (spec §18)

This module is pure logic — no IO. The engine persists instances and
executes the returned Actions (posts, deletes, state writes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import structure as st
from .detector import Candidate
from .models import Candle

STATE_DETECTED = "detected"
STATE_CONFIRMED = "confirmed"
STATE_COMPRESSING = "compressing"
STATE_BREAKOUT = "breakout"
STATE_INVALIDATED = "invalidated"

ACTIVE_STATES = (STATE_DETECTED, STATE_CONFIRMED, STATE_COMPRESSING)
TERMINAL_STATES = (STATE_BREAKOUT, STATE_INVALIDATED)

LEVEL_DETECTED = "detected"
LEVEL_CONFIRMED = "confirmed"
LEVEL_ESTABLISHED = "established"

_LEVEL_TO_STATE = {
    LEVEL_DETECTED: STATE_DETECTED,
    LEVEL_CONFIRMED: STATE_CONFIRMED,
    LEVEL_ESTABLISHED: STATE_COMPRESSING,
}

_NOTIFY_RANK = {STATE_DETECTED: 0, STATE_CONFIRMED: 1, STATE_COMPRESSING: 2}


def state_rank(state: str) -> int:
    """Rank for notify gates (breakout/invalidated rank -1 = never gates)."""
    return _NOTIFY_RANK.get(state, -1)


def level_for(pivot_count: int, type_name: str, cfg) -> str:
    extra = pivot_count - cfg.min_pivots[type_name]
    if extra >= cfg.established_extra_pivots:
        return LEVEL_ESTABLISHED
    if extra >= cfg.confirm_extra_pivots:
        return LEVEL_CONFIRMED
    return LEVEL_DETECTED


def state_for_level(level: str) -> str:
    return _LEVEL_TO_STATE[level]


@dataclass
class Action:
    """Something the engine must do as a result of a lifecycle decision.

    kind: compression_notify | breakout_keep | breakout_post |
          retract | invalidate | upgrade | skip_create
    instance is None only for skip_create (no instance exists by design).
    """

    kind: str
    instance: Optional["Instance"]
    detail: dict = field(default_factory=dict)


@dataclass
class Instance:
    id: str
    symbol: str
    tf: str
    type: str
    state: str
    level: str
    created_ts: int
    anchor_ts: int              # first pivot ts (ms)
    last_scan_ts: int
    last_eval_ts: int           # ts (ms) of last candle evaluated
    pivots: List[dict]          # PivotRef.to_dict() list
    upper_line: st.Line
    lower_line: st.Line
    atr: float
    upper_class: str
    lower_class: str
    metrics: dict
    notified: dict = field(default_factory=lambda: {"compression": False, "breakout": False})
    msg_ids: List[int] = field(default_factory=list)
    # probe bookkeeping (T−3:00 heads-up)
    probe_last_ms: int = 0      # candle open of last probe ATTEMPT
    probe_msg_id: Optional[int] = None
    probe_candle_ms: int = 0    # candle open of the POSTED message
    probe_side: str = ""        # "up" | "down"
    events: List[dict] = field(default_factory=list)

    # ── helpers ────────────────────────────────────────────────────────
    @property
    def pivot_tss(self) -> set:
        return {p["ts"] for p in self.pivots}

    @property
    def pivot_count(self) -> int:
        return len(self.pivots)

    def add_event(self, kind: str, ts: int, **detail) -> None:
        self.events.append({"ts": ts, "kind": kind, **detail})
        if len(self.events) > 60:
            self.events = self.events[-60:]

    def upper_at(self, abs_bar: float) -> float:
        return self.upper_line.at(abs_bar)

    def lower_at(self, abs_bar: float) -> float:
        return self.lower_line.at(abs_bar)

    def beyond_side(self, abs_bar: float, price: float, buffer: float) -> Optional[str]:
        """Which boundary (if any) the price is beyond right now."""
        if price > self.upper_at(abs_bar) + buffer:
            return "up"
        if price < self.lower_at(abs_bar) - buffer:
            return "down"
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "symbol": self.symbol, "tf": self.tf, "type": self.type,
            "state": self.state, "level": self.level,
            "created_ts": self.created_ts, "anchor_ts": self.anchor_ts,
            "last_scan_ts": self.last_scan_ts, "last_eval_ts": self.last_eval_ts,
            "pivots": self.pivots,
            "upper": {"slope": self.upper_line.slope, "intercept": self.upper_line.intercept,
                      "base_bar": self.upper_line.base_bar},
            "lower": {"slope": self.lower_line.slope, "intercept": self.lower_line.intercept,
                      "base_bar": self.lower_line.base_bar},
            "atr": self.atr, "upper_class": self.upper_class,
            "lower_class": self.lower_class, "metrics": self.metrics,
            "notified": dict(self.notified), "msg_ids": list(self.msg_ids),
            "probe_last_ms": self.probe_last_ms, "probe_msg_id": self.probe_msg_id,
            "probe_candle_ms": self.probe_candle_ms, "probe_side": self.probe_side,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        def _line(x):
            return st.Line(slope=x.get("slope", 0.0), intercept=x.get("intercept", 0.0),
                           base_bar=x.get("base_bar", 0.0))
        return cls(
            id=d["id"], symbol=d["symbol"], tf=d["tf"], type=d["type"],
            state=d["state"], level=d["level"], created_ts=d["created_ts"],
            anchor_ts=d["anchor_ts"], last_scan_ts=d.get("last_scan_ts", 0),
            last_eval_ts=d.get("last_eval_ts", 0), pivots=list(d.get("pivots") or []),
            upper_line=_line(d.get("upper") or {}), lower_line=_line(d.get("lower") or {}),
            atr=d.get("atr", 0.0), upper_class=d.get("upper_class", st.FLAT),
            lower_class=d.get("lower_class", st.FLAT), metrics=dict(d.get("metrics") or {}),
            notified={"compression": bool((d.get("notified") or {}).get("compression")),
                      "breakout": bool((d.get("notified") or {}).get("breakout"))},
            msg_ids=list(d.get("msg_ids") or []),
            probe_last_ms=d.get("probe_last_ms", 0),
            probe_msg_id=d.get("probe_msg_id"),
            probe_candle_ms=d.get("probe_candle_ms", 0),
            probe_side=d.get("probe_side", ""),
            events=list(d.get("events") or []),
        )


def instance_id(symbol: str, tf: str, type_name: str, anchor_ts: int) -> str:
    return f"{symbol}|{tf}|{type_name}|{anchor_ts}"


def create_instance(
    cand: Candidate, symbol: str, tf: str, now_s: int, cfg
) -> Instance:
    level = level_for(cand.pivot_count, cand.type, cfg)
    inst = Instance(
        id=instance_id(symbol, tf, cand.type, cand.first_ts),
        symbol=symbol, tf=tf, type=cand.type,
        state=state_for_level(level), level=level,
        created_ts=now_s, anchor_ts=cand.first_ts,
        last_scan_ts=now_s, last_eval_ts=0,
        pivots=[r.to_dict() for r in cand.refs],
        upper_line=cand.upper, lower_line=cand.lower, atr=cand.atr,
        upper_class=cand.upper_class, lower_class=cand.lower_class,
        metrics=dict(cand.metrics),
    )
    inst.add_event("detected", now_s, level=level, pivots=cand.pivot_count)
    return inst


def find_continuation(
    inst: Instance, candidates: Sequence[Candidate]
) -> Optional[Candidate]:
    """Best same-type candidate sharing >= 2 pivot timestamps."""
    for c in candidates:
        if c.type != inst.type:
            continue
        if len(c.pivot_tss & inst.pivot_tss) >= 2:
            return c
    return None


def _apply_candidate(inst: Instance, cand: Candidate, now_s: int, cfg) -> List[Action]:
    """Update a continuing instance from its fresh candidate."""
    actions: List[Action] = []
    inst.pivots = [r.to_dict() for r in cand.refs]
    inst.upper_line, inst.lower_line = cand.upper, cand.lower
    inst.atr, inst.upper_class, inst.lower_class = cand.atr, cand.upper_class, cand.lower_class
    inst.metrics = dict(cand.metrics)
    inst.last_scan_ts = now_s

    new_level = level_for(cand.pivot_count, cand.type, cfg)
    new_state = state_for_level(new_level)
    if new_state != inst.state:
        old = inst.state
        inst.state, inst.level = new_state, new_level
        inst.add_event("state_change", now_s, old=old, new=new_state, level=new_level)
        actions.append(Action("upgrade", inst, {"old": old, "new": new_state}))
    return actions


def _maybe_notify(inst: Instance, now_s: int, cfg) -> List[Action]:
    """Compression notification gate (once per instance)."""
    if inst.notified.get("compression"):
        return []
    if inst.state in TERMINAL_STATES:
        return []
    if state_rank(inst.state) < state_rank(cfg.notify_min_state):
        return []
    return [Action("compression_notify", inst, {"state": inst.state, "level": inst.level})]


def evaluate_closed_candles(
    inst: Instance,
    candles: Sequence[Candle],
    tf_ms: int,
    now_s: int,
    cfg,
) -> List[Action]:
    """Candle-close pass for one instance: probe verdicts + breakout posts.

    Only candles strictly newer than inst.last_eval_ts are processed
    (capped at cfg.catchup_max_candles newest ones). Runs BEFORE any
    candidate updates so the boundary used is the one known in real time.
    """
    actions: List[Action] = []
    new = [c for c in candles if c.ts > inst.last_eval_ts]
    if not new:
        return actions
    if len(new) > cfg.catchup_max_candles:
        new = new[-cfg.catchup_max_candles:]
    late_cutoff = new[-1].ts - 2 * tf_ms

    buffer = cfg.breakout_buffer_atr * inst.atr

    for c in new:
        abs_bar = c.ts // tf_ms
        up = inst.upper_at(abs_bar)
        lo = inst.lower_at(abs_bar)
        beyond_up = c.close > up + buffer
        beyond_dn = c.close < lo - buffer
        late = c.ts < late_cutoff

        if inst.probe_msg_id is not None and inst.probe_candle_ms == c.ts:
            # verdict for the posted heads-up
            if inst.probe_side == "up" and beyond_up:
                actions.append(Action("breakout_keep", inst, {
                    "side": "up", "boundary": up, "close": c.close,
                    "candle_ts": c.ts, "late": late, "msg_id": inst.probe_msg_id,
                }))
                inst.state = STATE_BREAKOUT
                inst.notified["breakout"] = True
                inst.add_event("breakout", int(c.ts // 1000), side="up", close=c.close,
                               boundary=up, kept=True)
                inst.probe_msg_id = None
                inst.last_eval_ts = new[-1].ts
                return actions
            if inst.probe_side == "down" and beyond_dn:
                actions.append(Action("breakout_keep", inst, {
                    "side": "down", "boundary": lo, "close": c.close,
                    "candle_ts": c.ts, "late": late, "msg_id": inst.probe_msg_id,
                }))
                inst.state = STATE_BREAKOUT
                inst.notified["breakout"] = True
                inst.add_event("breakout", int(c.ts // 1000), side="down", close=c.close,
                               boundary=lo, kept=True)
                inst.probe_msg_id = None
                inst.last_eval_ts = new[-1].ts
                return actions
            # failed close: retract the heads-up, stay active
            actions.append(Action("retract", inst, {
                "msg_id": inst.probe_msg_id, "candle_ts": c.ts,
                "reason": "no break by candle close",
            }))
            inst.add_event("retract", int(c.ts // 1000), side=inst.probe_side)
            inst.probe_msg_id = None
            inst.probe_candle_ms = 0
            inst.probe_side = ""
            # opposite-direction close = a genuine breakout the other way
            if beyond_up or beyond_dn:
                side = "up" if beyond_up else "down"
                boundary = up if beyond_up else lo
                actions.append(Action("breakout_post", inst, {
                    "side": side, "boundary": boundary, "close": c.close,
                    "candle_ts": c.ts, "late": late,
                }))
                inst.state = STATE_BREAKOUT
                inst.notified["breakout"] = True
                inst.add_event("breakout", int(c.ts // 1000), side=side, close=c.close,
                               boundary=boundary, kept=False)
                inst.last_eval_ts = new[-1].ts
                return actions
            continue

        # no pending probe for this candle
        if beyond_up or beyond_dn:
            side = "up" if beyond_up else "down"
            boundary = up if beyond_up else lo
            actions.append(Action("breakout_post", inst, {
                "side": side, "boundary": boundary, "close": c.close,
                "candle_ts": c.ts, "late": late,
            }))
            inst.state = STATE_BREAKOUT
            inst.notified["breakout"] = True
            inst.add_event("breakout", int(c.ts // 1000), side=side, close=c.close,
                           boundary=boundary, kept=False)
            inst.last_eval_ts = new[-1].ts
            return actions

    # stale probe: candle never came through the catchment (e.g. downtime)
    if inst.probe_msg_id is not None and inst.probe_candle_ms and \
            inst.probe_candle_ms < new[0].ts:
        actions.append(Action("retract", inst, {
            "msg_id": inst.probe_msg_id, "reason": "stale probe", "stale": True,
        }))
        inst.add_event("retract", now_s, side=inst.probe_side, stale=True)
        inst.probe_msg_id = None
        inst.probe_candle_ms = 0
        inst.probe_side = ""

    inst.last_eval_ts = new[-1].ts
    return actions


def update_for_scan(
    instances: Dict[str, Instance],
    symbol: str,
    tf: str,
    candidates: Sequence[Candidate],
    closed_candles: Sequence[Candle],
    tf_ms: int,
    now_s: int,
    cfg,
) -> List[Action]:
    """Full per-(symbol,tf) scan pass. Mutates `instances` in place.

    Order (locked): close verdicts → continuation/invalidation → new
    instances. Returns the actions for the engine to execute.
    """
    actions: List[Action] = []
    symbol_tf = [i for i in instances.values() if i.symbol == symbol and i.tf == tf
                 and i.state in ACTIVE_STATES]

    # 1) candle-close verdicts + breakout checks (boundary = pre-update)
    for inst in symbol_tf:
        if inst.state in TERMINAL_STATES:
            continue
        actions.extend(evaluate_closed_candles(inst, closed_candles, tf_ms, now_s, cfg))

    # 2) continuation / invalidation
    consumed: set = set()
    for inst in symbol_tf:
        if inst.state in TERMINAL_STATES:
            continue
        cand = find_continuation(inst, candidates)
        if cand is None:
            inst.state = STATE_INVALIDATED
            inst.add_event("invalidated", now_s)
            actions.append(Action("invalidate", inst, {}))
            continue
        consumed.add(id(cand))
        actions.extend(_apply_candidate(inst, cand, now_s, cfg))
        actions.extend(_maybe_notify(inst, now_s, cfg))

    # 3) new instances from unmatched candidates (with already-broken guard)
    skipped_types: set = set()
    for cand in candidates:
        if id(cand) in consumed:
            continue
        # one instance per type per (symbol,tf): continuation above covers
        # existing ones; skip if an active instance of this type exists
        if any(i.type == cand.type and i.state in ACTIVE_STATES
               for i in instances.values()
               if i.symbol == symbol and i.tf == tf):
            continue
        # anti-respawn: no new instance on the same anchor as a terminal one
        key = instance_id(symbol, tf, cand.type, cand.first_ts)
        if key in instances:
            continue
        # consumed pivots (PENDLE class): a terminal instance of the same
        # type sharing >=2 pivots = the same structure episode — a broken /
        # invalidated structure is watched ONCE; never re-created from its
        # own sub-windows or after a pullback. A genuinely new structure
        # forms with new pivots and passes this guard.
        if any(i.type == cand.type and i.state in TERMINAL_STATES
               and len(cand.pivot_tss & i.pivot_tss) >= 2
               for i in instances.values()
               if i.symbol == symbol and i.tf == tf):
            continue

        # already-broken guard (2026-09-16): a candidate whose post-last-pivot
        # closes already breach a boundary is a zombie detection — price left
        # the structure before it could ever be watched. Never create it;
        # record the skip once per type for the audit/report.
        beyond = _already_broken_side(cand, closed_candles, tf_ms, cfg)
        if beyond is not None:
            if cand.type not in skipped_types:
                skipped_types.add(cand.type)
                actions.append(Action("skip_create", None, {
                    "symbol": symbol, "tf": tf, "type": cand.type,
                    "reason": "already_broken", "side": beyond,
                    "first_ts": cand.first_ts, "pivot_count": cand.pivot_count,
                }))
            continue

        inst = create_instance(cand, symbol, tf, now_s, cfg)
        # a fresh instance watches candles strictly AFTER its creation scan
        if closed_candles:
            inst.last_eval_ts = closed_candles[-1].ts
        instances[inst.id] = inst
        if inst.state in TERMINAL_STATES:
            continue
        actions.extend(_maybe_notify(inst, now_s, cfg))

    return actions


def _already_broken_side(cand: Candidate, closed_candles, tf_ms: int, cfg) -> Optional[str]:
    """Which boundary the price already breached after the window's last
    pivot — None when all post-pivot closes sit inside the boundaries.

    Uses the same predicate as live breakout detection (parity: if a candle
    would have been an alert while watching, the structure is not createable).
    """
    if not closed_candles:
        return None
    last_ts = cand.refs[-1].ts
    buffer = cfg.breakout_buffer_atr * cand.atr
    for c in closed_candles:
        if c.ts <= last_ts:
            continue
        ab = c.ts // tf_ms
        if c.close > cand.upper.at(ab) + buffer:
            return "up"
        if c.close < cand.lower.at(ab) - buffer:
            return "down"
    return None


def best_per_coin_tf(actions: List[Action]) -> List[Action]:
    """Keep only the best compression_notify per (symbol, tf).

    Ranking matches the detector: pivot_count desc, fit error asc.
    Non-notify actions pass through untouched.
    """
    out: List[Action] = []
    best: Dict[tuple, Action] = {}
    passthrough: List[Action] = []
    for a in actions:
        if a.kind != "compression_notify":
            passthrough.append(a)
            continue
        inst = a.instance
        if inst is None:      # defensive; compression_notify always carries one
            passthrough.append(a)
            continue
        k = (inst.symbol, inst.tf)
        cur = best.get(k)
        if cur is None or _notify_rank(a) < _notify_rank(cur):
            best[k] = a
    out.extend(passthrough)
    out.extend(best.values())
    return out


def _notify_rank(a: Action):
    i = a.instance
    if i is None:
        return (0, 0.0)
    return (-i.pivot_count, i.metrics.get("fit_err_atr", 0.0))
