"""Scan report writer — the review artifact for dry-runs and audits.

One JSON per scan under scan_reports/: universe size, errors, detection
events (candidate dumps with full metrics — feeds spec §30 questions),
state transitions, notification sends/suppressions, and per-(tf,type,state)
instance counts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from .lifecycle import ACTIVE_STATES, Instance


def build_summary(
    now_ms: int,
    dry: bool,
    universe_count: int,
    errors: List[str],
    actions: List[dict],
    detections: List[dict],
    instances: Dict[str, Instance],
    sends: List[dict],
) -> dict:
    counts: Dict[str, int] = {}
    actives: List[dict] = []
    for inst in instances.values():
        if inst.state in ACTIVE_STATES:
            key = f"{inst.tf}|{inst.type}|{inst.state}"
            counts[key] = counts.get(key, 0) + 1
            last = inst.pivots[-1] if inst.pivots else {}
            actives.append({
                "id": inst.id, "symbol": inst.symbol, "tf": inst.tf,
                "type": inst.type, "state": inst.state, "level": inst.level,
                "pivot_count": inst.pivot_count,
                "upper": round(inst.upper_at(last.get("bar", 0)), 8),
                "lower": round(inst.lower_at(last.get("bar", 0)), 8),
                "metrics": {k: round(v, 6) if isinstance(v, float) else v
                            for k, v in inst.metrics.items()},
                "notified": dict(inst.notified),
            })
    actives.sort(key=lambda x: (x["tf"], x["type"], x["symbol"]))
    return {
        "generated_at": int(now_ms / 1000),
        "generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now_ms / 1000)),
        "dry": dry,
        "universe_count": universe_count,
        "errors": errors,
        "counts_by_tf_type_state": counts,
        "instances_active": len(actives),
        "active_instances": actives,
        "detections": detections,
        "actions": actions,
        "sends": sends,
    }


def write_report(dir_path, now_ms: int, summary: dict) -> Path:
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%Y%m%d_%H%M%S", time.gmtime(now_ms / 1000)) + ".json"
    path = d / name
    path.write_text(json.dumps(summary, indent=1, default=str))
    return path


def format_summary_text(summary: dict) -> str:
    """Compact human-readable summary for CLI output."""
    lines = [
        f"scan {summary['generated_utc']} UTC  dry={summary['dry']}  universe={summary['universe_count']}",
        f"active instances: {summary['instances_active']}",
    ]
    if summary["counts_by_tf_type_state"]:
        lines.append("counts (tf|type|state):")
        for k in sorted(summary["counts_by_tf_type_state"]):
            lines.append(f"  {k}: {summary['counts_by_tf_type_state'][k]}")
    if summary["detections"]:
        lines.append("detection events:")
        for d in summary["detections"][:40]:
            for c in d["candidates"][:2]:
                lines.append(
                    f"  {d['symbol']}({d['tf']}) {c['type']} x{len(c['pivots'])} "
                    f"fit={c.get('fit_err_atr', 0):.2f} conv={c.get('convergence_rate', 0):.2f}"
                )
        if len(summary["detections"]) > 40:
            lines.append(f"  ... {len(summary['detections']) - 40} more")
    if summary["actions"]:
        kinds: Dict[str, int] = {}
        for a in summary["actions"]:
            kinds[a["kind"]] = kinds.get(a["kind"], 0) + 1
        lines.append("actions: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    if summary["sends"]:
        lines.append(f"sends: {len(summary['sends'])}")
    if summary["errors"]:
        lines.append(f"errors: {len(summary['errors'])}")
        for e in summary["errors"][:10]:
            lines.append(f"  {e}")
    return "\n".join(lines)
