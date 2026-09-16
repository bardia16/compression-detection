"""Config loader — config.toml is the single source of truth."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .detector import ALL_TYPES, DetectConfig

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO / "config.toml"


@dataclass
class Config:
    timeframes: List[str]
    candle_limit: int
    zigzag_coef: float
    zigzag_atr_length: int
    atr14_length: int
    atr14_method: str
    det: DetectConfig
    scan_offset_s: int
    catchup_max_candles: int
    min_volume_btc: float
    lcw_limit: int
    probe_warn_s: int
    probe_loop_s: int
    breakout_buffer_atr: float
    notif_enabled: bool
    notify_min_state: str
    notify_on_compress_upgrade: bool
    notify_invalidation: bool
    best_only_per_coin_tf: bool
    chart_url: str
    state_path: Path
    prune_after_days: int

    # convenience delegates (lifecycle reads these off the config object)
    @property
    def min_pivots(self):
        return self.det.min_pivots

    @property
    def confirm_extra_pivots(self):
        return self.det.confirm_extra_pivots

    @property
    def established_extra_pivots(self):
        return self.det.established_extra_pivots

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        p = Path(path) if path else DEFAULT_PATH
        raw = tomllib.loads(p.read_text())

        det_raw = raw["detection"]
        det = DetectConfig(
            min_pivots={k: int(v) for k, v in det_raw["min_pivots"].items()},
            max_window_pivots=int(det_raw["max_window_pivots"]),
            max_pivot_age_bars=int(det_raw["max_pivot_age_bars"]),
            flat_tol_atr=float(det_raw["flat_tol_atr"]),
            boundary_tol_atr=float(det_raw["boundary_tol_atr"]),
            min_convergence_pct=float(det_raw["min_convergence_pct"]),
            max_width_bounce_frac=float(det_raw["max_width_bounce_frac"]),
            confirm_extra_pivots=int(det_raw["confirm_extra_pivots"]),
            established_extra_pivots=int(det_raw["established_extra_pivots"]),
            selection_order=tuple(det_raw.get("selection_order") or ALL_TYPES),
        )

        proj = raw["project"]
        zz = raw["zigzag"]
        lab = raw["labels"]
        scan = raw["scan"]
        univ = raw["universe"]
        brk = raw["breakout"]
        notif = raw["notifications"]
        state = raw["state"]

        return cls(
            timeframes=list(proj["timeframes"]),
            candle_limit=int(proj["candle_limit"]),
            zigzag_coef=float(zz["coef"]),
            zigzag_atr_length=int(zz["atr_length"]),
            atr14_length=int(lab["atr14_length"]),
            atr14_method=str(lab["atr14_method"]),
            det=det,
            scan_offset_s=int(scan["offset_s"]),
            catchup_max_candles=int(scan["catchup_max_candles"]),
            min_volume_btc=float(univ["min_volume_btc"]),
            lcw_limit=int(univ["lcw_limit"]),
            probe_warn_s=int(brk["probe_warn_s"]),
            probe_loop_s=int(brk["probe_loop_s"]),
            breakout_buffer_atr=float(brk["breakout_buffer_atr"]),
            notif_enabled=bool(notif["enabled"]),
            notify_min_state=str(notif["notify_min_state"]),
            notify_on_compress_upgrade=bool(notif["notify_on_compress_upgrade"]),
            notify_invalidation=bool(notif["notify_invalidation"]),
            best_only_per_coin_tf=bool(notif["best_only_per_coin_tf"]),
            chart_url=str(notif["chart_url"]),
            state_path=Path(state["path"]),
            prune_after_days=int(state["prune_after_days"]),
        )
