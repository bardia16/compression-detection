"""Config loader tests — pins the repo config.toml contract."""
from compression_detection.config import Config


def test_load_from_repo_config():
    cfg = Config.load()
    assert cfg.timeframes == ["15m", "1h", "4h"]
    assert cfg.candle_limit == 400
    assert cfg.zigzag_coef == 1.2
    assert cfg.zigzag_atr_length == 7
    assert cfg.atr14_length == 14
    assert cfg.atr14_method == "close_only"


def test_detection_config_defaults():
    cfg = Config.load()
    det = cfg.det
    assert det.min_pivots["box"] == 4
    assert det.min_pivots["falling_wedge"] == 5
    assert det.confirm_extra_pivots == 1
    assert det.established_extra_pivots == 2
    assert det.min_boundary_hits == 2
    assert det.flat_tol_atr == 1.0
    assert det.close_breach_tol_atr == 0.5
    assert det.max_close_breaches == 2
    assert det.box_range_mode is False
    # detection scope (user rule 2026-09-20): boxes + asc/desc triangles ONLY
    assert det.selection_order == ("ascending_triangle",
                                   "descending_triangle", "box")
    assert det.boundary_tol_atr == 1.0
    assert det.min_convergence_pct == 25.0
    # lifecycle delegates
    assert cfg.min_pivots is det.min_pivots
    assert cfg.confirm_extra_pivots == 1


def test_scan_breakout_notification_config():
    cfg = Config.load()
    assert cfg.scan_offset_s == 90
    assert cfg.catchup_max_candles == 8
    assert cfg.probe_warn_s == 180
    assert cfg.probe_loop_s == 30
    assert cfg.breakout_buffer_atr == 0.0
    assert isinstance(cfg.notif_enabled, bool)
    assert cfg.notify_min_state == "confirmed"
    assert cfg.best_only_per_coin_tf is True
    assert cfg.chart_url.startswith("http")
    assert cfg.state_path.name == "structure_state.json"
