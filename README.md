# compression-detection

DAW-theory **compression structure detector** — finds Box / Descending /
Ascending / Symmetrical Triangle / Falling & Rising Wedge formations on
confirmed ATR-ZigZag pivots, tracks their lifecycle, and posts
compression + breakout alerts to the **Breakouts** Telegram channel
(same bot as Trading-Alerts).

## What it does

- **Universe:** ALL LiveCoinWatch coins ≥ `min_volume_btc` (default 100
  BTC 24h volume), Binance-availability filtered. Completely independent
  of the daw watchlist — reuses only the same LCW fetch pattern.
- **Timeframes:** 15m / 1h / 4h (config).
- **Pipeline:** closed candles only → ZigZag (coef × ATR7, close-based
  pivots) → DAW labels (HH/LH/EH · HL/LL/EL vs prev same-type, 1×ATR14)
  → per-type candidate matchers (label gate + boundary geometry +
  convergence, all tolerances ATR-normalized) → lifecycle state →
  notifications.
- **Lifecycle:** `DETECTED → CONFIRMED → COMPRESSING → BREAKOUT |
  INVALIDATED`, with maturity levels detected/confirmed/established.
- **Breakout alerts (approved 2026-09-16):** probe at **T−3:00 before
  the structure TF's candle close** — if price is breaking → heads-up
  post. At the close a verdict runs: confirmed close beyond → message
  STAYS (breakout record); failed close → message DELETED, structure
  stays active. Breakout = candle **close** beyond the level; wicks never
  count. Structure confirmation uses **closed candles only**.
- **Breakout level (user rule 2026-09-16):** the price level shown and
  checked is the **last confirmed pivot in that direction** — e.g. the
  last EH for a range's long side, the last LH for a descending
  triangle's up-break, the last EL/HL for shorts. NOT the boundary line
  extrapolated to the break bar. Fitted boundary lines remain for
  geometry (slopes/convergence) and chart range display only.
- **Boundary-hit requirement (spec 2026-09-16):** a structure cannot
  reach CONFIRMED (or beyond) until at least one of its boundaries has
  **≥2 distinct confirmed pivot hits** (`min_boundary_hits` in config).
  Hits = confirmed pivots carrying that boundary's canonical label
  (EH/EL · LH/EL · EH/HL · LH/HL · LH/LL · HH/HL). Counts are exposed
  per instance as `upper_hits` / `lower_hits` (+ `hit_requirement`,
  `hit_boundary` in candidate metrics). Live pivots never count.

## Ops

```bash
# One-shot scan (dry: no sends, no state writes; writes a scan report)
cd /root/compression-detection
./venv/bin/python -m compression_detection scan --dry-run
./venv/bin/python -m compression_detection scan --symbol BTC

# Trace one symbol/TF (pivots + candidates)
./venv/bin/python -m compression_detection explain BTC 4h

# Daemon (also the pm2 entry): scan loop (15m grid + 90s) + probe loop (30s)
./venv/bin/python -m compression_detection.engine
```

**pm2 service:** `compression-detection` via `run_compression.sh`
(wrapper exports `.env` — pm2 strips inline env).

```bash
pm2 restart compression-detection    # after config/code changes
pm2 logs compression-detection
```

**Files:**
- `structure_state.json` — instance registry (restart-safe; atomic writes)
- `audit/compression_YYYYMMDD.jsonl` — every decision (scan/probe/verdict/post)
- `scan_reports/*.json` — per-scan report incl. candidate dumps + metrics
- `config.toml` — all thresholds; nothing is hardcoded

**Notifications toggle:** `[notifications] enabled` in `config.toml`
(false = pipeline runs, sends suppressed, action still recorded in
audit/report). Flip + `pm2 restart` to go live.

**Runbook — "why didn't X alert?" order:**
1. `audit/compression_<day>.jsonl` — was the instance active at the
   candle close? (scan_start → detect → probe_attempt/probe_post →
   retract/breakout_confirmed/post)
2. `scan_reports/` — candidate metrics at that scan (fit error,
   convergence, boundaries). `skip_create` = the structure resolved
   before it could be watched (price already left after its last pivot) —
   nothing alertable.
3. `explain SYM TF` — reproduce the current read offline.
4. Was a probe attempted for the right candle (`probe_attempt` id)? Was
   the live price beyond the boundary at T−3:00 (`probe_post` vs no
   `probe_post`)? Failure to probe = price wasn't beyond at the probe
   moment — the close verdict handles the rest.

## Development

```bash
./venv/bin/python -m pytest tests/ -q     # full suite
```

Test layers:
- vendored daw tests (zigzag/atr/dow — must pass unchanged)
- structure math (fits, slope classes, convergence)
- detector (synthetic pivot geometries; trend≠wedge negatives)
- lifecycle (states, verdicts, invalidation, anti-respawn, idempotency)
- engine integration (synthetic sine-wave Box through the real zigzag;
  breakout + probe + retract flows end-to-end with a fake Telegram)
