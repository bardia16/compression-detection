# compression-detection

DAW-theory **compression structure detector** — finds **Box /
Descending / Ascending Triangle** formations on confirmed ATR-ZigZag
pivots, tracks their lifecycle, and posts
compression + breakout alerts to the **Breakouts** Telegram channel
(same bot as Trading-Alerts). Symmetrical triangles and wedges are
disabled (user rules 2026-09-20 / 2026-09-16).

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
- **Breakout level (user rules 2026-09-16):** the price level shown and
  checked is the **last confirmed pivot in that direction** for **box and
  triangles** (e.g. the last EH for a box's long side, the last LH for a
  descending triangle's up-break). For **wedges the boundary LINE is the
  level** (close beyond the line = the breakout; level shown = line value
  at that bar). Fitted boundary lines remain for geometry
  (slopes/convergence) and chart drawing.
- **Triangle line-death (user rules 2026-09-16):** if a close goes through
  a triangle's boundary **line** WITHOUT breaking the pivot level, the
  pattern no longer holds — from that close it is a **range doing EH/EL**,
  silently invalidated (pending probe retracted). Never confused with a
  breakout (level breaks are classified first). A triangle whose line is
  already broken at detection time is never created.
- **Threading (user rule 2026-09-16):** breakout messages (heads-up and
  final) are posted as a **reply to the structure's compression
  confirmation message** — compression + its breakout read as one thread
  in the channel. If no compression message was ever sent for that
  structure, the breakout posts standalone.
- **DM mirror (user rule 2026-09-20):** ascending/descending triangle
  alerts are ALSO sent to Bardia's DM (`TELEGRAM_DM_CHAT_ID` in `.env`):
  compression message, breakout heads-ups (reply-threaded onto the DM
  compression message) and retracts. The channel is unchanged and boxes
  are untouched (channel only). Empty `TELEGRAM_DM_CHAT_ID` = off.
- **Charts (user rules 2026-09-16):** **triangles/wedges** draw the two
  extrapolated boundary lines, each starting at its own first pivot (no
  left overhang); the FLAT side of a triangle (desc base / asc top) is
  drawn strictly HORIZONTAL at that side's last pivot; **boxes** draw
  plain horizontals at their last pivot levels (same values as their
  breakout levels). No extrapolation for ranges. Equality of EH/EL is
  ATR-based across the pivot pair (the label logic).
- **Range-mode box (2026-09-17, JTO case, flag `box_range_mode`):** a
  consolidation range is defined by touch clusters (≥2 high taps within
  flat_tol_atr × ATR of the top, ≥2 lows near the bottom, the two sides'
  touch spans overlapping in time, closes inside the envelope) — internal
  swings between the taps are allowed. The strict model (every non-first
  pivot EH/EL) rejected JTO 4H 0.40/0.466 despite 3 taps per side.
- **Type priority (user rule 2026-09-19):** when more than one compression
  is confirmed for the same coin+tf, only the highest-priority type
  surfaces — triangles first (**ascending/descending**), **box** last
  (config `selection_order`). A
  lower-priority compression message is HELD while a notified
  higher-priority sibling is still active; it can surface later only if
  that sibling ends first. Same-scan conflicts resolve in
  `best_per_coin_tf` by the same order.
- **Detection scope (user rule 2026-09-20):** boxes + ascending/descending
  triangles ONLY. Symmetrical triangles disabled alongside the wedges
  (commented out of `TYPE_SPECS`/`ALL_TYPES`; re-enable only on user
  request).
- **Triangle pattern (user rule 2026-09-23, supersedes the 2026-09-20
  anchor rule):** LONG (ascending) = `[low, high, HL, EH, HL]` — a low, a
  high (first flat-top tap), a higher low, an equal high (second tap), a
  higher low; **the confirmation alert fires when that final higher low
  confirms** (min 5 pivots). The first low anchors the lower boundary
  LINE — it runs from it through the HLs, not just between two HLs. The
  first low and first high are label-exempt; every later low must be HL.
  SHORT (descending) mirrors: `[high, low, LH, EL, LH]` — the line starts
  from the first high. Boxes keep both first-pivot exemptions.
- **Interior close integrity (user rule 2026-09-17):** candle closes
  between the window's first and last pivot must stay within the
  boundaries. A close beyond a boundary by more than
  `close_breach_tol_atr` (0.5) × ATR counts as a breach; more than
  `max_close_breaches` (2) breaches on any side rejects the candidate
  (BTW case: 37 closes below the fitted lower line).
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
