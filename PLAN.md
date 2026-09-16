# compression-detection — Detailed Implementation Plan

> For the builder (me). High-level plan was approved-shape sent to Bardia; build starts only after his approval.

**Goal:** Standalone service that detects compression structures (Box, Descending/Ascending/Symmetrical Triangle, Falling/Rising Wedge) from DAW-theory ATR-ZigZag pivots, tracks their lifecycle, and posts compression + breakout notifications to the Breakouts Telegram channel (same bot as Trading-Alerts).

**Reference spec:** `DAW_Compression_Structure_Detection_—_Implementation_Specification.md` (user's md, cached at `/root/.hermes/cache/documents/doc_f21ed079af8c_*.md`).
**Project path:** `/root/compression-detection/` (new, standalone, own venv + git repo).
**Status:** PLAN — awaiting Bardia's approval on high-level plan.

---

## 1. Architecture

```
coin universe (LCW all coins ≥100 BTC 24h vol — same fetch as daw/watchlist, fully independent of the daw project)
   → fetch closed candles (Binance REST klines; spot → futures fallback; forming candle dropped)
   → ZigZag (coef × ATR7; close-based pivots; vendored from daw-theory, option-A ATR)
   → LabeledPivot list (HH/LH/EH · HL/LL/EL vs prev same-type, 1×ATR14 close_only; vendored dow.py +
     NEW label_all_pivots() that keeps unlabeled pivots with label=None)
   → StructureDetector: per-type candidate matchers (boundary fit, slope class, convergence, tolerances)
   → Lifecycle: instance registry (state file), state NONE→DETECTED→CONFIRMED→COMPRESSING→BREAKOUT/INVALIDATED
   → Notifier: compression + breakout messages (bot 8517752196 → channel -1004330149150) + chart (:8002)
   → Audit JSONL + scan reports
```

Daemon loop wakes every 15m grid + 90s (all of 15m/1h/4h close on the 15m grid), scans universe, evaluates.
pm2 service `compression-detection` via wrapper script.

### 1.1 Vendored from daw-theory (copy, no import coupling)
| File | What | Change |
|---|---|---|
| `models.py` | Candle, Pivot | none |
| `zigzag.py` | stateful ATR ZigZag, close-only reversal, true-range threshold ATR (option A), coef default 1.2 | default coef from local config; `shadow_reversal` kept but unused |
| `atr.py` | atr_close_only / atr_true_range / atr_series | none |
| `dow.py` | label_high/low, label_pivots | ADD `label_all_pivots(pivots, atr14_series) -> [(pivot, label|None), ...]` keeping every pivot (label None when undecidable) — detection needs the full sequence incl. first-of-type pivots |
| tests: `test_zigzag.py`, `test_atr.py`, `test_dow.py` | must pass unchanged in new repo |

Fetch pattern (rewrite from breakouts.py style): aiohttp REST `api.binance.com/api/v3/klines` → fallback `fapi.binance.com/fapi/v1/klines`, `limit` per TF, drop forming candle (`ts + TF_MS > now`).

### 1.2 Config defaults — INITIAL, verify in dry-run review with Bardia
```toml
[project]
timeframes = ["15m", "1h", "4h"]
candle_limit = 400             # per TF (15m≈4d, 1h≈16d, 4h≈66d)

[zigzag]
coef = 1.2                     # mirrors daw-theory live (2026-09-15)
atr_length = 7

[labels]
atr14_length = 14
atr14_method = "close_only"

[detection]
min_pivots  = { box = 4, descending_triangle = 4, ascending_triangle = 4, symmetrical_triangle = 4, falling_wedge = 5, rising_wedge = 5 }
max_window_pivots = 12         # cap on pivots used per candidate window
max_pivot_age_bars = 96        # pivots older than this (per own TF) are ignored for windows
flat_tol_atr = 1.0             # |Δ boundary across side span| ≤ this → flat; beyond → rising/falling
boundary_tol_atr = 1.0         # pivot must sit within this of its boundary line (fit respect / touch)
min_convergence_pct = 25.0     # % width shrink required (first→last) for converging types
max_width_bounce_frac = 0.10   # per-step non-monotonic slack in width samples
confirm_extra_pivots = 1       # extra pivots over min → CONFIRMED
established_extra_pivots = 2   # extra pivots over min → ESTABLISHED (→ COMPRESSING state)
selection_order = ["box", "descending_triangle", "ascending_triangle", "symmetrical_triangle", "falling_wedge", "rising_wedge"]

[scan]
offset_s = 90                  # run 90s after each 15m boundary
catchup_max_candles = 8        # per instance: cap catch-up breakout checks after downtime

[universe]
mode = "lcw_all"               # ALL LCW coins ≥ min_volume_btc — independent of daw watchlist
min_volume_btc = 100
lcw_limit = 300

[breakout]
probe_warn_s = 180             # probe at T−3:00 before the structure TF's candle close
probe_loop_s = 30              # probe check cadence
breakout_buffer_atr = 0.0      # live/close strictly beyond boundary

[notifications]
enabled = false                # flip to true after Bardia approves notification formats (mock step)
notify_min_state = "confirmed" # compression message when instance reaches this state
notify_on_compress_upgrade = false
notify_invalidation = false
best_only_per_coin_tf = true
bot_token_env = "TELEGRAM_BOT_TOKEN"
chat_id_env = "TELEGRAM_CHAT_ID"
chart_url = "http://localhost:8002"
```

---

## 2. Detection math (precise)

Inputs per (symbol, tf): closed candles C[0..n-1]; ATR14 series (close_only); confirmed pivots (zigzag); labeled list.
Notation: pivot has bar index b, price p, ts; side split: HIGH pivots → upper boundary, LOW pivots → lower boundary (spec §13).

**Helpers**
- `fit_line(pivots) -> (slope, intercept)` — OLS over (bar, price). Requires ≥2 pivots.
- `line(bar) = slope*bar + intercept`.
- `atr_ref` = ATR14 at the window's last pivot bar (fallback: latest non-None ATR14; None → skip detection this scan).
- `slope_class(line, first_bar, last_bar)`: `Δ = line(last_bar) - line(first_bar)`; flat if `|Δ| ≤ flat_tol_atr × atr_ref`; rising if `Δ > +tol`; falling if `Δ < −tol`.
- `fit_err = mean(|p_i − line(b_i)|) / atr_ref` per side; a pivot "respects" its boundary iff `|p_i − line(b_i)| ≤ boundary_tol_atr × atr_ref` (each respect = one boundary touch).
- `equality_err(side) = (max(p_i) − min(p_i)) / atr_ref` per side (metric; report only).
- `width(b) = upper.line(b) − lower.line(b)`; **convergence**: sample width at every pivot bar in window (≥3 samples needed); require: `shrink = (w_first − w_last)/w_first ≥ min_convergence_pct/100` AND for consecutive samples `w_{i+1} ≤ w_i × (1 + max_width_bounce_frac)` AND `w_last > 0`; expose `convergence_rate = shrink`.
- Ordering guard (all types): `upper.line(b) > lower.line(b)` for every bar in [first_pivot_bar, last_closed_bar].

**Windows**: candidates are built from the trailing (most recent) pivots, end-anchored at the latest pivot, sizes from `min_pivots..max_window_pivots`, ignoring pivots older than `max_pivot_age_bars`. For each type, all valid windows are candidates; rank = (pivot_count desc, fit_err_total asc, selection_order index asc). Best per (symbol, tf, type) feeds the instance; full candidate list is exposed in scan output (spec §18).

**Label gate** (`initial_exempt`): for each side, all pivots EXCEPT the first of that side must carry the canonical label; the first of each side is label-exempt (spec §5: initial reference pivots don't need literal labels when history already establishes the boundary).

| Type | Upper (highs, non-first) | Lower (lows, non-first) | Geometry | Extra |
|---|---|---|---|---|
| Box | EH | EL | upper flat, lower flat | — |
| Descending triangle | LH (≥1) | EL | upper falling, lower flat | — |
| Ascending triangle | EH | HL | upper flat, lower rising | — |
| Symmetrical triangle | LH | HL | upper falling, lower rising | convergence |
| Falling wedge | LH | LL | upper falling, lower falling | convergence |
| Rising wedge | HH | HL | upper rising, lower rising | convergence |

Negatives the tests must pin: HH→HL trend ≠ rising wedge (no convergence); LH→LL trend ≠ falling wedge; parallel falling channel (no convergence) ≠ falling wedge; boxes fail when a side's slope exceeds flat tol; equality tolerance is inclusive (no exact equality).

---

## 3. Lifecycle & state

**States**: `NONE → DETECTED → CONFIRMED → COMPRESSING → BREAKOUT | INVALIDATED`.
**Levels** (maturity, spec §27): `detected` = min pivots; `confirmed` = +`confirm_extra_pivots`; `established` = +`established_extra_pivots`.
Mapping: state = DETECTED if level detected; CONFIRMED if level confirmed (or established-but-converging-check-failing? no —) ; COMPRESSING if level established AND structure still passes all gates (for converging types convergence must still hold). Store both `state` and `level`.

**Instance registry** — `structure_state.json`:
```json
{ "instances": { "SYM|tf|type|anchor_first_pivot_ts": {
    "id": ..., "symbol": ..., "tf": ..., "type": ..., "state": ..., "level": ...,
    "created_ts": ..., "last_eval_ts": ..., "last_scan_ts": ...,
    "pivots": [{"ts","bar","price","side","label"}...],
    "boundaries": {"upper": {"slope","intercept"}, "lower": {"slope","intercept"}, "fit_atr": ...},
    "metrics": {"pivot_count","touches","fit_err_atr","equality_err_atr","convergence_rate",
                 "duration_s","dist_to_upper_atr","dist_to_lower_atr","width_reduction_pct"},
    "events": [{"ts","kind":"detected|confirmed|compressing|breakout|invalidated","detail":...}],
    "notified": {"compression": false, "breakout": false}, "msg_ids": [] } } }
```

**Per scan per (symbol, tf) order of operations** (critical — breakout BEFORE update):
1. Recompute pivots/labels/candidates from fresh candles.
2. For each active instance (same symbol/tf):
   a. **Candle-close verdict + breakout FIRST**: for each candle closed after `last_eval_ts` (cap `catchup_max_candles`; mark `late: true` if older than 2 candle periods):
      - probe was posted for this candle: confirmed close beyond boundary → KEEP message, BREAKOUT (terminal); failed close → DELETE heads-up message, clear probe, stay active (user rule: retract only when it didn't close through).
      - no probe posted (missed/downtime) but close beyond → post breakout message at close.
      - boundary used = stored from previous scan (what was known in real time).
   b. **Continuation match**: find candidate of same type sharing ≥2 pivot tss with the instance. If matched → update pivots/boundaries/metrics, recompute level & state, run convergence check for converging types (fail → INVALIDATED), notify on state upgrade if enabled. If no candidate matches → **INVALIDATED** (geometry ceased; no breakout occurred) — terminal.
3. Unmatched candidates (new structures): create instances; notify if level/state ≥ `notify_min_state`.
4. Per (symbol, tf): if multiple types would notify this scan and `best_only_per_coin_tf` → send only the top-ranked one.
5. Persist state (atomic write; multi-writer not expected but keep atomic-replace).

**Key correctness rules**
- Only CONFIRMED pivots drive structure (never `zigzag.provisional`).
- **Zombie guard (2026-09-16):** a candidate whose post-last-pivot closes already breach a boundary is never created — `skip_create` event (parity with the live breakout predicate: if a candle would have been an alert while watching, the structure is not createable).
- **Consumed pivots (PENDLE class):** a terminal instance of the same type sharing ≥2 pivots blocks re-creation of its own sub-windows/post-pullback clones — broken once = watched once.
- Wick vs close: pivots are close-based by construction; breakout = candle CLOSE beyond boundary; wick-only never breaks (test pinned).
- Recompute-from-scratch each scan (deterministic; no incremental drift) + idempotency: same candles scanned twice → zero new events (test pinned).
- A structure is never locked: every new pivot reruns matching; classification can dissolve (→ INVALIDATED) as spec §18/§29.

---

## 4. Notifications (formats require Bardia's mock approval BEFORE coding)

Events: (1) compression reached notify threshold; (2) breakout — two-stage (approved 2026-09-16): **probe at T−3:00 before the structure TF's candle close** → heads-up message if live price is beyond the boundary; at scan the close verdict runs — confirmed close beyond → message STAYS (breakout record, instance → BREAKOUT); failed close → message DELETED, instance stays active. Missed probe but closed beyond → post at close. Direction from the actual price event, never assumed. One probe per instance per candle.
Post 1 per instance per event; `notified` flags + `msg_ids` in state; survives restart.

- Same bot token 8517752196 ("Trading alerts bot"), chat `-1004330149150` (Breakouts channel) — values live in project `.env` (gitignored) exported by the pm2 wrapper (`run_compression.sh`: `set -a; . .env; set +a; exec venv/bin/python -u -m compression_detection.engine`).
- Chart: `GET {chart_url}/chart/alert?symbol=&timeframe=&cross_lines=u,l` — horizontal dashed lines are all :8002 supports; compression messages use boundary values at the latest closed bar; breakout messages (post-A1) draw the last-pivot level line. NOTE: sloped-line support in chart-generation = optional phase-2.
- **Mock step (mandatory, his standing rule):** send 2–3 candidate formats for compression + 2–3 for breakout as inline text blocks; implement the approved EXACT strings; add an equality test pinning the format.
- Draft starting points (for the mock, not final):
  - Compression: `🔷 <b>BTCUSDT</b> — 4H Compression · Falling Wedge` + `🎯 64.1 → 66.8 contracting`
  - Breakout: `🟢 <b>BTCUSDT</b> — 4H Compression · Breakout` + `🎯 closing <b>above</b> 66.8`
- Audit JSONL `audit/compression_YYYYMMDD.jsonl`: `scan`, `detect`, `state_change`, `breakout`, `invalidate`, `post`, `post_failed`, `skip` events (mirror breakouts.py audit doctrine: answers "did every event get caught?" later).
- Scan reports `scan_reports/YYYYMMDD_HHMM.json`: full per-coin/tf candidate dump + metrics (feeds §30 Q&A + dry-run review).

---

## 5. Engine & ops

- Two loops in one process: (1) SCAN loop — wakes at every 15m grid boundary + `offset_s` (15m/1h/4h candles all close on the 15m grid), fetches closed candles, updates state, runs candle-close verdicts; (2) PROBE loop — every `probe_loop_s`, for each active instance whose current candle closes within `probe_warn_s` (T−3:00, once per candle): fetch live price; if beyond its boundary → post heads-up + chart. Immediate scan at startup (catch-up). Per-coin errors logged, never crash; coin skipped for that TF this scan.
- Fetch: asyncio + aiohttp, semaphore ~8 concurrent, per-coin 3 TF fetches; spot→futures fallback; forming candle dropped.
- CLI:
  - `python -m compression_detection scan --dry-run` → scan + report, NO sends, NO state writes (state diff shown)
  - `python -m compression_detection scan` → full scan (respects config `notifications.enabled`)
  - `python -m compression_detection explain SYMBOL TF` → print labeled pivots + candidates + instance trace (debugging; one-shot)
- venv `/root/compression-detection/venv`; deps: `aiohttp`, `requests`, (universe: `ccxt` only if availability filter reused — decision: reuse `coins_source.py` logic incl. ccxt availability filter, copy file), `pytest`.
- pm2: `pm2 start run_compression.sh --name compression-detection` + `pm2 save`. Wrapper sources `.env`.
- Logs: pm2 logs + `logs/` if needed; audit separate.

**Universe (lcw_all):** ALL LCW coins with 24h vol ≥ `min_volume_btc` — same fetch pattern as daw/watchlist (POST coins/list sorted by volume, limit 300, drop stablecoins/asset-backed/BTC), Binance availability filter (spot + futures exchangeInfo REST — no ccxt dependency). Completely independent of the daw watchlist project. On LCW failure → log error + skip scan (no silent empty universe).

---

## 6. Test plan (TDD throughout)

1. Vendored tests pass: `test_zigzag.py`, `test_atr.py`, `test_dow.py` (+ new `label_all_pivots` cases).
2. `test_structure_math.py`: fit_line, slope_class (flat/rising/falling incl. exact-threshold edges), convergence (shrink pass/fail, bounce slack, crossing guard), touches/fit_err, equality_err.
3. `test_detector.py` — synthetic candle builders (`build_box`, `build_triangle`, `build_wedge`, `build_channel`, `build_trend`) where price paths are engineered so zigzag confirms the wanted pivot sequence:
   - each of 6 types detected at min pivots + after extra pivots (level upgrades);
   - negatives: HH/HL trend, LH/LL trend, parallel falling channel — NOT wedges; flat-tol edges; label-gate first-pivot exemption; equality inclusive tolerance.
4. `test_lifecycle.py`: instance matching (continuation vs new instance), state transitions by extra pivots, convergence loss → INVALIDATED, candidate vanish → INVALIDATED, breakout precedence (close beyond boundary in same scan as new pivot), wick-only poke → no breakout, close-based breakout both directions, idempotent double-scan, restart-safe state reload, catch-up cap.
5. `test_universe.py`: daw report filtering (S/A/B + direction), missing report → skip, lcw mode filter.
6. `test_notifier.py`: format equality tests (after approval), chart URL building, send-failure tolerated (fallback text-only; audit `post_failed`).
7. `test_engine_timing.py`: next-boundary computation, offset handling, skip-when-running-long.
8. Golden fixtures (best-effort): 2–3 real historical examples per structure fetched and pinned as JSON candle fixtures.

---

## 7. Build order (bite-sized, each with tests + commit)

- **P0 scaffold** — dir tree, git init, venv, pyproject, config.toml + loader, copy vendored modules + tests, run vendored tests.
- **P1 structure math** — helpers per §2 (TDD).
- **P2 detector** — per-type predicates + ranking (TDD, synthetic builders).
- **P3 lifecycle+state** — instances, transitions, persistence (TDD).
- **P4 breakout/invalidation** — close-based verdicts, T−3:00 probe + post/retract lifecycle, catch-up (TDD).
- **P5 universe+fetcher+engine** — LCW universe (lcw_all), CLI scan/explain, scan + probe loop timing, dry-run report (TDD).
- **P6 notifier** — mocks approved FIRST, then implement + equality tests + chart + audit.
- **P7 DRY-RUN REVIEW GATE with Bardia** — scan live universe, send him summary (counts per type/tf + examples + metric tables) → tune defaults within his approval → only then enable sends.
- **P8 ops** — pm2 wrapper, README/runbook, end-to-end verification post (he watches the channel), long-run check.
- **P9 (optional, ask first)** — sloped-line charts in chart-generation; T-3min pre-close probe; upgrade notifications; `/add`-style manual coins.

## 8. DECISIONS — locked by Bardia 2026-09-16 (approved)
- **Universe: ALL LCW coins** — completely separate from daw watchlist (reuses only the LCW fetch pattern). No S/A/B gating.
- **Structure confirmation: closed candles only.**
- **Breakout notification: probe at T−3:00 before the structure TF's candle close; if price is breaking → send.** Close verdict: confirmed → keep; failed → retract + stay active.
- TFs 15m/1h/4h; notify on CONFIRMED compression + breakout; invalidation silent; formats via mocks before enabling sends.

## 9. Acceptance checklist (from spec §30, §29)
Every candidate answers: type; establishing pivots; count; detected/confirmed/established; upper/lower boundaries; flat/rising/falling; converging?; still valid?; break event definition; breakout occurred/direction; interactions count — all as structured JSON. Prohibitions respected: no label-only classification, no trend→wedge confusion, no exact equality, no wick breakouts, no unconfirmed pivots, no assumed direction, no lock-in, no invalidation/breakout confusion, no hardcoded tolerances (all config), no visual judgment.

## 10. Amendments after approval (2026-09-16)

**A1 — Breakout level = last pivot in that direction (Bardia, live fix).**
Probe + close checks compare against the **newest confirmed pivot price on the broken side** (up → last high-side pivot, e.g. last EH/LH/HH; down → last low-side pivot). Fitted boundary lines remain for geometry (slopes/convergence/fit_err) and the compression message's range display ONLY. PENGU case that triggered this: line-at-bar 0.00683984 vs last LH 0.006859 (user: "a bit higher... from the last pivot in that direction"). Regression test: `test_level_not_extrapolated_boundary_line`; guard parity: `_already_broken_side` uses the same pivot levels. Action detail key renamed `boundary` → `level` (old events in state keep the historical key).

**A2 — Global boundary-hit requirement (spec: Add Minimum Confirmed Boundary-Hit Requirement).**
`[detection] min_boundary_hits = 2`. Hits = distinct **confirmed** pivots carrying that boundary's canonical label (label sets from TYPE_SPECS; live pivots can never appear — windows are built from confirmed pivots only). `state_for_candidate()`: below requirement → stays DETECTED even at confirmed/established levels; exposes `upper_hits`, `lower_hits`, `hit_requirement`, `hit_boundary` in candidate/instance metrics (spec §17 debug exposure). Since the label gate + alternation already imply ≥2 canonical pivots on one side at confirm-open counts, this is a formal, explicit gate + counters (bites if label gate loosens or counts change). Tests: `test_state_for_candidate_hit_gate`, `test_hit_requirement_exposed_on_instance`, detector hits assertions.

**A3 — Triangle line-death + wedge line-levels + chart semantics (Bardia, 2026-09-16, after BOME/IOST/ASTR).**
Three family rules, locked:
- **Triangles (desc/asc/sym):** alert level = last pivot in that direction (stands); additionally, a close through a boundary **LINE** without the level break = the pattern no longer holds — from that close it is a **range doing EH/EL**, NOT a triangle. Implemented: `classify_close()` line-death in `evaluate_closed_candles` + stateless `_post_pivot_line_death()` backfill in `update_for_scan` (retracts pending probe, silent INVALIDATED, event `line_break`); creation guard rejects candidates whose post-pivot closes already crossed a line (`_already_broken_side`, BOME/IOST class). PENGU flow unchanged: close beyond level (=breakout) is classified first, line-death only when no level hit.
- **Wedges:** the boundary LINE is the alert level — close beyond it = the breakout; displayed level = line value at the break bar (probe too). Guard parity: wedges vs lines.
- **Charts:** extrapolation ONLY for triangles/wedges — both boundary lines drawn from each side's FIRST pivot to the right edge (no left overhang); **boxes draw plain horizontals at their last pivot levels** (same values as their breakout levels — no extrapolation for ranges). Box compression text also uses last pivot levels. Equality/ATR: unchanged — the EH/EL ATR-based label logic is the single source of truth.
Tests: `test_triangle_line_break_kills_pattern_before_level`, `test_wedge_breakout_level_is_the_line`, `test_probe_level_wedge_uses_line`, `test_stale_triangle_line_break_dies_on_next_scan`, `test_triangle_line_already_broken_skips_creation`, `test_trend_segments_start_at_each_sides_first_pivot`, `test_fmt_compression_box_uses_last_pivot_levels`. 173 green.
