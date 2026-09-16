"""Package CLI.

    python -m compression_detection scan [--dry-run] [--symbol SYM]
    python -m compression_detection explain SYMBOL TF

The daemon entry point stays `python -m compression_detection.engine`
(used by the pm2 wrapper).
"""
from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    ap = argparse.ArgumentParser(prog="compression_detection")
    sub = ap.add_subparsers(dest="cmd")

    ps = sub.add_parser("scan", help="run one scan (respects config; --dry-run never sends or writes state)")
    ps.add_argument("--dry-run", action="store_true")
    ps.add_argument("--symbol", help="limit scan to one symbol (e.g. BTC)")

    pe = sub.add_parser("explain", help="trace one symbol/tf: pivots + candidates")
    pe.add_argument("symbol")
    pe.add_argument("tf")

    pr = sub.add_parser("rearm", help="clear compression-notify flags on active "
                                      "instances so they flush on the next enabled scan")
    pr.set_defaults(rearm=True)

    args = ap.parse_args()

    from .config import Config

    cfg = Config.load()

    if args.cmd == "explain":
        asyncio.run(_explain(cfg, args.symbol.upper(), args.tf))
        return

    if args.cmd == "rearm":
        from .engine import Engine
        from .lifecycle import ACTIVE_STATES
        eng = Engine(cfg, dry_run=False)
        n = 0
        for inst in eng.instances.values():
            if inst.state in ACTIVE_STATES and inst.notified.get("compression"):
                inst.notified["compression"] = False
                n += 1
        eng._save_state()
        print(f"re-armed {n} active instances (flush on the next enabled scan)")
        return

    if args.cmd == "scan":
        dry = bool(args.dry_run)
        from .engine import Engine
        from .report import format_summary_text
        eng = Engine(cfg, dry_run=dry)
        res = asyncio.run(eng.scan_once(symbol_filter=args.symbol))
        if "error" in res:
            print("ERROR:", res["error"])
            sys.exit(1)
        print(format_summary_text(res["summary"]))
        print("report:", res["report"])
        return

    ap.print_help()


async def _explain(cfg, symbol: str, tf: str) -> None:
    import aiohttp

    from .engine import Engine
    from .fetcher import fetch_klines
    from .notifier import pretty_type

    async with aiohttp.ClientSession() as ses:
        candles = await fetch_klines(ses, f"{symbol}/USDT", tf, cfg.candle_limit)
    if not candles:
        print(f"no data for {symbol} {tf}")
        return

    eng = Engine(cfg, dry_run=True)
    r = eng._analyze_candles(candles, tf)
    if r is None:
        print("not enough candles")
        return

    print(f"{symbol} {tf}: {len(candles)} closed candles, "
          f"last close {candles[-1].close:g}")
    print("\nlabeled pivots (last 16):")
    for p, lbl in r["labeled"][-16:]:
        print(f"  {'H' if p.is_high else 'L'} {p.price:<12g} ts={p.ts} "
              f"label={lbl.value if lbl else '-'}")
    print(f"\ncandidates: {len(r['candidates'])}")
    for c in r["candidates"]:
        m = c.metrics
        print(f"  {pretty_type(c.type):<22} pivots={len(c.refs):<3} "
              f"fit={m['fit_err_atr']:.2f}  conv={m.get('convergence_rate', 0):.2f}  "
              f"upper={c.upper_class:<7} lower={c.lower_class}")


if __name__ == "__main__":
    main()
