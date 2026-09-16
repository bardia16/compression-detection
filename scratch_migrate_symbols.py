"""Scratch: migrate state instances from bare symbols (SOL) to full Binance
symbols (SOLUSDT) — one-off after the 2026-09-16 symbol fix. Run with the
service STOPPED."""
import json

p = "/root/compression-detection/structure_state.json"
d = json.load(open(p))
n = 0
for key in list(d["instances"].keys()):
    inst = d["instances"][key]
    if not inst["symbol"].endswith("USDT"):
        inst["symbol"] = inst["symbol"] + "USDT"
        nid = f"{inst['symbol']}|{inst['tf']}|{inst['type']}|{inst['anchor_ts']}"
        inst["id"] = nid
        del d["instances"][key]
        d["instances"][nid] = inst
        n += 1
json.dump(d, open(p, "w"), indent=1)
print(f"migrated {n} instances to full symbols; total {len(d['instances'])}")
