import json, asyncio, sys, datetime, glob, aiohttp
sys.path.insert(0, '/root/compression-detection')
import os
os.chdir('/root/compression-detection')
from compression_detection.config import Config
from compression_detection.fetcher import fetch_klines

OFF = 3 * 3600 + 30 * 60


def secs(x):
    x = x or 0
    return x / 1000 if x > 1e12 else x


def fmt(x):
    return datetime.datetime.utcfromtimestamp(secs(x) + OFF).strftime("%d %H:%M")


st = json.load(open('structure_state.json'))['instances']

print("== all CCUSDT instances ==")
for k, v in sorted(st.items()):
    if v.get('symbol') == 'CCUSDT':
        print(f"  {k} | state={v['state']} created={fmt(v.get('created_ts'))} last_scan={fmt(v.get('last_scan_ts'))} last_eval={fmt(v.get('last_eval_ts'))} notified={v.get('notified')}")

conf = [(k, v) for k, v in st.items() if v.get('symbol') == 'CCUSDT' and v['state'] == 'confirmed']
if not conf:
    print("no confirmed CCUSDT instance")
    sys.exit(0)
k, v = conf[0]
print(f"\n== confirmed instance: {k}")
print(f"created={fmt(v.get('created_ts'))} last_scan={fmt(v.get('last_scan_ts'))} last_eval={fmt(v.get('last_eval_ts'))}")
print("pivots:", " ".join(f"{p.get('side')}:{p.get('price'):g}@{fmt(p.get('ts'))}" for p in v.get('pivots', [])))
print("level:", v.get('level'), "| notified:", v.get('notified'))

tf_ms = 15 * 60_000
up, lo = v['upper'], v['lower']


def up_at(ts):
    return up['slope'] * (ts // tf_ms - up['base_bar']) + up['intercept']


def lo_at(ts):
    return lo['slope'] * (ts // tf_ms - lo['base_bar']) + lo['intercept']


async def px():
    cfg = Config.load()
    async with aiohttp.ClientSession() as ses:
        cs = await fetch_klines(ses, "CC/USDT", "15m", 400)
    print("\nCC bars:", len(cs), "| first", fmt(cs[0].ts), "| last", fmt(cs[-1].ts), "close", cs[-1].close)
    below = [c for c in cs if c.close < lo_at(c.ts)]
    above = [c for c in cs if c.close > up_at(c.ts)]
    print("closes below LOWER line:", len(below), ("first: " + fmt(below[0].ts) + " close " + str(below[0].close) + " vs line " + str(round(lo_at(below[0].ts), 6))) if below else "")
    print("closes above UPPER line:", len(above), ("first: " + fmt(above[0].ts)) if above else "")
    # show around the freeze window
    print("\nlast 8 bars:")
    for c in cs[-8:]:
        print(f"  {fmt(c.ts)} O={c.open:g} H={c.high:g} L={c.low:g} C={c.close:g} | base={lo_at(c.ts):.6g} top={up_at(c.ts):.6g}")


asyncio.run(px())
