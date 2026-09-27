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
cc = st['CCUSDT|15m|ascending_triangle|1789587900000']
print("CC instance: state=%s | created=%s | last_scan=%s | last_eval=%s" % (
    cc['state'], fmt(cc['created_ts']), fmt(cc.get('last_scan_ts')), fmt(cc.get('last_eval_ts'))))
print("pivots:", " ".join(f"{p.get('side')}:{p.get('price'):g}@{fmt(p.get('ts'))}" for p in cc.get('pivots', [])[-5:]))

last = None
for f in sorted(glob.glob('scan_reports/*.json')):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    syms = {d['symbol'] for d in r.get('detections', [])}
    insts = [i for i in r.get('active_instances', []) if i.get('symbol') == 'CCUSDT']
    if 'CCUSDT' in syms or insts:
        last = f.split('/')[-1]
print("last report containing CCUSDT:", last)


async def px():
    cfg = Config.load()
    async with aiohttp.ClientSession() as ses:
        cs = await fetch_klines(ses, "CC/USDT", "15m", 400)
    tf_ms = 15 * 60_000
    lo = cc['lower']

    def line_at(ts):
        return lo['slope'] * (ts // tf_ms - lo['base_bar']) + lo['intercept']

    below = [c for c in cs if c.close < line_at(c.ts)]
    print("CC bars fetched:", len(cs), "| last close:", cs[-1].close, "at", fmt(cs[-1].ts))
    if below:
        b = below[0]
        print("first close below base line:", fmt(b.ts), "close", b.close, "vs line", round(line_at(b.ts), 6))
        print("total closes below base:", len(below), "of", len(cs))
    else:
        print("no closes below base in the fetched window")


asyncio.run(px())
