"""One-time cleanup (user rule 2026-09-23): retire the 6 out-of-universe
zombie instances — delete their pending breakout heads-up messages (channel
+ DM twin), clear probe bookkeeping, mark invalidated (reason
'out_of_universe'). Run with the app STOPPED (state races)."""
import asyncio
import json
import os
import sys
import time

REPO = '/root/compression-detection'
sys.path.insert(0, REPO)
os.chdir(REPO)


def load_env(path='.env'):
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

from compression_detection import notifier as nt

ZOMBIES = [
    'CCUSDT|15m|ascending_triangle|1790037000000',
    'DODOUSDT|1h|box|1789704000000',
    'ENSOUSDT|1h|box|1789948800000',
    'ENSOUSDT|15m|box|1790003700000',
    'FORMUSDT|1h|box|1790049600000',
    'LABUSDT|4h|box|1788580800000',
]


async def main():
    token = os.environ['TELEGRAM_BOT_TOKEN']
    chat = os.environ['TELEGRAM_CHAT_ID']
    dm = os.environ.get('TELEGRAM_DM_CHAT_ID') or None
    print(f"channel={chat} dm={dm}")

    st = json.load(open('structure_state.json'))
    instances = st['instances']
    tg = nt.Telegram(token, chat)
    now = int(time.time())
    for key in ZOMBIES:
        v = instances.get(key)
        if v is None:
            print(f"MISSING: {key}")
            continue
        mid = v.get('probe_msg_id')
        dmid = v.get('probe_msg_id_dm')
        if mid:
            ok = await tg.delete(mid, "coin out of universe - structure retired")
            print(f"{v['symbol']} {v['tf']}: heads-up msg {mid} deleted: {ok}")
        if dmid and dm:
            ok = await tg.delete(dmid, "coin out of universe - structure retired", chat_id=dm)
            print(f"{v['symbol']} {v['tf']}: DM heads-up msg {dmid} deleted: {ok}")
        old_state = v['state']
        v['probe_msg_id'] = None
        v['probe_candle_ms'] = 0
        v['probe_side'] = ''
        v['probe_msg_id_dm'] = None
        v['state'] = 'invalidated'
        v.setdefault('events', []).append(
            {'ts': now, 'kind': 'invalidated', 'reason': 'out_of_universe'})
        print(f"{v['symbol']} {v['tf']} {v['type']}: {old_state} -> invalidated")

    tmp = 'structure_state.json.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, 'structure_state.json')
    print("state saved")

asyncio.run(main())
