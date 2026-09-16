"""Scratch: repost the flushed compression messages with the corrected
symbol (SOLUSDT) + chart. Post-new-first; only delete the old message when
the new one lands. Run with the service STOPPED (state writer).

Usage: set -a; . ./.env; set +a; ./venv/bin/python scratch_repost.py
"""
import asyncio
import os
import sys

sys.path.insert(0, "/root/compression-detection")
os.chdir("/root/compression-detection")

import aiohttp

import compression_detection.engine as eng
from compression_detection import notifier as nt
from compression_detection.config import Config


async def main():
    cfg = Config.load()
    e = eng.Engine(cfg)  # loads migrated state
    tg = nt.Telegram(os.environ["TELEGRAM_BOT_TOKEN"],
                     os.environ["TELEGRAM_CHAT_ID"])
    reposted = 0
    async with aiohttp.ClientSession() as ses:
        for inst in list(e.instances.values()):
            if not inst.msg_ids or inst.state not in eng.ACTIVE_STATES:
                continue
            old = inst.msg_ids[-1]
            caption = nt.fmt_compression(inst)
            last = inst.pivots[-1]["bar"]
            img = await e._chart(ses, inst, [inst.lower_at(last), inst.upper_at(last)])
            mid = None
            if img is not None:
                mid = await tg.post_photo(caption, img)
            if mid is None:
                mid = await tg.post(caption)
            if mid:
                e.audit.write({"event": "repost", "id": inst.id, "old": old,
                               "new": mid, "chart": img is not None})
                ok = await tg.delete(old, why="format fix repost")
                inst.msg_ids = [mid]
                reposted += 1
                print(f"reposted {inst.id}: {old} -> {mid} (chart={img is not None}, old_deleted={ok})")
            else:
                print(f"REPOST FAILED (kept old {old}): {inst.id}")
    e._save_state()
    print(f"done: {reposted} reposted")


asyncio.run(main())
