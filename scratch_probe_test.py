"""Scratch: decisive probe test — simulate a due moment, watch audit writes."""
import asyncio
import sys
import time as _time

sys.path.insert(0, "/root/compression-detection")

import compression_detection.engine as eng
from compression_detection.config import Config

cfg = Config.load()
e = eng.Engine(cfg)
print("instances loaded:", len(e.instances))

# simulate now = 09:27:30 UTC on 2026-09-16
import datetime as dt
fake = dt.datetime(2026, 9, 16, 9, 27, 30, tzinfo=dt.timezone.utc).timestamp()

class FakeTime:
    @staticmethod
    def time():
        return fake

eng.time = FakeTime

async def fake_price(ses, symbol):
    return 0.0  # no post — we only want to see probe_attempt audit writes

eng.fetch_last_price = fake_price

before = 0
audit_path = "audit/compression_20260916.jsonl"
try:
    before = sum(1 for _ in open(audit_path) if "probe_attempt" in _)
except FileNotFoundError:
    pass

n = asyncio.run(e.probe_once(None))

after = sum(1 for _ in open(audit_path) if "probe_attempt" in _)
print("probe_once returned:", n, "| probe_attempt delta:", after - before)

# which instances were due?
import compression_detection.engine as em
for inst in list(e.instances.values()):
    tf_ms = em.TF_MS[inst.tf]
    due = em.probe_due(inst.probe_last_ms, int(fake * 1000), tf_ms, cfg.probe_warn_s)
    if due:
        print("  due:", inst.id, inst.state)
