"""JSONL audit trail — one file per UTC day (daw-breakouts doctrine:
every decision lands on disk so "did every event get caught?" is
answerable later from the record, not from log greps).

Never raises: auditing must not take the engine down.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class Audit:
    def __init__(self, dir_path):
        self.dir = Path(dir_path)
        self._path: Optional[Path] = None

    def write(self, event: dict) -> None:
        try:
            day = time.strftime("%Y%m%d", time.gmtime())
            p = self.dir / f"compression_{day}.jsonl"
            if p != self._path:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._path = p
            line = json.dumps({"ts": int(time.time()), **event}, default=str) + "\n"
            with open(p, "a") as f:
                f.write(line)
        except Exception:
            pass
