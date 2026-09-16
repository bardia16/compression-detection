#!/bin/bash
# compression-detection — pm2 entry point.
# pm2 strips inline env vars, so the .env file is the single source.
cd /root/compression-detection
set -a; . ./.env; set +a
exec ./venv/bin/python -u -m compression_detection.engine
