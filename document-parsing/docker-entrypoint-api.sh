#!/usr/bin/env bash
# API container entrypoint:
#   1. Wait for Postgres + run alembic upgrade head.
#   2. Start uvicorn.
set -euo pipefail

cd /app

echo "==> waiting for postgres..."
python - <<'PY'
import os, time, urllib.parse
from sqlalchemy import create_engine, text

url = os.environ.get("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
if not url:
    raise SystemExit("DATABASE_URL not set")

deadline = time.time() + 60
last_err = None
while time.time() < deadline:
    try:
        eng = create_engine(url, pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        print("postgres ready")
        break
    except Exception as e:
        last_err = e
        time.sleep(2)
else:
    raise SystemExit(f"postgres never ready: {last_err}")
PY

echo "==> running alembic upgrade head..."
alembic upgrade head

echo "==> starting uvicorn on 0.0.0.0:${PORT:-8002}"
cd /app/src
exec python main.py
