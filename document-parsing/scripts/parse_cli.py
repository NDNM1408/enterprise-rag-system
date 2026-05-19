"""CLI: upload a file → poll → save markdown locally.

Usage:
    python3 scripts/parse_cli.py /path/to/file.pdf
    python3 scripts/parse_cli.py /path/to/file.pdf --out ./output --poll 5

Requires: requests, python3.10+. The service must be running on localhost:8102
(set BASE via env var to override).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

BASE = os.getenv("PARSER_BASE_URL", "http://localhost:8102/api/v1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path, help="document to parse")
    ap.add_argument("--out", type=Path, default=Path.cwd(), help="output dir")
    ap.add_argument("--poll", type=int, default=5, help="poll interval seconds")
    ap.add_argument("--timeout", type=int, default=3600, help="abort after N s")
    args = ap.parse_args()

    if not args.file.exists():
        print(f"file not found: {args.file}", file=sys.stderr)
        return 2

    print(f"submitting {args.file.name} → {BASE}/jobs ...")
    with args.file.open("rb") as fh:
        r = requests.post(
            f"{BASE}/jobs",
            files={"file": (args.file.name, fh)},
            timeout=120,
        )
    r.raise_for_status()
    submit = r.json()
    job_id = submit["id"]
    print(f"  job_id={job_id} state={submit['state']}")

    deadline = time.time() + args.timeout
    last_progress = ""
    while time.time() < deadline:
        s = requests.get(f"{BASE}/jobs/{job_id}", timeout=10).json()
        prog = s.get("progress", {}) or {}
        cur = f"{s['state']} {prog.get('pages_done', 0)}/{prog.get('pages_total') or '?'}"
        if cur != last_progress:
            print(f"  [{time.strftime('%H:%M:%S')}] {cur}", flush=True)
            last_progress = cur
        if s["state"] == "done":
            args.out.mkdir(parents=True, exist_ok=True)
            md_path = args.out / f"{args.file.stem}.md"
            md_path.write_text(
                requests.get(f"{BASE}/jobs/{job_id}/markdown", timeout=300).text,
                encoding="utf-8",
            )
            print(f"  → {md_path} ({md_path.stat().st_size} B)")
            print(f"  parser={s['parser']} mode={s['mode']} duration={s['duration_ms']} ms")
            print(f"  images: {s['result']['image_count']} (S3 prefix: {s['result']['image_prefix']})")
            return 0
        if s["state"] == "failed":
            print(f"  FAILED: {s.get('error')}", file=sys.stderr)
            return 1
        time.sleep(args.poll)

    print("timeout", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
