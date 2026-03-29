from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def call_chat_api(base_url: str, line: str, mode: str | None, reset: bool, timeout_s: float) -> str:
    params: dict[str, str] = {"line": line}
    if mode:
        params["mode"] = mode
    if reset:
        params["reset"] = "true"

    url = f"{base_url.rstrip('/')}/chat?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def is_thai(text: str) -> bool:
    # Lightweight check: at least one Thai char
    return any("\u0E00" <= ch <= "\u0E7F" for ch in (text or ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Thai behavior by hitting running FastAPI.")
    ap.add_argument("--base-url", default="http://127.0.0.1:3001", help="API base URL")
    ap.add_argument("--jsonl", required=True, help="Filtered JSONL (route/chitchat/qa)")
    ap.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout seconds")
    ap.add_argument("--limit", type=int, default=0, help="If >0, limit number of examples")
    args = ap.parse_args()

    base_url = args.base_url
    path = Path(args.jsonl)

    n = 0
    thai_ok = 0
    by_mode = {"chat": 0, "qa": 0}
    by_mode_thai = {"chat": 0, "qa": 0}
    errors = 0
    latencies: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if args.limit and n >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            user = (ex.get("user") or "").strip()
            label = (ex.get("label") or "").strip().lower()

            mode = "chat" if label == "chat_mode" else "qa"
            reset = True  # independent calls

            t0 = time.perf_counter()
            try:
                ans = call_chat_api(base_url, user, mode=mode, reset=reset, timeout_s=args.timeout).strip()
            except Exception:
                errors += 1
                n += 1
                continue
            dt = time.perf_counter() - t0
            latencies.append(dt)

            by_mode[mode] += 1
            if is_thai(ans):
                thai_ok += 1
                by_mode_thai[mode] += 1

            n += 1

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    report = {
        "examples": n,
        "errors": errors,
        "thai_answer_rate": (thai_ok / max(1, (n - errors))),
        "by_mode": by_mode,
        "by_mode_thai": {
            "chat": (by_mode_thai["chat"] / max(1, by_mode["chat"])),
            "qa": (by_mode_thai["qa"] / max(1, by_mode["qa"])),
        },
        "avg_latency_s": avg_lat,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
