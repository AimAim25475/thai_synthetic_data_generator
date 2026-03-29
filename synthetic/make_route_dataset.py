from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract routing (qa/chat) training CSV from JSONL.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if (ex.get("task_type") or "").strip().lower() != "route":
                continue
            label = (ex.get("label") or "").strip()
            user = (ex.get("user") or "").strip()
            if not label or not user:
                continue
            rows.append({"question": user, "label": label})

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "label"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
