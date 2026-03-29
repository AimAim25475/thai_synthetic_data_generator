from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from synthetic.schema import thai_char_ratio


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _fingerprint(ex: dict) -> str:
    # Simple content-based hash (good enough for minimal workflow)
    user = _norm(str(ex.get("user") or ""))
    assistant = _norm(str(ex.get("assistant") or ""))
    label = _norm(str(ex.get("label") or ""))
    task_type = _norm(str(ex.get("task_type") or ""))
    raw = f"{task_type}\n{label}\n{user}\n{assistant}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter Thai synthetic dataset (language gate + dedupe).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL")
    ap.add_argument("--min-thai-ratio", type=float, default=0.70, help="Minimum Thai char ratio")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    kept = 0
    dropped_lang = 0
    dropped_dup = 0
    dropped_bad = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                dropped_bad += 1
                continue

            user = str(ex.get("user") or "")
            if thai_char_ratio(user) < args.min_thai_ratio:
                dropped_lang += 1
                continue

            fp = _fingerprint(ex)
            if fp in seen:
                dropped_dup += 1
                continue
            seen.add(fp)

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "input": str(in_path),
        "output": str(out_path),
        "kept": kept,
        "dropped_lang": dropped_lang,
        "dropped_dup": dropped_dup,
        "dropped_bad": dropped_bad,
        "min_thai_ratio": args.min_thai_ratio,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
