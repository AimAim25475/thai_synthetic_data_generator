"""
Stage 5: Validation and Testing
================================
Quality validation for Thai synthetic data examples.

Each example is scored on **6 criteria** (0 or 1 point each, max = 6).
An example is considered high-quality when it scores >= 4 out of 6.

Criteria
--------
1. thai_user        — ``user`` field has ≥ min_thai_ratio Thai characters.
2. thai_assistant   — ``assistant`` field (when present) has ≥ min_thai_ratio
                      Thai characters.  Examples with no assistant are scored 1
                      (N/A — does not penalise route examples).
3. schema_valid     — Required fields (id, task_type, user, label) are present
                      and non-empty.
4. label_valid      — ``label`` is one of ``chat_mode`` or ``qa_mode``.
5. length_ok        — ``user`` text has at least ``min_user_chars`` non-whitespace
                      characters (default 10).
6. source_present   — ``source`` field is present and non-empty.

Usage (CLI)
-----------
::

    python synthetic/validate_quality.py --in filtered.jsonl --out report.json

Outputs a JSON report containing per-criterion pass rates, an overall pass
rate (score ≥ 4), and a list of failing example IDs for diagnostics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from synthetic.schema import thai_char_ratio

# ---------------------------------------------------------------------------
# Criteria weights (all equal — 1 point each)
# ---------------------------------------------------------------------------
_CRITERIA = [
    "thai_user",
    "thai_assistant",
    "schema_valid",
    "label_valid",
    "length_ok",
    "source_present",
]

_VALID_LABELS = {"chat_mode", "qa_mode"}
_VALID_TASK_TYPES = {"route", "chitchat", "qa"}


@dataclass
class ValidationResult:
    """Per-example validation result."""

    id: str
    score: int
    max_score: int
    passed: bool  # score >= pass_threshold
    criteria: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Aggregate report over an entire JSONL file."""

    input: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    criteria_pass_rates: dict[str, float] = field(default_factory=dict)
    failing_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_example(
    ex: dict[str, Any],
    min_thai_ratio: float = 0.70,
    min_user_chars: int = 10,
) -> ValidationResult:
    """Score a single synthetic example and return a :class:`ValidationResult`.

    Parameters
    ----------
    ex:
        Parsed JSON object representing one synthetic example.
    min_thai_ratio:
        Minimum Thai character ratio required for *user* and *assistant* fields.
    min_user_chars:
        Minimum number of non-whitespace characters required in *user*.
    """
    ex_id = str(ex.get("id") or "")
    criteria: dict[str, bool] = {}

    # --- 1. Thai user ratio ---------------------------------------------------
    user = str(ex.get("user") or "")
    criteria["thai_user"] = thai_char_ratio(user) >= min_thai_ratio

    # --- 2. Thai assistant ratio (N/A → pass for route tasks) ----------------
    assistant = ex.get("assistant")
    if assistant is None or str(assistant).strip() == "":
        # Route examples legitimately have no assistant — not penalised.
        criteria["thai_assistant"] = True
    else:
        criteria["thai_assistant"] = thai_char_ratio(str(assistant)) >= min_thai_ratio

    # --- 3. Schema validity --------------------------------------------------
    required_non_empty = ["id", "task_type", "user", "label"]
    criteria["schema_valid"] = all(
        (ex.get(k) is not None and str(ex.get(k)).strip() != "")
        for k in required_non_empty
    )

    # --- 4. Label validity ---------------------------------------------------
    label = str(ex.get("label") or "").strip().lower()
    criteria["label_valid"] = label in _VALID_LABELS

    # --- 5. User text length -------------------------------------------------
    stripped = user.replace(" ", "").replace("\t", "").replace("\n", "")
    criteria["length_ok"] = len(stripped) >= min_user_chars

    # --- 6. Source field present ---------------------------------------------
    source = str(ex.get("source") or "").strip()
    criteria["source_present"] = source != ""

    score = sum(criteria.values())
    pass_threshold = 4  # at least 4 out of 6
    return ValidationResult(
        id=ex_id,
        score=score,
        max_score=len(_CRITERIA),
        passed=score >= pass_threshold,
        criteria=criteria,
    )


def validate_jsonl(
    in_path: Path,
    min_thai_ratio: float = 0.70,
    min_user_chars: int = 10,
) -> tuple[ValidationReport, list[ValidationResult]]:
    """Validate every example in a JSONL file.

    Returns
    -------
    report:
        Aggregate :class:`ValidationReport`.
    results:
        List of per-example :class:`ValidationResult` objects.
    """
    results: list[ValidationResult] = []
    criteria_totals: dict[str, int] = {c: 0 for c in _CRITERIA}

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            res = score_example(ex, min_thai_ratio=min_thai_ratio, min_user_chars=min_user_chars)
            results.append(res)
            for c in _CRITERIA:
                if res.criteria.get(c):
                    criteria_totals[c] += 1

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    total_score = sum(r.score for r in results)

    criteria_pass_rates = {
        c: (criteria_totals[c] / total if total > 0 else 0.0)
        for c in _CRITERIA
    }
    failing_ids = [r.id for r in results if not r.passed]

    report = ValidationReport(
        input=str(in_path),
        total=total,
        passed=passed_count,
        failed=total - passed_count,
        pass_rate=passed_count / total if total > 0 else 0.0,
        avg_score=total_score / total if total > 0 else 0.0,
        criteria_pass_rates=criteria_pass_rates,
        failing_ids=failing_ids,
    )
    return report, results


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Stage 5 – Validate quality of filtered Thai synthetic data "
            "(6-point scoring system)."
        )
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL (e.g. filtered.jsonl)")
    ap.add_argument("--out", dest="out_path", default=None, help="Output JSON report (optional)")
    ap.add_argument("--min-thai-ratio", type=float, default=0.70, help="Min Thai char ratio")
    ap.add_argument("--min-user-chars", type=int, default=10, help="Min user text length")
    ap.add_argument("--verbose", action="store_true", help="Print per-example results")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    report, results = validate_jsonl(
        in_path,
        min_thai_ratio=args.min_thai_ratio,
        min_user_chars=args.min_user_chars,
    )

    if args.verbose:
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] id={r.id[:12]}  score={r.score}/{r.max_score}  {r.criteria}")

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full = {
            "report": report.to_dict(),
            "results": [r.to_dict() for r in results],
        }
        out_path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nFull report saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
