# Thai Evaluation

This folder contains the evaluation harness for the running FastAPI endpoint.

## Pipeline Stages

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `synthetic/generate_typhoon.py` | Generate synthetic examples with Typhoon 3B |
| 2 | `synthetic/filter_thai.py` | Language gate (≥70% Thai) + deduplication |
| 3 | `synthetic/make_route_dataset.py` | Extract routing CSV for classifier training |
| 4 | `eval/eval_thai_api.py` | Hit live API, measure Thai answer rate & latency |
| **5** | **`synthetic/validate_quality.py`** | **6-point quality scoring on filtered JSONL** |

---

## Stage 4 — API Evaluation (`eval_thai_api.py`)

### What it measures

- Thai response rate (lightweight heuristic: response contains Thai characters)
- Split by forced mode: `chat` vs `qa`
- Average latency

### Run

1) Start the API (and Elasticsearch if QA enabled).
2) Generate + filter a JSONL file (see `synthetic/`).
3) Evaluate:

```bash
python eval/eval_thai_api.py --base-url http://127.0.0.1:3001 --jsonl filtered.jsonl
```

---

## Stage 5 — Validation and Testing (`validate_quality.py`)

### Overview

`synthetic/validate_quality.py` implements an **offline** 6-point quality scoring
system that validates every example in a filtered JSONL file **without** needing a
running API server.  It is the primary validation stage before training.

### Scoring criteria (1 point each, max = 6)

| # | Criterion | Rule |
|---|-----------|------|
| 1 | `thai_user` | `user` field has ≥ 70% Thai characters |
| 2 | `thai_assistant` | `assistant` field (when present) has ≥ 70% Thai characters; route examples (no assistant) score 1 automatically |
| 3 | `schema_valid` | Required fields `id`, `task_type`, `user`, `label` are present and non-empty |
| 4 | `label_valid` | `label` is `chat_mode` or `qa_mode` |
| 5 | `length_ok` | `user` text has ≥ 10 non-whitespace characters |
| 6 | `source_present` | `source` field is present and non-empty |

An example **passes** when `score ≥ 4`.

### Run

```bash
python synthetic/validate_quality.py --in filtered.jsonl --out report.json
```

Add `--verbose` to print a per-example breakdown:

```bash
python synthetic/validate_quality.py --in filtered.jsonl --verbose
```

### Example output

```json
{
  "input": "filtered.jsonl",
  "total": 120,
  "passed": 116,
  "failed": 4,
  "pass_rate": 0.9667,
  "avg_score": 5.85,
  "criteria_pass_rates": {
    "thai_user": 0.983,
    "thai_assistant": 0.975,
    "schema_valid": 1.0,
    "label_valid": 1.0,
    "length_ok": 0.992,
    "source_present": 1.0
  },
  "failing_ids": ["id-of-bad-example", "..."]
}
```

### Quality thresholds (targets)

| Metric | Target |
|--------|--------|
| `pass_rate` | ≥ 0.90 |
| `thai_user` rate | ≥ 0.96 |
| `thai_assistant` rate | ≥ 0.90 |
| `avg_score` | ≥ 5.0 / 6.0 |

---

## Unit Tests

Run the full test suite (Stage 5 includes 96 unit tests):

```bash
python -m pytest tests/ -v
```

Tests cover:
- `tests/test_schema.py` — `is_thai_char`, `thai_char_ratio`, `SynthExample`
- `tests/test_filter_thai.py` — normalisation, fingerprinting, filter pipeline
- `tests/test_generate_typhoon.py` — device/dtype selection, JSON extraction, field validation, Thai gate, prompt builder
- `tests/test_validate_quality.py` — 6-point scorer and aggregate report
