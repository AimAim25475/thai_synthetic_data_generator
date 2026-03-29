# Thai Evaluation

This folder contains a minimal evaluation harness for the running FastAPI endpoint.

## What it measures

- Thai response rate (lightweight heuristic: response contains Thai characters)
- Split by forced mode: `chat` vs `qa`
- Average latency

## Run

1) Start the API (and Elasticsearch if QA enabled).
2) Generate + filter a JSONL file (see `synthetic/`).
3) Evaluate:

```powershell
python .\eval\eval_thai_api.py --base-url http://127.0.0.1:3001 --jsonl .\data\thai_synth\filtered.jsonl
```

This is intentionally minimal; for a stronger internship report, add:
- a labeled Thai routing test set (macro-F1)
- QA correctness checks (answer string match or human rubric)
- chat win-rate judging
