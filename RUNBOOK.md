# RUNBOOK — CHITCHAT Thai Synthetic Data (Both: Chat + QA)

This runbook preserves the exact steps to reproduce the Topic 3 workflow after a PC reset.

Goal:
- Create Thai synthetic data using a **<= 7B** model (`typhoon-ai/llama3.2-typhoon2-3b`)
- Adapt the system (routing classifier + prompts/data)
- Verify it works in Thai (chat + QA)

---

## Prerequisites

- Windows 10/11
- Python 3.11+
- Docker Desktop
- (Recommended) NVIDIA GPU + NVIDIA Container Toolkit (you confirmed `docker run --gpus all ... nvidia-smi` works)

Repo location assumed: `D:\chitchat\chitchat_api`

Windows note (important):
- Prefer running scripts via `D:\chitchat\chitchat_api\.venv\Scripts\python.exe ...` to avoid accidentally using a different Python (PATH / WindowsApps alias).

---

## 1) Setup Python environment (local)

```powershell
cd D:\chitchat\chitchat_api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-qa.txt
python -m pip install sentence-transformers
```

Notes:
- `requirements-qa.txt` installs Haystack integration and pins `transformers`.
- `sentence-transformers` is required by Haystack inference in this environment.

---

## 2) Start Elasticsearch (QA dependency)

From `D:\chitchat\chitchat_api`:

```powershell
docker compose up -d elasticsearch
```

Health check:

```powershell
curl.exe http://127.0.0.1:9200
```

You should get JSON.

---

## 3) Start the API (local)

```powershell
cd D:\chitchat\chitchat_api
python -m uvicorn bot_api:app --host 127.0.0.1 --port 3001
```

Expected log:
- `preparing qa model ok`

If you see `QA disabled ...` then Haystack dependencies are missing.

---

## 4) Generate Thai synthetic data (<= 7B teacher)

Open a **new terminal** (keep uvicorn running) and run:

```powershell
cd D:\chitchat\chitchat_api
$env:TEACHER_MODEL_NAME = "typhoon-ai/llama3.2-typhoon2-3b"
.\.venv\Scripts\python.exe .\synthetic\generate_typhoon.py --model typhoon-ai/llama3.2-typhoon2-3b --device cuda --dtype float16 --out .\data\thai_synth\raw.jsonl --n 20 --task all --max-new-tokens 192 --min-thai-ratio 0.70
```

Notes:
- First run will download the model from HF Hub (can take time and disk).
- If generation is slow/bottlenecked (common on 6GB VRAM laptops due to offload), reduce `--max-new-tokens` (try `128` or `192`) and/or run one task at a time (`--task route`, then `chitchat`, then `qa`).

Generation tuning knobs (CLI flags + env vars):
- `--max-new-tokens` (env: `TEACHER_MAX_NEW_TOKENS`)
- `--temperature` (env: `TEACHER_TEMPERATURE`)
- `--top-p` (env: `TEACHER_TOP_P`)
- `--repetition-penalty` (env: `TEACHER_REPETITION_PENALTY`)

---

## 5) Filter + deduplicate (Thai-only gate)

```powershell
cd D:\chitchat\chitchat_api
python .\synthetic\filter_thai.py --in .\data\thai_synth\raw.jsonl --out .\data\thai_synth\filtered.jsonl --min-thai-ratio 0.70
```

---

## 6) Adapt routing classifier (fine-tuning/adaptation)

Create a small CSV from synthetic routing examples:

```powershell
cd D:\chitchat\chitchat_api
python .\synthetic\make_route_dataset.py --in .\data\thai_synth\filtered.jsonl --out .\data\thai_synth\route_train.csv
```

Restart the API using extra routing data:

```powershell
$env:ROUTING_EXTRA_TRAIN_CSV = (Resolve-Path .\data\thai_synth\route_train.csv)
python -m uvicorn bot_api:app --host 127.0.0.1 --port 3001
```

---

## 7) Evaluate “works in Thai” (Both chat + QA)

```powershell
cd D:\chitchat\chitchat_api
python .\eval\eval_thai_api.py --base-url http://127.0.0.1:3001 --jsonl .\data\thai_synth\filtered.jsonl
```

This prints a JSON report:
- `thai_answer_rate`
- `by_mode_thai.chat` and `by_mode_thai.qa`
- `avg_latency_s`

---

## Optional: Docker-only run

- Compose file: `docker-compose.yml`
- Starts `web` (API) on port `3000` and `elasticsearch` on `9200`.

```powershell
cd D:\chitchat\chitchat_api
docker compose up --build
```

Then evaluate against Docker API:

```powershell
python .\eval\eval_thai_api.py --base-url http://127.0.0.1:3000 --jsonl .\data\thai_synth\filtered.jsonl
```
