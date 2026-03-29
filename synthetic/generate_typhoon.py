from __future__ import annotations

# Allow running this file directly (python synthetic/generate_typhoon.py)
# without requiring users to set PYTHONPATH.
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from synthetic.schema import thai_char_ratio


Task = Literal["route", "chitchat", "qa", "all"]


@dataclass(frozen=True)
class GenConfig:
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    device: str = "auto"  # auto|cuda|cpu
    dtype: str = "auto"  # auto|float16|bfloat16|float32


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _pick_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pick_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return None


def _extract_json(text: str) -> dict[str, Any] | None:
    # Try to extract the first JSON object in the completion.
    m = _JSON_RE.search(text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _ensure_fields(obj: dict[str, Any]) -> dict[str, Any] | None:
    # Validate minimal schema fields; normalize task_type/label.
    task_type = str(obj.get("task_type") or "").strip().lower()
    if task_type not in {"route", "chitchat", "qa"}:
        return None

    user = str(obj.get("user") or "").strip()
    if not user:
        return None

    label = obj.get("label")
    if task_type in {"route", "chitchat", "qa"}:
        # label is required for routing and helpful for others
        label = str(label or ("chat_mode" if task_type == "chitchat" else "qa_mode")).strip()
        if label not in {"chat_mode", "qa_mode"}:
            return None

    assistant = obj.get("assistant")
    if task_type == "chitchat":
        assistant = str(assistant or "").strip()
        if not assistant:
            return None
    elif task_type == "qa":
        # QA tasks should have an assistant (answer)
        assistant = str(assistant or "").strip()
        if not assistant:
            return None
    else:
        # Route tasks should not have an assistant
        assistant = None if assistant is None else str(assistant).strip() or None

    style = obj.get("style")
    style = None if style is None else str(style).strip() or None

    tags = obj.get("tags")
    if tags is None:
        tags = []
    elif not isinstance(tags, list):
        tags = [str(tags)]
    else:
        tags = [str(t) for t in tags]

    out = {
        "id": str(obj.get("id") or str(uuid.uuid4())),
        "task_type": task_type,
        "user": user,
        "assistant": assistant,
        "label": label,
        "style": style,
        "tags": tags,
        "source": str(obj.get("source") or "synthetic"),
        "quality_score": obj.get("quality_score"),
    }
    return out


def _thai_gate(ex: dict[str, Any], min_ratio: float) -> bool:
    if thai_char_ratio(ex.get("user") or "") < min_ratio:
        return False
    if ex.get("assistant") and thai_char_ratio(ex.get("assistant") or "") < min_ratio:
        return False
    return True


def _build_prompt(task: str, label: str | None, style: str | None, seed_user: str) -> str:
    # Strict JSON-only instruction. Keep it short for smaller models.
    style_clause = f"\nstyle: {style}" if style else ""
    label_clause = f"\nlabel: {label}" if label else ""

    if task == "qa":
        # For QA tasks: generate both question and answer (NOT follow-up questions)
        return (
            "You are generating Thai-language synthetic Q&A data.\n"
            "IMPORTANT: Output ONLY one JSON object. No markdown. No explanations.\n"
            "The JSON MUST have keys: task_type, user, assistant, label, style, tags.\n"
            "Rules:\n"
            "- Use Thai language only.\n"
            "- task_type must be: qa\n"
            "- label must be: qa_mode\n"
            "- user: the Thai question or request for summary (based on the seed)\n"
            "- assistant: a DIRECT ANSWER or SUMMARY in Thai (3-5 sentences explaining facts, not asking more questions)\n"
            "- DO NOT respond with questions! Give explanatory statements instead.\n"
            "- tags must be a JSON array of short strings\n"
            "- style: null (not used for QA)\n"
            "\n"
            f"Seed input: {seed_user}\n"
            "Generate a Thai question and provide a clear, factual Thai answer/summary about financial/investment topics.\n"
            "Example format:\n"
            "  user: 'บอกความหมายของคำว่า...' → assistant: '[Topic] คือ... [explanation]...'\n"
            "NOT: user: '...' → assistant: '[Topic] คืออะไร? [more questions]?'\n"
            "\nJSON:"
        )
    else:
        # For route and chitchat tasks
        return (
            "You are generating Thai-language synthetic training data for a chatbot.\n"
            "IMPORTANT: Output ONLY one JSON object. No markdown. No explanations.\n"
            "The JSON MUST have keys: task_type, user, assistant, label, style, tags.\n"
            "Rules:\n"
            "- Use Thai language. Avoid English unless task requires code-mix style.\n"
            "- task_type must be one of: route, chitchat\n"
            "- label must be either chat_mode or qa_mode\n"
            "- For task_type=route: assistant must be null\n"
            "- For task_type=chitchat: assistant must be a natural Thai reply\n"
            "- tags must be a JSON array of short strings\n"
            "\n"
            f"Requested task_type: {task}{label_clause}{style_clause}\n"
            f"Seed user text: {seed_user}\n"
            "\nJSON:" 
        )


def _generate_one(
    model,
    tokenizer,
    prompt: str,
    cfg: GenConfig,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Keep only the completion region after "JSON:" if present.
    idx = decoded.rfind("JSON:")
    if idx >= 0:
        decoded = decoded[idx + len("JSON:") :]
    return decoded.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Thai synthetic JSONL using a local HF <=7B model.")
    ap.add_argument("--model", default=os.getenv("TEACHER_MODEL_NAME", ""), help="HF model name/path (<=7B)")
    ap.add_argument("--seeds", default=str(Path(__file__).with_name("seeds_thai.json")), help="Seed JSON")
    ap.add_argument("--out", required=True, help="Output JSONL")
    ap.add_argument("--task", choices=["route", "chitchat", "qa", "all"], default="all")
    ap.add_argument("--n", type=int, default=50, help="How many examples to generate per category")
    ap.add_argument("--max-new-tokens", type=int, default=int(os.getenv("TEACHER_MAX_NEW_TOKENS", "512")), help="Max new tokens per example")
    ap.add_argument("--temperature", type=float, default=float(os.getenv("TEACHER_TEMPERATURE", "0.7")), help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=float(os.getenv("TEACHER_TOP_P", "0.9")), help="Nucleus sampling top-p")
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(os.getenv("TEACHER_REPETITION_PENALTY", "1.05")),
        help="Repetition penalty",
    )
    ap.add_argument("--min-thai-ratio", type=float, default=0.70)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=os.getenv("TEACHER_DEVICE", "auto"))
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=os.getenv("TEACHER_DTYPE", "auto"))
    args = ap.parse_args()

    model_name = (args.model or "").strip()
    if not model_name:
        raise SystemExit("Missing --model or TEACHER_MODEL_NAME")

    device = _pick_device(args.device)
    torch_dtype = _pick_dtype(args.dtype)

    print(f"Loading teacher model: {model_name} (device={device}, dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to("cpu")

    seeds = json.loads(Path(args.seeds).read_text(encoding="utf-8"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted: set[str] = {args.task} if args.task != "all" else {"route", "chitchat", "qa"}

    # Seed pools
    route = seeds.get("route") or {}
    chitchat = seeds.get("chitchat") or {}
    qa = seeds.get("qa") or {}

    # Build minimal seed_user list per task
    route_seeds = []
    for label, utts in route.items():
        for u in utts or []:
            route_seeds.append(("route", label, None, u))

    chitchat_seeds = []
    for style, pairs in chitchat.items():
        for p in pairs or []:
            chitchat_seeds.append(("chitchat", "chat_mode", style, p.get("user") or "สวัสดี"))

    qa_seeds = []
    for q in qa.get("queries") or []:
        qa_seeds.append(("qa", "qa_mode", None, q))

    pools = []
    if "route" in wanted:
        pools.append(route_seeds)
    if "chitchat" in wanted:
        pools.append(chitchat_seeds)
    if "qa" in wanted:
        pools.append(qa_seeds)

    written = 0

    gen_cfg = GenConfig(
        model_name=model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
        dtype=args.dtype,
    )

    with out_path.open("w", encoding="utf-8") as fout:
        for pool in pools:
            if not pool:
                continue
            for i in range(args.n):
                task_type, label, style, seed_user = pool[i % len(pool)]
                prompt = _build_prompt(task_type, label, style, seed_user)

                ok = False
                for attempt in range(args.retries):
                    completion = _generate_one(model, tokenizer, prompt, gen_cfg)
                    obj = _extract_json(completion)
                    if not isinstance(obj, dict):
                        time.sleep(0.05)
                        continue

                    ex = _ensure_fields(obj)
                    if not ex:
                        time.sleep(0.05)
                        continue

                    # Force requested task/label/style if model deviated
                    ex["task_type"] = task_type
                    ex["label"] = label
                    ex["style"] = style

                    if not _thai_gate(ex, args.min_thai_ratio):
                        time.sleep(0.05)
                        continue

                    ex["tags"] = list(dict.fromkeys(["gen"] + (ex.get("tags") or [])))
                    fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    written += 1
                    ok = True
                    break

                if not ok:
                    # As a fallback, emit a seed example so the pipeline continues.
                    fallback = {
                        "id": str(uuid.uuid4()),
                        "task_type": task_type,
                        "user": seed_user,
                        "assistant": None if task_type != "chitchat" else "สวัสดี เราคุยกันได้เลยนะ",
                        "label": label,
                        "style": style,
                        "tags": ["fallback"],
                        "source": "seed_fallback",
                    }
                    if _thai_gate(fallback, args.min_thai_ratio):
                        fout.write(json.dumps(fallback, ensure_ascii=False) + "\n")
                        written += 1

    print(f"Wrote {written} examples to {out_path}")
    print("Next: run synthetic/filter_thai.py and eval/eval_thai_api.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
