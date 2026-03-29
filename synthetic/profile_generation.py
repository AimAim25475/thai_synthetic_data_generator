"""
Performance monitoring for generate_typhoon.py

Usage:
    python synthetic/profile_generation.py --model <model_name> --n 5

This runs a small test to measure actual performance metrics.
"""

import json
import os
import sys
import time
from pathlib import Path as _Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synthetic.generate_typhoon import (
    GenConfig,
    _build_prompt,
    _extract_json,
    _ensure_fields,
    _pick_device,
    _pick_dtype,
    _thai_gate,
)


class PerformanceMetrics:
    """Track performance metrics during generation."""

    def __init__(self):
        self.model_load_time = 0.0
        self.total_generation_time = 0.0
        self.total_tokens_generated = 0
        self.generation_count = 0
        self.peak_memory_gb = 0.0
        self.start_memory_gb = 0.0

    def get_memory_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        else:
            # CPU memory (rough estimate)
            import psutil
            return psutil.Process().memory_info().rss / (1024**3)

    def record_model_load(self, elapsed: float):
        self.model_load_time = elapsed
        self.start_memory_gb = self.get_memory_gb()

    def record_generation(self, elapsed: float, token_count: int):
        self.total_generation_time += elapsed
        self.total_tokens_generated += token_count
        self.generation_count += 1
        mem = self.get_memory_gb()
        if mem > self.peak_memory_gb:
            self.peak_memory_gb = mem

    def print_report(self, num_examples: int):
        """Print a formatted performance report."""
        print("\n" + "=" * 70)
        print("PERFORMANCE REPORT".center(70))
        print("=" * 70)

        print(f"\n📊 EXECUTION METRICS")
        print(f"  Model load time:        {self.model_load_time:.2f}s")
        print(f"  Total generation time:  {self.total_generation_time:.2f}s")
        print(f"  Examples generated:     {num_examples}")
        print(f"  Avg time per example:   {self.total_generation_time/max(num_examples, 1):.2f}s")

        print(f"\n🚀 TOKEN THROUGHPUT")
        if self.total_generation_time > 0:
            tokens_per_sec = self.total_tokens_generated / self.total_generation_time
            print(f"  Total tokens generated: {self.total_tokens_generated}")
            print(f"  Tokens/second:          {tokens_per_sec:.1f}")
            print(f"  Avg tokens per example: {self.total_tokens_generated/max(num_examples, 1):.0f}")
        else:
            print(f"  Total tokens generated: {self.total_tokens_generated}")

        print(f"\n💾 MEMORY USAGE")
        print(f"  Start memory:           {self.start_memory_gb:.2f} GB")
        print(f"  Peak memory:            {self.peak_memory_gb:.2f} GB")
        print(f"  Memory increase:        {(self.peak_memory_gb - self.start_memory_gb):.2f} GB")

        # Extrapolate
        print(f"\n📈 EXTRAPOLATION (to 150 examples across 3 tasks)")
        total_time_150 = (self.total_generation_time / max(num_examples, 1)) * 150
        hours = total_time_150 / 3600
        minutes = (total_time_150 % 3600) / 60
        print(f"  Estimated total time:   {hours:.1f}h {minutes:.0f}m ({total_time_150:.0f}s)")
        print(f"  Estimated tokens:       {int((self.total_tokens_generated/max(num_examples, 1)) * 150)}")

        print("\n" + "=" * 70)


def count_tokens(tokenizer, text: str) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text))


def profile_generation(model_name: str, num_examples: int = 5):
    """Run a small generation sample and collect metrics."""

    device = _pick_device("auto")
    torch_dtype = _pick_dtype("auto")

    metrics = PerformanceMetrics()

    print(f"🔧 Loading model: {model_name}")
    print(f"   Device: {device}, Dtype: {torch_dtype}")

    # Load model
    start = time.time()
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
    load_time = time.time() - start
    metrics.record_model_load(load_time)

    print(f"✓ Model loaded in {load_time:.2f}s\n")

    # Generate samples
    gen_cfg = GenConfig(
        model_name=model_name,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        device="auto",
        dtype="auto",
    )

    seed_user = "สวัสดีค่ะ วันนี้หากไหร่"
    successful = 0

    print(f"📝 Generating {num_examples} examples...\n")

    for i in range(num_examples):
        task_type = ["route", "chitchat", "qa"][i % 3]
        label = "chat_mode" if task_type == "chitchat" else "qa_mode"
        style = None if task_type != "chitchat" else "casual"

        prompt = _build_prompt(task_type, label, style, seed_user)
        prompt_tokens = count_tokens(tokenizer, prompt)

        # Generate
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt")
        device_obj = next(model.parameters()).device
        inputs = {k: v.to(device_obj) for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=True,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            repetition_penalty=gen_cfg.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.time() - start

        # Decode and count tokens
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        total_tokens = count_tokens(tokenizer, decoded)
        generated_tokens = total_tokens - prompt_tokens

        # Record metrics
        metrics.record_generation(elapsed, generated_tokens)

        # Parse and validate
        idx = decoded.rfind("JSON:")
        if idx >= 0:
            json_str = decoded[idx + len("JSON:") :].strip()
        else:
            json_str = decoded.strip()

        obj = _extract_json(json_str)
        ex = _ensure_fields(obj) if isinstance(obj, dict) else None
        is_valid = ex is not None and _thai_gate(ex, 0.7)

        status = "✓" if is_valid else "✗"
        print(
            f"  {status} Example {i+1} ({task_type:8s}): "
            f"{elapsed:.2f}s | {generated_tokens:3d} new tokens | {generated_tokens/elapsed:6.1f} tok/s"
        )
        if is_valid:
            successful += 1

    print(f"\n✓ Generated {successful}/{num_examples} valid examples\n")
    metrics.print_report(successful)

    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Profile generate_typhoon.py performance")
    ap.add_argument("--model", required=True, help="HF model name/path")
    ap.add_argument("--n", type=int, default=5, help="Number of examples to generate")
    args = ap.parse_args()

    profile_generation(args.model, args.n)
