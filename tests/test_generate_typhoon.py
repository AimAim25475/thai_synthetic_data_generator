"""Unit tests for non-model parts of synthetic/generate_typhoon.py."""

from __future__ import annotations

import json

import pytest

from synthetic.generate_typhoon import (
    _build_prompt,
    _ensure_fields,
    _extract_json,
    _pick_device,
    _pick_dtype,
    _thai_gate,
)


# ---------------------------------------------------------------------------
# _pick_device
# ---------------------------------------------------------------------------

class TestPickDevice:
    def test_cpu_always_cpu(self):
        assert _pick_device("cpu") == "cpu"

    def test_auto_returns_string(self):
        result = _pick_device("auto")
        assert result in ("cpu", "cuda")

    def test_cuda_returns_string(self):
        result = _pick_device("cuda")
        assert result in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# _pick_dtype
# ---------------------------------------------------------------------------

class TestPickDtype:
    def test_float16(self):
        import torch
        assert _pick_dtype("float16") == torch.float16

    def test_float32(self):
        import torch
        assert _pick_dtype("float32") == torch.float32

    def test_bfloat16(self):
        import torch
        assert _pick_dtype("bfloat16") == torch.bfloat16

    def test_auto_returns_none(self):
        assert _pick_dtype("auto") is None

    def test_unknown_returns_none(self):
        assert _pick_dtype("unknown_dtype") is None


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_valid_json_object(self):
        text = '{"task_type": "qa", "user": "สวัสดี"}'
        result = _extract_json(text)
        assert isinstance(result, dict)
        assert result["task_type"] == "qa"

    def test_json_embedded_in_text(self):
        text = 'Some preamble\n{"task_type":"chitchat"}\nSome suffix'
        result = _extract_json(text)
        assert isinstance(result, dict)
        assert result["task_type"] == "chitchat"

    def test_no_json_returns_none(self):
        assert _extract_json("no braces here") is None

    def test_malformed_json_returns_none(self):
        assert _extract_json("{bad json: }") is None

    def test_empty_string_returns_none(self):
        assert _extract_json("") is None

    def test_nested_json(self):
        obj = {"task_type": "qa", "tags": ["a", "b"], "user": "ทดสอบ"}
        result = _extract_json(json.dumps(obj))
        assert result == obj


# ---------------------------------------------------------------------------
# _ensure_fields
# ---------------------------------------------------------------------------

class TestEnsureFields:
    def _base(self, **kwargs):
        defaults = {
            "task_type": "chitchat",
            "user": "สวัสดีครับ",
            "assistant": "ดีครับ",
            "label": "chat_mode",
            "style": None,
            "tags": ["gen"],
            "source": "synthetic",
        }
        defaults.update(kwargs)
        return defaults

    def test_valid_chitchat(self):
        result = _ensure_fields(self._base())
        assert result is not None
        assert result["task_type"] == "chitchat"

    def test_valid_qa(self):
        obj = self._base(task_type="qa", label="qa_mode", assistant="คำตอบ")
        result = _ensure_fields(obj)
        assert result is not None
        assert result["task_type"] == "qa"

    def test_valid_route_no_assistant(self):
        obj = self._base(task_type="route", label="qa_mode", assistant=None)
        result = _ensure_fields(obj)
        assert result is not None

    def test_missing_user_returns_none(self):
        obj = self._base(user="")
        assert _ensure_fields(obj) is None

    def test_invalid_task_type_returns_none(self):
        obj = self._base(task_type="unknown")
        assert _ensure_fields(obj) is None

    def test_invalid_label_returns_none(self):
        obj = self._base(label="unknown_label")
        assert _ensure_fields(obj) is None

    def test_chitchat_missing_assistant_returns_none(self):
        obj = self._base(task_type="chitchat", assistant="")
        assert _ensure_fields(obj) is None

    def test_qa_missing_assistant_returns_none(self):
        obj = self._base(task_type="qa", label="qa_mode", assistant="")
        assert _ensure_fields(obj) is None

    def test_tags_normalised_to_list(self):
        obj = self._base(tags=None)
        result = _ensure_fields(obj)
        assert result is not None
        assert isinstance(result["tags"], list)

    def test_id_auto_generated_when_missing(self):
        obj = self._base()
        obj.pop("id", None)
        result = _ensure_fields(obj)
        assert result is not None
        assert result["id"] != ""


# ---------------------------------------------------------------------------
# _thai_gate
# ---------------------------------------------------------------------------

class TestThaiGate:
    def test_passes_all_thai(self):
        ex = {"user": "สวัสดีครับ", "assistant": "ยินดีครับ"}
        assert _thai_gate(ex, 0.70) is True

    def test_fails_non_thai_user(self):
        ex = {"user": "hello entirely in english", "assistant": "ยินดีครับ"}
        assert _thai_gate(ex, 0.70) is False

    def test_fails_non_thai_assistant(self):
        ex = {"user": "สวัสดีครับ", "assistant": "hello entirely in english"}
        assert _thai_gate(ex, 0.70) is False

    def test_passes_with_no_assistant(self):
        ex = {"user": "สวัสดีครับ", "assistant": None}
        assert _thai_gate(ex, 0.70) is True

    def test_threshold_boundary(self):
        # 7 Thai + 3 Latin = 0.7 exactly → should pass at threshold 0.70
        user = "กขคงจฉชabc"  # 7 Thai, 3 Latin = ratio 0.7
        ex = {"user": user, "assistant": None}
        assert _thai_gate(ex, 0.70) is True


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_qa_prompt_contains_seed(self):
        seed = "ถามเรื่องการลงทุน"
        prompt = _build_prompt("qa", "qa_mode", None, seed)
        assert seed in prompt

    def test_qa_prompt_mentions_task_type(self):
        prompt = _build_prompt("qa", "qa_mode", None, "test seed")
        assert "qa" in prompt.lower()

    def test_chitchat_prompt_contains_seed(self):
        seed = "สวัสดีครับ"
        prompt = _build_prompt("chitchat", "chat_mode", "casual", seed)
        assert seed in prompt

    def test_prompt_ends_with_json_marker(self):
        prompt = _build_prompt("qa", "qa_mode", None, "seed")
        assert prompt.strip().endswith("JSON:")

    def test_route_prompt_contains_label(self):
        prompt = _build_prompt("route", "qa_mode", None, "สอบถาม")
        assert "qa_mode" in prompt

    def test_chitchat_prompt_includes_style(self):
        prompt = _build_prompt("chitchat", "chat_mode", "casual", "สวัสดี")
        assert "casual" in prompt
