"""Unit tests for synthetic/validate_quality.py (Stage 5)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from synthetic.validate_quality import (
    ValidationReport,
    ValidationResult,
    score_example,
    validate_jsonl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_example(
    *,
    id: str | None = None,
    task_type: str = "qa",
    user: str = "ถามเรื่องการลงทุนในตลาดหุ้น",
    assistant: str | None = "การลงทุนในหุ้นคือการซื้อส่วนแบ่งของบริษัท",
    label: str = "qa_mode",
    source: str = "synthetic",
) -> dict:
    return {
        "id": id or str(uuid.uuid4()),
        "task_type": task_type,
        "user": user,
        "assistant": assistant,
        "label": label,
        "source": source,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# score_example
# ---------------------------------------------------------------------------

class TestScoreExample:
    def test_perfect_example_scores_6(self):
        ex = _make_example()
        res = score_example(ex)
        assert res.score == 6
        assert res.passed is True

    def test_result_has_correct_fields(self):
        ex = _make_example()
        res = score_example(ex)
        assert isinstance(res, ValidationResult)
        assert res.max_score == 6
        for c in ("thai_user", "thai_assistant", "schema_valid", "label_valid", "length_ok", "source_present"):
            assert c in res.criteria

    # --- criterion: thai_user ---

    def test_non_thai_user_fails_thai_user(self):
        ex = _make_example(user="entirely in english text here")
        res = score_example(ex)
        assert res.criteria["thai_user"] is False

    def test_thai_user_passes(self):
        ex = _make_example(user="ถามเรื่องตลาดหลักทรัพย์แห่งประเทศไทย")
        res = score_example(ex)
        assert res.criteria["thai_user"] is True

    # --- criterion: thai_assistant ---

    def test_non_thai_assistant_fails(self):
        ex = _make_example(assistant="this is entirely english text response")
        res = score_example(ex)
        assert res.criteria["thai_assistant"] is False

    def test_none_assistant_passes(self):
        # Route examples have no assistant — should not be penalised
        ex = _make_example(task_type="route", assistant=None)
        res = score_example(ex)
        assert res.criteria["thai_assistant"] is True

    def test_empty_assistant_passes(self):
        ex = _make_example(assistant="")
        res = score_example(ex)
        assert res.criteria["thai_assistant"] is True

    # --- criterion: schema_valid ---

    def test_missing_id_fails_schema(self):
        ex = _make_example()
        ex.pop("id")
        res = score_example(ex)
        assert res.criteria["schema_valid"] is False

    def test_empty_label_fails_schema(self):
        ex = _make_example()
        ex["label"] = ""
        res = score_example(ex)
        assert res.criteria["schema_valid"] is False

    # --- criterion: label_valid ---

    def test_invalid_label_fails(self):
        ex = _make_example(label="unknown_label")
        res = score_example(ex)
        assert res.criteria["label_valid"] is False

    def test_chat_mode_label_passes(self):
        ex = _make_example(label="chat_mode")
        res = score_example(ex)
        assert res.criteria["label_valid"] is True

    def test_qa_mode_label_passes(self):
        ex = _make_example(label="qa_mode")
        res = score_example(ex)
        assert res.criteria["label_valid"] is True

    # --- criterion: length_ok ---

    def test_very_short_user_fails_length(self):
        ex = _make_example(user="ก")  # single char < 10
        res = score_example(ex)
        assert res.criteria["length_ok"] is False

    def test_sufficient_user_length_passes(self):
        ex = _make_example(user="ถามเรื่องการลงทุน")  # > 10 chars
        res = score_example(ex)
        assert res.criteria["length_ok"] is True

    # --- criterion: source_present ---

    def test_missing_source_fails(self):
        ex = _make_example(source="")
        res = score_example(ex)
        assert res.criteria["source_present"] is False

    def test_source_present_passes(self):
        ex = _make_example(source="synthetic")
        res = score_example(ex)
        assert res.criteria["source_present"] is True

    # --- passed threshold ---

    def test_passes_at_4_out_of_6(self):
        # Fail thai_user and source_present but pass the other 4
        ex = _make_example(user="hello english", source="")
        res = score_example(ex)
        # thai_user=False, thai_assistant depends on assistant value,
        # schema_valid=True (id present), label_valid=True,
        # length_ok=True (13 chars), source_present=False
        assert res.score == res.criteria["thai_assistant"] + res.criteria["schema_valid"] + 1 + 1

    def test_fails_below_4(self):
        # Force many failures
        ex = {
            "id": "",        # schema fails (id empty)
            "task_type": "qa",
            "user": "hi",    # thai_user fails, length_ok fails
            "assistant": "english response",  # thai_assistant fails
            "label": "bad",  # label_valid fails
            "source": "",    # source_present fails
        }
        res = score_example(ex)
        assert res.passed is False
        assert res.score < 4


# ---------------------------------------------------------------------------
# validate_jsonl
# ---------------------------------------------------------------------------

class TestValidateJsonl:
    def test_empty_file_gives_zero_total(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        report, results = validate_jsonl(p)
        assert report.total == 0
        assert report.passed == 0
        assert report.pass_rate == 0.0

    def test_all_pass(self, tmp_path):
        p = tmp_path / "all_pass.jsonl"
        _write_jsonl(p, [_make_example() for _ in range(5)])
        report, results = validate_jsonl(p)
        assert report.total == 5
        assert report.passed == 5
        assert report.pass_rate == pytest.approx(1.0)

    def test_mixed_pass_fail(self, tmp_path):
        p = tmp_path / "mixed.jsonl"
        good = _make_example()
        bad = {"id": "", "task_type": "qa", "user": "hi", "assistant": "no", "label": "bad", "source": ""}
        _write_jsonl(p, [good, bad])
        report, results = validate_jsonl(p)
        assert report.total == 2
        assert report.passed == 1
        assert report.failed == 1

    def test_criteria_pass_rates_present(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [_make_example()])
        report, _ = validate_jsonl(p)
        for c in ("thai_user", "thai_assistant", "schema_valid", "label_valid", "length_ok", "source_present"):
            assert c in report.criteria_pass_rates

    def test_failing_ids_listed(self, tmp_path):
        p = tmp_path / "data.jsonl"
        bad_id = "bad-example-id"
        bad = {"id": bad_id, "task_type": "qa", "user": "hi", "assistant": "no", "label": "bad", "source": ""}
        _write_jsonl(p, [bad])
        report, _ = validate_jsonl(p)
        assert bad_id in report.failing_ids

    def test_skips_invalid_json_lines(self, tmp_path):
        p = tmp_path / "data.jsonl"
        with p.open("w", encoding="utf-8") as f:
            f.write("not json\n")
            f.write(json.dumps(_make_example(), ensure_ascii=False) + "\n")
        report, results = validate_jsonl(p)
        assert report.total == 1

    def test_report_to_dict_serialisable(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [_make_example()])
        report, _ = validate_jsonl(p)
        d = report.to_dict()
        json.dumps(d)  # must not raise

    def test_avg_score_is_float(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [_make_example() for _ in range(3)])
        report, _ = validate_jsonl(p)
        assert isinstance(report.avg_score, float)
        assert 0.0 <= report.avg_score <= 6.0
