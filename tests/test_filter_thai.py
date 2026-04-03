"""Unit tests for synthetic/filter_thai.py (non-I/O parts)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from synthetic.filter_thai import _fingerprint, _norm, main as filter_main


# ---------------------------------------------------------------------------
# _norm
# ---------------------------------------------------------------------------

class TestNorm:
    def test_strips_whitespace(self):
        assert _norm("  hello  ") == "hello"

    def test_lowercases(self):
        assert _norm("Hello World") == "hello world"

    def test_collapses_spaces(self):
        assert _norm("a  b   c") == "a b c"

    def test_none_like_empty(self):
        # _norm receives str(ex.get(...) or "") so empty string expected
        assert _norm("") == ""


# ---------------------------------------------------------------------------
# _fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:
    def _ex(self, user="สวัสดี", assistant="ดีมาก", label="chat_mode", task_type="chitchat"):
        return {"user": user, "assistant": assistant, "label": label, "task_type": task_type}

    def test_same_content_same_hash(self):
        ex = self._ex()
        assert _fingerprint(ex) == _fingerprint(ex)

    def test_different_user_different_hash(self):
        ex1 = self._ex(user="สวัสดี")
        ex2 = self._ex(user="โอเค")
        assert _fingerprint(ex1) != _fingerprint(ex2)

    def test_different_assistant_different_hash(self):
        ex1 = self._ex(assistant="ดีมาก")
        ex2 = self._ex(assistant="ไม่ดี")
        assert _fingerprint(ex1) != _fingerprint(ex2)

    def test_case_insensitive_latin_parts(self):
        # Normalisation lowercases labels/task_types
        ex1 = {"user": "สวัสดี", "assistant": "", "label": "chat_mode", "task_type": "chitchat"}
        ex2 = {"user": "สวัสดี", "assistant": "", "label": "CHAT_MODE", "task_type": "CHITCHAT"}
        assert _fingerprint(ex1) == _fingerprint(ex2)

    def test_returns_64_char_hex(self):
        fp = _fingerprint(self._ex())
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# filter pipeline (main) — integration via tmp files
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


class TestFilterMain:
    def _make_example(self, user="สวัสดีครับ", assistant="ตอบแบบนี้", label="chat_mode", task_type="chitchat"):
        return {
            "id": str(uuid.uuid4()),
            "task_type": task_type,
            "user": user,
            "assistant": assistant,
            "label": label,
            "source": "synthetic",
        }

    def test_passes_valid_thai_example(self, tmp_dir):
        inp = tmp_dir / "in.jsonl"
        out = tmp_dir / "out.jsonl"
        _write_jsonl(inp, [self._make_example()])

        filter_main(["--in", str(inp), "--out", str(out)])

        rows = _read_jsonl(out)
        assert len(rows) == 1

    def test_drops_non_thai_user(self, tmp_dir):
        inp = tmp_dir / "in.jsonl"
        out = tmp_dir / "out.jsonl"
        _write_jsonl(inp, [self._make_example(user="hello world entirely in english")])

        filter_main(["--in", str(inp), "--out", str(out)])

        rows = _read_jsonl(out)
        assert len(rows) == 0

    def test_deduplicates(self, tmp_dir):
        inp = tmp_dir / "in.jsonl"
        out = tmp_dir / "out.jsonl"
        ex = self._make_example()
        _write_jsonl(inp, [ex, ex])  # exact duplicate

        filter_main(["--in", str(inp), "--out", str(out)])

        rows = _read_jsonl(out)
        assert len(rows) == 1

    def test_skips_bad_json_lines(self, tmp_dir):
        inp = tmp_dir / "in.jsonl"
        out = tmp_dir / "out.jsonl"
        with inp.open("w", encoding="utf-8") as f:
            f.write("not json\n")
            f.write(json.dumps(self._make_example(), ensure_ascii=False) + "\n")

        filter_main(["--in", str(inp), "--out", str(out)])

        rows = _read_jsonl(out)
        assert len(rows) == 1

    def test_custom_min_thai_ratio(self, tmp_dir):
        # A text that is 50% Thai — passes 0.4 but not 0.7
        mixed_user = "กขคabc"  # 3 Thai / 6 non-space = 0.5
        inp = tmp_dir / "in.jsonl"
        out_strict = tmp_dir / "strict.jsonl"
        out_loose = tmp_dir / "loose.jsonl"
        _write_jsonl(inp, [self._make_example(user=mixed_user)])

        filter_main(["--in", str(inp), "--out", str(out_strict), "--min-thai-ratio", "0.7"])
        filter_main(["--in", str(inp), "--out", str(out_loose), "--min-thai-ratio", "0.4"])

        assert len(_read_jsonl(out_strict)) == 0
        assert len(_read_jsonl(out_loose)) == 1
