"""Unit tests for synthetic/schema.py."""

from __future__ import annotations

import pytest

from synthetic.schema import SynthExample, is_thai_char, thai_char_ratio


# ---------------------------------------------------------------------------
# is_thai_char
# ---------------------------------------------------------------------------

class TestIsThaiChar:
    def test_thai_consonant(self):
        assert is_thai_char("ก") is True

    def test_thai_vowel(self):
        assert is_thai_char("า") is True

    def test_thai_boundary_low(self):
        # U+0E00 is reserved but within the block
        assert is_thai_char("\u0E00") is True

    def test_thai_boundary_high(self):
        assert is_thai_char("\u0E7F") is True

    def test_latin_char(self):
        assert is_thai_char("a") is False

    def test_digit(self):
        assert is_thai_char("5") is False

    def test_space(self):
        assert is_thai_char(" ") is False

    def test_emoji(self):
        assert is_thai_char("😀") is False


# ---------------------------------------------------------------------------
# thai_char_ratio
# ---------------------------------------------------------------------------

class TestThaiCharRatio:
    def test_empty_string(self):
        assert thai_char_ratio("") == 0.0

    def test_whitespace_only(self):
        assert thai_char_ratio("   ") == 0.0

    def test_all_thai(self):
        ratio = thai_char_ratio("สวัสดี")
        assert ratio == pytest.approx(1.0)

    def test_all_latin(self):
        assert thai_char_ratio("hello") == pytest.approx(0.0)

    def test_mixed_50_percent(self):
        # 3 Thai chars + 3 Latin chars, no spaces → 0.5
        ratio = thai_char_ratio("กขคabc")
        assert ratio == pytest.approx(0.5)

    def test_spaces_ignored(self):
        # Spaces are excluded from denominator
        ratio = thai_char_ratio("กข ab")  # 2 Thai, 2 Latin = 0.5
        assert ratio == pytest.approx(0.5)

    def test_single_thai(self):
        assert thai_char_ratio("ก") == pytest.approx(1.0)

    def test_single_latin(self):
        assert thai_char_ratio("x") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SynthExample
# ---------------------------------------------------------------------------

class TestSynthExample:
    def _make(self, **kwargs) -> SynthExample:
        defaults = dict(
            id="abc-123",
            task_type="qa",
            user="ถามเรื่องนี้",
            assistant="ตอบแบบนี้",
            label="qa_mode",
            style=None,
            tags=["gen"],
            source="synthetic",
            quality_score=0.9,
        )
        defaults.update(kwargs)
        return SynthExample(**defaults)

    def test_to_dict_contains_all_keys(self):
        ex = self._make()
        d = ex.to_dict()
        for key in ("id", "task_type", "user", "assistant", "label", "style", "tags", "source", "quality_score"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_tags_default_empty_list(self):
        ex = self._make(tags=None)
        d = ex.to_dict()
        assert d["tags"] == []

    def test_to_dict_tags_preserved(self):
        ex = self._make(tags=["gen", "finance"])
        d = ex.to_dict()
        assert d["tags"] == ["gen", "finance"]

    def test_frozen_immutable(self):
        ex = self._make()
        with pytest.raises((AttributeError, TypeError)):
            ex.user = "changed"  # type: ignore[misc]

    def test_route_example_no_assistant(self):
        ex = self._make(task_type="route", assistant=None, label="qa_mode")
        d = ex.to_dict()
        assert d["assistant"] is None
