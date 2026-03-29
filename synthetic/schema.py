from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal


TaskType = Literal["chitchat", "qa", "route"]
ThaiStyle = Literal["polite", "casual", "slang", "typo", "code_mix"]


@dataclass(frozen=True)
class SynthExample:
    """Single synthetic training/eval example.

    - task_type=chitchat: train generative chat or response templates
    - task_type=qa: train query paraphrases / evaluation prompts for QA retrieval
    - task_type=route: train routing classifier (qa_mode vs chat_mode)
    """

    id: str
    task_type: TaskType
    user: str
    assistant: str | None = None
    label: Literal["chat_mode", "qa_mode"] | None = None
    style: ThaiStyle | None = None
    tags: list[str] | None = None
    source: str = "synthetic"
    quality_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = payload["tags"] or []
        return payload


def is_thai_char(ch: str) -> bool:
    # Thai block: U+0E00..U+0E7F
    return "\u0E00" <= ch <= "\u0E7F"


def thai_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    thai = sum(1 for c in chars if is_thai_char(c))
    return thai / len(chars)
