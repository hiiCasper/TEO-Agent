"""RSRCC annotation parsing with an explicit anti-leakage boundary."""

from __future__ import annotations

import re

from .schemas import ParsedRsrccTask


_QUESTION_LINE = re.compile(
    r"^\**Question(?:\**)?\s*:\s*\**\s*(.*?)\s*$", re.IGNORECASE
)
_OPTION_LINE = re.compile(r"^\*{0,2}([A-D])\)\*{0,2}\s*(.*?)\s*$", re.IGNORECASE)
_ANSWER_LINE = re.compile(
    r"^\**Answer(?:\**)?\s*:\s*\**\s*([A-D]|Yes|No)\)?", re.IGNORECASE
)


def _clean_markup(value: str) -> str:
    return " ".join(value.replace("**", "").strip(" *").split())


def parse_rsrcc_annotation(text: str) -> ParsedRsrccTask:
    """Split a dataset annotation and keep the answer out of ``agent_view``."""

    normalized = text.replace("\\n", "\n").strip()
    question: str | None = None
    answer: str | None = None
    options: dict[str, str] = {}
    preamble: list[str] = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        question_match = _QUESTION_LINE.match(line)
        if question_match:
            question = _clean_markup(question_match.group(1))
            continue
        answer_match = _ANSWER_LINE.match(line)
        if answer_match:
            answer = answer_match.group(1)
            continue
        option_match = _OPTION_LINE.match(line)
        if option_match:
            options[option_match.group(1).upper()] = _clean_markup(option_match.group(2))
            continue
        if not options and answer is None:
            preamble.append(_clean_markup(line))
    # One source row omits the Question marker entirely.  Preserve its leading
    # text verbatim (apart from weak whitespace/Markdown cleanup).
    if question is None and options and preamble:
        question = " ".join(preamble)
    if question is None or answer is None:
        raise ValueError("annotation does not contain a parseable Question and Answer")
    if options:
        if set(options) != {"A", "B", "C", "D"}:
            raise ValueError("multiple-choice annotations must contain A-D options")
        answer_format = "multiple_choice"
        gold_answer = answer.upper()
    else:
        answer_format = "binary"
        gold_answer = answer.lower()
    return ParsedRsrccTask(question, options, answer_format, gold_answer)
