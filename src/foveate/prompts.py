"""Prompt construction and answer scoring shared by every backend."""

from __future__ import annotations

import math
import re

from .data import LETTERS, Sample

MC_INSTRUCTION = "Answer with the option's letter from the given choices directly."
OPEN_INSTRUCTION = "Answer the question in one short sentence."


def question_text(s: Sample) -> str:
    if s.is_mc:
        opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(s.options or []))
        return f"Question: {s.question}\nOptions:\n{opts}\n{MC_INSTRUCTION}"
    return f"{s.question}\n{OPEN_INSTRUCTION}"


def chat_messages(s: Sample, video_item: dict | None = None) -> list[dict]:
    """HF-style chat messages with one video placeholder followed by the question."""
    content = [video_item or {"type": "video"}, {"type": "text", "text": question_text(s)}]
    return [{"role": "user", "content": content}]


def option_letters(s: Sample) -> list[str]:
    return list(LETTERS[: len(s.options or [])])


def softmax_over_options(letter_logprobs: dict[str, float], letters: list[str]) -> dict[str, float]:
    """Renormalise next-token log-probs over the valid option letters only.

    This is "logit-based" multiple-choice scoring: the prediction is the most
    probable valid letter, and the renormalised probability of that letter is
    the confidence the cascade thresholds. A letter missing from the top-k list
    gets a floor one nat below the smallest returned value.
    """
    if letter_logprobs:
        floor = min(letter_logprobs.values()) - 1.0
    else:
        floor = 0.0
    lps = [letter_logprobs.get(L, floor) for L in letters]
    m = max(lps)
    z = sum(math.exp(lp - m) for lp in lps)
    return {L: math.exp(lp - m) / z for L, lp in zip(letters, lps)}


def mc_summary(probs: dict[str, float]) -> dict:
    """Prediction plus three confidence signals: max prob, top-2 margin, entropy."""
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    p1 = ranked[0][1]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    ent = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return {"pred": ranked[0][0], "conf": p1, "margin": p1 - p2, "entropy": ent}


_LETTER_RE = re.compile(r"^\s*\(?([A-H])\b")


def parse_letter(text: str, letters: list[str]) -> str | None:
    """Fallback parser for generated text like 'B', '(B)', 'B. a car'."""
    m = _LETTER_RE.match(text or "")
    if m and m.group(1) in letters:
        return m.group(1)
    return None


def clean_token(tok: str) -> str:
    """Strip tokenizer artefacts (leading space markers) from a decoded token."""
    return tok.replace("Ġ", "").replace("▁", "").strip().lstrip("(")
