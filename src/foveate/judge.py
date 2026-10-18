"""Lingo-Judge scoring for LingoQA's free-form answers.

Lingo-Judge (wayveai/Lingo-Judge) is a small text classifier trained by the
LingoQA authors to predict whether a free-form answer matches the reference;
the paper reports it correlates better with human ratings than BLEU, METEOR,
CIDEr or a GPT-4 judge on this data. A prediction counts as correct when its
score against either reference answer exceeds 0.5.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .utils import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

JUDGE_ID = "wayveai/Lingo-Judge"


def judge_inputs(question: str, reference: str, prediction: str) -> str:
    return f"[CLS]\nQuestion: {question}\nAnswer: {reference}\nStudent: {prediction}"


def judge_dir(run_dir: Path, samples_by_qid: dict, device: int | str = 0, batch_size: int = 64):
    """Add `judge_score` and `correct` to every LingoQA grid row that lacks them."""
    from transformers import pipeline

    pipe = pipeline("text-classification", model=JUDGE_ID, device=device)
    files = sorted((run_dir / "grid").glob("*.jsonl"))
    for f in files:
        rows = list(read_jsonl(f))
        todo = [r for r in rows if "judge_score" not in r]
        if not todo:
            continue
        log.info("judging %d rows in %s", len(todo), f.name)
        texts, owners = [], []
        for r in todo:
            s = samples_by_qid[r["qid"]]
            for ref in s.references or [s.answer]:
                texts.append(judge_inputs(s.question, ref, r.get("pred") or ""))
                owners.append(r)
        scores = pipe(texts, batch_size=batch_size, truncation=True)
        best: dict[int, float] = {}
        for r, sc in zip(owners, scores):
            best[id(r)] = max(best.get(id(r), 0.0), float(sc["score"]))
        for r in todo:
            r["judge_score"] = best[id(r)]
            r["correct"] = r["judge_score"] > 0.5
        write_jsonl(f, rows)
