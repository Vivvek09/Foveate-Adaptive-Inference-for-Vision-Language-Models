"""A deterministic stand-in for a VLM: more frames and pixels -> more often right."""

import math

from foveate.data import stable_hash
from foveate.prompts import mc_summary, option_letters, softmax_over_options


class FakeBackend:
    def __init__(self, skill: float = 1.0):
        self.skill = skill

    def _row(self, s, fb):
        letters = option_letters(s)
        # Difficulty is a property of the question; capacity grows with tokens.
        tokens = len(fb.frames) * fb.height * fb.width / 784
        difficulty = (stable_hash(s.qid) % 1000) / 1000
        sharp = fb.stats.get("sharpness", 1.0)
        capacity = self.skill * (1 - math.exp(-tokens / 400)) * min(1.0, 0.3 + sharp / 2)
        right = capacity > difficulty
        lp = {L: -3.0 for L in letters}
        lp[s.answer if right else next(L for L in letters if L != s.answer)] = -3.0 + 4 * abs(capacity - difficulty) + 0.2
        row = mc_summary(softmax_over_options(lp, letters))
        row.update({"text": row["pred"], "prompt_tokens": int(tokens) + 40, "visual_tokens": int(tokens)})
        return row

    def predict_batch(self, samples, batches, max_new_tokens=48):
        return [self._row(s, fb) for s, fb in zip(samples, batches)], 0.001 * len(samples)

    def latency_probe(self, samples, batches, max_new_tokens=48):
        return [10.0 + len(fb.frames) * fb.height * fb.width / 784 * 0.05 for fb in batches]
