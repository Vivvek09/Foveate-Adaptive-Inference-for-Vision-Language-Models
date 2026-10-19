"""Datasets: a common Sample record, loaders, and the pilot/full subset logic."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LETTERS = "ABCDEFGH"
SMOKE_N = 8


@dataclass
class Sample:
    qid: str
    dataset: str
    task: str
    question: str
    answer: str  # gold option letter for multiple choice, reference text otherwise
    options: list[str] | None = None  # multiple-choice candidates (None = free-form)
    references: list[str] = field(default_factory=list)
    video: str | None = None  # path to a video file
    frame_paths: list[str] | None = None  # or a list of pre-extracted frame images
    start: float | None = None  # optional clip bounds in seconds
    end: float | None = None
    clip_id: str | None = None  # questions sharing a clip share this id

    @property
    def is_mc(self) -> bool:
        return self.options is not None

    @property
    def source_key(self) -> str:
        """Identifies the visual input (for caching and repeat-clip workloads)."""
        base = self.video or "|".join(self.frame_paths or [])
        return f"{base}#{self.start}-{self.end}"

    def exists(self) -> bool:
        if self.video:
            return Path(self.video).exists()
        if self.frame_paths:
            return all(Path(p).exists() for p in self.frame_paths)
        return False


def stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16)


def pilot_subset(samples: list[Sample], fraction: float = 0.30, seed: int = 0) -> list[Sample]:
    """Fixed-seed stratified subset: `fraction` of every task, at least one item each.

    Stratifying by task keeps per-task accuracy estimable on the pilot, and the
    fixed seed means pilot runs on different days hit exactly the same questions,
    so pilot results can be compared with each other (and later with full runs).
    """
    by_task: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_task[s.task].append(s)
    out: list[Sample] = []
    for task in sorted(by_task):
        items = sorted(by_task[task], key=lambda s: s.qid)
        rng = random.Random(stable_hash(f"{seed}:{task}"))
        k = max(1, round(len(items) * fraction))
        out.extend(rng.sample(items, k))
    return sorted(out, key=lambda s: (s.task, s.qid))


def shard(samples: list[Sample], shard_id: int, num_shards: int) -> list[Sample]:
    """Deterministic split for data-parallel runs (one process per GPU)."""
    if num_shards <= 1:
        return samples
    return [s for i, s in enumerate(samples) if i % num_shards == shard_id]


_LOADERS: dict[str, Callable[..., list[Sample]]] = {}


def register(name: str):
    def deco(fn):
        _LOADERS[name] = fn
        return fn

    return deco


def load_samples(
    dataset: str,
    subset: str = "pilot",
    *,
    fraction: float = 0.30,
    seed: int = 0,
    tasks: list[str] | None = None,
    limit: int | None = None,
    root: str | Path | None = None,
) -> list[Sample]:
    """Load a dataset, then select `subset`.

    "smoke" = the first `limit` (default 8) pilot questions, round-robin over tasks;
    "pilot" = fixed-seed 30% stratified by task; "full" = everything.
    """
    # Import loaders lazily so `import foveate.data` stays light.
    from . import lingoqa, mvbench, synthetic  # noqa: F401

    if dataset not in _LOADERS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(_LOADERS)}")
    samples = _LOADERS[dataset](root=root, tasks=tasks)
    if subset in ("pilot", "smoke"):
        samples = pilot_subset(samples, fraction=fraction, seed=seed)
        if subset == "smoke":
            limit = limit or SMOKE_N
    elif subset != "full":
        raise ValueError(f"subset must be 'smoke', 'pilot' or 'full', got {subset!r}")
    if limit:
        # Keep the task mix when truncating for smoke tests: round-robin over tasks.
        by_task: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_task[s.task].append(s)
        rr: list[Sample] = []
        queues = [by_task[t] for t in sorted(by_task)]
        while len(rr) < limit and any(queues):
            for q in queues:
                if q and len(rr) < limit:
                    rr.append(q.pop(0))
        samples = rr
    return samples
