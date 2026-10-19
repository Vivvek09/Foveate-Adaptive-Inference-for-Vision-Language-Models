"""LingoQA (Marcu et al., ECCV 2024): driving-scene video QA from Wayve.

The evaluation split has 500 questions over 100 driving clips (two reference
answers each, 1,000 rows), with a handful of frames per clip. Answers are free
text and are scored by Lingo-Judge, a small classifier released with the
dataset (see foveate/judge.py). This is the driving demo workload; MVBench is
the headline benchmark.

Read the dataset's LICENCE before redistributing anything derived from it.

Expected layout under <data_root>/lingoqa (what `foveate download lingoqa` makes):
    val.parquet (any *.parquet whose name contains "val" also works)
    images/...  (the extracted images archive)
"""

from __future__ import annotations

import logging
import zipfile
from collections import defaultdict
from pathlib import Path

from . import Sample, register

log = logging.getLogger(__name__)

EVAL_FOLDER = "https://drive.google.com/drive/folders/1oA7W8-Ej_uJEuUxZIjPP5K8hQGGzYsPq"


def download(root: Path) -> None:
    import gdown

    out = root / "lingoqa"
    out.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(EVAL_FOLDER, output=str(out), quiet=False, remaining_ok=True)
    for z in out.rglob("*.zip"):
        marker = z.with_suffix(".done")
        if not marker.exists():
            log.info("extracting %s", z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(z.parent)
            marker.touch()


def _find_parquet(base: Path) -> Path:
    cands = sorted(p for p in base.rglob("*.parquet") if "val" in p.name.lower())
    if not cands:
        cands = sorted(base.rglob("*.parquet"))
    if not cands:
        raise FileNotFoundError(f"no parquet under {base}; run `foveate download lingoqa`")
    return cands[0]


def _resolve_images(base: Path, rels, segment_id: str, seg_index: dict[str, list[Path]]):
    paths = []
    for r in rels if rels is not None else []:
        r = str(r)
        for cand in (base / r, base / "images" / r, Path(r)):
            if cand.exists():
                paths.append(str(cand))
                break
    if paths:
        return paths
    return [str(p) for p in seg_index.get(segment_id, [])]


@register("lingoqa")
def load(root: str | Path | None = None, tasks=None) -> list[Sample]:
    import pandas as pd

    from ..config import data_root

    base = Path(root or data_root()) / "lingoqa"
    df = pd.read_parquet(_find_parquet(base))
    # Fallback index: images grouped by their parent folder name (= segment id).
    seg_index: dict[str, list[Path]] = defaultdict(list)
    for p in base.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            seg_index[p.parent.name].append(p)
    for k in seg_index:
        seg_index[k].sort(key=lambda p: (len(p.stem), p.stem))

    ans_col = "answers" if "answers" in df.columns else "answer"
    groups: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        r = row._asdict()
        qid = str(r["question_id"])
        g = groups.setdefault(
            qid,
            {"segment_id": str(r["segment_id"]), "question": r["question"], "refs": [],
             "images": r.get("images")},
        )
        a = r[ans_col]
        if isinstance(a, str):
            g["refs"].append(a)
        else:
            g["refs"].extend(str(x) for x in a)
    samples = []
    for qid, g in groups.items():
        frames = _resolve_images(base, g["images"], g["segment_id"], seg_index)
        if not frames:
            continue
        samples.append(
            Sample(
                qid=qid,
                dataset="lingoqa",
                task="driving",
                question=str(g["question"]),
                answer=g["refs"][0],
                references=list(dict.fromkeys(g["refs"])),
                frame_paths=frames,
                clip_id=g["segment_id"],
            )
        )
    log.info("LingoQA: %d questions over %d clips", len(samples), len({s.clip_id for s in samples}))
    return samples
