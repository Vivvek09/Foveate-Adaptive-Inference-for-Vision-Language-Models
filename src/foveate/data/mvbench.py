"""MVBench (Li et al., CVPR 2024): 20 temporal-reasoning tasks x 200 multiple-choice questions.

Why this dataset: it is in the video table of nearly every recent VLM paper
(Qwen2.5-VL reports 67.0 for the 3B and 69.6 for the 7B model), so a harness
bug shows up as a gap to a published number. Clips are short and every answer
is one option letter, so a full pass over one config is cheap.

Layout expected under <data_root>/mvbench (what `foveate download mvbench` makes):
    json/<task>.json
    video/<extracted zip folders>   e.g. video/star/Charades_v1_480/XXXX.mp4
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path

from . import LETTERS, Sample, register

log = logging.getLogger(__name__)

HF_REPO = "OpenGVLab/MVBench"

# task json -> (video folder prefix inside video/, input kind, has start/end bounds)
# Mirrors the official evaluation script's data_list.
TASKS: dict[str, tuple[str, str, bool]] = {
    "action_sequence": ("star/Charades_v1_480/", "video", True),
    "action_prediction": ("star/Charades_v1_480/", "video", True),
    "action_antonym": ("ssv2_video/", "video", False),
    "fine_grained_action": ("Moments_in_Time_Raw/videos/", "video", False),
    "unexpected_action": ("FunQA_test/test/", "video", False),
    "object_existence": ("clevrer/video_validation/", "video", False),
    "object_interaction": ("star/Charades_v1_480/", "video", True),
    "object_shuffle": ("perception/videos/", "video", False),
    "moving_direction": ("clevrer/video_validation/", "video", False),
    "action_localization": ("sta/sta_video/", "video", True),
    "scene_transition": ("scene_qa/video/", "video", False),
    "action_count": ("perception/videos/", "video", False),
    "moving_count": ("clevrer/video_validation/", "video", False),
    "moving_attribute": ("clevrer/video_validation/", "video", False),
    "state_change": ("perception/videos/", "video", False),
    "fine_grained_pose": ("nturgbd/", "video", False),  # NTU videos are not in the HF repo
    "character_order": ("perception/videos/", "video", False),
    "egocentric_navigation": ("vlnqa/", "video", False),
    "episodic_reasoning": ("tvqa/frames_fps3_hq/", "frame", True),
    "counterfactual_inference": ("clevrer/video_validation/", "video", False),
}
TVQA_FPS = 3.0

# Zip archives needed per task (perception.zip alone is 11 GB).
TASK_ZIPS = {
    "star": ["action_sequence", "action_prediction", "object_interaction"],
    "ssv2_video": ["action_antonym"],
    "Moments_in_Time_Raw": ["fine_grained_action"],
    "FunQA_test": ["unexpected_action"],
    "clevrer": [
        "object_existence", "moving_direction", "moving_count",
        "moving_attribute", "counterfactual_inference",
    ],
    "perception": ["object_shuffle", "action_count", "state_change", "character_order"],
    "sta": ["action_localization"],
    "scene_qa": ["scene_transition"],
    "vlnqa": ["egocentric_navigation"],
    "tvqa": ["episodic_reasoning"],
}

# 15 tasks that skip perception.zip (11 GB) and the missing NTU videos:
# use when disk or download time is tight (e.g. a Kaggle session).
LITE_TASKS = sorted(
    t for t in TASKS if t not in TASK_ZIPS["perception"] and t != "fine_grained_pose"
)
DEFAULT_TASKS = sorted(t for t in TASKS if t != "fine_grained_pose")


def resolve_tasks(tasks: list[str] | str | None) -> list[str]:
    if tasks is None or tasks == "all" or tasks == ["all"]:
        return DEFAULT_TASKS
    if tasks == "lite" or tasks == ["lite"]:
        return LITE_TASKS
    if isinstance(tasks, str):
        tasks = tasks.split(",")
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"unknown MVBench tasks: {sorted(unknown)}")
    return list(tasks)


def download(root: Path, tasks: list[str] | str | None = None, keep_zips: bool = False) -> None:
    """Fetch the JSON files plus only the video zips the chosen tasks need, then unzip."""
    from huggingface_hub import hf_hub_download, snapshot_download

    tasks = resolve_tasks(tasks)
    out = root / "mvbench"
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=out, allow_patterns=["json/*"])
    zips = sorted({z for z, ts in TASK_ZIPS.items() if set(ts) & set(tasks)})
    vdir = out / "video"
    vdir.mkdir(exist_ok=True)
    for z in zips:
        marker = vdir / f".{z}.done"
        if marker.exists():
            log.info("already extracted %s", z)
            continue
        log.info("downloading %s.zip", z)
        path = hf_hub_download(HF_REPO, f"video/{z}.zip", repo_type="dataset", local_dir=out)
        log.info("extracting %s", path)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(vdir)
        if not keep_zips:
            os.remove(path)  # halve peak disk use
        marker.touch()


def _index_files(vdir: Path) -> dict[str, Path]:
    """basename -> path, a fallback when a zip's folder layout differs from the script's."""
    idx: dict[str, Path] = {}
    for p in vdir.rglob("*"):
        idx.setdefault(p.name, p)
    return idx


@register("mvbench")
def load(root: str | Path | None = None, tasks: list[str] | str | None = None) -> list[Sample]:
    from ..config import data_root

    base = Path(root or data_root()) / "mvbench"
    jdir, vdir = base / "json", base / "video"
    if not jdir.exists():
        raise FileNotFoundError(f"{jdir} not found; run `foveate download mvbench` first")
    index: dict[str, Path] | None = None
    samples: list[Sample] = []
    missing: dict[str, int] = {}
    for task in resolve_tasks(tasks):
        prefix, kind, bound = TASKS[task]
        with open(jdir / f"{task}.json") as f:
            items = json.load(f)
        for i, it in enumerate(items):
            cands = list(it["candidates"])
            if it["answer"] not in cands:
                continue
            letter = LETTERS[cands.index(it["answer"])]
            path = vdir / prefix / it["video"]
            if not path.exists():
                if index is None:
                    index = _index_files(vdir)
                path = index.get(Path(it["video"]).name, path)
            if not path.exists():
                missing[task] = missing.get(task, 0) + 1
                continue
            s = Sample(
                qid=f"{task}/{i:04d}",
                dataset="mvbench",
                task=task,
                question=it["question"],
                answer=letter,
                options=cands,
                start=float(it["start"]) if bound and "start" in it else None,
                end=float(it["end"]) if bound and "end" in it else None,
                clip_id=str(path),
            )
            if kind == "frame":
                frames = sorted(
                    (p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                    key=lambda p: (len(p.stem), p.stem),
                )
                if s.start is not None and s.end is not None:
                    lo, hi = int(s.start * TVQA_FPS), int(s.end * TVQA_FPS) + 1
                    frames = frames[lo:hi] or frames
                s.frame_paths = [str(p) for p in frames]
                s.start = s.end = None
            else:
                s.video = str(path)
            samples.append(s)
    if missing:
        log.warning("MVBench: skipped questions with missing media %s", missing)
    log.info("MVBench: %d questions over %d tasks", len(samples), len({s.task for s in samples}))
    return samples
