"""A tiny generated dataset: moving coloured squares on a road-grey background.

Used by the stage-1 profiler (compute cost does not depend on pixel content,
so you can profile before any dataset is downloaded) and by CPU tests that run
the whole pipeline with a fake backend.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import LETTERS, Sample, register

COLORS = {"red": (220, 40, 40), "green": (40, 180, 70), "blue": (40, 90, 220), "yellow": (230, 210, 40)}


def generate(root: Path, n_clips: int = 12, n_frames: int = 32, size=(1280, 720), seed: int = 0):
    rng = random.Random(seed)
    out = root / "synthetic"
    items = []
    for c in range(n_clips):
        clip = out / f"clip{c:03d}"
        clip.mkdir(parents=True, exist_ok=True)
        color = rng.choice(list(COLORS))
        direction = rng.choice(["left", "right"])
        for f in range(n_frames):
            p = clip / f"{f:04d}.jpg"
            if not p.exists():
                img = Image.new("RGB", size, (90, 90, 95))
                d = ImageDraw.Draw(img)
                d.rectangle([0, size[1] * 0.55, size[0], size[1] * 0.6], fill=(230, 230, 230))
                x = int((f / (n_frames - 1)) * (size[0] - 200))
                if direction == "left":
                    x = size[0] - 200 - x
                d.rectangle([x, 300, x + 160, 420], fill=COLORS[color])
                arr = np.asarray(img).astype(np.int16)
                arr += np.random.default_rng(c * 1000 + f).integers(-6, 7, arr.shape, dtype=np.int16)
                Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(p, quality=90)
        _write_mp4(clip)
        items.append((clip, color, direction))
    return items


def _write_mp4(clip: Path, fps: int = 8) -> Path | None:
    """Also encode the frames as clip.mp4 so the smoke test exercises real video
    decoding (PyAV) without downloading anything. Skipped if PyAV is missing."""
    out = clip / "clip.mp4"
    if out.exists():
        return out
    try:
        import av
    except ImportError:
        return None
    frames = sorted(clip.glob("*.jpg"))
    tmp = clip / "clip.tmp.mp4"
    with av.open(str(tmp), "w") as c:
        st = c.add_stream("mpeg4", rate=fps)
        w, h = Image.open(frames[0]).size
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.bit_rate = 4_000_000
        for p in frames:
            img = np.asarray(Image.open(p).convert("RGB"))
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    tmp.rename(out)
    return out


@register("synthetic")
def load(root: str | Path | None = None, tasks=None) -> list[Sample]:
    from ..config import data_root

    root = Path(root or data_root())
    samples = []
    for clip, color, direction in generate(root):
        frames = sorted(str(p) for p in clip.glob("*.jpg"))
        mp4 = clip / "clip.mp4"
        # Prefer the encoded video (real decode path); fall back to the JPEG frames.
        media = {"video": str(mp4)} if mp4.exists() else {"frame_paths": frames}
        colors = sorted(COLORS)
        samples.append(Sample(
            qid=f"color/{clip.name}", dataset="synthetic", task="color",
            question="What color is the moving block?",
            options=colors, answer=LETTERS[colors.index(color)],
            clip_id=clip.name, **media,
        ))
        dirs = ["left", "right"]
        samples.append(Sample(
            qid=f"direction/{clip.name}", dataset="synthetic", task="direction",
            question="Which direction is the block moving?",
            options=dirs, answer=LETTERS[dirs.index(direction)],
            clip_id=clip.name, **media,
        ))
    return samples
