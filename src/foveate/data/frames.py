"""Frame sampling, decoding, caching and resizing.

All frame handling lives on the client side (before the model), because frame
count and resolution are the two knobs the project controls. Doing it here
rather than inside the model processor means the Transformers profiler, the
vLLM grid runner and the serving gateway all see byte-identical inputs.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ..tokens import target_size
from . import Sample, stable_hash

log = logging.getLogger(__name__)


def sample_indices(lo: int, hi: int, n: int) -> list[int]:
    """n indices spread uniformly over [lo, hi], taking the middle of each segment.

    Middle-of-segment sampling avoids always picking the very first frame (often
    a fade-in) and matches what most video-QA evaluation code does. If n exceeds
    the number of frames available, indices repeat.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if hi < lo:
        hi = lo
    span = hi - lo + 1
    return [min(hi, lo + int((i + 0.5) * span / n)) for i in range(n)]


@dataclass
class VideoInfo:
    total_frames: int
    fps: float
    height: int
    width: int

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0


def probe_video(path: str) -> VideoInfo:
    import av

    with av.open(path) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate or st.guessed_rate or 25.0)
        n = st.frames
        if not n:
            dur = float(st.duration * st.time_base) if st.duration else float(c.duration or 0) / 1e6
            n = max(1, int(round(dur * fps)))
        return VideoInfo(int(n), fps, st.codec_context.height, st.codec_context.width)


def decode_indices(path: str, wanted: Iterable[int]) -> dict[int, np.ndarray]:
    """Decode the wanted frame indices in one sequential pass (robust for any codec).

    Sequential decoding avoids keyframe-seek inaccuracies across webm/mp4/gif.
    Indices past the real end of the stream (frame counts in headers lie) are
    filled with the last decoded frame.
    """
    import av

    wanted = sorted(set(wanted))
    if not wanted:
        return {}
    want_set, last_needed = set(wanted), wanted[-1]
    out: dict[int, np.ndarray] = {}
    last = None
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(st)):
            if i in want_set:
                out[i] = frame.to_ndarray(format="rgb24")
            last = (i, frame)
            if i >= last_needed:
                break
    if last is None:
        raise RuntimeError(f"no frames decoded from {path}")
    tail = None
    for idx in wanted:
        if idx not in out:
            if tail is None:
                tail = last[1].to_ndarray(format="rgb24") if last[0] not in out else out[last[0]]
            out[idx] = tail
    return out


@dataclass
class FrameBatch:
    frames: list[Image.Image]
    height: int
    width: int
    meta: dict  # fps, total_num_frames, frames_indices, duration (for the processor)
    load_s: float
    stats: dict = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        return np.stack([np.asarray(f) for f in self.frames])


def image_stats(frames: list[Image.Image]) -> dict[str, float]:
    """Cheap, model-free statistics used by the drift monitor (stage 4)."""
    arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.float32) for f in frames]) / 255.0
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    # Variance of a 4-neighbour Laplacian: low for blurry or foggy frames.
    lap = (
        -4 * luma[:, 1:-1, 1:-1]
        + luma[:, :-2, 1:-1] + luma[:, 2:, 1:-1] + luma[:, 1:-1, :-2] + luma[:, 1:-1, 2:]
    )
    mx, mn = arr.max(-1), arr.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    return {
        "brightness": float(luma.mean()),
        "contrast": float(luma.std()),
        "sharpness": float(lap.var() * 1e3),
        "saturation": float(sat.mean()),
    }


class FrameStore:
    """Loads n frames at a resolution for a sample, with an on-disk JPEG cache.

    The first request for a video decodes the union of frame indices for every
    frame count in `frame_counts` in a single pass and caches them at
    `store_res`, so later configs (other n or lower res) never touch the video
    decoder again. On Kaggle's 4 vCPUs this turns decoding from the bottleneck
    into a one-time cost.
    """

    def __init__(
        self,
        cache_dir: str | Path | None,
        frame_counts: Iterable[int] = (2, 4, 8, 16),
        store_res: int = 896,
        jpeg_quality: int = 92,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.frame_counts = sorted(set(frame_counts))
        self.store_res = store_res
        self.q = jpeg_quality
        self._locks: dict[str, threading.Lock] = {}
        self._glock = threading.Lock()

    def _lock(self, key: str) -> threading.Lock:
        with self._glock:
            return self._locks.setdefault(key, threading.Lock())

    # -- raw frames at store resolution ------------------------------------
    def _bounds(self, s: Sample, info: VideoInfo) -> tuple[int, int]:
        lo, hi = 0, info.total_frames - 1
        if s.start is not None:
            lo = max(0, min(hi, int(s.start * info.fps)))
        if s.end is not None:
            hi = max(lo, min(hi, int(s.end * info.fps)))
        return lo, hi

    def _shrink(self, img: Image.Image) -> Image.Image:
        h, w = img.height, img.width
        if h * w <= self.store_res**2:
            return img
        scale = (self.store_res**2 / (h * w)) ** 0.5
        return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)

    def raw_frames(self, s: Sample, n: int) -> tuple[list[Image.Image], dict]:
        if s.frame_paths:
            paths = s.frame_paths
            idx = sample_indices(0, len(paths) - 1, n)
            fps = 3.0 if s.dataset == "mvbench" else 1.0
            imgs = [self._shrink(Image.open(paths[i]).convert("RGB")) for i in idx]
            meta = {
                "fps": fps, "total_num_frames": len(paths),
                "frames_indices": idx, "duration": len(paths) / fps,
            }
            return imgs, meta
        if not s.video:
            raise ValueError(f"sample {s.qid} has no visual input")
        key = hashlib.sha1(s.source_key.encode()).hexdigest()[:20]
        with self._lock(key):
            info = self._info(key, s.video)
            lo, hi = self._bounds(s, info)
            idx = sample_indices(lo, hi, n)
            imgs = self._from_cache(key, idx)
            if imgs is None:
                union = set(idx)
                for k in self.frame_counts:
                    union.update(sample_indices(lo, hi, k))
                decoded = decode_indices(s.video, union)
                for i, a in decoded.items():
                    img = self._shrink(Image.fromarray(a))
                    if self.cache_dir:
                        p = self.cache_dir / key / f"{i:06d}.jpg"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        img.save(p, quality=self.q)
                    decoded[i] = img
                imgs = [decoded[i] for i in idx]
        meta = {
            "fps": info.fps, "total_num_frames": hi - lo + 1,
            "frames_indices": [i - lo for i in idx], "duration": (hi - lo + 1) / info.fps,
        }
        return imgs, meta

    def _info(self, key: str, path: str) -> VideoInfo:
        if self.cache_dir:
            p = self.cache_dir / key / "info.txt"
            if p.exists():
                n, fps, h, w = p.read_text().split()
                return VideoInfo(int(n), float(fps), int(h), int(w))
            info = probe_video(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"{info.total_frames} {info.fps} {info.height} {info.width}")
            return info
        return probe_video(path)

    def _from_cache(self, key: str, idx: list[int]) -> list[Image.Image] | None:
        if not self.cache_dir:
            return None
        paths = [self.cache_dir / key / f"{i:06d}.jpg" for i in idx]
        if not all(p.exists() for p in paths):
            return None
        return [Image.open(p).convert("RGB") for p in paths]

    # -- public -------------------------------------------------------------
    def load(
        self, s: Sample, n: int, res: int, corruption: str | None = None, with_stats: bool = True
    ) -> FrameBatch:
        t0 = time.perf_counter()
        imgs, meta = self.raw_frames(s, n)
        if corruption:
            from .corrupt import apply_corruption

            imgs = [
                apply_corruption(im, corruption, seed=stable_hash(f"{s.qid}:{j}:{corruption}") & 0xFFFFFFFF)
                for j, im in enumerate(imgs)
            ]
        h, w = target_size(imgs[0].height, imgs[0].width, res)
        imgs = [im.resize((w, h), Image.BICUBIC) if im.size != (w, h) else im for im in imgs]
        stats = image_stats(imgs) if with_stats else {}
        return FrameBatch(imgs, h, w, meta, time.perf_counter() - t0, stats)
