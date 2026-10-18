"""Visual-token and FLOP arithmetic for Qwen2.5-VL style models.

Qwen2.5-VL cuts each frame into 14x14 patches, merges every 2x2 block of
patches into one LLM token (so one token covers 28x28 pixels) and, for video,
fuses every 2 consecutive frames into one temporal patch. Everything the
profiler measures should line up with these formulas; when it does not, that
gap is a finding (see theory/01_anatomy_of_vlm_inference.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PATCH = 14
MERGE = 2
FACTOR = PATCH * MERGE  # 28 px per token side
TEMPORAL_PATCH = 2


def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = 4 * FACTOR * FACTOR,
    max_pixels: int = 16384 * FACTOR * FACTOR,
) -> tuple[int, int]:
    """Same rounding rule as Qwen2-VL's image processor.

    Returns (h, w), both multiples of `factor`, with h*w inside
    [min_pixels, max_pixels] and the aspect ratio kept as close as possible.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"bad size {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio too extreme: {height}x{width}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def target_size(height: int, width: int, res: int) -> tuple[int, int]:
    """Size a frame is resized to for a `res` setting.

    `res` is a pixel budget: the frame keeps its aspect ratio and gets about
    res*res pixels, rounded to multiples of 28. res=448 on a 16:9 frame gives
    588x336 (= 21x12 tokens = 252 tokens per frame pair).
    """
    return smart_resize(height, width, max_pixels=res * res, min_pixels=FACTOR * FACTOR)


def tokens_per_frame(height: int, width: int) -> int:
    """LLM tokens for one already-resized frame (or one frame pair in video mode)."""
    if height % FACTOR or width % FACTOR:
        raise ValueError(f"{height}x{width} is not a multiple of {FACTOR}")
    return (height // FACTOR) * (width // FACTOR)


def visual_tokens(height: int, width: int, n_frames: int, mode: str = "video") -> int:
    """Visual tokens the LLM sees for n frames of a resized size.

    mode="video": frames are fused in pairs (odd counts are padded by
    repeating the last frame, as the HF processor does).
    mode="images": every frame is its own image; a single image is internally
    duplicated to fill the temporal patch, so it costs one frame pair.
    """
    per = tokens_per_frame(height, width)
    if mode == "video":
        return math.ceil(n_frames / TEMPORAL_PATCH) * per
    if mode == "images":
        return n_frames * per
    raise ValueError(mode)


def vit_patches(height: int, width: int, n_frames: int, mode: str = "video") -> int:
    """Patches the vision encoder processes (4x the LLM tokens, before merging)."""
    return visual_tokens(height, width, n_frames, mode) * MERGE * MERGE


@dataclass(frozen=True)
class ModelShape:
    """The few numbers needed for back-of-envelope cost models."""

    name: str
    llm_params: float  # language model parameters (incl. embeddings is fine at this precision)
    vit_params: float
    layers: int
    kv_heads: int
    head_dim: int
    weight_bytes: float  # bytes per weight as loaded (2 for fp16, ~0.55 for AWQ int4)


MODEL_SHAPES = {
    "Qwen/Qwen2.5-VL-3B-Instruct": ModelShape(
        "qwen2.5-vl-3b", 3.09e9, 0.67e9, 36, 2, 128, 2.0
    ),
    "Qwen/Qwen2.5-VL-7B-Instruct": ModelShape(
        "qwen2.5-vl-7b", 7.6e9, 0.67e9, 28, 4, 128, 2.0
    ),
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ": ModelShape(
        "qwen2.5-vl-7b-awq", 7.6e9, 0.67e9, 28, 4, 128, 0.55
    ),
}


def shape_for(model_id: str) -> ModelShape | None:
    return MODEL_SHAPES.get(model_id)


def kv_bytes_per_token(shape: ModelShape, dtype_bytes: int = 2) -> int:
    """K and V, for every layer, for every KV head (GQA keeps this small)."""
    return 2 * shape.layers * shape.kv_heads * shape.head_dim * dtype_bytes


def prefill_flops(shape: ModelShape, llm_tokens: int, vit_patch_count: int) -> dict[str, float]:
    """~2 FLOPs per parameter per token for the matmuls (attention FLOPs ignored:
    they are small at these lengths, and Qwen's ViT uses windowed attention)."""
    llm = 2.0 * shape.llm_params * llm_tokens
    vit = 2.0 * shape.vit_params * vit_patch_count
    return {"llm": llm, "vit": vit, "total": llm + vit}


def decode_floor_seconds(shape: ModelShape, mem_bw_gbs: float) -> float:
    """Lower bound per decode step at batch 1: every weight is read once."""
    total_params = shape.llm_params  # the ViT does not run during decode
    return total_params * shape.weight_bytes / (mem_bw_gbs * 1e9)


# Dense fp16/bf16 tensor-core peak and memory bandwidth from vendor spec sheets.
# They are ceilings, not expectations; override with --peak-tflops / --mem-bw.
GPU_SPECS = {
    "T4": {"tflops": 65.0, "mem_bw": 320.0},
    "L4": {"tflops": 121.0, "mem_bw": 300.0},
    "A10G": {"tflops": 70.0, "mem_bw": 600.0},
    "A10": {"tflops": 125.0, "mem_bw": 600.0},
    "RTX 4090": {"tflops": 165.0, "mem_bw": 1008.0},
    "A100": {"tflops": 312.0, "mem_bw": 1555.0},
    "H100": {"tflops": 989.0, "mem_bw": 3350.0},
    "P100": {"tflops": 19.0, "mem_bw": 732.0},
}


def gpu_spec(gpu_name: str | None) -> dict[str, float] | None:
    if not gpu_name:
        return None
    up = gpu_name.upper()
    # Longest key first so "A10G" wins over "A10".
    for key in sorted(GPU_SPECS, key=len, reverse=True):
        if key.upper() in up:
            return GPU_SPECS[key]
    return None
