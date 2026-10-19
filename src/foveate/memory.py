"""Will this model + visual config fit on one GPU? Decide before loading, and record why not.

Each process owns exactly one GPU (CUDA_VISIBLE_DEVICES), so every check is
against a single device's memory: the two T4s of a Kaggle T4 x2 box are two
separate 15 GiB pools, not one 30 GiB pool.

The estimate is deliberately simple and conservative:

    weights            checkpoint bytes on the GPU (AWQ keeps ViT/embeddings fp16)
  + KV cache           kv_bytes_per_token x max_model_len (vLLM must hold one full sequence)
  + activations        ViT + LLM prefill working set for the worst-case input of the config
  + runtime overhead   CUDA context, cuBLAS workspace, CUDA graphs, allocator slack

and the context-length check is exact: the prompt of the largest frame size a
config can produce must fit in max_model_len. A config that fails either check
is skipped with the reason written to `skipped.jsonl`, never replaced by a
different config. The estimate is a guard, not a measurement: the smoke test
and the profiler record the real peak memory.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import VisualConfig
from .tokens import (
    FACTOR,
    MERGE,
    TEMPORAL_PATCH,
    kv_bytes_per_token,
    shape_for,
    weight_bytes_total,
)

log = logging.getLogger(__name__)

GIB = 2**30
T4_TOTAL_GIB = 14.6  # what torch reports for a 16 GB T4 (ECC on)
TEXT_TOKENS = 320  # system prompt + question + 4-8 options + chat template, generous
RUNTIME_OVERHEAD_GIB = 1.2  # CUDA context + cuBLAS workspace + CUDA graphs + slack
VIT_HIDDEN, VIT_MLP = 1280, 3420
LLM_MLP = {2048: 11008, 3584: 18944}


@dataclass
class FitCheck:
    config: str
    model_id: str
    backend: str
    dtype: str
    ok: bool
    reason: str
    max_visual_tokens: int
    max_prompt_tokens: int
    weights_gib: float
    kv_gib: float
    activations_gib: float
    need_gib: float
    budget_gib: float


def worst_case_visual_tokens(cfg: VisualConfig) -> int:
    """Upper bound on visual tokens for any aspect ratio at this res.

    target_size caps h*w at res^2 pixels, so tokens per frame pair <= res^2 / 28^2
    (rounded up for smart_resize's rounding to multiples of 28).
    """
    per_pair = math.ceil(cfg.res * cfg.res / (FACTOR * FACTOR)) + 2 * math.ceil(cfg.res / FACTOR)
    return math.ceil(cfg.frames / TEMPORAL_PATCH) * per_pair


def gpu_total_gib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / GIB
    except ImportError:
        pass
    return None


def check_fit(
    cfg: VisualConfig,
    model_id: str,
    *,
    backend: str,
    dtype: str = "float16",
    max_model_len: int = 8192,
    gpu_mem_util: float = 0.90,
    total_gib: float | None = None,
    batch_size: int = 1,
) -> FitCheck:
    shape = shape_for(model_id)
    vis = worst_case_visual_tokens(cfg)
    prompt = vis + TEXT_TOKENS
    total = total_gib or gpu_total_gib() or T4_TOTAL_GIB
    dbytes = 4.0 if dtype == "float32" else 2.0
    if shape is None:
        return FitCheck(cfg.name, model_id, backend, dtype, True, "unknown model shape: not checked",
                        vis, prompt, 0, 0, 0, 0, total)
    weights = weight_bytes_total(shape, dbytes) / GIB
    kv_tok = kv_bytes_per_token(shape, 2) / GIB
    patches = vis * MERGE * MERGE
    vit_act = patches * (2 * VIT_MLP + 4 * VIT_HIDDEN) * dbytes / GIB
    mlp = LLM_MLP.get(shape.hidden, 4 * shape.hidden)
    llm_act = prompt * (2 * mlp + 4 * shape.hidden) * dbytes / GIB
    if backend == "vllm":
        # vLLM pre-allocates its KV pool from what is left after profiling; it
        # needs room for at least one full max_model_len sequence.
        kv = kv_tok * max_model_len
        budget = total * gpu_mem_util
        act = max(vit_act, llm_act)
    else:  # Transformers, batch `batch_size`, KV only for the actual prompt
        kv = kv_tok * (prompt + 64) * batch_size
        budget = total
        act = (vit_act + llm_act) * batch_size
    need = weights + kv + act + RUNTIME_OVERHEAD_GIB
    reasons = []
    if backend == "vllm" and prompt > max_model_len:
        reasons.append(f"worst-case prompt {prompt} tokens > max_model_len {max_model_len}")
    if need > budget:
        reasons.append(f"needs ~{need:.1f} GiB > {budget:.1f} GiB budget on one GPU "
                       f"(weights {weights:.1f}, KV {kv:.1f}, activations {act:.1f})")
    return FitCheck(cfg.name, model_id, backend, dtype, not reasons, "; ".join(reasons) or "fits",
                    vis, prompt, round(weights, 2), round(kv, 3), round(act, 3), round(need, 2),
                    round(budget, 2))


def record_skips(run_dir: Path, checks: list[FitCheck], extra: dict | None = None) -> None:
    """Append skipped configs (with reasons) to <run_dir>/skipped.jsonl, deduplicated."""
    bad = [c for c in checks if not c.ok]
    if not bad:
        return
    path = run_dir / "skipped.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            seen.add((r["config"], r["backend"], r["dtype"], r.get("corruption")))
    with path.open("a") as f:
        for c in bad:
            rec = asdict(c) | (extra or {})
            key = (rec["config"], rec["backend"], rec["dtype"], rec.get("corruption"))
            if key not in seen:
                f.write(json.dumps(rec) + "\n")
                seen.add(key)
    for c in bad:
        log.warning("SKIPPED %s on %s (%s): %s", c.config, c.backend, c.dtype, c.reason)


def record_skip(run_dir: Path, config: str, model_id: str, backend: str, dtype: str,
                reason: str, **extra) -> None:
    """Skip a config for a reason other than memory (e.g. AWQ without vLLM)."""
    record_skips(run_dir, [FitCheck(config, model_id, backend, dtype, False, reason,
                                    0, 0, 0, 0, 0, 0, 0)], extra)
