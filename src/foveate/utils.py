"""Small shared helpers: JSONL I/O, environment capture, logging."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("foveate")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def write_jsonl(path: str | Path, rows: Iterable[dict], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def capture_env() -> dict[str, Any]:
    """Everything needed to reproduce a number: GPU, driver, library versions."""
    env: dict[str, Any] = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
        "packages": {
            p: _pkg_version(p)
            for p in ["torch", "transformers", "vllm", "foveate", "av", "numpy", "accelerate"]
        },
    }
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["torch_cuda"] = torch.version.cuda
            env["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                    "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        env["cuda_available"] = False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            env["driver"] = out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return env


def gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def default_dtype() -> str:
    """bf16 where the GPU supports it (Ampere+), fp16 otherwise (T4, V100)."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            return "bfloat16"
    except ImportError:
        pass
    return "float16"


def gpu_capability() -> tuple[int, int] | None:
    try:
        import torch

        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability(0))
    except ImportError:
        pass
    return None


def resolve_dtype(requested: str = "auto") -> str:
    """The dtype to run in, never silently bf16 on a GPU that lacks it.

    "auto" -> bfloat16 on compute capability >= 8.0 (Ampere+), float16 below
    (T4 is 7.5). An explicit bfloat16 request on a pre-Ampere GPU is an error
    rather than a silent emulation: T4 has no bf16 tensor cores.
    """
    aliases = {"half": "float16", "fp16": "float16", "bf16": "bfloat16", "fp32": "float32",
               "float": "float32"}
    req = aliases.get(requested, requested)
    if req == "auto":
        return default_dtype()
    if req not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype {requested!r}; use auto, float16, bfloat16 or float32")
    cap = gpu_capability()
    if req == "bfloat16" and cap is not None and cap[0] < 8:
        raise ValueError(f"bfloat16 requested but this GPU has compute capability "
                         f"{cap[0]}.{cap[1]} (T4 = 7.5) without bf16 support; use float16")
    return req


def percentile(values: list[float], q: float) -> float:
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")
