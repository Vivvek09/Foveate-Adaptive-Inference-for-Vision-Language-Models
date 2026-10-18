"""Experiment configuration: model aliases, visual configs and paths."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"

_SPEC = re.compile(r"^(?:(?P<model>[\w.\-]+):)?(?P<frames>\d+)x(?P<res>\d+)$")


@dataclass(frozen=True, order=True)
class VisualConfig:
    """One point in the compute space: which model, how many frames, what resolution."""

    model: str
    frames: int
    res: int

    @property
    def name(self) -> str:
        return f"{self.model}:{self.frames}x{self.res}"

    @classmethod
    def parse(cls, spec: str, default_model: str | None = None) -> "VisualConfig":
        m = _SPEC.match(spec.strip())
        if not m:
            raise ValueError(f"bad config spec {spec!r}; expected MODEL:FRAMESxRES like 3b:8x448")
        model = m.group("model") or default_model
        if model is None:
            raise ValueError(f"config {spec!r} needs a model prefix")
        return cls(model, int(m.group("frames")), int(m.group("res")))

    def __str__(self) -> str:  # pragma: no cover
        return self.name


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def model_id(cfg: dict, alias: str) -> str:
    if alias in cfg["models"]:
        return cfg["models"][alias]["id"]
    return alias  # allow raw HF ids


def model_quantization(cfg: dict, alias: str) -> str | None:
    return cfg["models"].get(alias, {}).get("quantization")


def grid_configs(cfg: dict, dataset: str, model_alias: str) -> list[VisualConfig]:
    specs = cfg["grid"][dataset][model_alias]
    return [VisualConfig.parse(s, default_model=model_alias) for s in specs]


def ladder(cfg: dict, dataset: str) -> list[VisualConfig]:
    return [VisualConfig.parse(s) for s in cfg["ladder"][dataset]]


def data_root() -> Path:
    return Path(os.environ.get("FOVEATE_DATA", "data")).expanduser()


def results_root() -> Path:
    return Path(os.environ.get("FOVEATE_RESULTS", "results")).expanduser()


def cache_root() -> Path:
    return Path(os.environ.get("FOVEATE_CACHE", str(data_root() / "frame_cache"))).expanduser()
