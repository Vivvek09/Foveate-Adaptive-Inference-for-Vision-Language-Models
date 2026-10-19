"""Synthetic distribution shift for driving-style footage: fog, night, motion blur.

Modelled loosely on ImageNet-C (Hendrycks & Dietterich, 2019): each corruption
has severities 1..5. They are deliberately simple and deterministic (seeded per
frame) so a shifted run can be reproduced exactly.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

CORRUPTIONS = ("fog", "night", "blur")


def _fog(a: np.ndarray, sev: int, rng: np.random.Generator) -> np.ndarray:
    # Atmospheric scattering model: I = J*t + A*(1-t), with transmission t
    # falling towards the top of the frame (farther scene = more haze).
    h, w, _ = a.shape
    t_min = [0.75, 0.6, 0.45, 0.32, 0.2][sev - 1]
    depth = np.linspace(1.0, 0.25, h)[:, None]  # top rows ~far
    t = t_min + (1 - t_min) * (1 - depth)
    # Low-frequency density noise so the haze is not a perfect gradient.
    noise = rng.normal(0, 1, (max(2, h // 32), max(2, w // 32))).astype(np.float32)
    noise = np.asarray(Image.fromarray(noise).resize((w, h), Image.BICUBIC))
    t = np.clip(t + 0.05 * noise, 0.05, 1.0)[..., None]
    airlight = 0.85
    return a * t + airlight * (1 - t)


def _night(a: np.ndarray, sev: int, rng: np.random.Generator) -> np.ndarray:
    # Gamma darkening, a blue cast, and sensor noise that grows as light drops.
    gamma = [1.6, 2.0, 2.5, 3.0, 3.6][sev - 1]
    gain = [0.75, 0.6, 0.5, 0.4, 0.3][sev - 1]
    out = gain * np.power(a, gamma)
    out = out * np.array([0.85, 0.9, 1.1], dtype=np.float32)
    sigma = [0.01, 0.02, 0.03, 0.04, 0.05][sev - 1]
    return out + rng.normal(0, sigma, a.shape).astype(np.float32)


def _blur(img: Image.Image, sev: int, rng: np.random.Generator) -> np.ndarray:
    # Horizontal motion blur (camera pan / ego-motion) via a box filter along x,
    # with length relative to frame width so severity is resolution-independent.
    a = np.asarray(img, dtype=np.float32) / 255.0
    frac = [0.01, 0.02, 0.035, 0.05, 0.07][sev - 1]
    k = max(3, int(round(frac * a.shape[1])) | 1)
    pad = k // 2
    p = np.pad(a, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(p, axis=1, dtype=np.float64)
    c = np.concatenate([np.zeros_like(c[:, :1]), c], axis=1)
    out = (c[:, k:] - c[:, :-k]) / k
    # A touch of defocus as well.
    out_img = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    out_img = out_img.filter(ImageFilter.GaussianBlur(radius=0.3 * sev))
    return np.asarray(out_img, dtype=np.float32) / 255.0


def parse(spec: str) -> tuple[str, int]:
    """'fog:3' -> ('fog', 3); 'fog' -> ('fog', 3)."""
    name, _, sev = spec.partition(":")
    sev_i = int(sev) if sev else 3
    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {name!r}; choose from {CORRUPTIONS}")
    if not 1 <= sev_i <= 5:
        raise ValueError("severity must be 1..5")
    return name, sev_i


def apply_corruption(img: Image.Image, spec: str, seed: int = 0) -> Image.Image:
    name, sev = parse(spec)
    rng = np.random.default_rng(seed)
    if name == "blur":
        out = _blur(img.convert("RGB"), sev, rng)
    else:
        a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        out = _fog(a, sev, rng) if name == "fog" else _night(a, sev, rng)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
