"""Single preprocessing registry used by the entire project."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .contrast import clahe_prior
from .edge import canny_prior, sobel_prior
from .gray import gray_prior
from .texture import gabor_prior, lbp_prior

PriorFn = Callable[[np.ndarray], np.ndarray]


def rgb_prior(image_rgb: np.ndarray) -> np.ndarray:
    return image_rgb.astype(np.float32) / 255.0


PREPROCESS_REGISTRY: dict[str, PriorFn] = {
    "sobel": sobel_prior,
    "canny": canny_prior,
    "gabor": gabor_prior,
    "clahe": clahe_prior,
    "lbp": lbp_prior,
    "gray": gray_prior,
}

INPUT_CONFIG_REGISTRY: dict[str, tuple[str, ...]] = {
    "rgb": (),
    "sobel": ("sobel",),
    "canny": ("canny",),
    "gabor": ("gabor",),
    "sobel_gabor": ("sobel", "gabor"),
    "clahe": ("clahe",),
    "lbp": ("lbp",),
    "gray": ("gray",),
}


def validate_preprocess_name(name: str) -> str:
    key = name.lower()
    if key not in INPUT_CONFIG_REGISTRY:
        raise KeyError(f"Unsupported preprocess: {name}")
    return key


def build_prior(image_rgb: np.ndarray, name: str) -> np.ndarray:
    key = name.lower()
    if key not in PREPROCESS_REGISTRY:
        raise KeyError(f"Unsupported prior component: {name}")
    return PREPROCESS_REGISTRY[key](image_rgb)


def prior_component_names(name: str) -> tuple[str, ...]:
    key = validate_preprocess_name(name)
    return INPUT_CONFIG_REGISTRY[key]
