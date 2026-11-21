"""Single degradation registry used by SEG-true."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .blur import gaussian_blur
from .contrast import brightness_contrast
from .noise import gaussian_noise, poisson_noise, salt_pepper_noise, stable_rng_seed


@dataclass(frozen=True)
class DegradationSpec:
    name: str
    label: str
    params: dict[str, float]


DEGRADATION_LEVELS: dict[str, list[DegradationSpec]] = {
    "gaussian": [
        DegradationSpec("gaussian", "gaussian_l1", {"sigma": 10.0 / 255.0}),
        DegradationSpec("gaussian", "gaussian_l2", {"sigma": 30.0 / 255.0}),
        DegradationSpec("gaussian", "gaussian_l3", {"sigma": 50.0 / 255.0}),
        DegradationSpec("gaussian", "gaussian_l4", {"sigma": 70.0 / 255.0}),
    ],
    "salt_pepper": [
        DegradationSpec("salt_pepper", "sp_l1", {"amount": 0.01, "salt_vs_pepper": 0.5}),
        DegradationSpec("salt_pepper", "sp_l2", {"amount": 0.03, "salt_vs_pepper": 0.5}),
        DegradationSpec("salt_pepper", "sp_l3", {"amount": 0.05, "salt_vs_pepper": 0.5}),
        DegradationSpec("salt_pepper", "sp_l4", {"amount": 0.10, "salt_vs_pepper": 0.5}),
    ],
    "poisson": [
        # In this parameterization, smaller scale produces stronger shot noise.
        DegradationSpec("poisson", "poisson_l1", {"scale": 8.0}),
        DegradationSpec("poisson", "poisson_l2", {"scale": 5.0}),
        DegradationSpec("poisson", "poisson_l3", {"scale": 3.0}),
        DegradationSpec("poisson", "poisson_l4", {"scale": 1.0}),
    ],
    "blur": [
        DegradationSpec("blur", "blur_l1", {"kernel_size": 3}),
        DegradationSpec("blur", "blur_l2", {"kernel_size": 5}),
        DegradationSpec("blur", "blur_l3", {"kernel_size": 7}),
        DegradationSpec("blur", "blur_l4", {"kernel_size": 9}),
    ],
    "brightness_contrast": [
        DegradationSpec("brightness_contrast", "bc_l1", {"brightness": 0.1, "contrast": 0.1}),
        DegradationSpec("brightness_contrast", "bc_l2", {"brightness": 0.2, "contrast": 0.2}),
        DegradationSpec("brightness_contrast", "bc_l3", {"brightness": 0.3, "contrast": 0.3}),
        DegradationSpec("brightness_contrast", "bc_l4", {"brightness": 0.5, "contrast": 0.3}),
    ],
}


def apply_degradation(
    image_rgb: np.ndarray,
    image_name: str,
    name: str,
    params: dict[str, float],
    base_seed: int = 0,
) -> np.ndarray:
    key = name.lower()
    seed = stable_rng_seed(image_name=image_name, name=key, params=params, base_seed=base_seed)
    rng = np.random.default_rng(seed)
    if key == "gaussian":
        return gaussian_noise(image_rgb, sigma=float(params["sigma"]), rng=rng)
    if key == "salt_pepper":
        return salt_pepper_noise(
            image_rgb,
            amount=float(params["amount"]),
            salt_vs_pepper=float(params.get("salt_vs_pepper", 0.5)),
            rng=rng,
        )
    if key == "poisson":
        return poisson_noise(image_rgb, scale=float(params["scale"]), rng=rng)
    if key == "blur":
        return gaussian_blur(image_rgb, kernel_size=int(params["kernel_size"]))
    if key == "brightness_contrast":
        return brightness_contrast(
            image_rgb,
            brightness=float(params["brightness"]),
            contrast=float(params["contrast"]),
        )
    raise KeyError(f"Unsupported degradation: {name}")
