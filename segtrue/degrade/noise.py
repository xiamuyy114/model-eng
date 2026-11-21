"""Noise degradations used by evaluation and paper figures."""

from __future__ import annotations

import hashlib

import numpy as np


def stable_rng_seed(image_name: str, name: str, params: dict[str, object], base_seed: int) -> int:
    digest = hashlib.md5(
        f"{base_seed}|{image_name}|{name}|{sorted(params.items())}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def gaussian_noise(image_rgb: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    noisy = np.clip(image + rng.normal(loc=0.0, scale=sigma, size=image.shape), 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)


def salt_pepper_noise(
    image_rgb: np.ndarray,
    amount: float,
    salt_vs_pepper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    image = image_rgb.copy()
    total_pixels = image.shape[0] * image.shape[1]
    num_noisy = int(total_pixels * amount)
    coords = rng.choice(total_pixels, size=num_noisy, replace=False)
    ys = coords // image.shape[1]
    xs = coords % image.shape[1]
    salt_cutoff = int(num_noisy * salt_vs_pepper)
    image[ys[:salt_cutoff], xs[:salt_cutoff]] = 255
    image[ys[salt_cutoff:], xs[salt_cutoff:]] = 0
    return image


def poisson_noise(image_rgb: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    scaled = np.clip(image * max(scale, 1e-6), 0.0, None)
    noisy = rng.poisson(scaled * 255.0) / (255.0 * max(scale, 1e-6))
    return (np.clip(noisy, 0.0, 1.0) * 255.0).astype(np.uint8)
