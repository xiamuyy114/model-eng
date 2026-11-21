"""Base preprocessing utilities."""

from __future__ import annotations

import numpy as np


def rgb_to_gray_u8(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    gray = (
        0.299 * image_rgb[..., 0]
        + 0.587 * image_rgb[..., 1]
        + 0.114 * image_rgb[..., 2]
    )
    return np.clip(gray, 0, 255).astype(np.uint8)


def minmax_to_unit(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    low = float(image.min())
    high = float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return (image - low) / (high - low)
