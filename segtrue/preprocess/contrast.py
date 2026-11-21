"""Contrast-style preprocessing operators."""

from __future__ import annotations

import cv2
import numpy as np

from .base import rgb_to_gray_u8


def clahe_prior(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    gray = rgb_to_gray_u8(image_rgb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced.astype(np.float32) / 255.0
