"""Brightness and contrast degradations."""

from __future__ import annotations

import cv2
import numpy as np


def brightness_contrast(image_rgb: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    alpha = 1.0 + contrast
    beta = brightness * 255.0
    adjusted = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)
    return adjusted.astype(np.uint8)
