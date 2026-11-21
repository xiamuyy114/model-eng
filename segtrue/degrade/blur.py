"""Blur degradations."""

from __future__ import annotations

import cv2
import numpy as np


def gaussian_blur(image_rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    return cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), sigmaX=0)
