"""Edge-style preprocessing operators."""

from __future__ import annotations

import cv2
import numpy as np

from .base import minmax_to_unit, rgb_to_gray_u8


def sobel_prior(image_rgb: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = rgb_to_gray_u8(image_rgb)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return minmax_to_unit(magnitude)


def canny_prior(
    image_rgb: np.ndarray,
    threshold1: int = 100,
    threshold2: int = 200,
) -> np.ndarray:
    gray = rgb_to_gray_u8(image_rgb)
    edges = cv2.Canny(gray, threshold1=threshold1, threshold2=threshold2)
    return (edges.astype(np.float32) / 255.0).clip(0.0, 1.0)
