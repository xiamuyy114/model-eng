"""Texture-oriented preprocessing operators."""

from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from .base import minmax_to_unit, rgb_to_gray_u8


def gabor_prior(
    image_rgb: np.ndarray,
    kernel_size: int = 9,
    sigma: float = 2.0,
    lambd: float = 5.0,
    gamma: float = 0.5,
    psi: float = 0.0,
    thetas: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
) -> np.ndarray:
    gray = rgb_to_gray_u8(image_rgb).astype(np.float32) / 255.0
    responses: list[np.ndarray] = []
    for theta_deg in thetas:
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma,
            math.radians(theta_deg),
            lambd,
            gamma,
            psi,
            ktype=cv2.CV_32F,
        )
        response = cv2.filter2D(gray, cv2.CV_32F, kernel)
        responses.append(np.abs(response))
    return minmax_to_unit(np.max(np.stack(responses, axis=0), axis=0))


def lbp_prior(
    image_rgb: np.ndarray,
    points: int = 8,
    radius: int = 1,
    method: str = "uniform",
) -> np.ndarray:
    gray = rgb_to_gray_u8(image_rgb)
    lbp = local_binary_pattern(gray, P=points, R=radius, method=method)
    return minmax_to_unit(lbp)
