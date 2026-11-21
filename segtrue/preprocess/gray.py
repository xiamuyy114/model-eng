"""Equal-channel control: one grayscale channel that is a linear combination of RGB."""

from __future__ import annotations

import numpy as np

from .base import rgb_to_gray_u8


def gray_prior(image_rgb: np.ndarray) -> np.ndarray:
    """0.299R + 0.587G + 0.114B (same conversion the other priors use), scaled to [0, 1]."""
    return rgb_to_gray_u8(image_rgb).astype(np.float32) / 255.0
