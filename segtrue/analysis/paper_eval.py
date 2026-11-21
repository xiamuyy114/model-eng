"""Shared evaluation core: per-image IoU metrics (identical formulas to SEG-true
engine/metrics.py) and grain statistics over the sensitivity grid.

Inputs for one (dataset, seed, condition) are built once and shared by every
model of that seed, so the expensive CPU work (degradation, priors) is done
once instead of once per model.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import torch

from segtrue.data.datasets import _normalize_rgb
from segtrue.degrade.registry import DEGRADATION_LEVELS, apply_degradation
from segtrue.preprocess.registry import build_prior, prior_component_names

EPS = 1e-7
PLANES = ("sobel", "canny", "gabor", "clahe", "lbp", "gray")  # planes 3..8 after the three RGB planes
CONDITIONS = [("clean", "clean", 0, {})] + [
    (spec.name, spec.label, level, spec.params)
    for name, levels in DEGRADATION_LEVELS.items() for level, spec in enumerate(levels, 1)
]
THRESHOLDS = (0.3, 0.5, 0.7)
MIN_AREAS = (20, 50, 100)
BORDER_OPTIONS = (False, True)
DEFAULT_RULE = (0.5, 50, False)


def unicode_imread(filename, flags=cv2.IMREAD_COLOR):
    with open(filename, "rb") as handle:
        encoded = np.frombuffer(handle.read(), dtype=np.uint8)
    return cv2.imdecode(encoded, flags)


def scheme_channel_indices(scheme: str) -> list[int]:
    return [0, 1, 2] + [3 + PLANES.index(name) for name in prior_component_names(scheme)]


def build_planes(image_rgb: np.ndarray) -> np.ndarray:
    """(9, H, W) float32: ImageNet-normalised RGB followed by the six prior planes."""
    rgb = _normalize_rgb(image_rgb)
    priors = [build_prior(image_rgb, name).astype(np.float32)[None] for name in PLANES]
    return np.concatenate([rgb.astype(np.float32), *priors], axis=0)


class SeedInputs:
    """Clean test images and masks of one (dataset, seed); degraded planes built on demand."""

    def __init__(self, samples, workers: int = 4):
        self.samples = samples
        self.names = [s.image_name for s in samples]
        self.rgb = []
        self.masks = []
        for sample in samples:
            bgr = unicode_imread(str(sample.image_path), cv2.IMREAD_COLOR)
            mask = unicode_imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None or mask is None:
                raise FileNotFoundError(sample.image_name)
            self.rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            self.masks.append((mask > 127).astype(np.uint8))
        self.mask_tensor = torch.from_numpy(np.stack(self.masks).astype(np.float32)[:, None])
        self.gt_components = [connected_components(m) for m in self.masks]
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def condition_planes(self, name: str, params: dict, base_seed: int = 0) -> torch.Tensor:
        def one(index):
            image = self.rgb[index]
            if name != "clean":
                image = apply_degradation(image_rgb=image, image_name=self.names[index], name=name,
                                          params=params, base_seed=base_seed)
            return build_planes(image)
        planes = list(self.pool.map(one, range(len(self.rgb))))
        return torch.from_numpy(np.stack(planes))

    def close(self):
        self.pool.shutdown(wait=True)


def connected_components(mask_u8: np.ndarray):
    """Stats table of 8-connected components of an interior-positive mask (background row removed)."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8.astype(np.uint8), connectivity=8)
    return stats[1:count] if count > 1 else stats[:0]


def grain_stats(stats: np.ndarray, height: int, width: int, min_area: int, exclude_border: bool):
    """(num_grains, mean_area, mean_diameter) with the SEG-true analyze_grains rule plus optional border exclusion."""
    if len(stats) == 0:
        return 0, 0.0, 0.0
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    if exclude_border:
        x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
        w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
        touching = (x == 0) | (y == 0) | (x + w == width) | (y + h == height)
        keep &= ~touching
    areas = stats[keep, cv2.CC_STAT_AREA].astype(np.float64)
    if areas.size == 0:
        return 0, 0.0, 0.0
    diameters = 2.0 * np.sqrt(areas / np.pi)
    return int(areas.size), float(areas.mean()), float(diameters.mean())


def binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-image dice/iou exactly as SEG-true _compute_binary_metrics (float32 sums, float64 arithmetic)."""
    dims = (1, 2, 3)
    tp = (pred * target).sum(dims).double().cpu().numpy()
    fp = (pred * (1.0 - target)).sum(dims).double().cpu().numpy()
    fn = ((1.0 - pred) * target).sum(dims).double().cpu().numpy()
    union = tp + fp + fn
    has = union > 0.0
    iou = np.where(has, (tp + EPS) / (union + EPS), 1.0)
    dice = np.where(has, (2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS), 1.0)
    return {"iou": iou, "dice": dice}


@torch.no_grad()
def evaluate_planes(model, planes: torch.Tensor, channel_idx: list[int], masks: torch.Tensor,
                    gt_components, device, batch_size: int = 8, mask_output_dir=None, image_names=None):
    """Run one model over one condition. Returns per-image dicts with GI/GB IoU, Dice and the 18-rule grain stats."""
    model.eval()
    height, width = masks.shape[-2:]
    rows = []
    for start in range(0, planes.shape[0], batch_size):
        images = planes[start:start + batch_size][:, channel_idx].to(device, non_blocking=False)
        target = masks[start:start + batch_size].to(device)
        probs = torch.sigmoid(model(images).float())
        interior = (probs >= 0.5).float()
        gi = binary_metrics(interior, target)
        gb = binary_metrics(1.0 - interior, 1.0 - target)
        pred_masks = {t: (probs >= t).to(torch.uint8).cpu().numpy()[:, 0] for t in THRESHOLDS}
        for j in range(images.shape[0]):
            index = start + j
            if mask_output_dir is not None:
                folder = Path(mask_output_dir)
                folder.mkdir(parents=True, exist_ok=True)
                name = Path(image_names[index]).stem if image_names is not None else f"{index:05d}"
                Image.fromarray(pred_masks[0.5][j] * 255).save(folder / (name + ".png"))
            row = {"GI_IoU": float(gi["iou"][j]), "GB_IoU": float(gb["iou"][j]),
                   "GI_Dice": float(gi["dice"][j]), "GB_Dice": float(gb["dice"][j]), "grain": {}}
            gt_stats = gt_components[index]
            for t in THRESHOLDS:
                pred_stats = connected_components(pred_masks[t][j])
                for a in MIN_AREAS:
                    for border in BORDER_OPTIONS:
                        n_p, a_p, d_p = grain_stats(pred_stats, height, width, a, border)
                        n_g, a_g, d_g = grain_stats(gt_stats, height, width, a, border)
                        row["grain"][(t, a, border)] = (n_p, n_g, a_p, a_g, d_p, d_g)
            rows.append(row)
    return rows


def rule_key(threshold: float, min_area: int, exclude_border: bool) -> str:
    return f"t{threshold:.1f}_a{min_area}_b{int(exclude_border)}"
