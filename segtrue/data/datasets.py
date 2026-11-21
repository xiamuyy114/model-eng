"""Unified dataset definitions for SEG-true."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from segtrue.constants import RGB_MEAN, RGB_STD
from segtrue.degrade.registry import apply_degradation
from segtrue.preprocess.registry import build_prior, prior_component_names, validate_preprocess_name


def read_image(path, flags):
    """Decode bytes so Windows paths containing Chinese characters work as in the archive evaluator."""
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), flags)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    image_dir: Path
    mask_dir: Path
    image_prefix: str = "RG"
    mask_prefix: str = "RGMask"
    group_pattern: str = r"^RG\d+"

    def resolve_group(self, image_stem: str) -> str:
        match = re.match(self.group_pattern, image_stem)
        if not match:
            raise ValueError(f"Unable to infer group from image stem: {image_stem}")
        return match.group(0)

    def mask_name_from_image(self, image_name: str) -> str:
        stem = Path(image_name).stem
        suffix = Path(image_name).suffix
        if not stem.startswith(self.image_prefix):
            raise ValueError(f"Image stem does not start with {self.image_prefix}: {stem}")
        return f"{self.mask_prefix}{stem[len(self.image_prefix):]}{suffix}"


@dataclass(frozen=True)
class SampleRecord:
    dataset_key: str
    image_name: str
    image_path: Path
    mask_path: Path
    group_id: str


def load_dataset_spec(payload: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        key=str(payload["dataset_key"]).lower(),
        display_name=str(payload["display_name"]),
        image_dir=Path(payload["image_dir"]),
        mask_dir=Path(payload["mask_dir"]),
        image_prefix=str(payload.get("image_prefix", "RG")),
        mask_prefix=str(payload.get("mask_prefix", "RGMask")),
        group_pattern=str(payload.get("group_pattern", r"^RG\d+")),
    )


def discover_samples(spec: DatasetSpec) -> list[SampleRecord]:
    samples: list[SampleRecord] = []
    for image_path in sorted(spec.image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
            continue
        image_name = image_path.name
        mask_name = spec.mask_name_from_image(image_name)
        mask_path = spec.mask_dir / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_name}: {mask_path}")
        group_id = spec.resolve_group(image_path.stem)
        samples.append(
            SampleRecord(
                dataset_key=spec.key,
                image_name=image_name,
                image_path=image_path,
                mask_path=mask_path,
                group_id=group_id,
            )
        )
    if not samples:
        raise RuntimeError(f"No samples found in {spec.image_dir}")
    return samples


def filter_samples_by_image_names(
    samples: list[SampleRecord],
    image_names: list[str],
) -> list[SampleRecord]:
    image_name_set = set(image_names)
    selected = [sample for sample in samples if sample.image_name in image_name_set]
    if len(selected) != len(image_name_set):
        available = {sample.image_name for sample in samples}
        missing = sorted(image_name_set - available)
        raise KeyError(f"Some image names are not present in discovered samples: {missing[:10]}")
    return selected


class SegmentationDataset(Dataset):
    """Builds RGB-only or RGB+prior tensors from a fixed sample list."""

    def __init__(
        self,
        samples: list[SampleRecord],
        preprocess: str,
        degradation_name: str | None = None,
        degradation_params: dict[str, Any] | None = None,
        degradation_seed: int = 0,
    ) -> None:
        self.samples = samples
        self.preprocess = validate_preprocess_name(preprocess)
        self.degradation_name = degradation_name
        self.degradation_params = degradation_params or {}
        self.degradation_seed = degradation_seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_bgr = read_image(sample.image_path, cv2.IMREAD_COLOR)
        mask_gray = read_image(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {sample.image_path}")
        if mask_gray is None:
            raise FileNotFoundError(f"Failed to read mask: {sample.mask_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.degradation_name is not None:
            image_rgb = apply_degradation(
                image_rgb=image_rgb,
                image_name=sample.image_name,
                name=self.degradation_name,
                params=self.degradation_params,
                base_seed=self.degradation_seed,
            )
        image_tensor = _build_input_tensor(image_rgb, self.preprocess)
        mask_tensor = torch.from_numpy((mask_gray > 127).astype(np.float32)[None, ...])

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_name": sample.image_name,
            "group_id": sample.group_id,
            "dataset_key": sample.dataset_key,
            "degradation_name": self.degradation_name or "clean",
        }


def _normalize_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    mean = np.asarray(RGB_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(RGB_STD, dtype=np.float32).reshape(3, 1, 1)
    image_chw = image.transpose(2, 0, 1)
    return (image_chw - mean) / std


def _build_input_tensor(image_rgb: np.ndarray, preprocess: str) -> torch.Tensor:
    rgb = _normalize_rgb(image_rgb)
    prior_names = prior_component_names(preprocess)
    if not prior_names:
        return torch.from_numpy(rgb.astype(np.float32))
    prior_channels: list[np.ndarray] = []
    for prior_name in prior_names:
        prior = build_prior(image_rgb, prior_name).astype(np.float32)
        if prior.ndim != 2:
            raise ValueError(f"Expected 2D prior map for {prior_name}, got shape {prior.shape}")
        prior_channels.append(prior[None, ...])
    stacked = np.concatenate([rgb, *prior_channels], axis=0)
    return torch.from_numpy(stacked.astype(np.float32))
