"""Deterministic group-aware split manifest helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

from segtrue.utils.io import dump_json

from .datasets import SampleRecord


@dataclass(frozen=True)
class SplitManifest:
    dataset_key: str
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]
    train_images: list[str]
    val_images: list[str]
    test_images: list[str]


def build_group_split_manifest(
    dataset_key: str,
    samples: list[SampleRecord],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitManifest:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")

    groups = sorted({sample.group_id for sample in samples})
    rng = random.Random(seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, round(total * train_ratio))
    val_count = max(1, round(total * val_ratio))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count >= val_count:
            train_count -= 1
        else:
            val_count -= 1

    train_groups = sorted(shuffled[:train_count])
    val_groups = sorted(shuffled[train_count : train_count + val_count])
    test_groups = sorted(shuffled[train_count + val_count :])

    train_images, val_images, test_images = _partition_images(
        samples=samples,
        train_groups=set(train_groups),
        val_groups=set(val_groups),
        test_groups=set(test_groups),
    )

    return SplitManifest(
        dataset_key=dataset_key,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
    )


def save_split_manifest(manifest: SplitManifest, path: Path) -> None:
    dump_json(asdict(manifest), path)


def load_split_manifest(path: Path) -> SplitManifest:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SplitManifest(**payload)


def image_names_for_split(manifest: SplitManifest, split_name: str) -> list[str]:
    key = split_name.lower()
    if key == "train":
        return manifest.train_images
    if key == "val":
        return manifest.val_images
    if key == "test":
        return manifest.test_images
    raise KeyError(f"Unsupported split name: {split_name}")


def _partition_images(
    samples: list[SampleRecord],
    train_groups: set[str],
    val_groups: set[str],
    test_groups: set[str],
) -> tuple[list[str], list[str], list[str]]:
    train_images: list[str] = []
    val_images: list[str] = []
    test_images: list[str] = []

    for sample in samples:
        if sample.group_id in train_groups:
            train_images.append(sample.image_name)
        elif sample.group_id in val_groups:
            val_images.append(sample.image_name)
        elif sample.group_id in test_groups:
            test_images.append(sample.image_name)
        else:
            raise RuntimeError(f"Sample group not assigned: {sample.group_id}")

    return sorted(train_images), sorted(val_images), sorted(test_images)
