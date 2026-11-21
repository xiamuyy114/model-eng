"""Unified grain-analysis helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from segtrue.data.datasets import SampleRecord, SegmentationDataset
from segtrue.data.loaders import LoaderConfig, build_dataloader
from segtrue.degrade.registry import DEGRADATION_LEVELS


DEFAULT_MIN_GRAIN_AREA = 50


@dataclass(frozen=True)
class GrainStats:
    num_grains: int
    mean_area: float
    median_area: float
    std_area: float
    mean_diameter: float


def analyze_grains(interior_mask: np.ndarray, min_grain_area: int = DEFAULT_MIN_GRAIN_AREA) -> GrainStats:
    """Estimate grain components from an interior-positive binary mask."""
    grain_region = interior_mask.astype(np.uint8).clip(0, 1)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(grain_region, connectivity=8)
    areas: list[int] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_grain_area:
            areas.append(area)
    if not areas:
        return GrainStats(0, 0.0, 0.0, 0.0, 0.0)

    areas_array = np.asarray(areas, dtype=np.float64)
    diameters = 2.0 * np.sqrt(areas_array / np.pi)
    return GrainStats(
        num_grains=len(areas),
        mean_area=float(areas_array.mean()),
        median_area=float(np.median(areas_array)),
        std_area=float(areas_array.std()),
        mean_diameter=float(diameters.mean()),
    )


@torch.no_grad()
def collect_grain_pairs(
    model: torch.nn.Module,
    dataset: SegmentationDataset,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 4,
    min_grain_area: int = DEFAULT_MIN_GRAIN_AREA,
) -> list[tuple[GrainStats, GrainStats]]:
    loader = build_dataloader(
        dataset,
        LoaderConfig(batch_size=batch_size, shuffle=False, drop_last=False),
    )
    model.eval()
    pairs: list[tuple[GrainStats, GrainStats]] = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"]
        logits = model(images)
        preds = (torch.sigmoid(logits).cpu() >= threshold).float()
        for pred_mask, gt_mask in zip(preds[:, 0], masks[:, 0]):
            gt_stats = analyze_grains(gt_mask.numpy().astype(np.uint8), min_grain_area=min_grain_area)
            pred_stats = analyze_grains(pred_mask.numpy().astype(np.uint8), min_grain_area=min_grain_area)
            pairs.append((gt_stats, pred_stats))
    return pairs


def aggregate_grain_metrics(pairs: list[tuple[GrainStats, GrainStats]]) -> dict[str, float] | None:
    if not pairs:
        return None

    gt_counts: list[float] = []
    pred_counts: list[float] = []
    gt_areas: list[float] = []
    pred_areas: list[float] = []
    gt_diams: list[float] = []
    pred_diams: list[float] = []
    count_errs: list[float] = []
    area_errs: list[float] = []
    diam_errs: list[float] = []

    for gt_stats, pred_stats in pairs:
        if gt_stats.num_grains == 0:
            continue
        gt_counts.append(gt_stats.num_grains)
        pred_counts.append(pred_stats.num_grains)
        gt_areas.append(gt_stats.mean_area)
        pred_areas.append(pred_stats.mean_area)
        gt_diams.append(gt_stats.mean_diameter)
        pred_diams.append(pred_stats.mean_diameter)
        count_errs.append(abs(pred_stats.num_grains - gt_stats.num_grains) / gt_stats.num_grains)
        if gt_stats.mean_area > 0:
            area_errs.append(abs(pred_stats.mean_area - gt_stats.mean_area) / gt_stats.mean_area)
        if gt_stats.mean_diameter > 0:
            diam_errs.append(abs(pred_stats.mean_diameter - gt_stats.mean_diameter) / gt_stats.mean_diameter)

    if not count_errs:
        return None

    return {
        "n_images": float(len(count_errs)),
        "gt_count": float(np.mean(gt_counts)),
        "pred_count": float(np.mean(pred_counts)),
        "count_err": float(np.mean(count_errs) * 100.0),
        "count_err_std": float(np.std(count_errs) * 100.0),
        "gt_area": float(np.mean(gt_areas)),
        "pred_area": float(np.mean(pred_areas)),
        "area_err": float(np.mean(area_errs) * 100.0) if area_errs else 0.0,
        "area_err_std": float(np.std(area_errs) * 100.0) if area_errs else 0.0,
        "gt_diam": float(np.mean(gt_diams)),
        "pred_diam": float(np.mean(pred_diams)),
        "diam_err": float(np.mean(diam_errs) * 100.0) if diam_errs else 0.0,
        "diam_err_std": float(np.std(diam_errs) * 100.0) if diam_errs else 0.0,
    }


def run_grain_suite(
    model: torch.nn.Module,
    samples: list[SampleRecord],
    preprocess: str,
    device: torch.device,
    batch_size: int,
    threshold: float,
    degradation_seed: int,
    min_grain_area: int = DEFAULT_MIN_GRAIN_AREA,
    degradation_names: tuple[str, ...] | None = None,
    include_clean: bool = True,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    allowed = {name.lower() for name in degradation_names} if degradation_names else None

    if include_clean:
        clean_pairs = collect_grain_pairs(
            model=model,
            dataset=SegmentationDataset(samples=samples, preprocess=preprocess),
            device=device,
            threshold=threshold,
            batch_size=batch_size,
            min_grain_area=min_grain_area,
        )
        clean_metrics = aggregate_grain_metrics(clean_pairs)
        if clean_metrics is not None:
            rows.append({"condition": "clean", "degradation": "clean", **clean_metrics})

    for degradation_name, levels in DEGRADATION_LEVELS.items():
        if allowed is not None and degradation_name not in allowed:
            continue
        for level in levels:
            degraded_pairs = collect_grain_pairs(
                model=model,
                dataset=SegmentationDataset(
                    samples=samples,
                    preprocess=preprocess,
                    degradation_name=degradation_name,
                    degradation_params=level.params,
                    degradation_seed=degradation_seed,
                ),
                device=device,
                threshold=threshold,
                batch_size=batch_size,
                min_grain_area=min_grain_area,
            )
            degraded_metrics = aggregate_grain_metrics(degraded_pairs)
            if degraded_metrics is not None:
                rows.append(
                    {
                        "condition": level.label,
                        "degradation": degradation_name,
                        **degraded_metrics,
                    }
                )
    return rows


def write_grain_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
