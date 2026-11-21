"""Evaluation helpers for clean and degraded benchmarks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from segtrue.data.datasets import SampleRecord, SegmentationDataset
from segtrue.data.loaders import LoaderConfig, build_dataloader
from segtrue.degrade.registry import DEGRADATION_LEVELS

from .metrics import compute_batch_metrics, summarize_metrics


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, float]
    per_image: list[dict[str, float | str]]


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> EvaluationResult:
    model.eval()
    all_metrics: list[dict[str, float]] = []
    per_image: list[dict[str, float | str]] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        batch_metrics = compute_batch_metrics(logits, masks, threshold=threshold)
        all_metrics.extend(batch_metrics)
        for image_name, metrics in zip(batch["image_name"], batch_metrics):
            per_image.append({"image_name": image_name, **metrics})

    return EvaluationResult(summary=summarize_metrics(all_metrics), per_image=per_image)


@torch.no_grad()
def evaluate_degradation_suite(
    model: torch.nn.Module,
    samples: list[SampleRecord],
    preprocess: str,
    device: torch.device,
    batch_size: int,
    threshold: float,
    degradation_seed: int,
    degradation_names: tuple[str, ...] | None = None,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    allowed = {name.lower() for name in degradation_names} if degradation_names else None
    for degradation_name, levels in DEGRADATION_LEVELS.items():
        if allowed is not None and degradation_name not in allowed:
            continue
        for level in levels:
            dataset = SegmentationDataset(
                samples=samples,
                preprocess=preprocess,
                degradation_name=degradation_name,
                degradation_params=level.params,
                degradation_seed=degradation_seed,
            )
            loader = build_dataloader(
                dataset,
                LoaderConfig(batch_size=batch_size, shuffle=False, drop_last=False),
            )
            result = evaluate_model(model, loader, device=device, threshold=threshold)
            rows.append(
                {
                    "degradation": degradation_name,
                    "level": level.label,
                    **result.summary,
                }
            )
    return rows


def write_degradation_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    metric_fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in {"degradation", "level"} and key not in metric_fields:
                metric_fields.append(key)
    fieldnames = ["degradation", "level", *metric_fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
