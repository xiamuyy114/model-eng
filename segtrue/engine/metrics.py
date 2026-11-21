"""Binary segmentation metrics.

The dataset masks are stored as interior-positive binary regions
(`mask > 127`). Boundary-positive metrics are computed by complementing both
prediction and target so trained interior-positive checkpoints can be reused.
"""

from __future__ import annotations

import math
from statistics import mean

import torch


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> list[dict[str, float]]:
    probs = torch.sigmoid(logits)
    interior_preds = (probs >= threshold).float()
    metrics: list[dict[str, float]] = []
    for interior_pred, interior_target in zip(interior_preds, targets):
        interior = _compute_binary_metrics(
            interior_pred.view(-1),
            interior_target.view(-1),
            eps=eps,
        )
        boundary = _compute_binary_metrics(
            (1.0 - interior_pred).view(-1),
            (1.0 - interior_target).view(-1),
            eps=eps,
        )

        item = {
            "dice": interior["dice"],
            "miou": interior["miou"],
            "precision": interior["precision"],
            "recall": interior["recall"],
            "accuracy": interior["accuracy"],
        }
        item.update({f"interior_{key}": value for key, value in interior.items()})
        item.update({f"boundary_{key}": value for key, value in boundary.items()})
        metrics.append(item)
    return metrics


def _compute_binary_metrics(
    pred_flat: torch.Tensor,
    target_flat: torch.Tensor,
    eps: float,
) -> dict[str, float]:
    tp = float((pred_flat * target_flat).sum().item())
    fp = float((pred_flat * (1.0 - target_flat)).sum().item())
    fn = float(((1.0 - pred_flat) * target_flat).sum().item())
    tn = float(((1.0 - pred_flat) * (1.0 - target_flat)).sum().item())

    has_union_error = (tp + fp + fn) > 0.0
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps) if has_union_error else 1.0
    iou = (tp + eps) / (tp + fp + fn + eps) if has_union_error else 1.0
    precision = (tp + eps) / (tp + fp + eps) if (tp + fp) > 0.0 else 1.0
    recall = (tp + eps) / (tp + fn + eps) if (tp + fn) > 0.0 else 1.0
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": dice,
        "miou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def summarize_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {
            key: math.nan
            for key in (
                "dice",
                "miou",
                "precision",
                "recall",
                "accuracy",
                "interior_dice",
                "interior_miou",
                "interior_precision",
                "interior_recall",
                "interior_accuracy",
                "boundary_dice",
                "boundary_miou",
                "boundary_precision",
                "boundary_recall",
                "boundary_accuracy",
            )
        }
    keys = items[0].keys()
    return {key: mean(item[key] for item in items) for key in keys}
