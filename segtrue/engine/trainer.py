"""Single-run training loop for SEG-true."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .evaluator import evaluate_model


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    threshold: float = 0.5
    amp: bool = True


def train_run(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    run_dir: Path,
    config: TrainerConfig,
) -> dict[str, dict[str, float]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    log_path = run_dir / "train_log.csv"

    best_val_dice = float("-inf")
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_dice", "val_miou"])
        writer.writeheader()

        for epoch in range(1, config.epochs + 1):
            model.train()
            loss_sum = 0.0
            num_batches = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                    logits = model(images)
                    loss = loss_fn(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.item())
                num_batches += 1

            train_loss = loss_sum / max(num_batches, 1)
            val_result = evaluate_model(model, val_loader, device=device, threshold=config.threshold)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_dice": f"{val_result.summary['dice']:.6f}",
                    "val_miou": f"{val_result.summary['miou']:.6f}",
                }
            )
            handle.flush()

            save_checkpoint(model, run_dir / "last.ckpt")
            if val_result.summary["dice"] > best_val_dice:
                best_val_dice = val_result.summary["dice"]
                save_checkpoint(model, run_dir / "best.ckpt")
                _write_json(run_dir / "val_metrics.json", val_result.summary)

    best_state = torch.load(run_dir / "best.ckpt", map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    test_result = evaluate_model(model, test_loader, device=device, threshold=config.threshold)
    _write_json(run_dir / "test_clean_metrics.json", test_result.summary)
    _write_json(run_dir / "per_image_clean.json", test_result.per_image)
    return {
        "val": json.loads((run_dir / "val_metrics.json").read_text(encoding="utf-8")),
        "test_clean": test_result.summary,
    }


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
