"""Run-level result aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def collect_clean_rows(runs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config_path in sorted(runs_root.rglob("config.json")):
        run_dir = config_path.parent
        clean_metrics_path = run_dir / "test_clean_metrics.json"
        if not clean_metrics_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(clean_metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_dir": str(run_dir),
                "dataset_key": config["dataset_key"],
                "dataset": config["dataset"],
                "architecture": config["architecture"],
                "preprocess": config["preprocess"],
                "seed": config["seed"],
                **metrics,
            }
        )
    return rows


def collect_degraded_rows(runs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config_path in sorted(runs_root.rglob("config.json")):
        run_dir = config_path.parent
        degraded_csv = run_dir / "test_degraded_metrics.csv"
        if not degraded_csv.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with degraded_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "dataset_key": config["dataset_key"],
                        "dataset": config["dataset"],
                        "architecture": config["architecture"],
                        "preprocess": config["preprocess"],
                        "seed": config["seed"],
                        **item,
                    }
                )
    return rows


def collect_grain_rows(runs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config_path in sorted(runs_root.rglob("config.json")):
        run_dir = config_path.parent
        grain_csv = run_dir / "grain_results.csv"
        if not grain_csv.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with grain_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "dataset_key": config["dataset_key"],
                        "dataset": config["dataset"],
                        "architecture": config["architecture"],
                        "preprocess": config["preprocess"],
                        "seed": config["seed"],
                        **item,
                    }
                )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
