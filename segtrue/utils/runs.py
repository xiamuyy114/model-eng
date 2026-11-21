"""Helpers for building canonical run and summary paths."""

from __future__ import annotations

from pathlib import Path

from segtrue.constants import RUNS_DIR, SUMMARIES_DIR


def build_run_dir(
    task: str,
    dataset_key: str,
    architecture: str,
    preprocess: str,
    seed: int,
) -> Path:
    return RUNS_DIR / task / dataset_key / architecture / preprocess / f"seed_{seed}"


def build_summary_dir(task: str) -> Path:
    return SUMMARIES_DIR / task
