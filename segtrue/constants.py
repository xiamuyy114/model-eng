"""Project-wide constants."""

from __future__ import annotations

from pathlib import Path

# This directory is also the standalone repository/package root.
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

SUPPORTED_PREPROCESS = ("rgb", "sobel", "canny", "gabor", "sobel_gabor", "clahe", "lbp", "gray")

PAPER_SEEDS_DIM1 = {
    "tio": (42, 2024, 12345, 7, 123, 456, 789, 1000, 2000, 3000),
    "316l": (42, 2024, 12345, 7, 123, 456, 789, 1000, 2000, 3000),
}

PAPER_SEEDS_DIM2 = {
    "tio": (42, 2024, 12345, 7, 123),
    "316l": (42, 2024, 12345, 7, 123),
}

PAPER_SPLIT_SEEDS = {
    "tio": PAPER_SEEDS_DIM1["tio"],
    "316l": PAPER_SEEDS_DIM1["316l"],
}

DEGRADATION_EVAL_SEED = 0
