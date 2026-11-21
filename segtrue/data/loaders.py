"""DataLoader builders for the unified project structure."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from .datasets import SegmentationDataset


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True
    drop_last: bool = False


def build_dataloader(dataset: SegmentationDataset, config: LoaderConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
