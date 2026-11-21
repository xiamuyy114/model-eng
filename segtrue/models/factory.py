"""Unified model factory."""

from __future__ import annotations

from torch import nn

from .deeplabv3plus import ResNet34DeepLabV3Plus
from .unet import ResNet34UNet


def build_model(
    architecture: str,
    in_channels: int,
    classes: int = 1,
    pretrained: bool = False,
) -> nn.Module:
    key = architecture.lower()
    if key == "unet_r34":
        return ResNet34UNet(
            in_channels=in_channels,
            classes=classes,
            pretrained=pretrained,
        )
    if key in {"deeplabv3plus_r34", "deeplabv3+_r34"}:
        return ResNet34DeepLabV3Plus(
            in_channels=in_channels,
            classes=classes,
            pretrained=pretrained,
        )
    if key in {"unet_r18", "unet_r50"}:
        raise KeyError(
            f"Architecture '{architecture}' was removed from SEG-true when the "
            "encoder comparison task was dropped. Use 'unet_r34'."
        )
    if key == "deeplabv3_r34":
        raise KeyError(
            "Architecture 'deeplabv3_r34' was a temporary placeholder. "
            "Use 'deeplabv3plus_r34' for the paper-aligned benchmark."
        )
    raise KeyError(f"Unsupported architecture: {architecture}")
