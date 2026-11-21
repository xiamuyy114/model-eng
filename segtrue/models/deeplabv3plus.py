"""DeepLabV3+ wrapper used by the unified benchmark pipeline."""

from __future__ import annotations

from torch import nn

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DeepLabV3+ requires segmentation-models-pytorch. "
        "Install dependencies from pyproject.toml before training."
    ) from exc


class ResNet34DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int, classes: int = 1, pretrained: bool = False) -> None:
        super().__init__()
        encoder_weights = "imagenet" if pretrained else None
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        return self.model(x)
