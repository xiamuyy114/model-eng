"""U-Net wrapper used by the retained experiment matrix."""

from __future__ import annotations

from torch import nn

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "U-Net backbones require segmentation-models-pytorch. "
        "Install dependencies from pyproject.toml before training."
    ) from exc

class ResNet34UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        classes: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        encoder_weights = "imagenet" if pretrained else None
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        return self.model(x)
