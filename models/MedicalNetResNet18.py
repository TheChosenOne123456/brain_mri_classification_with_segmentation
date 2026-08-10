"""MedicalNet ResNet-18 backbone adapted for single-sequence classification.

The backbone follows Tencent MedicalNet's 23-dataset ResNet-18 with shortcut A.
Only the segmentation decoder is replaced: global average pooling and a linear
three-class head are used for this project.
"""

from hashlib import sha256
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MEDICALNET_WEIGHTS_SHA256 = (
    "61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4"
)
MEDICALNET_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "pretrained-medicalnet"
    / "resnet_18_23dataset.pth"
)


class _ShortcutA(nn.Module):
    """MedicalNet shortcut A: spatial pooling followed by zero channel padding."""

    def __init__(self, out_channels, stride):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        residual = F.avg_pool3d(x, kernel_size=1, stride=self.stride)
        channel_padding = self.out_channels - residual.shape[1]
        if channel_padding < 0:
            raise ValueError("Shortcut A cannot reduce the channel dimension")
        if channel_padding == 0:
            return residual
        zeros = residual.new_zeros(
            residual.shape[0],
            channel_padding,
            *residual.shape[2:],
        )
        return torch.cat((residual, zeros), dim=1)


class _MedicalNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class _MedicalNetResNet18Backbone(nn.Module):
    """Encoder portion of the official MedicalNet ResNet-18 architecture."""

    def __init__(self, in_channels=1):
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                "MedicalNetResNet18 uses the official one-channel pretraining and "
                "therefore requires in_channels=1"
            )

        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, dilation=2)
        self.layer4 = self._make_layer(512, blocks=2, dilation=4)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, channels, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = _ShortcutA(channels, stride)

        layers = [
            _MedicalNetBasicBlock(
                self.inplanes,
                channels,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        ]
        self.inplanes = channels
        layers.extend(
            _MedicalNetBasicBlock(
                self.inplanes,
                channels,
                dilation=dilation,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class MedicalNetResNet18(nn.Module):
    """MedicalNet-pretrained classification model for T1/T2 volumes."""

    has_classification_head = True
    has_subtype_head = False
    has_segmentation_head = False
    uses_capability_interface = True

    def __init__(
        self,
        num_classes=3,
        in_channels=1,
        weights_path=None,
        load_pretrained=True,
    ):
        super().__init__()
        self.backbone = _MedicalNetResNet18Backbone(in_channels=in_channels)
        self.pretrained_loaded = False
        self.pretrained_metadata = None
        if load_pretrained:
            source_path = Path(weights_path or MEDICALNET_WEIGHTS_PATH)
            self.pretrained_metadata = self._load_medicalnet_weights(source_path)
            self.pretrained_loaded = True

        # Match the existing FoundationModel_ori normalization choice while
        # retaining every pretrained convolution and normalization affine value.
        self._replace_bn3d_with_in3d(self.backbone)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(512, num_classes)

    @staticmethod
    def _file_sha256(path):
        digest = sha256()
        with path.open("rb") as file_object:
            for chunk in iter(lambda: file_object.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_medicalnet_weights(self, weights_path):
        if not weights_path.is_file():
            raise FileNotFoundError(
                "MedicalNet pretrained weights were not found at "
                f"{weights_path}. Expected resnet_18_23dataset.pth."
            )

        actual_sha256 = self._file_sha256(weights_path)
        if actual_sha256 != MEDICALNET_WEIGHTS_SHA256:
            raise ValueError(
                "MedicalNet weights failed SHA-256 verification: "
                f"expected {MEDICALNET_WEIGHTS_SHA256}, got {actual_sha256}"
            )

        checkpoint = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        source_state = checkpoint.get("state_dict", checkpoint)
        source_state = {
            key.removeprefix("module."): value
            for key, value in source_state.items()
            if torch.is_tensor(value)
        }
        target_state = self.backbone.state_dict()
        matched_state = {
            key: value
            for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        missing_parameters = sorted(
            key
            for key in dict(self.backbone.named_parameters())
            if key not in matched_state
        )
        if missing_parameters:
            preview = ", ".join(missing_parameters[:8])
            raise ValueError(
                "MedicalNet checkpoint does not fully cover the backbone "
                f"parameters; missing: {preview}"
            )
        self.backbone.load_state_dict(matched_state, strict=False)
        return {
            "path": str(weights_path.resolve()),
            "sha256": actual_sha256,
            "loaded_tensors": len(matched_state),
            "source_tensors": len(source_state),
        }

    def _replace_bn3d_with_in3d(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm3d):
                instance_norm = nn.InstanceNorm3d(
                    child.num_features,
                    eps=child.eps,
                    affine=True,
                    track_running_stats=False,
                )
                with torch.no_grad():
                    instance_norm.weight.copy_(child.weight)
                    instance_norm.bias.copy_(child.bias)
                setattr(module, name, instance_norm)
            else:
                self._replace_bn3d_with_in3d(child)

    def forward_features(self, x):
        features = self.backbone(x)
        return self.pool(features).flatten(1)

    def forward(
        self,
        x,
        return_seg=False,
        return_subtype=False,
        return_dict=False,
    ):
        if return_seg:
            raise ValueError("MedicalNetResNet18 has no segmentation head")
        if return_subtype:
            raise ValueError("MedicalNetResNet18 has no subtype head")

        logits = self.head(self.forward_features(x))
        if return_dict:
            return {"classification": logits}
        return logits
