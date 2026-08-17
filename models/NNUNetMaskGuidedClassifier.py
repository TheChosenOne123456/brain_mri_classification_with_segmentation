"""使用冻结 nnU-Net soft mask、重新训练分类参数的 FLAIR 分类模型。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18


STAGE_CHANNELS = (128, 256, 512)
MASK_STAT_NAMES = (
    "log_soft_volume_ml",
    "log_hard_volume_ml",
    "soft_fraction",
    "hard_fraction",
    "high_confidence_fraction",
    "probability_std",
    "max_probability",
    "mean_entropy",
    "centroid_z",
    "centroid_y",
    "centroid_x",
    "spread_z",
    "spread_y",
    "spread_x",
)
MASK_STAT_DIM = len(MASK_STAT_NAMES)
POOL_EPS = 1e-6


class NNUNetMaskGuidedClassifier(nn.Module):
    """独立的全局分类分支和病灶引导亚型分支。

    本模型只使用 torchvision 的 Kinetics 预训练初始化，不加载任何已经训练好的
    Foundation 分类 checkpoint。nnU-Net 也不作为子模块参与反向传播；调用方需要
    提供与病例对应的 OOF soft mask 及其统计量。
    """

    model_name = "NNUNetMaskGuidedClassifier"
    has_classification_head = True
    has_subtype_head = True
    has_segmentation_head = False
    uses_capability_interface = False

    def __init__(
        self,
        num_classes=3,
        in_channels=1,
        stage_projection_dim=64,
        stats_projection_dim=32,
        subtype_hidden_dim=128,
        subtype_dropout=0.2,
    ):
        super().__init__()
        if int(num_classes) != 3:
            raise ValueError("NNUNetMaskGuidedClassifier requires three classes")
        if int(in_channels) != 1:
            raise ValueError("NNUNetMaskGuidedClassifier requires one FLAIR channel")

        original = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self._replace_bn3d_with_in3d(original)
        original.stem[0] = self._adapt_first_conv_to_1ch(original.stem[0])
        self.backbone = nn.Sequential(
            original.stem,
            original.layer1,
            original.layer2,
            original.layer3,
            original.layer4,
        )
        feature_dim = int(original.fc.in_features)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classification_head = nn.Linear(feature_dim, int(num_classes))

        self.stage_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(3 * channels),
                    nn.Linear(3 * channels, int(stage_projection_dim)),
                    nn.GELU(),
                )
                for channels in STAGE_CHANNELS
            ]
        )
        self.stats_projector = nn.Sequential(
            nn.LayerNorm(MASK_STAT_DIM),
            nn.Linear(MASK_STAT_DIM, int(stats_projection_dim)),
            nn.GELU(),
        )
        subtype_input_dim = (
            len(STAGE_CHANNELS) * int(stage_projection_dim)
            + int(stats_projection_dim)
        )
        self.subtype_head = nn.Sequential(
            nn.LayerNorm(subtype_input_dim),
            nn.Linear(subtype_input_dim, int(subtype_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(subtype_dropout)),
            nn.Linear(int(subtype_hidden_dim), 1),
        )
        self.register_buffer("guidance_mix_alpha", torch.zeros(()))

    @staticmethod
    def _replace_bn3d_with_in3d(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm3d):
                replacement = nn.InstanceNorm3d(
                    child.num_features,
                    eps=child.eps,
                    affine=True,
                    track_running_stats=False,
                )
                with torch.no_grad():
                    if child.weight is not None:
                        replacement.weight.copy_(child.weight)
                    if child.bias is not None:
                        replacement.bias.copy_(child.bias)
                setattr(module, name, replacement)
            else:
                NNUNetMaskGuidedClassifier._replace_bn3d_with_in3d(child)

    @staticmethod
    def _adapt_first_conv_to_1ch(conv3d):
        replacement = nn.Conv3d(
            in_channels=1,
            out_channels=conv3d.out_channels,
            kernel_size=conv3d.kernel_size,
            stride=conv3d.stride,
            padding=conv3d.padding,
            bias=(conv3d.bias is not None),
        )
        with torch.no_grad():
            replacement.weight.copy_(conv3d.weight.mean(dim=1, keepdim=True))
            if conv3d.bias is not None:
                replacement.bias.copy_(conv3d.bias)
        return replacement

    @staticmethod
    def _weighted_pool(features, weights, fallback):
        numerator = (features * weights).sum(dim=(2, 3, 4))
        denominator = weights.sum(dim=(2, 3, 4))
        pooled = numerator / denominator.clamp_min(POOL_EPS)
        return torch.where(denominator <= POOL_EPS, fallback, pooled)

    def forward_multiscale_features(self, inputs):
        features = self.backbone[0](inputs)
        features = self.backbone[1](features)
        layer2 = self.backbone[2](features)
        layer3 = self.backbone[3](layer2)
        layer4 = self.backbone[4](layer3)
        pooled = torch.flatten(self.global_pool(layer4), 1)
        logits = self.classification_head(pooled)
        return logits, (layer2, layer3, layer4)

    def guidance_embedding(
        self,
        inputs,
        multiscale_features,
        lesion_probability,
        lesion_statistics,
    ):
        if lesion_probability.dim() != 5 or lesion_probability.size(1) != 1:
            raise ValueError(
                "lesion_probability must have shape [B,1,D,H,W], got "
                f"{tuple(lesion_probability.shape)}"
            )
        if lesion_statistics.shape != (inputs.size(0), MASK_STAT_DIM):
            raise ValueError(
                "lesion_statistics must have shape "
                f"[B,{MASK_STAT_DIM}], got {tuple(lesion_statistics.shape)}"
            )
        if lesion_probability.size(0) != inputs.size(0):
            raise ValueError("Input and lesion-probability batch sizes differ")

        brain_mask = (inputs.abs().sum(dim=1, keepdim=True) > 1e-6).float()
        probability = lesion_probability.float().clamp(0.0, 1.0)
        projected_stages = []
        for features, projector in zip(
            multiscale_features,
            self.stage_projectors,
        ):
            features = features.float()
            brain_weights = F.interpolate(
                brain_mask,
                size=features.shape[2:],
                mode="area",
            ).clamp(0.0, 1.0)
            feature_probability = F.interpolate(
                probability,
                size=features.shape[2:],
                mode="trilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            lesion_weights = feature_probability * brain_weights
            dilated_probability = F.max_pool3d(
                feature_probability,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            ring_weights = (
                dilated_probability - feature_probability
            ).clamp_min(0.0) * brain_weights
            fallback = features.mean(dim=(2, 3, 4))
            global_features = self._weighted_pool(
                features,
                brain_weights,
                fallback,
            )
            lesion_features = self._weighted_pool(
                features,
                lesion_weights,
                global_features,
            )
            ring_features = self._weighted_pool(
                features,
                ring_weights,
                global_features,
            )
            stage_features = torch.cat(
                (
                    global_features,
                    lesion_features,
                    lesion_features - ring_features,
                ),
                dim=1,
            )
            projected_stages.append(projector(stage_features))
        projected_statistics = self.stats_projector(
            lesion_statistics.float()
        )
        return torch.cat((*projected_stages, projected_statistics), dim=1)

    @staticmethod
    def mixed_subtype_probability(
        base_probabilities,
        expert_probability,
        alpha,
    ):
        abnormal_probability = base_probabilities[:, 1:].sum(dim=1)
        base_subtype_probability = (
            base_probabilities[:, 2]
            / abnormal_probability.clamp_min(POOL_EPS)
        )
        alpha = torch.as_tensor(
            alpha,
            dtype=base_probabilities.dtype,
            device=base_probabilities.device,
        ).clamp(0.0, 1.0)
        return (
            (1.0 - alpha) * base_subtype_probability
            + alpha * expert_probability
        )

    @staticmethod
    def corrected_probabilities(base_probabilities, subtype_probability):
        abnormal_probability = base_probabilities[:, 1:].sum(dim=1)
        return torch.stack(
            (
                base_probabilities[:, 0],
                abnormal_probability * (1.0 - subtype_probability),
                abnormal_probability * subtype_probability,
            ),
            dim=1,
        )

    def set_guidance_mix_alpha(self, alpha):
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("guidance mixture alpha must lie in [0, 1]")
        with torch.no_grad():
            self.guidance_mix_alpha.fill_(alpha)

    def forward_guided(
        self,
        inputs,
        lesion_probability,
        lesion_statistics,
    ):
        base_logits, multiscale = self.forward_multiscale_features(inputs)
        embedding = self.guidance_embedding(
            inputs,
            multiscale,
            lesion_probability,
            lesion_statistics,
        )
        subtype_logit = self.subtype_head(embedding).squeeze(1)
        base_probabilities = F.softmax(base_logits.float(), dim=1)
        expert_probability = torch.sigmoid(subtype_logit.float())
        subtype_probability = self.mixed_subtype_probability(
            base_probabilities,
            expert_probability,
            self.guidance_mix_alpha,
        )
        probabilities = self.corrected_probabilities(
            base_probabilities,
            subtype_probability,
        )
        return {
            "classification": base_logits,
            "subtype_logit": subtype_logit,
            "base_probabilities": base_probabilities,
            "expert_probability": expert_probability,
            "subtype_probability": subtype_probability,
            "probabilities": probabilities,
        }

    def forward(
        self,
        inputs,
        lesion_probability=None,
        lesion_statistics=None,
    ):
        if lesion_probability is None and lesion_statistics is None:
            logits, _ = self.forward_multiscale_features(inputs)
            return logits
        if lesion_probability is None or lesion_statistics is None:
            raise ValueError(
                "lesion_probability and lesion_statistics must be provided together"
            )
        return self.forward_guided(
            inputs,
            lesion_probability,
            lesion_statistics,
        )
