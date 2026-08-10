"""Foundation 分类主干与冻结 nnU-Net soft mask 的病灶引导混合模型。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FoundationModel import FoundationModel


GUIDANCE_STAGE_CHANNELS = (128, 256, 512)
GUIDANCE_STAT_NAMES = (
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
GUIDANCE_FEATURE_DIM = 3 * sum(GUIDANCE_STAGE_CHANNELS) + len(
    GUIDANCE_STAT_NAMES
)
GUIDANCE_POOL_EPS = 1e-6
FLAIR_VOXEL_VOLUME_ML = 3.0 * 0.75 * 0.75 / 1000.0


class FoundationModelNNUNetGuided(FoundationModel):
    """融合独立 subtype expert，并保持 Foundation 正常类门控不变。

    nnU-Net 不作为本类的子模块。训练和推理入口先生成与输入对齐的 soft
    lesion probability，再调用 :meth:`forward_guided`。这样可以冻结、复用
    原始 Foundation checkpoint，同时独立选择无数据泄漏的 nnU-Net fold。
    """

    model_name = "FoundationModelNNUNetGuided"

    def __init__(
        self,
        num_classes=3,
        in_channels=1,
    ):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            num_seg_classes=3,
        )
        self.guidance_feature_dim = GUIDANCE_FEATURE_DIM
        self.guidance_stat_names = GUIDANCE_STAT_NAMES
        # 首轮诊断中线性 probe 已显示有效的 validation 工作点。这里保持单层
        # expert，减少 2702 维小样本上的过拟合和不必要的优化不确定性。
        self.guidance_expert_head = nn.Linear(GUIDANCE_FEATURE_DIM, 1)
        self.register_buffer(
            "guidance_feature_mean",
            torch.zeros(GUIDANCE_FEATURE_DIM),
        )
        self.register_buffer(
            "guidance_feature_std",
            torch.ones(GUIDANCE_FEATURE_DIM),
        )
        self.register_buffer("guidance_mix_alpha", torch.zeros(()))
        self._zero_initialize_expert()

    def _zero_initialize_expert(self):
        nn.init.zeros_(self.guidance_expert_head.weight)
        nn.init.zeros_(self.guidance_expert_head.bias)

    def initialize_from_foundation_state(self, source_state):
        """严格复制 Foundation 权重，新增残差模块保持自身初始化。"""
        target_state = self.state_dict()
        source_keys = set(source_state)
        guidance_keys = {
            key
            for key in target_state
            if key.startswith("guidance_")
        }
        expected_source_keys = set(target_state) - guidance_keys
        if source_keys != expected_source_keys:
            missing = sorted(expected_source_keys - source_keys)
            unexpected = sorted(source_keys - expected_source_keys)
            raise RuntimeError(
                "Foundation checkpoint structure mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for key, value in source_state.items():
            if target_state[key].shape != value.shape:
                raise ValueError(
                    f"Foundation tensor shape mismatch for {key}: "
                    f"{tuple(target_state[key].shape)} vs {tuple(value.shape)}"
                )
            target_state[key] = value.detach().clone()
        self.load_state_dict(target_state, strict=True)
        return len(source_state)

    def load_guidance_state(self, guidance_state):
        """加载训练产出的轻量 guidance checkpoint，并拒绝静默缺键。"""
        target_state = self.state_dict()
        expected_keys = {
            key for key in target_state if key.startswith("guidance_")
        }
        source_keys = set(guidance_state)
        if source_keys != expected_keys:
            missing = sorted(expected_keys - source_keys)
            unexpected = sorted(source_keys - expected_keys)
            raise RuntimeError(
                "Guidance checkpoint structure mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for key, value in guidance_state.items():
            if target_state[key].shape != value.shape:
                raise ValueError(
                    f"Guidance tensor shape mismatch for {key}: "
                    f"{tuple(target_state[key].shape)} vs {tuple(value.shape)}"
                )
            target_state[key] = value.detach().clone()
        self.load_state_dict(target_state, strict=True)
        return len(guidance_state)

    def freeze_foundation(self):
        for module in (
            self.backbone,
            self.neck,
            self.head,
            self.aux_heads,
        ):
            module.requires_grad_(False)
            module.eval()
        self.guidance_expert_head.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
            self.neck.eval()
            self.head.eval()
            self.aux_heads.eval()
        return self

    def set_guidance_standardization(self, feature_mean, feature_std):
        feature_mean = torch.as_tensor(feature_mean).flatten()
        feature_std = torch.as_tensor(feature_std).flatten()
        if feature_mean.numel() != GUIDANCE_FEATURE_DIM:
            raise ValueError(
                f"Expected {GUIDANCE_FEATURE_DIM} feature means, "
                f"got {feature_mean.numel()}"
            )
        if feature_std.numel() != GUIDANCE_FEATURE_DIM:
            raise ValueError(
                f"Expected {GUIDANCE_FEATURE_DIM} feature stds, "
                f"got {feature_std.numel()}"
            )
        with torch.no_grad():
            self.guidance_feature_mean.copy_(
                feature_mean.to(self.guidance_feature_mean)
            )
            self.guidance_feature_std.copy_(
                feature_std.clamp_min(1e-6).to(self.guidance_feature_std)
            )

    def set_guidance_mix_alpha(self, alpha):
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("guidance mixture alpha must lie in [0, 1]")
        with torch.no_grad():
            self.guidance_mix_alpha.fill_(alpha)

    def forward_multiscale_features(self, x):
        """返回 base logits 以及 R3D-18 layer2/3/4 空间特征。"""
        features = self.backbone[0](x)
        features = self.backbone[1](features)
        layer2 = self.backbone[2](features)
        layer3 = self.backbone[3](layer2)
        layer4 = self.backbone[4](layer3)
        pooled = torch.flatten(self.global_pool(layer4), 1)
        base_logits = self.head(self.neck(pooled))
        return base_logits, (layer2, layer3, layer4)

    @staticmethod
    def _weighted_pool(features, weights, fallback):
        numerator = (features * weights).sum(dim=(2, 3, 4))
        denominator = weights.sum(dim=(2, 3, 4))
        pooled = numerator / denominator.clamp_min(GUIDANCE_POOL_EPS)
        use_fallback = denominator <= GUIDANCE_POOL_EPS
        return torch.where(use_fallback, fallback, pooled)

    @staticmethod
    def _axis_moments(weights, axis):
        spatial_dims = (2, 3, 4)
        reduce_dims = tuple(dim for dim in spatial_dims if dim != axis + 2)
        marginal = weights.sum(dim=reduce_dims).squeeze(1)
        coordinates = torch.linspace(
            -1.0,
            1.0,
            marginal.size(1),
            device=weights.device,
            dtype=weights.dtype,
        ).unsqueeze(0)
        denominator = marginal.sum(dim=1).clamp_min(GUIDANCE_POOL_EPS)
        center = (marginal * coordinates).sum(dim=1) / denominator
        variance = (
            marginal * (coordinates - center.unsqueeze(1)).square()
        ).sum(dim=1) / denominator
        has_lesion = marginal.sum(dim=1) > GUIDANCE_POOL_EPS
        center = torch.where(has_lesion, center, torch.zeros_like(center))
        spread = torch.where(
            has_lesion,
            variance.clamp_min(0).sqrt(),
            torch.zeros_like(variance),
        )
        return center, spread

    def lesion_statistics(self, lesion_probability, brain_mask):
        probability = lesion_probability.float().clamp(0.0, 1.0)
        brain = brain_mask.float()
        probability = probability * brain
        brain_voxels = brain.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
        soft_voxels = probability.sum(dim=(1, 2, 3, 4))
        hard_voxels = (
            (probability >= 0.5).to(probability.dtype) * brain
        ).sum(dim=(1, 2, 3, 4))
        high_confidence_voxels = (
            (probability >= 0.9).to(probability.dtype) * brain
        ).sum(dim=(1, 2, 3, 4))
        mean_probability = soft_voxels / brain_voxels
        probability_square_mean = probability.square().sum(
            dim=(1, 2, 3, 4)
        ) / brain_voxels
        probability_std = (
            probability_square_mean - mean_probability.square()
        ).clamp_min(0).sqrt()
        max_probability = probability.flatten(1).amax(dim=1)
        bounded = probability.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(
            bounded * bounded.log()
            + (1.0 - bounded) * (1.0 - bounded).log()
        )
        mean_entropy = (entropy * brain).sum(dim=(1, 2, 3, 4)) / brain_voxels

        centers = []
        spreads = []
        for axis in range(3):
            center, spread = self._axis_moments(probability, axis)
            centers.append(center)
            spreads.append(spread)

        return torch.stack(
            (
                torch.log1p(soft_voxels * FLAIR_VOXEL_VOLUME_ML),
                torch.log1p(hard_voxels * FLAIR_VOXEL_VOLUME_ML),
                mean_probability,
                hard_voxels / brain_voxels,
                high_confidence_voxels / brain_voxels,
                probability_std,
                max_probability,
                mean_entropy,
                *centers,
                *spreads,
            ),
            dim=1,
        )

    def guidance_feature_vector(
        self,
        inputs,
        multiscale_features,
        lesion_probability,
    ):
        if lesion_probability.shape[:2] != (inputs.size(0), 1):
            raise ValueError(
                "lesion_probability must have shape [B,1,D,H,W], got "
                f"{tuple(lesion_probability.shape)}"
            )
        if lesion_probability.shape[2:] != inputs.shape[2:]:
            raise ValueError(
                "lesion_probability and inputs must be spatially aligned: "
                f"{tuple(lesion_probability.shape[2:])} vs "
                f"{tuple(inputs.shape[2:])}"
            )
        brain_mask = (inputs.abs().sum(dim=1, keepdim=True) > 1e-6).float()
        probability = lesion_probability.float().clamp(0.0, 1.0) * brain_mask
        pooled_features = []
        for features in multiscale_features:
            features = features.float()
            brain_weights = F.interpolate(
                brain_mask,
                size=features.shape[2:],
                mode="area",
            ).clamp(0.0, 1.0)
            feature_probability = F.interpolate(
                probability,
                size=features.shape[2:],
                mode="area",
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
            global_features = self._weighted_pool(
                features,
                brain_weights,
                features.mean(dim=(2, 3, 4)),
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
            pooled_features.extend(
                (global_features, lesion_features, lesion_features - ring_features)
            )
        statistics = self.lesion_statistics(probability, brain_mask).to(
            pooled_features[0].dtype
        )
        features = torch.cat((*pooled_features, statistics), dim=1)
        if features.size(1) != GUIDANCE_FEATURE_DIM:
            raise RuntimeError(
                f"Guidance feature mismatch: {features.size(1)} vs "
                f"{GUIDANCE_FEATURE_DIM}"
            )
        return features

    def guidance_expert_logit(self, guidance_features):
        standardized = (
            guidance_features - self.guidance_feature_mean
        ) / self.guidance_feature_std
        return self.guidance_expert_head(standardized).squeeze(1)

    @staticmethod
    def mixed_subtype_probability(base_logits, expert_logit, alpha):
        if base_logits.size(1) != 3:
            raise ValueError("base_logits must contain three classification logits")
        base_subtype_logit = base_logits[:, 2] - base_logits[:, 1]
        base_probability = torch.sigmoid(base_subtype_logit)
        expert_probability = torch.sigmoid(expert_logit)
        alpha = torch.as_tensor(
            alpha,
            device=base_logits.device,
            dtype=base_probability.dtype,
        ).clamp(0.0, 1.0)
        return (1.0 - alpha) * base_probability + alpha * expert_probability

    @staticmethod
    def hierarchical_predictions(base_logits, metastasis_probability):
        predictions = base_logits.argmax(dim=1)
        abnormal = predictions != 0
        if abnormal.any():
            predictions = predictions.clone()
            predictions[abnormal] = (
                metastasis_probability[abnormal] >= 0.5
            ).long() + 1
        return predictions

    @staticmethod
    def corrected_probabilities(base_logits, metastasis_probability):
        base_probabilities = F.softmax(base_logits, dim=1)
        abnormal_probability = base_probabilities[:, 1:].sum(dim=1)
        return torch.stack(
            (
                base_probabilities[:, 0],
                abnormal_probability * (1.0 - metastasis_probability),
                abnormal_probability * metastasis_probability,
            ),
            dim=1,
        )

    def forward_guided(self, inputs, lesion_probability):
        base_logits, multiscale_features = self.forward_multiscale_features(inputs)
        guidance_features = self.guidance_feature_vector(
            inputs,
            multiscale_features,
            lesion_probability,
        )
        expert_logit = self.guidance_expert_logit(guidance_features)
        metastasis_probability = self.mixed_subtype_probability(
            base_logits,
            expert_logit,
            self.guidance_mix_alpha,
        )
        return {
            "classification": base_logits,
            "guidance_features": guidance_features,
            "guidance_expert_logit": expert_logit,
            "metastasis_probability": metastasis_probability,
            "probabilities": self.corrected_probabilities(
                base_logits,
                metastasis_probability,
            ),
            "segmentation_probability": lesion_probability,
        }
