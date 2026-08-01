import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18


FOUNDATION_BACKBONE_NAME = "official_r3d18"
FOUNDATION_USE_PRETRAINED = True
FOUNDATION_DROPOUT = 0.0


class FoundationModelHierarchical(nn.Module):
    """
    共享一个 3D backbone 的层级多任务模型。

    classification: normal / inflammation / metastasis
    subtype: inflammation / metastasis，仅对异常样本计算损失
    segmentation: 可选；当前仅 seq3/FLAIR 在模型工厂中启用
    """

    has_classification_head = True
    has_subtype_head = True
    has_segmentation_head = False
    uses_capability_interface = True

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        num_subtype_classes: int = 2,
        num_seg_classes: int = 3,
        enable_segmentation: bool = False,
    ):
        super().__init__()
        if num_classes != 3:
            raise ValueError(
                "FoundationModelHierarchical requires exactly three classes in "
                "the order normal, inflammation, metastasis"
            )
        if num_subtype_classes != 2:
            raise ValueError("The subtype head requires exactly two classes")

        self.backbone_name = FOUNDATION_BACKBONE_NAME
        self.in_channels = in_channels
        self.has_segmentation_head = bool(enable_segmentation)

        self.backbone, feat_dim = self._build_backbone(
            FOUNDATION_BACKBONE_NAME,
            FOUNDATION_USE_PRETRAINED,
            in_channels,
        )
        self.neck = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = self._build_head(
            feat_dim,
            num_classes,
            FOUNDATION_DROPOUT,
        )
        self.subtype_head = self._build_head(
            feat_dim,
            num_subtype_classes,
            FOUNDATION_DROPOUT,
        )

        self.aux_heads = nn.ModuleDict()
        if self.has_segmentation_head:
            self.aux_heads["seg_head"] = nn.Sequential(
                nn.Conv3d(feat_dim, 128, kernel_size=3, padding=1),
                nn.InstanceNorm3d(128, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(128, 64, kernel_size=3, padding=1),
                nn.InstanceNorm3d(64, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, num_seg_classes, kernel_size=1),
            )

    def _build_backbone(self, backbone_name, use_pretrained, in_channels):
        if backbone_name != "official_r3d18":
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        weights = R3D_18_Weights.DEFAULT if use_pretrained else None
        original_model = r3d_18(weights=weights)
        self._replace_bn3d_with_in3d(original_model)

        if in_channels == 1:
            original_model.stem[0] = self._adapt_first_conv_to_1ch(
                original_model.stem[0]
            )
        elif in_channels != 3:
            raise ValueError("official_r3d18 only supports in_channels=1 or 3")

        feat_dim = original_model.fc.in_features
        backbone = nn.Sequential(
            original_model.stem,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4,
        )
        return backbone, feat_dim

    def _replace_bn3d_with_in3d(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm3d):
                inorm = nn.InstanceNorm3d(
                    child.num_features,
                    eps=child.eps,
                    affine=True,
                    track_running_stats=False,
                )
                with torch.no_grad():
                    if child.weight is not None:
                        inorm.weight.copy_(child.weight)
                    if child.bias is not None:
                        inorm.bias.copy_(child.bias)
                setattr(module, name, inorm)
            else:
                self._replace_bn3d_with_in3d(child)

    @staticmethod
    def _build_head(feat_dim, num_classes, dropout):
        if dropout > 0:
            return nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feat_dim, num_classes),
            )
        return nn.Linear(feat_dim, num_classes)

    @staticmethod
    def _adapt_first_conv_to_1ch(conv3d):
        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=conv3d.out_channels,
            kernel_size=conv3d.kernel_size,
            stride=conv3d.stride,
            padding=conv3d.padding,
            bias=(conv3d.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv3d.weight.mean(dim=1, keepdim=True))
            if conv3d.bias is not None:
                new_conv.bias.copy_(conv3d.bias)
        return new_conv

    def initialize_from_baseline(self, baseline_model):
        """
        从现有 FoundationModel 系列复制共享主干和主分类头。

        FoundationModel_ori 保留 torchvision r3d_18 的模块外壳，而带分割头的
        FoundationModel 与本模型一样将 stem/layer1-4 拆成 Sequential。这里按
        模块语义复制，避免依赖两种 state_dict 键名恰好一致。
        """
        source_backbone = baseline_model.backbone
        if all(
            hasattr(source_backbone, name)
            for name in ("stem", "layer1", "layer2", "layer3", "layer4")
        ):
            source_stages = (
                source_backbone.stem,
                source_backbone.layer1,
                source_backbone.layer2,
                source_backbone.layer3,
                source_backbone.layer4,
            )
        elif isinstance(source_backbone, nn.Sequential) and len(source_backbone) >= 5:
            source_stages = tuple(source_backbone[index] for index in range(5))
        else:
            raise TypeError(
                "Unsupported baseline backbone layout: "
                f"{source_backbone.__class__.__name__}"
            )

        for target_stage, source_stage in zip(self.backbone, source_stages):
            target_stage.load_state_dict(source_stage.state_dict(), strict=True)
        self.neck.load_state_dict(baseline_model.neck.state_dict(), strict=True)
        self.head.load_state_dict(baseline_model.head.state_dict(), strict=True)

        copied_parts = ["backbone", "neck", "head"]
        if self.has_segmentation_head:
            source_aux_heads = getattr(baseline_model, "aux_heads", None)
            if source_aux_heads is None or "seg_head" not in source_aux_heads:
                raise ValueError(
                    "The seq3 hierarchical model requires a baseline checkpoint "
                    "with a segmentation head"
                )
            self.aux_heads["seg_head"].load_state_dict(
                source_aux_heads["seg_head"].state_dict(),
                strict=True,
            )
            copied_parts.append("aux_heads.seg_head")

        return tuple(copied_parts)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(
        self,
        x,
        return_seg=False,
        return_subtype=False,
        return_dict=False,
    ):
        if return_seg and not self.has_segmentation_head:
            raise ValueError(
                "Segmentation was requested, but this model instance has no "
                "segmentation head"
            )

        spatial_feat = self.forward_features(x)
        pooled_feat = torch.flatten(self.global_pool(spatial_feat), 1)
        feat = self.neck(pooled_feat)
        cls_logits = self.head(feat)

        subtype_logits = self.subtype_head(feat) if return_subtype else None
        seg_logits = None
        if return_seg:
            seg_features = self.aux_heads["seg_head"](spatial_feat)
            seg_logits = F.interpolate(
                seg_features,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        if return_dict:
            outputs = {"classification": cls_logits}
            if subtype_logits is not None:
                outputs["subtype"] = subtype_logits
            if seg_logits is not None:
                outputs["segmentation"] = seg_logits
            return outputs

        requested_aux = []
        if return_subtype:
            requested_aux.append(subtype_logits)
        if return_seg:
            requested_aux.append(seg_logits)
        if requested_aux:
            return (cls_logits, *requested_aux)
        return cls_logits
