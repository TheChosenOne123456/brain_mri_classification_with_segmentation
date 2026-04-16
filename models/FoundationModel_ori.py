import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


# 固定在模型文件内的配置（非构造参数）
FOUNDATION_BACKBONE_NAME = "official_r3d18"
FOUNDATION_USE_PRETRAINED = True
FOUNDATION_DROPOUT = 0.0


class FoundationModel(nn.Module):
    """
    可扩展“壳子”：
    - backbone: 当前仅支持 official_r3d18（官方预训练）
    - neck: 预留（默认 Identity）
    - head: 分类头（可替换）
    - aux_heads: 预留多任务/蒸馏/对比学习头
    """
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
    ):
        super().__init__()
        self.backbone_name = FOUNDATION_BACKBONE_NAME
        self.in_channels = in_channels

        self.backbone, feat_dim = self._build_backbone(
            FOUNDATION_BACKBONE_NAME,
            FOUNDATION_USE_PRETRAINED,
            in_channels
        )
        self.neck = self._build_neck(feat_dim)   # 预留创新点1
        self.head = self._build_head(feat_dim, num_classes, FOUNDATION_DROPOUT)
        self.aux_heads = nn.ModuleDict()         # 预留创新点2

    # ===== 可扩展点：backbone =====
    def _build_backbone(self, backbone_name: str, use_pretrained: bool, in_channels: int):
        if backbone_name == "official_r3d18":
            weights = R3D_18_Weights.DEFAULT if use_pretrained else None
            model = r3d_18(weights=weights)

            # [新增] 仅做 BN -> IN
            self._replace_bn3d_with_in3d(model)

            # 适配输入通道（MRI 常见 1 通道）
            if in_channels == 1:
                model.stem[0] = self._adapt_first_conv_to_1ch(model.stem[0])
            elif in_channels != 3:
                raise ValueError("official_r3d18 仅支持 in_channels=1 或 3")

            feat_dim = model.fc.in_features
            model.fc = nn.Identity()  # 把分类头拆掉，交给壳子自己的 head
            return model, feat_dim

        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def _replace_bn3d_with_in3d(self, module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm3d):
                inorm = nn.InstanceNorm3d(
                    child.num_features,
                    eps=child.eps,
                    affine=True,
                    track_running_stats=False,
                )
                # 可选：拷贝 affine 参数，减少突变
                with torch.no_grad():
                    if child.weight is not None:
                        inorm.weight.copy_(child.weight)
                    if child.bias is not None:
                        inorm.bias.copy_(child.bias)
                setattr(module, name, inorm)
            else:
                self._replace_bn3d_with_in3d(child)

    # ===== 可扩展点：neck =====
    def _build_neck(self, feat_dim: int):
        # 后续可替换为 attention / adapter / projector
        return nn.Identity()

    # ===== 可扩展点：head =====
    def _build_head(self, feat_dim: int, num_classes: int, dropout: float):
        if dropout > 0:
            return nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feat_dim, num_classes),
            )
        return nn.Linear(feat_dim, num_classes)

    def _adapt_first_conv_to_1ch(self, conv3d: nn.Conv3d):
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

    def forward_features(self, x):
        feat = self.backbone(x)   # [B, feat_dim]
        feat = self.neck(feat)
        return feat

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.head(feat)
        return logits