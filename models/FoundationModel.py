import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn.functional as F
import types

# 固定在模型文件内的配置（非构造参数）
FOUNDATION_BACKBONE_NAME = "official_r3d18"
FOUNDATION_USE_PRETRAINED = True
FOUNDATION_DROPOUT = 0.0


class FoundationModel(nn.Module):
    """
    多任务网络：
    - backbone: 提取带有空间维度的深层特征图 [B, C, D, H, W]
    - classification head: 将特征图池化后，进行全局病种三分类
    - segmentation head: 基于同一个特征图，输出像素级病灶 Mask
    """
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        num_seg_classes: int = 3,  # 新增：分割类别数（如 0:背景, 1:炎症, 2:转移瘤）
    ):
        super().__init__()
        self.backbone_name = FOUNDATION_BACKBONE_NAME
        self.in_channels = in_channels

        self.backbone, feat_dim = self._build_backbone(
            FOUNDATION_BACKBONE_NAME,
            FOUNDATION_USE_PRETRAINED,
            in_channels
        )
        self.neck = self._build_neck(feat_dim)
        
        # [新增] 分类用的全局池化层
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.head = self._build_head(feat_dim, num_classes, FOUNDATION_DROPOUT)
        
        # [新增] 分割辅助头：轻量级卷积降维网络，提取病灶轮廓
        self.aux_heads = nn.ModuleDict({
            "seg_head": nn.Sequential(
                nn.Conv3d(feat_dim, 128, kernel_size=3, padding=1),
                nn.InstanceNorm3d(128, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(128, 64, kernel_size=3, padding=1),
                nn.InstanceNorm3d(64, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, num_seg_classes, kernel_size=1)
            )
        })

    # ===== 可扩展点：backbone =====
    def _build_backbone(self, backbone_name: str, use_pretrained: bool, in_channels: int):
        if backbone_name == "official_r3d18":
            weights = R3D_18_Weights.DEFAULT if use_pretrained else None
            
            # 拿到原始的 model，这时候它在 CPU 上，还只是个躯壳
            original_model = r3d_18(weights=weights)

            # 仅做 BN -> IN
            self._replace_bn3d_with_in3d(original_model)

            # 适配输入通道（MRI 常见 1 通道）
            if in_channels == 1:
                original_model.stem[0] = self._adapt_first_conv_to_1ch(original_model.stem[0])
            elif in_channels != 3:
                raise ValueError("official_r3d18 仅支持 in_channels=1 或 3")

            feat_dim = original_model.fc.in_features
            
            # 【最标准的 PyTorch 做法】
            # 我们不要它原本自带扁平化的 forward，直接把它需要的组件拆下来组装成全新的序列网络。
            # 这样既完美避开了 flatten，又在 DataParallel 复制时能够保证所有权重严格对齐属于哪张卡。
            clean_backbone = nn.Sequential(
                original_model.stem,
                original_model.layer1,
                original_model.layer2,
                original_model.layer3,
                original_model.layer4
            )
            
            return clean_backbone, feat_dim

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
        # 此时提取出的是尚未池化的 3D 张量：[B, feat_dim, D', H', W']
        spatial_feat = self.backbone(x)   
        return spatial_feat

    def forward(self, x, return_seg=False):
        # 1. 提取具有空间维度的深层特征
        spatial_feat = self.forward_features(x)
        
        # 2. 分类分支：对特征图进行全局池化 -> 展平 -> 传入 Neck 和 Head
        pooled_feat = self.global_pool(spatial_feat)       # -> [B, feat_dim, 1, 1, 1]
        pooled_feat = torch.flatten(pooled_feat, 1)        # -> [B, feat_dim]
        feat = self.neck(pooled_feat)
        cls_logits = self.head(feat)
        
        # 3. 分割分支（开启 return_seg 时才会发生计算，节约普通推理期间的开销）
        if return_seg:
            seg_features = self.aux_heads["seg_head"](spatial_feat)
            
            # 使用差值上采样，将网络深层迷你的特征图等比放大回输入图像的原尺寸
            # 例如：x.shape[2:] 可能是 (160, 192, 160)
            seg_logits = F.interpolate(
                seg_features, 
                size=x.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
            return cls_logits, seg_logits
            
        return cls_logits