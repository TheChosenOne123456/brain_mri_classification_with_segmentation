import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    ResNet 的基本残差块 (Basic Residual Block)
    包含两个 3x3x3 卷积层和一条 Shortcut 连接
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # 第一层卷积：如果 stride > 1，在这里进行下采样
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # 使用 InstanceNorm3d 代替 BatchNorm3d
        # 原因：你的 Batch Size = 2，BN 的统计量会非常不稳定，导致训练困难。
        # IN 对 Batch Size 不敏感，是 3D 医学图像任务的首选。
        # affine=True 让 IN 层拥有可学习的参数 (gamma, beta)
        self.bn1 = nn.InstanceNorm3d(planes, affine=True)
        
        # 第二层卷积：保持尺寸不变
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.InstanceNorm3d(planes, affine=True)

        # Shortcut (跳跃连接)
        # 如果输入输出维度不一致（stride!=1 或通道数改变），需要用 1x1 卷积调整 x 的形状
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_planes, self.expansion * planes, 
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.InstanceNorm3d(self.expansion * planes, affine=True)
            )

    def forward(self, x):
        # 主路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 残差连接：F(x) + x
        # 这让梯度可以直接流向浅层，解决了深层网络难以训练的问题
        out += self.shortcut(x)
        
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    has_classification_head = True
    has_subtype_head = False
    has_segmentation_head = False
    uses_capability_interface = False

    def __init__(self, block, num_blocks, num_classes=2, in_channels=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # ================== 初始层 ==================
        # 标准 ResNet 使用 7x7 stride=2，但对于 MRI（特别是 Z 轴层数较少时），
        # 过早的下采样会丢失细节。这里改用 3x3 stride=1 保留分辨率。
        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.InstanceNorm3d(64, affine=True)
        
        # ================== 残差层 (Layer 1-4) ==================
        # Layer 1: 64通道，不降采样
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # Layer 2: 128通道，降采样 (stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # Layer 3: 256通道，降采样 (stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # Layer 4: 512通道，降采样 (stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # ================== 分类头 ==================
        # 全局平均池化：无论输入尺寸多大，都压缩成 1x1x1
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 1, D, H, W]
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out) # [B, 64, D, H, W]
        out = self.layer2(out) # [B, 128, D/2, H/2, W/2]
        out = self.layer3(out) # [B, 256, D/4, H/4, W/4]
        out = self.layer4(out) # [B, 512, D/8, H/8, W/8]

        out = self.avgpool(out) # [B, 512, 1, 1, 1]
        out = out.flatten(1)    # [B, 512]
        
        return self.fc(out)


# ================== 快捷入口 ==================

def ResNet10(num_classes=2, in_channels=1):
    """
    ResNet-10: 较浅的网络，适合数据量较少的情况
    结构: [1, 1, 1, 1] 个 Block
    """
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, in_channels=in_channels)

def ResNet18(num_classes=2, in_channels=1):
    """
    ResNet-18: 标准轻量级 ResNet
    结构: [2, 2, 2, 2] 个 Block
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)
