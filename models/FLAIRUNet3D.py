import torch
import torch.nn as nn
import torch.nn.functional as F


FLAIR_UNET_CHANNELS = (16, 32, 64, 128, 256)
FLAIR_NNUNET_CHANNELS = (32, 64, 128, 256, 320, 320)
FLAIR_NNUNET_KERNELS = (
    (1, 3, 3),
    (1, 3, 3),
    (3, 3, 3),
    (3, 3, 3),
    (3, 3, 3),
    (3, 3, 3),
)
FLAIR_NNUNET_STRIDES = (
    (1, 1, 1),
    (1, 2, 2),
    (1, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
)
FLAIR_UNET_CLASSIFIER_DIM = 128
FLAIR_UNET_CLASSIFIER_DROPOUT = 0.2
FLAIR_UNET_ATTENTION_EPS = 1e-6


class DoubleConv3d(nn.Module):
    """标准 U-Net 双卷积块；小 batch 场景使用 InstanceNorm。"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = tuple(size // 2 for size in kernel_size)
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AnisotropicUNetEncoder3d(nn.Module):
    """
    适配 (3.0, 0.75, 0.75) mm FLAIR 的 U-Net 编码器。

    前两级只在平面内卷积/下采样；当平面 spacing 接近层厚后，再使用
    各向同性 3D 卷积和下采样。
    """

    def __init__(self, in_channels, channels):
        super().__init__()
        if len(channels) != 5:
            raise ValueError("FLAIRUNet3D requires exactly five encoder levels")

        kernels = (
            (1, 3, 3),
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        )
        self.downsample_strides = (
            (1, 2, 2),
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        )

        blocks = []
        current_channels = in_channels
        for output_channels, kernel_size in zip(channels, kernels):
            blocks.append(
                DoubleConv3d(
                    current_channels,
                    output_channels,
                    kernel_size,
                )
            )
            current_channels = output_channels
        self.blocks = nn.ModuleList(blocks)
        self.pools = nn.ModuleList(
            nn.MaxPool3d(kernel_size=stride, stride=stride)
            for stride in self.downsample_strides
        )

    def forward(self, x):
        skip_features = []
        for level, pool in enumerate(self.pools):
            x = self.blocks[level](x)
            skip_features.append(x)
            x = pool(x)
        bottleneck = self.blocks[-1](x)
        return bottleneck, tuple(skip_features)


class UNetDecoder3d(nn.Module):
    """转置卷积上采样、同层 skip connection 和多尺度辅助输出。"""

    def __init__(self, channels):
        super().__init__()
        encoder_strides = (
            (1, 2, 2),
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        )
        decoder_kernels = (
            (3, 3, 3),
            (3, 3, 3),
            (1, 3, 3),
            (1, 3, 3),
        )

        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.deep_supervision_heads = nn.ModuleList()

        current_channels = channels[-1]
        decoder_levels = zip(
            reversed(channels[:-1]),
            reversed(encoder_strides),
            decoder_kernels,
        )
        for level, (output_channels, stride, kernel_size) in enumerate(
            decoder_levels
        ):
            self.upconvs.append(
                nn.ConvTranspose3d(
                    current_channels,
                    output_channels,
                    kernel_size=stride,
                    stride=stride,
                )
            )
            self.blocks.append(
                DoubleConv3d(
                    output_channels * 2,
                    output_channels,
                    kernel_size,
                )
            )
            if level < len(channels) - 2:
                self.deep_supervision_heads.append(
                    nn.Conv3d(output_channels, 1, kernel_size=1)
                )
            current_channels = output_channels

    @staticmethod
    def _match_skip_shape(x, skip):
        if x.shape[2:] == skip.shape[2:]:
            return x
        return F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, bottleneck, skip_features):
        x = bottleneck
        decoder_features = []
        auxiliary_logits = []
        decoder_levels = zip(
            self.upconvs,
            self.blocks,
            reversed(skip_features),
        )
        for level, (upconv, block, skip) in enumerate(decoder_levels):
            x = self._match_skip_shape(upconv(x), skip)
            x = block(torch.cat((skip, x), dim=1))
            decoder_features.append(x)
            if level < len(self.deep_supervision_heads):
                auxiliary_logits.append(
                    self.deep_supervision_heads[level](x)
                )
        return x, tuple(decoder_features), tuple(auxiliary_logits)


class StridedDoubleConv3d(nn.Module):
    """nnU-Net PlainConvUNet 风格的两层卷积 stage。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        first_stride=(1, 1, 1),
    ):
        super().__init__()
        padding = tuple(size // 2 for size in kernel_size)
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=first_stride,
                padding=padding,
                bias=True,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class NNUNetStyleEncoder3d(nn.Module):
    """与 Dataset501 计划一致的六级 PlainConvUNet 编码器。"""

    def __init__(self, in_channels, channels, kernels, strides):
        super().__init__()
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("channels, kernels and strides must have equal length")
        if tuple(strides[0]) != (1, 1, 1):
            raise ValueError("The first encoder stage must have unit stride")

        stages = []
        current_channels = in_channels
        for output_channels, kernel_size, stride in zip(
            channels,
            kernels,
            strides,
        ):
            stages.append(
                StridedDoubleConv3d(
                    current_channels,
                    output_channels,
                    kernel_size,
                    first_stride=stride,
                )
            )
            current_channels = output_channels
        self.stages = nn.ModuleList(stages)
        self.stage_strides = tuple(tuple(value) for value in strides)

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features[-1], tuple(features[:-1])


class NNUNetStyleDecoder3d(nn.Module):
    """与 nnU-Net PlainConvUNet 对齐的转置卷积 skip decoder。"""

    def __init__(self, channels, kernels, strides):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.deep_supervision_heads = nn.ModuleList()

        current_channels = channels[-1]
        decoder_spec = zip(
            reversed(channels[:-1]),
            reversed(kernels[:-1]),
            reversed(strides[1:]),
        )
        for level, (output_channels, kernel_size, stride) in enumerate(
            decoder_spec
        ):
            self.upconvs.append(
                nn.ConvTranspose3d(
                    current_channels,
                    output_channels,
                    kernel_size=stride,
                    stride=stride,
                    bias=True,
                )
            )
            self.blocks.append(
                StridedDoubleConv3d(
                    output_channels * 2,
                    output_channels,
                    kernel_size,
                )
            )
            if level < len(channels) - 2:
                self.deep_supervision_heads.append(
                    nn.Conv3d(output_channels, 2, kernel_size=1, bias=True)
                )
            current_channels = output_channels

    @staticmethod
    def _match_skip_shape(x, skip):
        if x.shape[2:] == skip.shape[2:]:
            return x
        return F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, bottleneck, skip_features):
        x = bottleneck
        decoder_features = []
        auxiliary_logits = []
        for level, (upconv, block, skip) in enumerate(
            zip(self.upconvs, self.blocks, reversed(skip_features))
        ):
            x = self._match_skip_shape(upconv(x), skip)
            x = block(torch.cat((skip, x), dim=1))
            decoder_features.append(x)
            if level < len(self.deep_supervision_heads):
                auxiliary_logits.append(
                    self.deep_supervision_heads[level](x)
                )
        return x, tuple(decoder_features), tuple(auxiliary_logits)


class FLAIRUNet3D(nn.Module):
    """
    FLAIR 专用的病灶分割/三分类联合模型。

    segmentation 输出单通道二值 lesion logit；classification 同时使用编码器
    bottleneck 的全局特征与预测 soft lesion mask 引导的高分辨率局部特征。
    """

    has_classification_head = True
    has_subtype_head = False
    has_segmentation_head = True
    uses_capability_interface = True
    supports_segmentation_only_forward = True
    segmentation_target_mode = "binary_lesion"
    segmentation_num_classes = 1

    def __init__(
        self,
        num_classes=3,
        in_channels=1,
        channels=FLAIR_UNET_CHANNELS,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError("FLAIRUNet3D requires a single FLAIR input channel")
        if num_classes != 3:
            raise ValueError(
                "FLAIRUNet3D requires normal/inflammation/metastasis classes"
            )

        channels = tuple(channels)
        self.in_channels = in_channels
        self.channels = channels
        self.encoder = AnisotropicUNetEncoder3d(in_channels, channels)
        self.decoder = UNetDecoder3d(channels)
        self.aux_heads = nn.ModuleDict(
            {"seg_head": nn.Conv3d(channels[0], 1, kernel_size=1)}
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_projection = nn.Sequential(
            nn.Linear(channels[-1], FLAIR_UNET_CLASSIFIER_DIM),
            nn.LayerNorm(FLAIR_UNET_CLASSIFIER_DIM),
            nn.GELU(),
        )
        self.lesion_projection = nn.Sequential(
            nn.Linear(channels[0], FLAIR_UNET_CLASSIFIER_DIM),
            nn.LayerNorm(FLAIR_UNET_CLASSIFIER_DIM),
            nn.GELU(),
        )
        self.classification_head = nn.Sequential(
            nn.Dropout(FLAIR_UNET_CLASSIFIER_DROPOUT),
            nn.Linear(FLAIR_UNET_CLASSIFIER_DIM * 2, num_classes),
        )

    @staticmethod
    def _soft_lesion_pool(features, lesion_logits):
        attention = torch.sigmoid(lesion_logits.float()).to(features.dtype)
        if attention.shape[2:] != features.shape[2:]:
            attention = F.interpolate(
                attention,
                size=features.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        numerator = (features * attention).sum(dim=(2, 3, 4))
        denominator = attention.sum(dim=(2, 3, 4)).clamp_min(
            FLAIR_UNET_ATTENTION_EPS
        )
        return numerator / denominator

    def forward_features(self, x):
        bottleneck, skip_features = self.encoder(x)
        high_resolution_features, decoder_features, auxiliary_logits = (
            self.decoder(bottleneck, skip_features)
        )
        lesion_logits = self.aux_heads["seg_head"](high_resolution_features)
        return {
            "bottleneck": bottleneck,
            "decoder_features": decoder_features,
            "segmentation": lesion_logits,
            "segmentation_aux": auxiliary_logits,
        }

    def forward(
        self,
        x,
        return_seg=False,
        return_subtype=False,
        return_dict=False,
        segmentation_only=False,
    ):
        if return_subtype:
            raise ValueError("FLAIRUNet3D does not have a subtype head")

        features = self.forward_features(x)
        bottleneck = features["bottleneck"]
        lesion_logits = features["segmentation"]
        high_resolution_features = features["decoder_features"][-1]

        if segmentation_only:
            if not return_seg or not return_dict:
                raise ValueError(
                    "segmentation_only requires structured segmentation output"
                )
            return {
                "classification": lesion_logits.new_zeros(
                    (x.size(0), self.classification_head[-1].out_features)
                ),
                "segmentation": lesion_logits,
                "segmentation_aux": features["segmentation_aux"],
            }

        global_features = torch.flatten(self.global_pool(bottleneck), 1)
        lesion_features = self._soft_lesion_pool(
            high_resolution_features,
            lesion_logits,
        )
        classification_logits = self.classification_head(
            torch.cat(
                (
                    self.global_projection(global_features),
                    self.lesion_projection(lesion_features),
                ),
                dim=1,
            )
        )

        if return_dict:
            outputs = {"classification": classification_logits}
            if return_seg:
                outputs["segmentation"] = lesion_logits
                outputs["segmentation_aux"] = features["segmentation_aux"]
            return outputs
        if return_seg:
            return classification_logits, lesion_logits
        return classification_logits


class FLAIRUNet3DNNUNet(FLAIRUNet3D):
    """保留双路分类接口的 nnU-Net Dataset501 结构复现版本。"""

    def __init__(self, num_classes=3, in_channels=1):
        nn.Module.__init__(self)
        if in_channels != 1:
            raise ValueError(
                "FLAIRUNet3DNNUNet requires a single FLAIR input channel"
            )
        if num_classes != 3:
            raise ValueError(
                "FLAIRUNet3DNNUNet requires normal/inflammation/metastasis classes"
            )

        channels = FLAIR_NNUNET_CHANNELS
        self.in_channels = in_channels
        self.channels = channels
        self.encoder = NNUNetStyleEncoder3d(
            in_channels,
            channels,
            FLAIR_NNUNET_KERNELS,
            FLAIR_NNUNET_STRIDES,
        )
        self.decoder = NNUNetStyleDecoder3d(
            channels,
            FLAIR_NNUNET_KERNELS,
            FLAIR_NNUNET_STRIDES,
        )
        self.aux_heads = nn.ModuleDict(
            {"seg_head": nn.Conv3d(channels[0], 2, kernel_size=1, bias=True)}
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_projection = nn.Sequential(
            nn.Linear(channels[-1], FLAIR_UNET_CLASSIFIER_DIM),
            nn.LayerNorm(FLAIR_UNET_CLASSIFIER_DIM),
            nn.GELU(),
        )
        self.lesion_projection = nn.Sequential(
            nn.Linear(channels[0], FLAIR_UNET_CLASSIFIER_DIM),
            nn.LayerNorm(FLAIR_UNET_CLASSIFIER_DIM),
            nn.GELU(),
        )
        self.classification_head = nn.Sequential(
            nn.Dropout(FLAIR_UNET_CLASSIFIER_DROPOUT),
            nn.Linear(FLAIR_UNET_CLASSIFIER_DIM * 2, num_classes),
        )
        self.apply(self._initialize_nnunet_convolutions)

    @staticmethod
    def _initialize_nnunet_convolutions(module):
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def _foreground_logit(two_class_logits):
        if two_class_logits.size(1) != 2:
            raise ValueError(
                "nnU-Net style segmentation heads must emit two logits"
            )
        return two_class_logits[:, 1:2] - two_class_logits[:, 0:1]

    def forward_features(self, x):
        bottleneck, skip_features = self.encoder(x)
        high_resolution_features, decoder_features, auxiliary_logits = (
            self.decoder(bottleneck, skip_features)
        )
        lesion_logits = self._foreground_logit(
            self.aux_heads["seg_head"](high_resolution_features)
        )
        return {
            "bottleneck": bottleneck,
            "decoder_features": decoder_features,
            "segmentation": lesion_logits,
            "segmentation_aux": tuple(
                self._foreground_logit(logits)
                for logits in auxiliary_logits
            ),
        }
