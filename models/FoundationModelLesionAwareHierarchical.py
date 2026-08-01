import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FoundationModelHierarchical import FoundationModelHierarchical


LESION_SUBTYPE_HIDDEN_DIM = 256
LESION_SUBTYPE_DROPOUT = 0.2
LESION_POOL_EPS = 1e-6


class FoundationModelLesionAwareHierarchical(FoundationModelHierarchical):
    """
    FLAIR 专用层级模型。

    保留历史三分类主头，同时使用分割头产生的 soft lesion attention 对空间
    backbone 特征做病灶加权池化；subtype MLP 联合全局和病灶特征区分炎症/转移。
    """

    has_classification_head = True
    has_subtype_head = True
    has_segmentation_head = True
    uses_capability_interface = True

    def __init__(
        self,
        num_classes=3,
        in_channels=1,
        num_subtype_classes=2,
        num_seg_classes=3,
    ):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            num_subtype_classes=num_subtype_classes,
            num_seg_classes=num_seg_classes,
            enable_segmentation=True,
        )
        feat_dim = self.head.in_features
        self.subtype_head = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, LESION_SUBTYPE_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(LESION_SUBTYPE_DROPOUT),
            nn.Linear(LESION_SUBTYPE_HIDDEN_DIM, num_subtype_classes),
        )

    @staticmethod
    def _lesion_attention(segmentation_logits):
        segmentation_probabilities = F.softmax(segmentation_logits, dim=1)
        return segmentation_probabilities[:, 1:].sum(dim=1, keepdim=True)

    @staticmethod
    def _attention_pool(spatial_features, lesion_attention):
        weighted_features = spatial_features * lesion_attention
        numerator = weighted_features.sum(dim=(2, 3, 4))
        denominator = lesion_attention.sum(dim=(2, 3, 4)).clamp_min(
            LESION_POOL_EPS
        )
        return numerator / denominator

    def forward(
        self,
        x,
        return_seg=False,
        return_subtype=False,
        return_dict=False,
    ):
        spatial_features = self.forward_features(x)
        global_features = torch.flatten(
            self.global_pool(spatial_features),
            1,
        )
        global_features = self.neck(global_features)
        classification_logits = self.head(global_features)

        low_resolution_segmentation_logits = None
        subtype_logits = None
        if return_subtype or return_seg:
            low_resolution_segmentation_logits = self.aux_heads["seg_head"](
                spatial_features
            )

        if return_subtype:
            lesion_attention = self._lesion_attention(
                low_resolution_segmentation_logits
            )
            lesion_features = self._attention_pool(
                spatial_features,
                lesion_attention,
            )
            subtype_features = torch.cat(
                (global_features, lesion_features),
                dim=1,
            )
            subtype_logits = self.subtype_head(subtype_features)

        segmentation_logits = None
        if return_seg:
            segmentation_logits = F.interpolate(
                low_resolution_segmentation_logits,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        if return_dict:
            outputs = {"classification": classification_logits}
            if subtype_logits is not None:
                outputs["subtype"] = subtype_logits
            if segmentation_logits is not None:
                outputs["segmentation"] = segmentation_logits
            return outputs

        requested_auxiliary_outputs = []
        if return_subtype:
            requested_auxiliary_outputs.append(subtype_logits)
        if return_seg:
            requested_auxiliary_outputs.append(segmentation_logits)
        if requested_auxiliary_outputs:
            return (
                classification_logits,
                *requested_auxiliary_outputs,
            )
        return classification_logits
