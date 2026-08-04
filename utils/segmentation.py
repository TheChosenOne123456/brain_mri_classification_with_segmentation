import torch
import torch.nn.functional as F


def prepare_binary_mask(mask, spatial_shape=None):
    """把项目中的 0/1/2 mask 统一转换为单通道二值 lesion mask。"""
    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    if mask.dim() != 5:
        raise ValueError(f"Expected mask [B,1,D,H,W], got {tuple(mask.shape)}")
    mask = (mask > 0).float()
    if spatial_shape is not None and mask.shape[2:] != tuple(spatial_shape):
        mask = F.interpolate(mask, size=spatial_shape, mode="nearest")
    return mask


def binary_segmentation_loss(
    logits,
    mask,
    mask_flag,
    *,
    bce_weight=1.0,
    dice_weight=1.0,
    positive_weight=None,
    bce_mode="mean",
    hard_negative_fraction=0.002,
    hard_negative_min_voxels=256,
    normal_dense_bce_weight=0.5,
    smooth=1e-5,
):
    """逐病例计算 BCE + soft Dice，只聚合有可靠 mask 的病例。

    ``hard_negative`` 模式分别归一化正体素与最难负体素，避免局部假阳性被
    全体积简单背景稀释；全零 mask 只使用 dense/hard-negative BCE，不计算
    几乎无梯度的 empty-target Dice。
    """
    if logits.dim() != 5 or logits.size(1) != 1:
        raise ValueError(
            "Binary lesion logits must have shape [B,1,D,H,W], got "
            f"{tuple(logits.shape)}"
        )
    original_target = prepare_binary_mask(mask).to(logits.device)
    target = prepare_binary_mask(mask, logits.shape[2:]).to(logits.device)
    valid = mask_flag.to(device=logits.device, dtype=torch.float32).reshape(-1)
    if valid.numel() != logits.size(0):
        raise ValueError("mask_flag batch size does not match segmentation logits")
    if bce_mode not in ("mean", "hard_negative"):
        raise ValueError(
            "bce_mode must be 'mean' or 'hard_negative', got "
            f"{bce_mode!r}"
        )
    if not 0 < float(hard_negative_fraction) <= 1:
        raise ValueError("hard_negative_fraction must lie in (0, 1]")
    if int(hard_negative_min_voxels) <= 0:
        raise ValueError("hard_negative_min_voxels must be positive")
    if not 0 <= float(normal_dense_bce_weight) <= 1:
        raise ValueError("normal_dense_bce_weight must lie in [0, 1]")

    pos_weight = None
    if positive_weight is not None:
        pos_weight = torch.as_tensor(
            positive_weight,
            device=logits.device,
            dtype=torch.float32,
        )
    voxel_bce = F.binary_cross_entropy_with_logits(
        logits.float(),
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    dense_bce = voxel_bce.mean(dim=(1, 2, 3, 4))
    resized_has_positive = target.flatten(1).sum(dim=1) > 0
    original_has_positive = original_target.flatten(1).sum(dim=1) > 0
    vanished_positive = original_has_positive & ~resized_has_positive

    if bce_mode == "mean":
        per_sample_bce = dense_bce
    else:
        # 最近邻缩放可能让极小病灶从低分辨率辅助目标中消失；这种辅助层
        # 不应把该阳性病例误当成全零负例。
        valid = valid * (~vanished_positive).to(valid.dtype)
        flat_bce = voxel_bce.flatten(1)
        flat_target = target.flatten(1) > 0.5
        hard_bce = []
        for sample_index in range(logits.size(0)):
            if valid[sample_index] <= 0:
                hard_bce.append(flat_bce[sample_index].sum() * 0.0)
                continue
            negative_losses = flat_bce[sample_index][~flat_target[sample_index]]
            negative_count = int(negative_losses.numel())
            if negative_count:
                hard_count = min(
                    negative_count,
                    max(
                        int(hard_negative_min_voxels),
                        int(
                            round(
                                float(hard_negative_fraction) * negative_count
                            )
                        ),
                    ),
                )
                hard_negative_bce = negative_losses.topk(
                    hard_count,
                    sorted=False,
                ).values.mean()
            else:
                hard_negative_bce = flat_bce[sample_index].sum() * 0.0
            if resized_has_positive[sample_index]:
                positive_bce = flat_bce[sample_index][
                    flat_target[sample_index]
                ].mean()
                sample_bce = 0.5 * (positive_bce + hard_negative_bce)
            else:
                dense_weight = float(normal_dense_bce_weight)
                sample_bce = (
                    dense_weight * dense_bce[sample_index]
                    + (1.0 - dense_weight) * hard_negative_bce
                )
            hard_bce.append(sample_bce)
        per_sample_bce = torch.stack(hard_bce)

    probabilities = torch.sigmoid(logits.float())
    intersection = (probabilities * target).sum(dim=(1, 2, 3, 4))
    cardinality = probabilities.sum(dim=(1, 2, 3, 4)) + target.sum(
        dim=(1, 2, 3, 4)
    )
    per_sample_dice_loss = 1.0 - (
        (2.0 * intersection + smooth) / (cardinality + smooth)
    )
    if bce_mode == "hard_negative":
        per_sample_dice_loss = torch.where(
            resized_has_positive,
            per_sample_dice_loss,
            torch.zeros_like(per_sample_dice_loss),
        )
    per_sample_loss = (
        float(bce_weight) * per_sample_bce
        + float(dice_weight) * per_sample_dice_loss
    )

    valid_count = valid.sum()
    if valid_count > 0:
        return (per_sample_loss * valid).sum() / valid_count
    return logits.sum() * 0.0


def binary_lesion_predictions(logits, threshold=0.5):
    if logits.dim() != 5 or logits.size(1) != 1:
        raise ValueError(
            "Binary lesion logits must have shape [B,1,D,H,W], got "
            f"{tuple(logits.shape)}"
        )
    return (torch.sigmoid(logits.float()) >= threshold).long().squeeze(1)


def binary_dice_per_sample(prediction, mask, smooth=1e-5):
    """计算二值病灶 Dice；调用方负责筛选 positive-case。"""
    if prediction.dim() == 5 and prediction.size(1) == 1:
        prediction = prediction.squeeze(1)
    target = prepare_binary_mask(mask).squeeze(1).to(prediction.device)
    prediction = (prediction > 0).float()
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    cardinality = prediction.sum(dim=(1, 2, 3)) + target.sum(
        dim=(1, 2, 3)
    )
    return (2.0 * intersection + smooth) / (cardinality + smooth)


def binary_recall_per_sample(prediction, mask, smooth=1e-5):
    """计算病灶体素 recall，仅适合在真实 mask 非空的病例上聚合。"""
    if prediction.dim() == 5 and prediction.size(1) == 1:
        prediction = prediction.squeeze(1)
    target = prepare_binary_mask(mask).squeeze(1).to(prediction.device)
    prediction = (prediction > 0).float()
    true_positive = (prediction * target).sum(dim=(1, 2, 3))
    positive = target.sum(dim=(1, 2, 3))
    return (true_positive + smooth) / (positive + smooth)
