import numpy as np
import SimpleITK as sitk

def normalize_intensity(
    img,
    mask_img=None,
    clip_percentiles=None,
    robust=False,
):
    """
    输入 SimpleITK.Image，返回归一化后的 Image
    默认使用 z-score 归一化。

    mask_img 不为空时，只在 mask 内计算统计量，避免背景、padding、
    FOV 和脖颈比例影响强度归一化。
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    if mask_img is not None:
        mask = sitk.GetArrayFromImage(mask_img) > 0
        if mask.shape != arr.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {arr.shape}"
            )
        values = arr[mask]
        if values.size == 0:
            values = arr.reshape(-1)
    else:
        values = arr.reshape(-1)

    if clip_percentiles is not None:
        low, high = clip_percentiles
        lo = np.percentile(values, low)
        hi = np.percentile(values, high)
        if hi > lo:
            arr = np.clip(arr, lo, hi)
            if mask_img is not None:
                values = arr[mask]
            else:
                values = arr.reshape(-1)

    if robust:
        center = np.median(values)
        q1, q3 = np.percentile(values, [25, 75])
        scale = q3 - q1
    else:
        center = values.mean()
        scale = values.std()

    if scale > 0:
        arr = (arr - center) / scale
    else:
        arr = arr - center  # 避免除0

    out = sitk.GetImageFromArray(arr, isVector=False)
    out.CopyInformation(img)  # 还原完整的物理空间信息(Spacing, Direction, Origin)
    return out


def zero_outside_mask(img, mask_img):
    """
    将 mask 外部置 0，保留 mask 内归一化后的真实信号。
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    mask = sitk.GetArrayFromImage(mask_img) > 0
    if mask.shape != arr.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {arr.shape}")
    arr[~mask] = 0.0

    out = sitk.GetImageFromArray(arr, isVector=False)
    out.CopyInformation(img)
    return out
