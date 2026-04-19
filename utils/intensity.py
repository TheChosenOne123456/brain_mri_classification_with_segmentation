import numpy as np
import SimpleITK as sitk

def normalize_intensity(img):
    """
    输入 SimpleITK.Image，返回归一化后的 Image
    使用 z-score 归一化
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    mean = arr.mean()
    std = arr.std()
    if std > 0:
        arr = (arr - mean) / std
    else:
        arr = arr - mean  # 避免除0
    out = sitk.GetImageFromArray(arr, isVector=False)
    out.CopyInformation(img)  # 还原完整的物理空间信息(Spacing, Direction, Origin)
    return out
