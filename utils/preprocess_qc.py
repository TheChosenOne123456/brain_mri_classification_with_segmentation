"""
预处理输出质量检查。

这些检查用于在数据进入训练前拦截明显污染样本，例如全零体数据、
几乎全零体数据、非零区域异常小、shape/spacing 不符合配置等。
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def validate_brain_extraction_input(
    img: sitk.Image,
    max_zero_ratio=0.995,
    normalization_percentile=99.0,
):
    """在调用脑提取器前拦截空体积和无法安全归一化的影像。

    SynthStrip 会先减去最小值，再除以第 99 百分位数。若输入全零、近乎
    全零或只有极少数异常非零点，该分母会是 0，随后网络输出全部 NaN。
    这里仅做输入有效性检查，不改变通过检查的影像强度或后续预处理流程。
    """
    reasons = []
    arr = sitk.GetArrayViewFromImage(img)

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count:
        reasons.append(f"nan:{nan_count}")
    if inf_count:
        reasons.append(f"inf:{inf_count}")

    finite_values = np.asarray(arr[np.isfinite(arr)], dtype=np.float64)
    if finite_values.size == 0:
        reasons.append("no_finite_voxels")
        return False, reasons

    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    scale = max(abs(minimum), abs(maximum), 1.0)
    tolerance = max(1e-6, np.finfo(np.float32).eps * scale * 16)
    if maximum - minimum <= tolerance:
        reasons.append(
            f"constant_intensity:min={minimum:.6g},max={maximum:.6g}"
        )

    percentile_value = float(
        np.percentile(finite_values, normalization_percentile)
    )
    if percentile_value - minimum <= tolerance:
        reasons.append(
            f"no_p99_contrast:min={minimum:.6g},p99={percentile_value:.6g},"
            f"max={maximum:.6g}"
        )

    zero_ratio = float(np.mean(arr == 0))
    if zero_ratio >= max_zero_ratio:
        reasons.append(f"mostly_zero:{zero_ratio:.6f}")

    return len(reasons) == 0, reasons


def validate_preprocessed_image(
    img: sitk.Image,
    target_shape,
    target_spacing,
    max_zero_ratio=0.995,
    min_nonzero_bbox_fraction=0.25,
    spacing_atol=1e-3,
):
    """
    返回 (ok, reasons)。
    target_shape 使用 SimpleITK array 维度顺序: (D, H, W)。
    target_spacing 使用 SimpleITK spacing 顺序: (X, Y, Z)，沿用项目配置。
    """
    reasons = []
    arr = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)

    if tuple(arr.shape) != tuple(target_shape):
        reasons.append(f"bad_shape:{tuple(arr.shape)}")

    spacing = tuple(float(x) for x in img.GetSpacing())
    if any(abs(a - b) > spacing_atol for a, b in zip(spacing, target_spacing)):
        reasons.append(f"bad_spacing:{spacing}")

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count:
        reasons.append(f"nan:{nan_count}")
    if inf_count:
        reasons.append(f"inf:{inf_count}")

    finite = np.isfinite(arr)
    if not finite.any():
        reasons.append("no_finite_voxels")
        return False, reasons

    zero_ratio = float(np.mean(arr == 0))
    if zero_ratio >= max_zero_ratio:
        reasons.append(f"mostly_zero:{zero_ratio:.6f}")

    nonzero = np.argwhere(arr != 0)
    if nonzero.size == 0:
        reasons.append("all_zero")
    else:
        bbox_shape = nonzero.max(axis=0) - nonzero.min(axis=0) + 1
        min_bbox_fraction = float(np.min(bbox_shape / np.asarray(target_shape)))
        if min_bbox_fraction < min_nonzero_bbox_fraction:
            reasons.append(f"tiny_nonzero_bbox:{tuple(int(x) for x in bbox_shape)}")

    return len(reasons) == 0, reasons


def validate_saved_file_size(path: Path, min_file_size_mb):
    if not path.exists():
        return False, ["missing_output_file"]

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb < min_file_size_mb:
        return False, [f"small_file:{file_size_mb:.6f}MB"]

    return True, []
