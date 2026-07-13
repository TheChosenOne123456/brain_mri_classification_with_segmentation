'''
定义center_crop_or_pad函数，统一图像尺寸
'''

import SimpleITK as sitk
import numpy as np

def _crop_or_pad_array(arr, target_shape, center):
    cropped, _ = _crop_or_pad_array_with_meta(arr, target_shape, center)
    return cropped


def _crop_or_pad_array_with_meta(arr, target_shape, center):
    out = np.zeros(target_shape, dtype=arr.dtype)

    for_dim = []
    for size, target, c in zip(arr.shape, target_shape, center):
        src_start = int(round(c - target / 2))
        src_end = src_start + target

        dst_start = max(-src_start, 0)
        src_start = max(src_start, 0)
        src_end = min(src_end, size)
        length = max(src_end - src_start, 0)
        dst_end = dst_start + length

        for_dim.append((slice(src_start, src_end), slice(dst_start, dst_end)))

    out[
        for_dim[0][1],
        for_dim[1][1],
        for_dim[2][1],
    ] = arr[
        for_dim[0][0],
        for_dim[1][0],
        for_dim[2][0],
    ]

    meta = {
        "source_shape_zyx": [int(x) for x in arr.shape],
        "target_shape_zyx": [int(x) for x in target_shape],
        "crop_center_zyx": [float(x) for x in center],
        "src_start_zyx": [int(dim[0].start) for dim in for_dim],
        "src_end_zyx": [int(dim[0].stop) for dim in for_dim],
        "dst_start_zyx": [int(dim[1].start) for dim in for_dim],
        "dst_end_zyx": [int(dim[1].stop) for dim in for_dim],
    }

    return out, meta


def center_crop_or_pad(img: sitk.Image, target_shape):
    """
    target_shape: (D, H, W)
    """
    arr = sitk.GetArrayFromImage(img)  # [D, H, W]
    d, h, w = arr.shape
    td, th, tw = target_shape

    # ---- crop ----
    d_start = max((d - td) // 2, 0)
    h_start = max((h - th) // 2, 0)
    w_start = max((w - tw) // 2, 0)

    arr = arr[
        d_start:d_start + td,
        h_start:h_start + th,
        w_start:w_start + tw
    ]

    # ---- pad ----
    pd = max(td - arr.shape[0], 0)
    ph = max(th - arr.shape[1], 0)
    pw = max(tw - arr.shape[2], 0)

    arr = np.pad(
        arr,
        (
            (pd // 2, pd - pd // 2),
            (ph // 2, ph - ph // 2),
            (pw // 2, pw - pw // 2),
        ),
        mode="constant",
        constant_values=0
    )

    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(img.GetSpacing())
    out.SetOrigin(img.GetOrigin())
    out.SetDirection(img.GetDirection())

    return out


def center_crop_or_pad_with_meta(img: sitk.Image, target_shape):
    """
    返回中心 crop/pad 后的图像及可复用的 crop/pad 参数。
    """
    arr = sitk.GetArrayFromImage(img)
    center = (np.asarray(arr.shape, dtype=np.float64) - 1) / 2.0
    arr, crop_meta = _crop_or_pad_array_with_meta(arr, target_shape, center)

    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(img.GetSpacing())
    out.SetOrigin(img.GetOrigin())
    out.SetDirection(img.GetDirection())

    return out, crop_meta


def crop_or_pad_around_mask(img: sitk.Image, mask_img: sitk.Image, target_shape):
    """
    以 mask 的 bbox 中心为裁剪中心，输出固定尺寸体数据。

    target_shape: (D, H, W)，对应 SimpleITK array 的 [Z, Y, X]。
    如果 mask 为空，回退到原来的数组中心裁剪/填充。
    """
    arr = sitk.GetArrayFromImage(img)
    mask = sitk.GetArrayFromImage(mask_img) > 0
    if mask.shape != arr.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {arr.shape}")

    coords = np.argwhere(mask)
    if coords.size == 0:
        return center_crop_or_pad(img, target_shape)

    min_zyx = coords.min(axis=0)
    max_zyx = coords.max(axis=0)
    center = (min_zyx + max_zyx) / 2.0

    arr = _crop_or_pad_array(arr, target_shape, center)

    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(img.GetSpacing())
    out.SetOrigin(img.GetOrigin())
    out.SetDirection(img.GetDirection())

    return out


def crop_or_pad_around_mask_with_meta(img: sitk.Image, mask_img: sitk.Image, target_shape):
    """
    以 mask bbox 中心裁剪/填充，并返回可复用的 crop/pad 参数。
    """
    arr = sitk.GetArrayFromImage(img)
    mask = sitk.GetArrayFromImage(mask_img) > 0
    if mask.shape != arr.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {arr.shape}")

    coords = np.argwhere(mask)
    if coords.size == 0:
        return center_crop_or_pad_with_meta(img, target_shape)

    min_zyx = coords.min(axis=0)
    max_zyx = coords.max(axis=0)
    center = (min_zyx + max_zyx) / 2.0

    arr, crop_meta = _crop_or_pad_array_with_meta(arr, target_shape, center)
    crop_meta["foreground_bbox_min_zyx"] = [int(x) for x in min_zyx]
    crop_meta["foreground_bbox_max_zyx"] = [int(x) for x in max_zyx]

    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(img.GetSpacing())
    out.SetOrigin(img.GetOrigin())
    out.SetDirection(img.GetDirection())

    return out, crop_meta


def apply_crop_or_pad_from_meta(img: sitk.Image, crop_meta):
    """
    对另一个已处于相同 resampled reference 的图像应用保存的 crop/pad 参数。
    常用于医生 mask，保证与 MRI 图像逐体素对齐。
    """
    arr = sitk.GetArrayFromImage(img)
    expected_shape = tuple(crop_meta["source_shape_zyx"])
    if tuple(arr.shape) != expected_shape:
        raise ValueError(f"Image shape {arr.shape} does not match metadata source shape {expected_shape}")

    target_shape = tuple(crop_meta["target_shape_zyx"])
    out_arr = np.zeros(target_shape, dtype=arr.dtype)

    src_slices = tuple(
        slice(start, end)
        for start, end in zip(crop_meta["src_start_zyx"], crop_meta["src_end_zyx"])
    )
    dst_slices = tuple(
        slice(start, end)
        for start, end in zip(crop_meta["dst_start_zyx"], crop_meta["dst_end_zyx"])
    )
    out_arr[dst_slices] = arr[src_slices]

    out = sitk.GetImageFromArray(out_arr)
    out.SetSpacing(img.GetSpacing())
    out.SetOrigin(img.GetOrigin())
    out.SetDirection(img.GetDirection())

    return out
