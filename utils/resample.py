import SimpleITK as sitk
from pathlib import Path


def read_image_3d(img_path):
    try:
        # 尝试读取图像
        img = sitk.ReadImage(str(img_path))
    except Exception:
        # 捕获所有读取错误（包括格式不支持、文件损坏等）
        # 返回 None 表示失败
        return None

    if img.GetDimension() == 4:
        size = list(img.GetSize())
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(size[:3] + [0])
        extractor.SetIndex([0, 0, 0, 0])
        img = extractor.Execute(img)

    return img


def resample_image(img_path, target_spacing=(1.0, 1.0, 1.0), is_label=False):
    img = read_image_3d(img_path)
    if img is None:
        return None

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    # ---- 构造 reference image（关键）----
    reference = sitk.Image(new_size, img.GetPixelID())
    reference.SetSpacing(target_spacing)
    reference.SetOrigin(img.GetOrigin())
    reference.SetDirection(img.GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)

    return resampler.Execute(img)


def resample_to_reference(img_path, reference_img, is_label=False):
    img = read_image_3d(img_path)
    if img is None:
        return None

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetDefaultPixelValue(0)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)

    return resampler.Execute(img)


def save_image(img, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))
