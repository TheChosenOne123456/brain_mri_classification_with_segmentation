"""
脑部 foreground mask 生成工具。

支持 HD-BET 和 SynthStrip。两者都以 NIfTI 路径为接口，因此这里用临时
目录做包装：预处理过程中临时写入输入和 mask，读取后即删除。项目最终
只保存模型需要的预处理图像。
"""

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import subprocess
import sys

import SimpleITK as sitk


def _resolve_hd_bet_device(device):
    if device != "auto":
        if device == "cuda":
            return 0
        if isinstance(device, str) and device.isdigit():
            return int(device)
        return device

    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_orientation(img: sitk.Image):
    """
    返回 SimpleITK 根据 direction cosines 推断出的三字母方向码，例如 RAS/LPS。
    """
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        img.GetDirection()
    )


def orient_to_target(img: sitk.Image, target_orientation):
    """
    如果图像方向不是 target_orientation，重定向到目标方向。

    SimpleITK 会通过轴翻转/置换保持物理空间一致；后续 mask 会再重采样回原始
    预处理网格。
    """
    current_orientation = get_orientation(img)
    if current_orientation == target_orientation:
        return img, False, current_orientation

    oriented = sitk.DICOMOrient(img, target_orientation)
    return oriented, True, current_orientation


def resample_mask_to_reference(mask_img: sitk.Image, reference_img: sitk.Image):
    """
    将 mask 最近邻重采样回 reference_img 网格。
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampled = resampler.Execute(mask_img)
    resampled = sitk.Cast(resampled > 0, sitk.sitkUInt8)
    resampled.CopyInformation(reference_img)
    return resampled


def extract_foreground_mask_hd_bet(
    img: sitk.Image,
    device="auto",
    mode="fast",
    do_tta=False,
    postprocess=True,
    target_orientation="RAS",
    verbose=False,
):
    """
    使用 HD-BET 为 SimpleITK.Image 生成 brain/foreground mask。

    返回的 mask 与输入 img 处于同一 voxel grid。
    """
    try:
        from HD_BET.run import run_hd_bet
    except ImportError as e:
        raise RuntimeError(
            "HD-BET is not installed. Install it in the active environment with "
            "`pip install hd-bet`, or run preprocessing with `--brain_extractor none`."
        ) from e

    resolved_device = _resolve_hd_bet_device(device)
    hdbet_img, was_reoriented, original_orientation = orient_to_target(
        img,
        target_orientation,
    )

    with TemporaryDirectory(prefix="brain_mri_hdbet_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        in_path = tmp_dir / "input.nii.gz"
        out_path = tmp_dir / "output.nii.gz"
        mask_path = tmp_dir / "output_mask.nii.gz"

        sitk.WriteImage(hdbet_img, str(in_path))

        if verbose and was_reoriented:
            print(
                f"Reoriented image for HD-BET: "
                f"{original_orientation} -> {target_orientation}"
            )

        if verbose:
            run_hd_bet(
                str(in_path),
                str(out_path),
                mode=mode,
                device=resolved_device,
                postprocess=postprocess,
                do_tta=do_tta,
                keep_mask=True,
                overwrite=True,
                bet=False,
            )
        else:
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    run_hd_bet(
                        str(in_path),
                        str(out_path),
                        mode=mode,
                        device=resolved_device,
                        postprocess=postprocess,
                        do_tta=do_tta,
                        keep_mask=True,
                        overwrite=True,
                        bet=False,
                    )

        if not mask_path.exists():
            raise RuntimeError("HD-BET did not produce the expected mask file.")

        mask_img = sitk.ReadImage(str(mask_path))
        mask_img = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
        mask_img.CopyInformation(hdbet_img)

        if was_reoriented:
            mask_img = resample_mask_to_reference(mask_img, img)
        else:
            mask_img.CopyInformation(img)

        return mask_img


def _resolve_synthstrip_gpu(device):
    if device == "cpu":
        return False
    if device in {"cuda", "gpu"}:
        return True
    if device != "auto":
        raise ValueError(
            f"Unsupported SynthStrip device: {device}. "
            "Expected 'auto', 'cpu', or 'cuda'."
        )

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def extract_foreground_mask_synthstrip(
    img: sitk.Image,
    script_path,
    model_path,
    device="auto",
    border_mm=1.0,
    threads=None,
    verbose=False,
):
    """使用官方 ``mri_synthstrip`` 脚本生成与 ``img`` 同网格的 brain mask。"""
    script_path = Path(script_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    use_gpu = _resolve_synthstrip_gpu(device)

    with TemporaryDirectory(prefix="brain_mri_synthstrip_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        in_path = tmp_dir / "input.nii.gz"
        mask_path = tmp_dir / "mask.nii.gz"
        sitk.WriteImage(img, str(in_path))

        command = [
            sys.executable,
            str(script_path),
            "--image",
            str(in_path),
            "--mask",
            str(mask_path),
            "--model",
            str(model_path),
            "--border",
            str(float(border_mm)),
        ]
        if use_gpu:
            command.append("--gpu")
        if threads is not None:
            command.extend(["--threads", str(int(threads))])

        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE,
        )
        if completed.returncode != 0:
            details = ""
            if not verbose:
                output = "\n".join(
                    part.strip()
                    for part in (completed.stdout or "", completed.stderr or "")
                    if part.strip()
                )
                details = f"\n{output[-4000:]}" if output else ""
            raise RuntimeError(
                f"SynthStrip failed with exit code {completed.returncode}.{details}"
            )
        if not mask_path.exists():
            raise RuntimeError("SynthStrip did not produce the expected mask file.")

        mask_img = sitk.ReadImage(str(mask_path))
        mask_img = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
        return resample_mask_to_reference(mask_img, img)


def dilate_mask(mask_img: sitk.Image, dilation_mm):
    """
    按物理距离膨胀 mask，避免 skull stripping 边界过紧切掉脑膜相关信息。
    """
    if dilation_mm <= 0:
        return mask_img

    spacing = mask_img.GetSpacing()
    radius = [max(1, int(round(dilation_mm / s))) for s in spacing]
    dilated = sitk.BinaryDilate(mask_img > 0, radius)
    dilated = sitk.Cast(dilated, sitk.sitkUInt8)
    dilated.CopyInformation(mask_img)
    return dilated
