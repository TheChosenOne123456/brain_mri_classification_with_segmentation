"""二值分割的全体积直接推理与 nnU-Net 风格滑窗推理。"""

import itertools

import torch
from monai.inferers import sliding_window_inference

from models.model_factory import forward_model


def _nonzero_bbox(volume):
    nonzero = volume.abs().sum(dim=0) > 0
    coordinates = nonzero.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        return tuple(slice(0, int(size)) for size in volume.shape[1:])
    lower = coordinates.min(dim=0).values
    upper = coordinates.max(dim=0).values + 1
    return tuple(
        slice(int(start), int(stop))
        for start, stop in zip(lower, upper)
    )


def _mirror_combinations(mirror_axes):
    axes = tuple(sorted({int(axis) for axis in mirror_axes}))
    if any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("mirror_axes entries must be 0, 1 or 2")
    combinations = [()]
    for length in range(1, len(axes) + 1):
        combinations.extend(itertools.combinations(axes, length))
    return tuple(combinations)


def sliding_window_segmentation_logits(
    model,
    inputs,
    *,
    roi_size,
    overlap=0.5,
    sw_batch_size=1,
    mirror_axes=(),
    crop_nonzero=True,
    renormalize_nonzero=False,
):
    """返回与输入同尺寸的单通道 lesion logits。"""
    if inputs.dim() != 5 or inputs.size(1) != 1:
        raise ValueError(
            f"Expected FLAIR inputs [B,1,D,H,W], got {tuple(inputs.shape)}"
        )
    roi_size = tuple(int(value) for value in roi_size)
    if len(roi_size) != 3 or any(value <= 0 for value in roi_size):
        raise ValueError("roi_size must contain three positive integers")
    if not 0 <= float(overlap) < 1:
        raise ValueError("overlap must lie in [0, 1)")
    if int(sw_batch_size) <= 0:
        raise ValueError("sw_batch_size must be positive")

    mirror_combinations = _mirror_combinations(mirror_axes)

    def predictor(patches):
        return forward_model(
            model,
            patches,
            return_seg=True,
            segmentation_only=True,
        )["segmentation"]

    outputs = []
    for sample in inputs:
        spatial_slices = (
            _nonzero_bbox(sample)
            if crop_nonzero
            else tuple(slice(0, int(size)) for size in sample.shape[1:])
        )
        cropped = sample[(slice(None), *spatial_slices)].unsqueeze(0)
        if renormalize_nonzero:
            nonzero = cropped.abs().sum(dim=1, keepdim=True) > 0
            values = cropped[nonzero]
            if values.numel() > 1:
                standard_deviation = values.std(unbiased=False)
                if standard_deviation > 0:
                    cropped = torch.where(
                        nonzero,
                        (cropped - values.mean()) / standard_deviation,
                        0.0,
                    )
        prediction_sum = None
        for axes in mirror_combinations:
            tensor_axes = tuple(axis + 2 for axis in axes)
            augmented = (
                torch.flip(cropped, dims=tensor_axes)
                if tensor_axes
                else cropped
            )
            prediction = sliding_window_inference(
                augmented,
                roi_size=roi_size,
                sw_batch_size=int(sw_batch_size),
                predictor=predictor,
                overlap=float(overlap),
                mode="gaussian",
                padding_mode="constant",
                cval=0.0,
            )
            if tensor_axes:
                prediction = torch.flip(prediction, dims=tensor_axes)
            prediction_sum = (
                prediction
                if prediction_sum is None
                else prediction_sum + prediction
            )
        cropped_logits = prediction_sum / len(mirror_combinations)
        full_logits = inputs.new_full(
            (1, 1, *inputs.shape[2:]),
            -20.0,
            dtype=cropped_logits.dtype,
        )
        full_logits[(slice(None), slice(None), *spatial_slices)] = cropped_logits
        outputs.append(full_logits)
    return torch.cat(outputs, dim=0)


def segmentation_logits_from_config(model, inputs, config):
    mode = getattr(config, "SEGMENTATION_VALIDATION_INFERENCE", "direct")
    if mode == "direct":
        return forward_model(
            model,
            inputs,
            return_seg=True,
            segmentation_only=True,
        )["segmentation"]
    if mode != "sliding_window":
        raise ValueError(
            "SEGMENTATION_VALIDATION_INFERENCE must be direct or "
            f"sliding_window, got {mode!r}"
        )
    return sliding_window_segmentation_logits(
        model,
        inputs,
        roi_size=getattr(config, "SEG_VALIDATION_ROI_SIZE"),
        overlap=getattr(config, "SEG_VALIDATION_OVERLAP", 0.5),
        sw_batch_size=getattr(config, "SEG_VALIDATION_SW_BATCH_SIZE", 1),
        mirror_axes=getattr(config, "SEG_VALIDATION_MIRROR_AXES", ()),
        crop_nonzero=getattr(config, "SEG_VALIDATION_CROP_NONZERO", True),
        renormalize_nonzero=getattr(
            config,
            "SEG_VALIDATION_RENORMALIZE_NONZERO",
            False,
        ),
    )
