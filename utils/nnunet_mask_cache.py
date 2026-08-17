"""生成不绑定分类 checkpoint 的 OOF nnU-Net soft-mask 缓存。"""

import os
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from models.NNUNetMaskGuidedClassifier import MASK_STAT_DIM
from train_foundation_nnunet_guided import (
    GuidanceInputDataset,
    amp_context,
    load_nnunet_model,
    normalize_case_id,
)
from utils.dataset import load_nii_as_tensor
from utils.segmentation_inference import segmentation_logits_from_config


CACHE_SCHEMA_VERSION = 1
CACHE_QUANTIZATION_LEVELS = 255
CACHE_EPS = 1e-6


def _axis_moments(weights, axis):
    spatial_dims = (2, 3, 4)
    reduce_dims = tuple(dim for dim in spatial_dims if dim != axis + 2)
    marginal = weights.sum(dim=reduce_dims).squeeze(1)
    coordinates = torch.linspace(
        -1.0,
        1.0,
        marginal.size(1),
        device=weights.device,
        dtype=weights.dtype,
    ).unsqueeze(0)
    raw_denominator = marginal.sum(dim=1)
    denominator = raw_denominator.clamp_min(CACHE_EPS)
    center = (marginal * coordinates).sum(dim=1) / denominator
    variance = (
        marginal * (coordinates - center.unsqueeze(1)).square()
    ).sum(dim=1) / denominator
    has_lesion = raw_denominator > CACHE_EPS
    center = torch.where(has_lesion, center, torch.zeros_like(center))
    spread = torch.where(
        has_lesion,
        variance.clamp_min(0.0).sqrt(),
        torch.zeros_like(variance),
    )
    return center, spread


def mask_statistics(inputs, lesion_probability, voxel_volume_ml):
    """在量化和降采样前计算每例 soft mask 的十四项统计量。"""
    brain = (inputs.abs().sum(dim=1, keepdim=True) > 1e-6).float()
    probability = lesion_probability.float().clamp(0.0, 1.0) * brain
    brain_voxels = brain.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    soft_voxels = probability.sum(dim=(1, 2, 3, 4))
    hard_voxels = (
        (probability >= 0.5).to(probability.dtype) * brain
    ).sum(dim=(1, 2, 3, 4))
    high_confidence_voxels = (
        (probability >= 0.9).to(probability.dtype) * brain
    ).sum(dim=(1, 2, 3, 4))
    mean_probability = soft_voxels / brain_voxels
    probability_square_mean = probability.square().sum(
        dim=(1, 2, 3, 4)
    ) / brain_voxels
    probability_std = (
        probability_square_mean - mean_probability.square()
    ).clamp_min(0.0).sqrt()
    max_probability = probability.flatten(1).amax(dim=1)
    bounded = probability.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(
        bounded * bounded.log()
        + (1.0 - bounded) * (1.0 - bounded).log()
    )
    mean_entropy = (entropy * brain).sum(dim=(1, 2, 3, 4)) / brain_voxels

    centers = []
    spreads = []
    for axis in range(3):
        center, spread = _axis_moments(probability, axis)
        centers.append(center)
        spreads.append(spread)
    statistics = torch.stack(
        (
            torch.log1p(soft_voxels * float(voxel_volume_ml)),
            torch.log1p(hard_voxels * float(voxel_volume_ml)),
            mean_probability,
            hard_voxels / brain_voxels,
            high_confidence_voxels / brain_voxels,
            probability_std,
            max_probability,
            mean_entropy,
            *centers,
            *spreads,
        ),
        dim=1,
    )
    if statistics.shape[1] != MASK_STAT_DIM:
        raise RuntimeError(
            f"Mask-statistic dimension mismatch: {statistics.shape[1]}"
        )
    return probability, statistics


def quantized_mask_representation(
    inputs,
    lesion_probability,
    cache_size,
    voxel_volume_ml,
):
    probability, statistics = mask_statistics(
        inputs,
        lesion_probability,
        voxel_volume_ml,
    )
    low_resolution = F.interpolate(
        probability,
        size=tuple(int(value) for value in cache_size),
        mode="area",
    ).clamp(0.0, 1.0)
    quantized = torch.round(
        low_resolution * CACHE_QUANTIZATION_LEVELS
    ).to(torch.uint8)
    return quantized, statistics.float()


def mask_cache_signature(
    *,
    split,
    target_fold,
    nnunet_paths,
    nnunet_sha256,
    config,
):
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "split": str(split),
        "target_fold": int(target_fold),
        "nnunet_checkpoints": {
            str(fold): str(path) for fold, path in nnunet_paths.items()
        },
        "nnunet_sha256": {
            str(fold): str(value) for fold, value in nnunet_sha256.items()
        },
        "inference": {
            "roi_size": tuple(int(x) for x in config.SEG_VALIDATION_ROI_SIZE),
            "overlap": float(config.SEG_VALIDATION_OVERLAP),
            "sw_batch_size": int(config.SEG_VALIDATION_SW_BATCH_SIZE),
            "mirror_axes": tuple(int(x) for x in config.SEG_VALIDATION_MIRROR_AXES),
            "crop_nonzero": bool(config.SEG_VALIDATION_CROP_NONZERO),
            "renormalize_nonzero": bool(
                config.SEG_VALIDATION_RENORMALIZE_NONZERO
            ),
        },
        "mask_cache_size": tuple(int(x) for x in config.GUIDANCE_MASK_CACHE_SIZE),
        "voxel_volume_ml": float(config.GUIDANCE_VOXEL_VOLUME_ML),
        "quantization_levels": CACHE_QUANTIZATION_LEVELS,
        "mask_stat_dim": MASK_STAT_DIM,
    }


def validate_mask_cache(payload, signature, case_ids, labels, source_folds):
    if payload.get("signature") != signature:
        raise RuntimeError("Cached nnU-Net mask signature does not match this run")
    if tuple(payload.get("case_ids", ())) != tuple(case_ids):
        raise RuntimeError("Cached nnU-Net mask case order does not match")
    if tuple(int(x) for x in payload.get("source_folds", ())) != tuple(
        int(x) for x in source_folds
    ):
        raise RuntimeError("Cached OOF nnU-Net source folds do not match")
    cached_labels = payload.get("labels")
    if cached_labels is None or not torch.equal(
        cached_labels.long(),
        torch.as_tensor(labels, dtype=torch.long),
    ):
        raise RuntimeError("Cached nnU-Net mask labels do not match")
    expected_mask_shape = (
        len(case_ids),
        1,
        *tuple(int(x) for x in signature["mask_cache_size"]),
    )
    if tuple(payload["masks_uint8"].shape) != expected_mask_shape:
        raise RuntimeError(
            "Cached nnU-Net mask tensor has the wrong shape: "
            f"{tuple(payload['masks_uint8'].shape)} vs {expected_mask_shape}"
        )
    if payload["masks_uint8"].dtype != torch.uint8:
        raise RuntimeError("Cached nnU-Net masks must use uint8 quantization")
    if tuple(payload["mask_statistics"].shape) != (
        len(case_ids),
        MASK_STAT_DIM,
    ):
        raise RuntimeError("Cached nnU-Net mask statistics have the wrong shape")


def _save_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _chunk_payload(signature, source_fold, records):
    records = sorted(records, key=lambda record: record[0])
    return {
        "signature": signature,
        "source_fold": int(source_fold),
        "indices": torch.tensor(
            [record[0] for record in records],
            dtype=torch.long,
        ),
        "case_ids": [record[1] for record in records],
        "labels": torch.tensor(
            [record[2] for record in records],
            dtype=torch.long,
        ),
        "masks_uint8": torch.stack([record[3] for record in records]),
        "mask_statistics": torch.stack([record[4] for record in records]),
    }


def _records_from_chunk(path, signature, source_fold):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        raise RuntimeError(
            f"Partial mask-cache signature mismatch: {path}; "
            "use --rebuild-mask-cache"
        )
    if int(payload.get("source_fold", -1)) != int(source_fold):
        raise RuntimeError(f"Partial mask-cache source fold mismatch: {path}")
    return [
        (
            int(index),
            str(case_id),
            int(label),
            mask.to(torch.uint8),
            statistics.float(),
        )
        for index, case_id, label, mask, statistics in zip(
            payload["indices"].tolist(),
            payload["case_ids"],
            payload["labels"].tolist(),
            payload["masks_uint8"],
            payload["mask_statistics"],
        )
    ]


def extract_mask_cache(
    *,
    split,
    dataset,
    source_folds,
    target_fold,
    nnunet_paths,
    nnunet_sha256,
    cache_dir,
    config,
    device,
    rebuild=False,
):
    """生成或验证一个 split 的 OOF soft-mask 缓存。"""
    case_ids = tuple(
        normalize_case_id(case["case_id"]) for case in dataset.cases
    )
    labels = tuple(int(case["label"]) for case in dataset.cases)
    source_folds = tuple(int(value) for value in source_folds)
    if len(source_folds) != len(case_ids):
        raise ValueError("source_folds length does not match the dataset")
    signature = mask_cache_signature(
        split=split,
        target_fold=target_fold,
        nnunet_paths=nnunet_paths,
        nnunet_sha256=nnunet_sha256,
        config=config,
    )
    cache_dir = Path(cache_dir).expanduser().resolve()
    final_path = cache_dir / f"fold{int(target_fold)}_{split}.pt"
    if final_path.is_file() and not rebuild:
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        validate_mask_cache(
            payload,
            signature,
            case_ids,
            labels,
            source_folds,
        )
        print(f"复用完整 {split} nnU-Net mask 缓存：{final_path}")
        return final_path

    records_by_index = {}
    grouped_indices = {}
    for index, source_fold in enumerate(source_folds):
        grouped_indices.setdefault(source_fold, []).append(index)

    for source_fold, expected_indices in sorted(grouped_indices.items()):
        part_dir = (
            cache_dir
            / "parts"
            / f"fold{int(target_fold)}_{split}_source_fold{source_fold}"
        )
        if rebuild and part_dir.is_dir():
            # 只有显式 --rebuild-mask-cache 才清理本实验的可恢复分块；不触碰
            # nnU-Net、数据集或其他实验目录。
            for chunk_path in part_dir.glob("chunk_*.pt"):
                chunk_path.unlink()
        expected_set = set(expected_indices)
        for chunk_path in sorted(part_dir.glob("chunk_*.pt")):
            for record in _records_from_chunk(
                chunk_path,
                signature,
                source_fold,
            ):
                index = record[0]
                if index not in expected_set:
                    raise RuntimeError(
                        f"Unexpected index {index} in partial cache {chunk_path}"
                    )
                if index in records_by_index:
                    raise RuntimeError(
                        f"Duplicate cached case index {index} in {chunk_path}"
                    )
                records_by_index[index] = record

        missing_indices = [
            index for index in expected_indices if index not in records_by_index
        ]
        if not missing_indices:
            continue
        segmenter = load_nnunet_model(
            nnunet_paths[source_fold],
            device,
            bool(getattr(config, "NNUNET_USE_DATAPARALLEL", True)),
        )
        feature_workers = int(config.FEATURE_EXTRACTION_NUM_WORKERS)
        loader_options = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": feature_workers,
            "pin_memory": device.type == "cuda",
        }
        if feature_workers > 0:
            loader_options.update(
                multiprocessing_context="spawn",
                persistent_workers=True,
            )
        loader = DataLoader(
            Subset(GuidanceInputDataset(dataset.cases), missing_indices),
            **loader_options,
        )
        id_to_index = {case_ids[index]: index for index in missing_indices}
        chunk_size = int(getattr(config, "MASK_CACHE_CHUNK_SIZE", 25))
        if chunk_size <= 0:
            raise ValueError("MASK_CACHE_CHUNK_SIZE must be positive")
        pending = []
        progress = tqdm(
            loader,
            desc=f"缓存 {split}：OOF nnU-Net fold {source_fold}",
        )
        for inputs, batch_labels, batch_case_ids in progress:
            case_id = normalize_case_id(batch_case_ids[0])
            index = id_to_index[case_id]
            inputs = inputs.to(device, non_blocking=True)
            with torch.no_grad(), amp_context(device):
                lesion_logits = segmentation_logits_from_config(
                    segmenter,
                    inputs,
                    config,
                )
                lesion_probability = torch.sigmoid(lesion_logits.float())
                masks_uint8, statistics = quantized_mask_representation(
                    inputs,
                    lesion_probability,
                    config.GUIDANCE_MASK_CACHE_SIZE,
                    config.GUIDANCE_VOXEL_VOLUME_ML,
                )
            record = (
                index,
                case_id,
                int(batch_labels.item()),
                masks_uint8[0].cpu(),
                statistics[0].cpu(),
            )
            pending.append(record)
            records_by_index[index] = record
            if len(pending) >= chunk_size:
                first_index = min(item[0] for item in pending)
                last_index = max(item[0] for item in pending)
                chunk_path = part_dir / (
                    f"chunk_{first_index:05d}_{last_index:05d}.pt"
                )
                if chunk_path.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite partial mask cache: {chunk_path}"
                    )
                _save_atomic(
                    _chunk_payload(signature, source_fold, pending),
                    chunk_path,
                )
                pending = []
        if pending:
            first_index = min(item[0] for item in pending)
            last_index = max(item[0] for item in pending)
            chunk_path = part_dir / (
                f"chunk_{first_index:05d}_{last_index:05d}.pt"
            )
            if chunk_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite partial mask cache: {chunk_path}"
                )
            _save_atomic(
                _chunk_payload(signature, source_fold, pending),
                chunk_path,
            )
        del segmenter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected_all = set(range(len(dataset)))
    if set(records_by_index) != expected_all:
        missing = sorted(expected_all - set(records_by_index))
        raise RuntimeError(
            f"Mask cache is incomplete; missing indices: {missing[:20]}"
        )
    ordered = [records_by_index[index] for index in range(len(dataset))]
    payload = {
        "signature": signature,
        "case_ids": [record[1] for record in ordered],
        "source_folds": source_folds,
        "labels": torch.tensor(
            [record[2] for record in ordered],
            dtype=torch.long,
        ),
        "masks_uint8": torch.stack([record[3] for record in ordered]),
        "mask_statistics": torch.stack([record[4] for record in ordered]),
    }
    validate_mask_cache(
        payload,
        signature,
        case_ids,
        labels,
        source_folds,
    )
    _save_atomic(payload, final_path)
    print(f"完成 {split} nnU-Net mask 缓存：{final_path}")
    return final_path


class MaskGuidedClassificationDataset(Dataset):
    """按已验证的病例顺序组合 FLAIR 与量化 OOF mask。"""

    def __init__(self, cases, cache_payload):
        self.cases = tuple(cases)
        self.labels = [int(case["label"]) for case in self.cases]
        expected_case_ids = tuple(
            normalize_case_id(case["case_id"]) for case in self.cases
        )
        if tuple(cache_payload.get("case_ids", ())) != expected_case_ids:
            raise RuntimeError("Mask cache does not align with classification cases")
        if not torch.equal(
            cache_payload["labels"].long(),
            torch.tensor(self.labels, dtype=torch.long),
        ):
            raise RuntimeError("Mask cache labels do not align with classification cases")
        self.masks_uint8 = cache_payload["masks_uint8"]
        self.mask_statistics = cache_payload["mask_statistics"].float()

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        image = load_nii_as_tensor(Path(case["nii_path"]))
        probability = (
            self.masks_uint8[index].float() / CACHE_QUANTIZATION_LEVELS
        )
        return (
            image,
            torch.tensor(self.labels[index], dtype=torch.long),
            probability,
            self.mask_statistics[index],
            normalize_case_id(case["case_id"]),
        )


def load_mask_guided_dataset(dataset, cache_path):
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    return MaskGuidedClassificationDataset(dataset.cases, payload)
