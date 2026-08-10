"""冻结 FLAIR nnU-Net 的滑窗全脑/病灶特征提取与磁盘缓存。"""

import hashlib
import math
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.inferers.utils import compute_importance_map
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from models.model_factory import unwrap_model
from utils.dataset import load_nii_as_tensor


FEATURE_CACHE_FORMAT_VERSION = 2
ATTENTION_EPS = 1e-6


class CachedClassificationFeatureDataset(Dataset):
    """保存每例一个全脑向量和一个 soft-mask 病灶向量。"""

    def __init__(self, payload):
        self.global_features = payload["global_features"].float()
        self.lesion_features = payload["lesion_features"].float()
        self.labels = payload["labels"].long()
        self.case_ids = tuple(str(case_id) for case_id in payload["case_ids"])
        sample_count = int(self.labels.numel())
        if self.global_features.dim() != 2 or self.lesion_features.dim() != 2:
            raise ValueError("Cached global/lesion features must be matrices")
        if not (
            self.global_features.size(0)
            == self.lesion_features.size(0)
            == sample_count
            == len(self.case_ids)
        ):
            raise ValueError("Cached feature tensors have inconsistent sample counts")

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, index):
        return (
            self.global_features[index],
            self.lesion_features[index],
            self.labels[index],
            self.case_ids[index],
        )


class ClassificationImageDataset(Dataset):
    """特征缓存只读取 FLAIR，避免无用地解码/传输完整分割 mask。"""

    def __init__(self, source_dataset):
        cases = getattr(source_dataset, "cases", None)
        if cases is None:
            raise ValueError("Classification feature extraction requires case metadata")
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        return (
            load_nii_as_tensor(Path(case["nii_path"])),
            torch.tensor(int(case["label"]), dtype=torch.long),
            str(case["case_id"]),
        )


@lru_cache(maxsize=8)
def checkpoint_sha256(checkpoint_path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as checkpoint_file:
        while True:
            chunk = checkpoint_file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def feature_cache_metadata(
    *,
    checkpoint_path,
    fold,
    split,
    model_name,
    roi_size,
    overlap,
    sw_batch_size,
    crop_nonzero,
    renormalize_nonzero,
    dataset_fingerprint,
    sample_count,
    feature_layout,
):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    return {
        "format_version": FEATURE_CACHE_FORMAT_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "fold": int(fold),
        "split": str(split),
        "model_name": str(model_name),
        "roi_size": tuple(int(value) for value in roi_size),
        "overlap": float(overlap),
        "sw_batch_size": int(sw_batch_size),
        "crop_nonzero": bool(crop_nonzero),
        "renormalize_nonzero": bool(renormalize_nonzero),
        "dataset_fingerprint": str(dataset_fingerprint),
        "sample_count": int(sample_count),
        "mirror_axes": (),
        "aggregation": (
            "gaussian_multiscale_global_soft_lesion_local_and_spatial_statistics"
        ),
        "feature_layout": feature_layout,
    }


def _nonzero_bbox(volume):
    nonzero = volume.abs().sum(dim=0) > 0
    coordinates = nonzero.nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        return tuple(slice(0, int(size)) for size in volume.shape[1:])
    lower = coordinates.min(dim=0).values
    upper = coordinates.max(dim=0).values + 1
    return tuple(
        slice(int(start), int(stop)) for start, stop in zip(lower, upper)
    )


def _renormalize_nonzero(volume):
    brain = volume.abs().sum(dim=1, keepdim=True) > 0
    values = volume[brain]
    if values.numel() > 1:
        standard_deviation = values.std(unbiased=False)
        if standard_deviation > 0:
            volume = torch.where(
                brain,
                (volume - values.mean()) / standard_deviation,
                0.0,
            )
    return volume


def _symmetric_pad_to_roi(volume, roi_size):
    padding_per_axis = []
    for size, target in zip(volume.shape[2:], roi_size):
        required = max(int(target) - int(size), 0)
        lower = required // 2
        padding_per_axis.append((lower, required - lower))
    depth_padding, height_padding, width_padding = padding_per_axis
    padding = (
        width_padding[0],
        width_padding[1],
        height_padding[0],
        height_padding[1],
        depth_padding[0],
        depth_padding[1],
    )
    if any(padding):
        volume = F.pad(volume, padding, value=0.0)
    return volume


def _sliding_starts(image_size, roi_size, overlap):
    starts_per_axis = []
    for size, tile in zip(image_size, roi_size):
        size = int(size)
        tile = int(tile)
        if size <= tile:
            starts_per_axis.append((0,))
            continue
        target_step = tile * (1.0 - float(overlap))
        step_count = int(math.ceil((size - tile) / target_step)) + 1
        actual_step = (size - tile) / (step_count - 1)
        starts_per_axis.append(
            tuple(int(round(actual_step * index)) for index in range(step_count))
        )
    return tuple(starts_per_axis)


def _patch_slices(image_size, roi_size, overlap):
    depth_starts, height_starts, width_starts = _sliding_starts(
        image_size,
        roi_size,
        overlap,
    )
    return tuple(
        (
            slice(depth, depth + roi_size[0]),
            slice(height, height + roi_size[1]),
            slice(width, width + roi_size[2]),
        )
        for depth in depth_starts
        for height in height_starts
        for width in width_starts
    )


def dataset_fingerprint(dataset):
    digest = hashlib.sha256()
    cases = getattr(dataset, "cases", None)
    if cases is None:
        raise ValueError("Feature caching requires a dataset with case metadata")
    for case in cases:
        record = (
            str(case.get("case_id")),
            str(int(case.get("label"))),
            str(Path(case.get("nii_path")).expanduser().resolve()),
        )
        digest.update("\t".join(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _patch_coordinate_tensor(patch, image_size, device, dtype):
    axes = []
    for spatial_slice, full_size in zip(patch, image_size):
        start = int(spatial_slice.start)
        stop = int(spatial_slice.stop)
        if int(full_size) <= 1:
            axis = torch.zeros(stop - start, device=device, dtype=dtype)
        else:
            axis = torch.arange(start, stop, device=device, dtype=dtype)
            axis = axis * (2.0 / (int(full_size) - 1)) - 1.0
        axes.append(axis)
    coordinates = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(coordinates, dim=0).unsqueeze(0)


def _amp_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def sliding_window_pooled_features(
    model,
    inputs,
    *,
    roi_size,
    overlap=0.5,
    sw_batch_size=1,
    crop_nonzero=True,
    renormalize_nonzero=True,
    voxel_volume_ml=None,
):
    """按 nnU-Net ROI 提取一个病例的全脑及 soft-mask 病灶特征。"""
    if inputs.dim() != 5 or inputs.size(0) != 1 or inputs.size(1) != 1:
        raise ValueError(
            "Sliding classification feature extraction requires [1,1,D,H,W], "
            f"got {tuple(inputs.shape)}"
        )
    roi_size = tuple(int(value) for value in roi_size)
    if len(roi_size) != 3 or any(value <= 0 for value in roi_size):
        raise ValueError("roi_size must contain three positive integers")
    if not 0 <= float(overlap) < 1:
        raise ValueError("overlap must lie in [0, 1)")
    if int(sw_batch_size) <= 0:
        raise ValueError("sw_batch_size must be positive")

    base_model = unwrap_model(model)
    device = next(base_model.parameters()).device
    sample = inputs[0]
    spatial_slices = (
        _nonzero_bbox(sample)
        if crop_nonzero
        else tuple(slice(0, int(size)) for size in sample.shape[1:])
    )
    volume = sample[(slice(None), *spatial_slices)].unsqueeze(0).to(
        device,
        non_blocking=True,
    )
    if renormalize_nonzero:
        volume = _renormalize_nonzero(volume)
    volume = _symmetric_pad_to_roi(volume, roi_size)
    slices = _patch_slices(volume.shape[2:], roi_size, overlap)
    importance = compute_importance_map(
        roi_size,
        mode="gaussian",
        sigma_scale=0.125,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)
    blend_normalizer = torch.zeros(
        (1, 1, *volume.shape[2:]),
        device=device,
        dtype=torch.float32,
    )
    for patch in slices:
        blend_normalizer[(slice(None), slice(None), *patch)] += importance

    global_indices = tuple(base_model.classification_global_encoder_indices)
    lesion_indices = tuple(base_model.classification_lesion_decoder_indices)
    global_numerators = [None] * len(global_indices)
    global_denominators = [None] * len(global_indices)
    lesion_numerators = [None] * len(lesion_indices)
    lesion_denominators = [None] * len(lesion_indices)

    statistic_dtype = torch.float32
    brain_sum = torch.zeros((), device=device, dtype=statistic_dtype)
    soft_sum = torch.zeros_like(brain_sum)
    hard_sum = torch.zeros_like(brain_sum)
    high_confidence_sum = torch.zeros_like(brain_sum)
    probability_square_sum = torch.zeros_like(brain_sum)
    entropy_sum = torch.zeros_like(brain_sum)
    max_probability = torch.zeros_like(brain_sum)
    coordinate_sum = torch.zeros(3, device=device, dtype=statistic_dtype)
    coordinate_square_sum = torch.zeros_like(coordinate_sum)

    for offset in range(0, len(slices), int(sw_batch_size)):
        chunk_slices = slices[offset : offset + int(sw_batch_size)]
        patches = torch.cat(
            [volume[(slice(None), slice(None), *patch)] for patch in chunk_slices],
            dim=0,
        )
        brain = patches.abs().sum(dim=1, keepdim=True) > 0
        with torch.no_grad(), _amp_context(device):
            outputs = base_model.forward_features(patches)
        encoder_features = tuple(
            feature.float() for feature in outputs["encoder_features"]
        )
        decoder_features = tuple(
            feature.float() for feature in outputs["decoder_features"]
        )
        lesion_logits = outputs["segmentation"].float()

        high_importance = torch.cat(
            [
                importance
                / blend_normalizer[(slice(None), slice(None), *patch)].clamp_min(
                    ATTENTION_EPS
                )
                for patch in chunk_slices
            ],
            dim=0,
        )
        brain_weight = brain.float() * high_importance
        probability = torch.sigmoid(lesion_logits)
        local_weight = probability * brain_weight

        for position, feature_index in enumerate(global_indices):
            feature = encoder_features[feature_index]
            feature_weight = F.interpolate(
                brain_weight,
                size=feature.shape[2:],
                mode="area",
            )
            numerator = (feature * feature_weight).sum((0, 2, 3, 4))
            denominator = feature_weight.sum()
            global_numerators[position] = (
                numerator
                if global_numerators[position] is None
                else global_numerators[position] + numerator
            )
            global_denominators[position] = (
                denominator
                if global_denominators[position] is None
                else global_denominators[position] + denominator
            )

        for position, feature_index in enumerate(lesion_indices):
            feature = decoder_features[feature_index]
            feature_weight = F.interpolate(
                local_weight,
                size=feature.shape[2:],
                mode="area",
            )
            numerator = (feature * feature_weight).sum((0, 2, 3, 4))
            denominator = feature_weight.sum()
            lesion_numerators[position] = (
                numerator
                if lesion_numerators[position] is None
                else lesion_numerators[position] + numerator
            )
            lesion_denominators[position] = (
                denominator
                if lesion_denominators[position] is None
                else lesion_denominators[position] + denominator
            )

        brain_sum += brain_weight.sum()
        soft_sum += local_weight.sum()
        hard_sum += (
            (probability >= 0.5).to(probability.dtype) * brain_weight
        ).sum()
        high_confidence_sum += (
            (probability >= 0.9).to(probability.dtype) * brain_weight
        ).sum()
        probability_square_sum += (probability.square() * brain_weight).sum()
        bounded_probability = probability.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(
            bounded_probability * bounded_probability.log()
            + (1.0 - bounded_probability)
            * (1.0 - bounded_probability).log()
        )
        entropy_sum += (entropy * brain_weight).sum()
        max_probability = torch.maximum(
            max_probability,
            (probability * brain.float()).amax(),
        )
        coordinates = torch.cat(
            [
                _patch_coordinate_tensor(
                    patch,
                    volume.shape[2:],
                    device,
                    probability.dtype,
                )
                for patch in chunk_slices
            ],
            dim=0,
        )
        coordinate_sum += (local_weight * coordinates).sum((0, 2, 3, 4))
        coordinate_square_sum += (
            local_weight * coordinates.square()
        ).sum((0, 2, 3, 4))

    global_features = torch.cat(
        tuple(
            numerator / denominator.clamp_min(ATTENTION_EPS)
            for numerator, denominator in zip(
                global_numerators,
                global_denominators,
            )
        ),
        dim=0,
    )
    lesion_features = torch.cat(
        tuple(
            numerator / denominator.clamp_min(ATTENTION_EPS)
            for numerator, denominator in zip(
                lesion_numerators,
                lesion_denominators,
            )
        ),
        dim=0,
    )
    if base_model.classification_lesion_stat_names:
        statistics = base_model.format_lesion_statistics(
            soft_sum=soft_sum.unsqueeze(0),
            hard_sum=hard_sum.unsqueeze(0),
            high_confidence_sum=high_confidence_sum.unsqueeze(0),
            probability_square_sum=probability_square_sum.unsqueeze(0),
            entropy_sum=entropy_sum.unsqueeze(0),
            max_probability=max_probability.unsqueeze(0),
            brain_sum=brain_sum.unsqueeze(0),
            coordinate_sum=coordinate_sum.unsqueeze(0),
            coordinate_square_sum=coordinate_square_sum.unsqueeze(0),
            voxel_volume_ml=voxel_volume_ml,
        ).squeeze(0)
        lesion_features = torch.cat((lesion_features, statistics), dim=0)

    expected_layout = base_model.classification_feature_layout()
    if global_features.numel() != expected_layout["global_feature_dim"]:
        raise RuntimeError(
            "Sliding global feature layout mismatch: "
            f"{global_features.numel()} vs "
            f"{expected_layout['global_feature_dim']}"
        )
    if lesion_features.numel() != expected_layout["lesion_feature_dim"]:
        raise RuntimeError(
            "Sliding lesion feature layout mismatch: "
            f"{lesion_features.numel()} vs "
            f"{expected_layout['lesion_feature_dim']}"
        )
    return global_features.unsqueeze(0).cpu(), lesion_features.unsqueeze(0).cpu()


def _save_payload_atomically(payload, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _extract_dataset_features(
    model,
    dataset,
    metadata,
    config,
    device,
    split,
    partial_path,
):
    global_features = []
    lesion_features = []
    labels = []
    case_ids = []
    if partial_path.is_file():
        partial_payload = torch.load(
            partial_path,
            map_location="cpu",
            weights_only=False,
        )
        if partial_payload.get("metadata") != metadata:
            raise ValueError(
                "Existing partial feature cache does not match the current "
                f"checkpoint/config: {partial_path}"
            )
        partial_dataset = CachedClassificationFeatureDataset(partial_payload)
        expected_prefix = tuple(
            str(case["case_id"])
            for case in dataset.cases[: len(partial_dataset)]
        )
        if partial_dataset.case_ids != expected_prefix:
            raise ValueError(
                f"Partial feature cache case order mismatch: {partial_path}"
            )
        global_features.append(partial_dataset.global_features)
        lesion_features.append(partial_dataset.lesion_features)
        labels.append(partial_dataset.labels)
        case_ids.extend(partial_dataset.case_ids)
        print(
            f"Resuming {split} feature extraction at "
            f"{len(partial_dataset)}/{len(dataset)} cases"
        )

    start_index = len(case_ids)
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": int(getattr(config, "FEATURE_EXTRACTION_NUM_WORKERS", 4)),
        "pin_memory": device.type == "cuda",
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    image_dataset = ClassificationImageDataset(dataset)
    remaining_dataset = Subset(image_dataset, range(start_index, len(dataset)))
    loader = DataLoader(remaining_dataset, **loader_kwargs)
    model.eval()
    progress = tqdm(
        loader,
        desc=f"Extract {split} features",
        initial=start_index,
        total=len(dataset),
    )
    save_every = max(1, int(getattr(config, "FEATURE_CACHE_SAVE_EVERY", 25)))
    for inputs, targets, batch_case_ids in progress:
        pooled_global, pooled_lesion = sliding_window_pooled_features(
            model,
            inputs,
            roi_size=metadata["roi_size"],
            overlap=metadata["overlap"],
            sw_batch_size=metadata["sw_batch_size"],
            crop_nonzero=metadata["crop_nonzero"],
            renormalize_nonzero=metadata["renormalize_nonzero"],
            voxel_volume_ml=metadata["feature_layout"]["voxel_volume_ml"],
        )
        global_features.append(pooled_global)
        lesion_features.append(pooled_lesion)
        labels.append(targets.long().cpu())
        case_ids.extend(str(case_id) for case_id in batch_case_ids)
        if len(case_ids) % save_every == 0:
            _save_payload_atomically(
                {
                    "metadata": metadata,
                    "global_features": torch.cat(global_features, dim=0),
                    "lesion_features": torch.cat(lesion_features, dim=0),
                    "labels": torch.cat(labels, dim=0),
                    "case_ids": case_ids,
                },
                partial_path,
            )
    payload = {
        "metadata": metadata,
        "global_features": torch.cat(global_features, dim=0),
        "lesion_features": torch.cat(lesion_features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "case_ids": case_ids,
    }
    if int(payload["labels"].numel()) != len(dataset):
        raise RuntimeError(
            f"Feature extraction ended with {payload['labels'].numel()} of "
            f"{len(dataset)} cases"
        )
    return payload


def load_or_create_feature_cache(
    cache_path,
    *,
    model,
    dataset,
    metadata,
    config,
    device,
):
    """严格校验已有缓存；不存在时原子写入，绝不覆盖不匹配缓存。"""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("metadata") != metadata:
            raise ValueError(
                "Existing classification feature cache does not match the current "
                f"checkpoint/config; choose a new output root: {cache_path}"
            )
        print(f"Reusing feature cache: {cache_path}")
        return CachedClassificationFeatureDataset(payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = cache_path.with_suffix(".partial.pt")
    payload = _extract_dataset_features(
        model,
        dataset,
        metadata,
        config,
        device,
        metadata["split"],
        partial_path,
    )
    _save_payload_atomically(payload, cache_path)
    if partial_path.exists():
        partial_path.unlink()
    print(f"Saved feature cache: {cache_path}")
    return CachedClassificationFeatureDataset(payload)


def cached_feature_loaders(
    *,
    model,
    train_dataset,
    val_dataset,
    config,
    output_root,
    initialization_path,
    fold,
    model_name,
    device,
):
    roi_size = tuple(int(value) for value in config.CLASSIFICATION_FEATURE_ROI_SIZE)
    common_metadata = {
        "checkpoint_path": initialization_path,
        "fold": fold,
        "model_name": model_name,
        "roi_size": roi_size,
        "overlap": float(
            getattr(config, "CLASSIFICATION_FEATURE_OVERLAP", 0.5)
        ),
        "sw_batch_size": int(
            getattr(config, "CLASSIFICATION_FEATURE_SW_BATCH_SIZE", 1)
        ),
        "crop_nonzero": bool(
            getattr(config, "CLASSIFICATION_FEATURE_CROP_NONZERO", True)
        ),
        "renormalize_nonzero": bool(
            getattr(config, "CLASSIFICATION_FEATURE_RENORMALIZE_NONZERO", True)
        ),
        "feature_layout": unwrap_model(model).classification_feature_layout(),
    }
    cache_root = (
        Path(output_root).expanduser().resolve()
        / "feature_cache"
        / "seq3_FLAIR"
        / model_name
    )
    datasets = {}
    metadata_by_split = {}
    for split, dataset in (("train", train_dataset), ("val", val_dataset)):
        metadata = feature_cache_metadata(
            split=split,
            dataset_fingerprint=dataset_fingerprint(dataset),
            sample_count=len(dataset),
            **common_metadata,
        )
        metadata_by_split[split] = metadata
        datasets[split] = load_or_create_feature_cache(
            cache_root / f"fold{fold}_{split}.pt",
            model=model,
            dataset=dataset,
            metadata=metadata,
            config=config,
            device=device,
        )

    loader_kwargs = {
        "batch_size": int(config.BATCH_SIZE),
        "num_workers": int(config.NUM_WORKERS),
        "pin_memory": device.type == "cuda",
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(datasets["train"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(datasets["val"], shuffle=False, **loader_kwargs)
    return train_loader, val_loader, metadata_by_split
