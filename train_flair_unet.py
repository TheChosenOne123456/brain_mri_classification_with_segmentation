"""FLAIRUNet3D 的三阶段 K-Fold 训练入口。

阶段：
1. segmentation：只训练 encoder + U-Net decoder，数据仅含有医生 mask 的异常
   病例和正常全零 mask 病例。
2. classification-warmup：加载阶段一 checkpoint，冻结分割网络，只训练全局/
   局部双路分类头。
3. joint：加载阶段二 checkpoint，使用分组学习率联合微调，分割损失仍只作用于
   有可靠监督的病例。
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.config_utils import (
    infer_data_dir,
    load_python_config,
    resolve_input_artifact_dir,
    resolve_output_artifact_dir,
)
from configs.global_config import CLASS_NAMES, K_FOLDS, NUM_CLASSES, SEED
from models.FLAIRUNet3D import load_flair_segmentation_state
from models.model_factory import create_model, forward_model, unwrap_model
from utils.classification_features import cached_feature_loaders
from utils.segmentation import (
    binary_dice_per_sample,
    binary_lesion_predictions,
    binary_recall_per_sample,
    binary_segmentation_loss,
    prepare_binary_mask,
)
from utils.segmentation_data import FixedIterationBatchSampler, NNUNetPatchDataset
from utils.segmentation_inference import segmentation_logits_from_config
from utils.train_and_test import load_pt_dataset, set_seed


MODEL_NAME = "FLAIRUNet3D"
SUPPORTED_MODEL_NAMES = ("FLAIRUNet3D", "FLAIRUNet3DNNUNet")
SEQUENCE_ID = 3
SEQUENCE_NAME = "FLAIR"
STAGES = ("segmentation", "classification-warmup", "joint")
VOXEL_VOLUME_ML = 3.0 * 0.75 * 0.75 / 1000.0

COMMON_CONFIG_FIELDS = (
    "NUM_EPOCHS",
    "MIN_EPOCHS",
    "BATCH_SIZE",
    "WEIGHT_DECAY",
    "DEVICE",
    "NUM_WORKERS",
    "PATIENCE",
    "CLASSIFICATION_LOSS",
    "CLASS_WEIGHT_POWER",
    "CLASSIFICATION_ALPHA",
    "SEG_ALPHA",
    "SEG_BCE_WEIGHT",
    "SEG_DICE_WEIGHT",
    "SEG_POSITIVE_WEIGHT",
    "DEEP_SUPERVISION_WEIGHTS",
    "ENCODER_LEARNING_RATE",
    "SEGMENTATION_LEARNING_RATE",
    "CLASSIFICATION_LEARNING_RATE",
    "CHECKPOINT_SELECTION_METRIC",
    "METASTASIS_F_BETA",
    "CHECKPOINT_MIN_ACCURACY",
    "CHECKPOINT_MIN_METASTASIS_PRECISION",
    "CHECKPOINT_MIN_POSITIVE_DICE",
)


@dataclass(frozen=True)
class ValidationResult:
    loss: float
    classification_loss: float
    segmentation_loss: float
    accuracy: float
    macro_f1: float
    metastasis_precision: float
    metastasis_recall: float
    metastasis_fbeta: float
    positive_dice: float
    positive_voxel_precision: float
    positive_voxel_recall: float
    positive_complete_miss_rate: float
    normal_false_positive_fraction: float
    normal_false_positive_volume_ml: float
    normal_p95_false_positive_volume_ml: float
    positive_mask_cases: int
    normal_mask_cases: int
    nnunet_foreground_mean_dice: float = 0.0


def require_config_fields(config):
    missing = [name for name in COMMON_CONFIG_FIELDS if not hasattr(config, name)]
    if missing:
        raise ValueError(
            f"Config {config.__config_path__} is missing fields: "
            f"{', '.join(missing)}"
        )


def validate_stage_config(config, stage):
    if int(config.BATCH_SIZE) <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    if stage == "segmentation":
        if float(config.SEG_ALPHA) <= 0:
            raise ValueError("The segmentation stage requires SEG_ALPHA > 0")
    elif stage == "classification-warmup":
        if float(config.CLASSIFICATION_ALPHA) <= 0:
            raise ValueError(
                "The classification-warmup stage requires CLASSIFICATION_ALPHA > 0"
            )
        if float(config.SEG_ALPHA) != 0:
            raise ValueError("The classification-warmup stage requires SEG_ALPHA = 0")
    elif stage == "joint":
        if float(config.CLASSIFICATION_ALPHA) <= 0 or float(config.SEG_ALPHA) <= 0:
            raise ValueError(
                "The joint stage requires CLASSIFICATION_ALPHA > 0 and SEG_ALPHA > 0"
            )

    weights = tuple(float(value) for value in config.DEEP_SUPERVISION_WEIGHTS)
    if len(weights) > 4 or any(weight < 0 for weight in weights):
        raise ValueError(
            "DEEP_SUPERVISION_WEIGHTS must contain at most four non-negative values"
        )
    if (
        config.SEG_POSITIVE_WEIGHT is not None
        and float(config.SEG_POSITIVE_WEIGHT) <= 0
    ):
        raise ValueError("SEG_POSITIVE_WEIGHT must be positive or None")
    bce_mode = getattr(config, "SEG_BCE_MODE", "mean")
    if bce_mode not in ("mean", "hard_negative"):
        raise ValueError("SEG_BCE_MODE must be mean or hard_negative")
    hard_negative_fraction = float(
        getattr(config, "SEG_HARD_NEGATIVE_FRACTION", 0.002)
    )
    if not 0 < hard_negative_fraction <= 1:
        raise ValueError("SEG_HARD_NEGATIVE_FRACTION must lie in (0, 1]")
    if int(getattr(config, "SEG_HARD_NEGATIVE_MIN_VOXELS", 256)) <= 0:
        raise ValueError("SEG_HARD_NEGATIVE_MIN_VOXELS must be positive")
    normal_dense_weight = float(
        getattr(config, "SEG_NORMAL_DENSE_BCE_WEIGHT", 0.5)
    )
    if not 0 <= normal_dense_weight <= 1:
        raise ValueError("SEG_NORMAL_DENSE_BCE_WEIGHT must lie in [0, 1]")
    if config.CHECKPOINT_SELECTION_METRIC not in (
        "macro_f1",
        "metastasis_fbeta",
    ):
        raise ValueError(
            "CHECKPOINT_SELECTION_METRIC must be macro_f1 or metastasis_fbeta"
        )
    model_name = getattr(config, "MODEL_NAME", MODEL_NAME)
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"MODEL_NAME must be one of {SUPPORTED_MODEL_NAMES}, got {model_name!r}"
        )
    patch_size = getattr(config, "SEG_PATCH_SIZE", None)
    if patch_size is not None:
        patch_size = tuple(int(value) for value in patch_size)
        if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
            raise ValueError("SEG_PATCH_SIZE must contain three positive integers")
        if int(getattr(config, "TRAIN_STEPS_PER_EPOCH", 0)) <= 0:
            raise ValueError(
                "TRAIN_STEPS_PER_EPOCH must be positive when SEG_PATCH_SIZE is set"
            )
    optimizer_name = getattr(config, "OPTIMIZER", "adamw")
    if optimizer_name not in ("adamw", "sgd_nesterov"):
        raise ValueError("OPTIMIZER must be adamw or sgd_nesterov")
    scheduler_name = getattr(config, "LR_SCHEDULER", "none")
    if scheduler_name not in ("none", "poly"):
        raise ValueError("LR_SCHEDULER must be none or poly")
    inference_mode = getattr(
        config,
        "SEGMENTATION_VALIDATION_INFERENCE",
        "direct",
    )
    if inference_mode not in ("direct", "sliding_window"):
        raise ValueError(
            "SEGMENTATION_VALIDATION_INFERENCE must be direct or sliding_window"
        )
    if inference_mode == "sliding_window" and not hasattr(
        config,
        "SEG_VALIDATION_ROI_SIZE",
    ):
        raise ValueError(
            "Sliding-window validation requires SEG_VALIDATION_ROI_SIZE"
        )

    feature_mode = getattr(config, "CLASSIFICATION_FEATURE_MODE", "direct")
    if feature_mode not in ("direct", "sliding_cache"):
        raise ValueError(
            "CLASSIFICATION_FEATURE_MODE must be direct or sliding_cache"
        )
    if feature_mode == "sliding_cache":
        if stage != "classification-warmup":
            raise ValueError(
                "sliding_cache is currently supported only for "
                "classification-warmup"
            )
        if not hasattr(config, "CLASSIFICATION_FEATURE_ROI_SIZE"):
            raise ValueError(
                "sliding_cache requires CLASSIFICATION_FEATURE_ROI_SIZE"
            )
        feature_roi = tuple(
            int(value) for value in config.CLASSIFICATION_FEATURE_ROI_SIZE
        )
        if len(feature_roi) != 3 or any(value <= 0 for value in feature_roi):
            raise ValueError(
                "CLASSIFICATION_FEATURE_ROI_SIZE must contain three positive integers"
            )
        feature_overlap = float(
            getattr(config, "CLASSIFICATION_FEATURE_OVERLAP", 0.5)
        )
        if not 0 <= feature_overlap < 1:
            raise ValueError(
                "CLASSIFICATION_FEATURE_OVERLAP must lie in [0, 1)"
            )
        if int(
            getattr(config, "CLASSIFICATION_FEATURE_SW_BATCH_SIZE", 1)
        ) <= 0:
            raise ValueError(
                "CLASSIFICATION_FEATURE_SW_BATCH_SIZE must be positive"
            )
        if tuple(getattr(config, "CLASSIFICATION_FEATURE_MIRROR_AXES", ())) != ():
            raise ValueError(
                "Cached intermediate features do not support mirror TTA; set "
                "CLASSIFICATION_FEATURE_MIRROR_AXES = ()"
            )
        frozen_metric_fields = (
            "FROZEN_SEGMENTATION_POSITIVE_DICE",
            "FROZEN_SEGMENTATION_NNUNET_FOREGROUND_DICE",
            "FROZEN_SEGMENTATION_POSITIVE_PRECISION",
            "FROZEN_SEGMENTATION_POSITIVE_RECALL",
            "FROZEN_SEGMENTATION_POSITIVE_MISS_RATE",
            "FROZEN_SEGMENTATION_NORMAL_MEAN_FP_VOLUME_ML",
            "FROZEN_SEGMENTATION_NORMAL_P95_FP_VOLUME_ML",
            "FROZEN_SEGMENTATION_POSITIVE_CASES",
            "FROZEN_SEGMENTATION_NORMAL_CASES",
        )
        missing_frozen_metrics = [
            name for name in frozen_metric_fields if not hasattr(config, name)
        ]
        if missing_frozen_metrics:
            raise ValueError(
                "sliding_cache requires the frozen validation metrics: "
                + ", ".join(missing_frozen_metrics)
            )


def config_snapshot(config):
    return {
        name: value
        for name, value in vars(config).items()
        if name.isupper() and not name.startswith("__")
    }


def case_has_positive_mask(dataset, index):
    case = dataset.cases[index]
    if int(case["label"]) == 0:
        return False
    if case.get("has_mask") and case.get("mask_path") is not None:
        return True
    valid_sequences = dataset.mask_index.get(str(case["case_id"]), ())
    return SEQUENCE_ID in valid_sequences


def segmentation_supervision_indices(dataset):
    positive_indices = []
    normal_indices = []
    for index, case in enumerate(dataset.cases):
        if int(case["label"]) == 0:
            normal_indices.append(index)
        elif case_has_positive_mask(dataset, index):
            positive_indices.append(index)
    return positive_indices, normal_indices


def build_loaders(config, data_root, fold, stage):
    dataset_root = resolve_input_artifact_dir(data_root, "datasets")
    processed_data_root = infer_data_dir(data_root)
    fold_root = dataset_root / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}" / f"fold{fold}"
    if not fold_root.is_dir():
        raise FileNotFoundError(f"FLAIR fold directory not found: {fold_root}")

    train_dataset = load_pt_dataset(
        fold_root / "train.pt",
        data_root=processed_data_root,
    )
    val_dataset = load_pt_dataset(
        fold_root / "val.pt",
        data_root=processed_data_root,
    )

    train_positive, train_normal = segmentation_supervision_indices(train_dataset)
    val_positive, val_normal = segmentation_supervision_indices(val_dataset)
    print(
        "Segmentation supervision (train): "
        f"positive={len(train_positive)}, normal-zero={len(train_normal)}, "
        f"ratio={len(train_positive) / max(len(train_normal), 1):.2f}:1"
    )
    print(
        "Segmentation supervision (val)  : "
        f"positive={len(val_positive)}, normal-zero={len(val_normal)}"
    )

    if stage == "segmentation":
        train_data = Subset(train_dataset, train_positive + train_normal)
        val_data = Subset(val_dataset, val_positive + val_normal)
    else:
        train_data = train_dataset
        val_data = val_dataset

    loader_kwargs = {
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": str(config.DEVICE).startswith("cuda"),
    }
    patch_size = getattr(config, "SEG_PATCH_SIZE", None)
    if stage == "segmentation" and patch_size is not None:
        patch_dataset = NNUNetPatchDataset(
            train_data,
            patch_size,
            augment=getattr(config, "SEG_AUGMENT", True),
            renormalize_nonzero=getattr(
                config,
                "SEG_RENORMALIZE_NONZERO",
                True,
            ),
        )
        batch_sampler = FixedIterationBatchSampler(
            len(patch_dataset),
            config.BATCH_SIZE,
            getattr(config, "TRAIN_STEPS_PER_EPOCH"),
            foreground_oversample=getattr(
                config,
                "SEG_FOREGROUND_OVERSAMPLE",
                0.33,
            ),
            seed=SEED + int(fold) * 10000,
        )
        patch_loader_kwargs = {
            "batch_sampler": batch_sampler,
            "num_workers": config.NUM_WORKERS,
            "pin_memory": str(config.DEVICE).startswith("cuda"),
        }
        if int(config.NUM_WORKERS) > 0:
            patch_loader_kwargs["persistent_workers"] = True
        train_loader = DataLoader(patch_dataset, **patch_loader_kwargs)
        print(
            "Segmentation patch training: "
            f"patch={tuple(int(v) for v in patch_size)}, "
            f"batches/epoch={len(batch_sampler)}, "
            "foreground oversample="
            f"{float(getattr(config, 'SEG_FOREGROUND_OVERSAMPLE', 0.33)):.2f}"
        )
    else:
        train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, dataset_root


def set_module_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def configure_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if stage in ("segmentation", "joint"):
        set_module_trainable(model.encoder, True)
        set_module_trainable(model.decoder, True)
        set_module_trainable(model.aux_heads, True)
    if stage in ("classification-warmup", "joint"):
        set_module_trainable(model.global_projection, True)
        set_module_trainable(model.lesion_projection, True)
        set_module_trainable(model.classification_head, True)


def add_parameter_group(groups, module, learning_rate, name):
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if parameters:
        groups.append(
            {
                "params": parameters,
                "lr": float(learning_rate),
                "name": name,
            }
        )


def build_optimizer(model, config):
    groups = []
    add_parameter_group(
        groups,
        model.encoder,
        config.ENCODER_LEARNING_RATE,
        "encoder",
    )
    add_parameter_group(
        groups,
        model.decoder,
        config.SEGMENTATION_LEARNING_RATE,
        "decoder",
    )
    add_parameter_group(
        groups,
        model.aux_heads,
        config.SEGMENTATION_LEARNING_RATE,
        "segmentation_head",
    )
    for module_name in (
        "global_projection",
        "lesion_projection",
        "classification_head",
    ):
        add_parameter_group(
            groups,
            getattr(model, module_name),
            config.CLASSIFICATION_LEARNING_RATE,
            module_name,
        )
    if not groups:
        raise ValueError("The selected stage has no trainable parameters")
    for group in groups:
        parameter_count = sum(parameter.numel() for parameter in group["params"])
        print(
            f"Optimizer group {group['name']}: lr={group['lr']:g}, "
            f"parameters={parameter_count:,}"
        )
    optimizer_name = getattr(config, "OPTIMIZER", "adamw")
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            groups,
            weight_decay=float(config.WEIGHT_DECAY),
        )
    if optimizer_name == "sgd_nesterov":
        return torch.optim.SGD(
            groups,
            weight_decay=float(config.WEIGHT_DECAY),
            momentum=float(getattr(config, "SGD_MOMENTUM", 0.99)),
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_lr_scheduler(optimizer, config):
    scheduler_name = getattr(config, "LR_SCHEDULER", "none")
    if scheduler_name == "none":
        return None
    if scheduler_name == "poly":
        total_epochs = int(config.NUM_EPOCHS)
        power = float(getattr(config, "POLY_LR_POWER", 0.9))

        def learning_rate_factor(epoch):
            progress = min(max(float(epoch) / total_epochs, 0.0), 1.0)
            return (1.0 - progress) ** power

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=learning_rate_factor,
        )
    raise ValueError(f"Unsupported LR scheduler: {scheduler_name}")


def unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def build_classification_criterion(dataset, config, device):
    labels = torch.as_tensor(dataset.labels, dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=NUM_CLASSES)
    if torch.any(class_counts <= 0):
        raise ValueError(f"Every class needs samples, got {class_counts.tolist()}")

    if config.CLASSIFICATION_LOSS == "cross_entropy":
        print("Classification loss: CrossEntropyLoss")
        return nn.CrossEntropyLoss().to(device), class_counts
    if config.CLASSIFICATION_LOSS == "weighted_cross_entropy":
        raw_weights = len(labels) / (NUM_CLASSES * class_counts.float())
        weights = raw_weights.pow(float(config.CLASS_WEIGHT_POWER)).to(device)
        print(f"Classification weights: {weights.tolist()}")
        return nn.CrossEntropyLoss(weight=weights), class_counts
    raise ValueError(
        "FLAIR U-Net trainer supports cross_entropy or weighted_cross_entropy, "
        f"got {config.CLASSIFICATION_LOSS}"
    )


def class_precision_recall_fbeta(targets, predictions, class_index, beta):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    true_positive = int(
        np.sum((targets == class_index) & (predictions == class_index))
    )
    predicted_positive = int(np.sum(predictions == class_index))
    actual_positive = int(np.sum(targets == class_index))
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    beta_squared = float(beta) ** 2
    denominator = beta_squared * precision + recall
    fbeta = (
        (1.0 + beta_squared) * precision * recall / denominator
        if denominator > 0
        else 0.0
    )
    return float(precision), float(recall), float(fbeta)


def load_initial_checkpoint(
    model,
    checkpoint_root,
    fold,
    expected_source_stage,
    model_name,
):
    checkpoint_dir = resolve_input_artifact_dir(checkpoint_root, "checkpoints")
    checkpoint_path = (
        checkpoint_dir
        / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}"
        / model_name
        / f"fold{fold}_model_best.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_stage = checkpoint.get("stage")
    if source_stage != expected_source_stage:
        raise ValueError(
            f"Expected a {expected_source_stage} checkpoint, got {source_stage!r}: "
            f"{checkpoint_path}"
        )
    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name not in (None, model_name):
        raise ValueError(
            f"Checkpoint model mismatch: {checkpoint_model_name!r} vs "
            f"{model_name!r}: {checkpoint_path}"
        )
    source_state = checkpoint["model_state"]
    if expected_source_stage == "segmentation":
        # 阶段一的分类头从未训练，且增强版允许改变分类输入维度。这里只迁移
        # 分割路径，并严格要求 encoder/decoder/seg head 的键和形状完全一致。
        loaded_tensor_count = load_flair_segmentation_state(model, source_state)
        print(
            "Loaded segmentation backbone: "
            f"{loaded_tensor_count} tensors; "
            "classification modules initialized for this stage"
        )
    else:
        model.load_state_dict(source_state, strict=True)
    print(f"Initialized from: {checkpoint_path}")
    return str(checkpoint_path)


def amp_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def segmentation_loss_kwargs(config):
    return {
        "bce_weight": config.SEG_BCE_WEIGHT,
        "dice_weight": config.SEG_DICE_WEIGHT,
        "positive_weight": config.SEG_POSITIVE_WEIGHT,
        "bce_mode": getattr(config, "SEG_BCE_MODE", "mean"),
        "hard_negative_fraction": getattr(
            config,
            "SEG_HARD_NEGATIVE_FRACTION",
            0.002,
        ),
        "hard_negative_min_voxels": getattr(
            config,
            "SEG_HARD_NEGATIVE_MIN_VOXELS",
            256,
        ),
        "normal_dense_bce_weight": getattr(
            config,
            "SEG_NORMAL_DENSE_BCE_WEIGHT",
            0.5,
        ),
    }


def segmentation_loss_from_outputs(outputs, mask, mask_flag, config, deep_supervision):
    loss_kwargs = segmentation_loss_kwargs(config)
    primary_loss = binary_segmentation_loss(
        outputs["segmentation"],
        mask,
        mask_flag,
        **loss_kwargs,
    )
    if not deep_supervision:
        return primary_loss

    auxiliary_logits = outputs.get("segmentation_aux", ())
    weights = tuple(float(value) for value in config.DEEP_SUPERVISION_WEIGHTS)
    if len(weights) > len(auxiliary_logits):
        raise ValueError(
            f"Got {len(weights)} deep-supervision weights for "
            f"{len(auxiliary_logits)} auxiliary outputs"
        )
    weighted_loss = primary_loss
    total_weight = 1.0
    for weight, logits in zip(weights, auxiliary_logits):
        if weight <= 0:
            continue
        weighted_loss = weighted_loss + weight * binary_segmentation_loss(
            logits,
            mask,
            mask_flag,
            **loss_kwargs,
        )
        total_weight += weight
    return weighted_loss / total_weight


def compute_batch_loss(outputs, y, mask, mask_flag, criterion, config, stage, train):
    zero = outputs["classification"].sum() * 0.0
    classification_loss = zero
    segmentation_loss = zero

    if stage in ("classification-warmup", "joint"):
        classification_loss = criterion(outputs["classification"], y)
    if stage in ("segmentation", "joint"):
        segmentation_loss = segmentation_loss_from_outputs(
            outputs,
            mask,
            mask_flag,
            config,
            deep_supervision=train,
        )
    total_loss = (
        float(config.CLASSIFICATION_ALPHA) * classification_loss
        + float(config.SEG_ALPHA) * segmentation_loss
    )
    return total_loss, classification_loss, segmentation_loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    config,
    stage,
    device,
    fold,
    epoch,
):
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    progress = tqdm(loader, desc=f"Fold {fold} Ep {epoch}", leave=False)
    for x, y, mask, mask_flag, _ in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        mask_flag = mask_flag.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            outputs = forward_model(
                model,
                x,
                return_seg=True,
                segmentation_only=(stage == "segmentation"),
            )
            loss, _, _ = compute_batch_loss(
                outputs,
                y,
                mask,
                mask_flag,
                criterion,
                config,
                stage,
                train=True,
            )
        scaler.scale(loss).backward()
        gradient_clip_norm = getattr(config, "GRADIENT_CLIP_NORM", None)
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(gradient_clip_norm),
            )
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach())
        predictions.extend(outputs["classification"].argmax(dim=1).cpu().tolist())
        targets.extend(y.cpu().tolist())
    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(targets, predictions),
    }


def train_cached_classifier_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    config,
    device,
    fold,
    epoch,
):
    """只对缓存向量运行双路 projection 和 classification head。"""
    model.train()
    base_model = unwrap_model(model)
    total_loss = 0.0
    sample_count = 0
    predictions = []
    targets = []
    progress = tqdm(loader, desc=f"Fold {fold} Ep {epoch}", leave=False)
    for global_features, lesion_features, y, _ in progress:
        global_features = global_features.to(device, non_blocking=True)
        lesion_features = lesion_features.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            logits = base_model.classify_pooled_features(
                global_features,
                lesion_features,
            )
            classification_loss = criterion(logits, y)
            loss = float(config.CLASSIFICATION_ALPHA) * classification_loss
        scaler.scale(loss).backward()
        gradient_clip_norm = getattr(config, "GRADIENT_CLIP_NORM", None)
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                base_model.parameters(),
                float(gradient_clip_norm),
            )
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(y.size(0))
        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(y.cpu().tolist())
    return {
        "loss": total_loss / max(sample_count, 1),
        "accuracy": accuracy_score(targets, predictions),
    }


def validate_cached_classifier(model, loader, criterion, config, device):
    model.eval()
    base_model = unwrap_model(model)
    total_loss = 0.0
    total_classification_loss = 0.0
    sample_count = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for global_features, lesion_features, y, _ in loader:
            global_features = global_features.to(device, non_blocking=True)
            lesion_features = lesion_features.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with amp_context(device):
                logits = base_model.classify_pooled_features(
                    global_features,
                    lesion_features,
                )
                classification_loss = criterion(logits, y)
                loss = float(config.CLASSIFICATION_ALPHA) * classification_loss
            batch_size = int(y.size(0))
            total_loss += float(loss) * batch_size
            total_classification_loss += float(classification_loss) * batch_size
            sample_count += batch_size
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(y.cpu().tolist())

    metastasis_index = CLASS_NAMES.index("metastasis")
    metastasis_precision, metastasis_recall, metastasis_fbeta = (
        class_precision_recall_fbeta(
            targets,
            predictions,
            metastasis_index,
            config.METASTASIS_F_BETA,
        )
    )
    return ValidationResult(
        loss=total_loss / max(sample_count, 1),
        classification_loss=total_classification_loss / max(sample_count, 1),
        segmentation_loss=0.0,
        accuracy=float(accuracy_score(targets, predictions)),
        macro_f1=float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        metastasis_precision=metastasis_precision,
        metastasis_recall=metastasis_recall,
        metastasis_fbeta=metastasis_fbeta,
        positive_dice=float(config.FROZEN_SEGMENTATION_POSITIVE_DICE),
        positive_voxel_precision=float(
            config.FROZEN_SEGMENTATION_POSITIVE_PRECISION
        ),
        positive_voxel_recall=float(config.FROZEN_SEGMENTATION_POSITIVE_RECALL),
        positive_complete_miss_rate=float(
            config.FROZEN_SEGMENTATION_POSITIVE_MISS_RATE
        ),
        normal_false_positive_fraction=float(
            getattr(config, "FROZEN_SEGMENTATION_NORMAL_FP_FRACTION", 0.0)
        ),
        normal_false_positive_volume_ml=float(
            config.FROZEN_SEGMENTATION_NORMAL_MEAN_FP_VOLUME_ML
        ),
        normal_p95_false_positive_volume_ml=float(
            config.FROZEN_SEGMENTATION_NORMAL_P95_FP_VOLUME_ML
        ),
        positive_mask_cases=int(config.FROZEN_SEGMENTATION_POSITIVE_CASES),
        normal_mask_cases=int(config.FROZEN_SEGMENTATION_NORMAL_CASES),
        nnunet_foreground_mean_dice=float(
            config.FROZEN_SEGMENTATION_NNUNET_FOREGROUND_DICE
        ),
    )


def validate(model, loader, criterion, config, stage, device):
    model.eval()
    losses = []
    classification_losses = []
    segmentation_losses = []
    predictions = []
    targets = []
    positive_dices = []
    positive_precisions = []
    positive_recalls = []
    positive_detected = []
    nnunet_foreground_dices = []
    normal_false_positive_fractions = []
    normal_false_positive_volumes = []

    with torch.no_grad():
        for x, y, mask, mask_flag, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            mask_flag = mask_flag.to(device, non_blocking=True)
            with amp_context(device):
                if (
                    stage == "segmentation"
                    and getattr(
                        config,
                        "SEGMENTATION_VALIDATION_INFERENCE",
                        "direct",
                    )
                    == "sliding_window"
                ):
                    segmentation_logits = segmentation_logits_from_config(
                        model,
                        x,
                        config,
                    )
                    outputs = {
                        "classification": segmentation_logits.new_zeros(
                            (x.size(0), NUM_CLASSES)
                        ),
                        "segmentation": segmentation_logits,
                    }
                else:
                    outputs = forward_model(
                        model,
                        x,
                        return_seg=True,
                        segmentation_only=(stage == "segmentation"),
                    )
                loss, classification_loss, segmentation_loss = compute_batch_loss(
                    outputs,
                    y,
                    mask,
                    mask_flag,
                    criterion,
                    config,
                    stage,
                    train=False,
                )

            losses.append(float(loss))
            classification_losses.append(float(classification_loss))
            segmentation_losses.append(float(segmentation_loss))
            predictions.extend(outputs["classification"].argmax(dim=1).cpu().tolist())
            targets.extend(y.cpu().tolist())

            lesion_prediction = binary_lesion_predictions(outputs["segmentation"])
            dice = binary_dice_per_sample(lesion_prediction, mask)
            voxel_recall = binary_recall_per_sample(lesion_prediction, mask)
            target_mask = prepare_binary_mask(mask).squeeze(1).to(device) > 0
            predicted_mask = lesion_prediction > 0
            true_positive = (predicted_mask & target_mask).sum(dim=(1, 2, 3))
            predicted_count = predicted_mask.sum(dim=(1, 2, 3))
            target_count = target_mask.sum(dim=(1, 2, 3))
            foreground_denominator = predicted_count + target_count
            foreground_defined = foreground_denominator > 0
            if foreground_defined.any():
                nnunet_foreground_dices.extend(
                    (
                        2.0
                        * true_positive[foreground_defined].float()
                        / foreground_denominator[foreground_defined].float()
                    )
                    .cpu()
                    .tolist()
                )
            voxel_precision = torch.where(
                predicted_count > 0,
                true_positive.float() / predicted_count.clamp_min(1).float(),
                torch.zeros_like(true_positive, dtype=torch.float32),
            )
            positive = (y > 0) & (mask_flag > 0.5)
            normal = (y == 0) & (mask_flag > 0.5)
            positive_dices.extend(dice[positive].cpu().tolist())
            positive_precisions.extend(
                voxel_precision[positive].cpu().tolist()
            )
            positive_recalls.extend(voxel_recall[positive].cpu().tolist())
            positive_detected.extend(
                (true_positive[positive] > 0).cpu().tolist()
            )
            if normal.any():
                normal_predictions = lesion_prediction[normal].float()
                false_positive_voxels = normal_predictions.sum(dim=(1, 2, 3))
                normal_false_positive_fractions.extend(
                    normal_predictions.mean(dim=(1, 2, 3)).cpu().tolist()
                )
                normal_false_positive_volumes.extend(
                    (false_positive_voxels * VOXEL_VOLUME_ML).cpu().tolist()
                )

    metastasis_index = CLASS_NAMES.index("metastasis")
    metastasis_precision, metastasis_recall, metastasis_fbeta = (
        class_precision_recall_fbeta(
            targets,
            predictions,
            metastasis_index,
            config.METASTASIS_F_BETA,
        )
    )
    return ValidationResult(
        loss=float(np.mean(losses)),
        classification_loss=float(np.mean(classification_losses)),
        segmentation_loss=float(np.mean(segmentation_losses)),
        accuracy=float(accuracy_score(targets, predictions)),
        macro_f1=float(f1_score(targets, predictions, average="macro", zero_division=0)),
        metastasis_precision=metastasis_precision,
        metastasis_recall=metastasis_recall,
        metastasis_fbeta=metastasis_fbeta,
        positive_dice=float(np.mean(positive_dices)) if positive_dices else 0.0,
        positive_voxel_precision=(
            float(np.mean(positive_precisions)) if positive_precisions else 0.0
        ),
        positive_voxel_recall=(
            float(np.mean(positive_recalls)) if positive_recalls else 0.0
        ),
        positive_complete_miss_rate=(
            1.0 - float(np.mean(positive_detected))
            if positive_detected
            else 0.0
        ),
        normal_false_positive_fraction=(
            float(np.mean(normal_false_positive_fractions))
            if normal_false_positive_fractions
            else 0.0
        ),
        normal_false_positive_volume_ml=(
            float(np.mean(normal_false_positive_volumes))
            if normal_false_positive_volumes
            else 0.0
        ),
        normal_p95_false_positive_volume_ml=(
            float(np.percentile(normal_false_positive_volumes, 95))
            if normal_false_positive_volumes
            else 0.0
        ),
        positive_mask_cases=len(positive_dices),
        normal_mask_cases=len(normal_false_positive_fractions),
        nnunet_foreground_mean_dice=(
            float(np.mean(nnunet_foreground_dices))
            if nnunet_foreground_dices
            else 0.0
        ),
    )


def classification_constraints_met(result, config):
    return (
        result.positive_dice >= float(config.CHECKPOINT_MIN_POSITIVE_DICE)
        and result.accuracy >= float(config.CHECKPOINT_MIN_ACCURACY)
        and result.macro_f1
        >= float(getattr(config, "CHECKPOINT_MIN_MACRO_F1", 0.0))
        and result.metastasis_precision
        >= float(config.CHECKPOINT_MIN_METASTASIS_PRECISION)
        and result.metastasis_recall
        >= float(getattr(config, "CHECKPOINT_MIN_METASTASIS_RECALL", 0.0))
    )


def segmentation_constraints_met(result, config):
    maximum_fp_volume = float(
        getattr(config, "CHECKPOINT_MAX_NORMAL_FP_VOLUME_ML", float("inf"))
    )
    return (
        result.positive_dice >= float(config.CHECKPOINT_MIN_POSITIVE_DICE)
        and result.positive_voxel_precision
        >= float(getattr(config, "CHECKPOINT_MIN_POSITIVE_PRECISION", 0.0))
        and result.positive_voxel_recall
        >= float(getattr(config, "CHECKPOINT_MIN_POSITIVE_RECALL", 0.0))
        and result.positive_complete_miss_rate
        <= float(getattr(config, "CHECKPOINT_MAX_POSITIVE_MISS_RATE", 1.0))
        and result.normal_false_positive_volume_ml <= maximum_fp_volume
    )


def segmentation_reference_met(result, config):
    required_fields = (
        "REFERENCE_POSITIVE_DICE",
        "REFERENCE_NNUNET_FOREGROUND_MEAN_DICE",
        "REFERENCE_POSITIVE_PRECISION",
        "REFERENCE_POSITIVE_RECALL",
        "REFERENCE_MAX_POSITIVE_MISS_RATE",
        "REFERENCE_MAX_NORMAL_MEAN_FP_VOLUME_ML",
        "REFERENCE_MAX_NORMAL_P95_FP_VOLUME_ML",
    )
    if not all(hasattr(config, name) for name in required_fields):
        return None
    return (
        result.positive_dice >= float(config.REFERENCE_POSITIVE_DICE)
        and result.nnunet_foreground_mean_dice
        >= float(config.REFERENCE_NNUNET_FOREGROUND_MEAN_DICE)
        and result.positive_voxel_precision
        >= float(config.REFERENCE_POSITIVE_PRECISION)
        and result.positive_voxel_recall
        >= float(config.REFERENCE_POSITIVE_RECALL)
        and result.positive_complete_miss_rate
        <= float(config.REFERENCE_MAX_POSITIVE_MISS_RATE)
        and result.normal_false_positive_volume_ml
        <= float(config.REFERENCE_MAX_NORMAL_MEAN_FP_VOLUME_ML)
        and result.normal_p95_false_positive_volume_ml
        <= float(config.REFERENCE_MAX_NORMAL_P95_FP_VOLUME_ML)
    )


def primary_classification_metric(result, config):
    if config.CHECKPOINT_SELECTION_METRIC == "macro_f1":
        return result.macro_f1
    if config.CHECKPOINT_SELECTION_METRIC == "metastasis_fbeta":
        return result.metastasis_fbeta
    raise ValueError(
        "CHECKPOINT_SELECTION_METRIC must be macro_f1 or metastasis_fbeta, "
        f"got {config.CHECKPOINT_SELECTION_METRIC}"
    )


def is_better_result(candidate, best, stage, config, tolerance=1e-5):
    if best is None:
        return True
    if stage == "segmentation":
        candidate_meets_constraints = segmentation_constraints_met(
            candidate,
            config,
        )
        best_meets_constraints = segmentation_constraints_met(best, config)
        if candidate_meets_constraints != best_meets_constraints:
            return candidate_meets_constraints
        if abs(candidate.positive_dice - best.positive_dice) > tolerance:
            return candidate.positive_dice > best.positive_dice
        if (
            abs(
                candidate.normal_false_positive_volume_ml
                - best.normal_false_positive_volume_ml
            )
            > tolerance
        ):
            return (
                candidate.normal_false_positive_volume_ml
                < best.normal_false_positive_volume_ml
            )
        return candidate.segmentation_loss < best.segmentation_loss

    candidate_meets_constraints = classification_constraints_met(candidate, config)
    best_meets_constraints = classification_constraints_met(best, config)
    if candidate_meets_constraints != best_meets_constraints:
        return candidate_meets_constraints
    candidate_primary = primary_classification_metric(candidate, config)
    best_primary = primary_classification_metric(best, config)
    if abs(candidate_primary - best_primary) > tolerance:
        return candidate_primary > best_primary
    if abs(candidate.macro_f1 - best.macro_f1) > tolerance:
        return candidate.macro_f1 > best.macro_f1
    if abs(candidate.positive_dice - best.positive_dice) > tolerance:
        return candidate.positive_dice > best.positive_dice
    return candidate.loss < best.loss


def expected_source_stage(stage):
    if stage == "classification-warmup":
        return "segmentation"
    if stage == "joint":
        return "classification-warmup"
    return None


def main(args):
    config = load_python_config(args.config)
    require_config_fields(config)
    validate_stage_config(config, args.stage)
    model_name = getattr(config, "MODEL_NAME", MODEL_NAME)
    set_seed(SEED)
    device = torch.device(config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA, but CUDA is not available")

    source_stage = expected_source_stage(args.stage)
    if source_stage is None and args.init_checkpoint_root is not None:
        raise ValueError("The segmentation stage starts from scratch; omit init root")
    if source_stage is not None and args.init_checkpoint_root is None:
        raise ValueError(
            f"The {args.stage} stage requires --init-checkpoint-root from "
            f"the {source_stage} stage"
        )

    train_loader, val_loader, dataset_root = build_loaders(
        config,
        args.data_root,
        args.fold,
        args.stage,
    )
    model = create_model(
        model_name,
        num_classes=NUM_CLASSES,
        in_channels=1,
        sequence_id=SEQUENCE_ID,
    )
    initialization_path = None
    if source_stage is not None:
        initialization_path = load_initial_checkpoint(
            model,
            args.init_checkpoint_root,
            args.fold,
            source_stage,
            model_name,
        )

    configure_stage(model, args.stage)
    optimizer = build_optimizer(model, config)
    lr_scheduler = build_lr_scheduler(optimizer, config)
    model = model.to(device)
    cached_feature_mode = (
        args.stage == "classification-warmup"
        and getattr(config, "CLASSIFICATION_FEATURE_MODE", "direct")
        == "sliding_cache"
    )
    if (
        device.type == "cuda"
        and torch.cuda.device_count() > 1
        and not cached_feature_mode
    ):
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    elif cached_feature_mode and device.type == "cuda":
        print(
            "Sliding feature extraction uses the primary GPU; cached classifier "
            "training does not benefit from DataParallel"
        )

    classification_dataset = unwrap_dataset(train_loader.dataset)
    validation_dataset = unwrap_dataset(val_loader.dataset)
    criterion, class_counts = build_classification_criterion(
        classification_dataset,
        config,
        device,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    checkpoint_root = resolve_output_artifact_dir(args.output_root, "checkpoints")
    checkpoint_dir = (
        checkpoint_root / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}" / model_name
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fold{args.fold}_model_best.pth"
    unconstrained_checkpoint_path = (
        checkpoint_dir / f"fold{args.fold}_model_best_unconstrained.pth"
    )
    latest_checkpoint_path = (
        checkpoint_dir / f"fold{args.fold}_model_latest.pth"
    )

    best_result = None
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1
    if args.resume:
        if not latest_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {latest_checkpoint_path}"
            )
        resume_checkpoint = torch.load(
            latest_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if resume_checkpoint.get("model_name") != model_name:
            raise ValueError(
                "Resume checkpoint model mismatch: "
                f"{resume_checkpoint.get('model_name')!r} vs {model_name!r}"
            )
        unwrap_model(model).load_state_dict(
            resume_checkpoint["model_state"],
            strict=True,
        )
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(
                resume_checkpoint["lr_scheduler_state"]
            )
        if resume_checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
        saved_best = resume_checkpoint.get("best_validation")
        if saved_best is not None:
            best_result = ValidationResult(**saved_best)
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))
        patience_counter = int(
            resume_checkpoint.get("patience_counter", 0)
        )
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        if hasattr(train_loader.batch_sampler, "epoch"):
            train_loader.batch_sampler.epoch = start_epoch - 1
        print(f"Resuming from: {latest_checkpoint_path} (epoch {start_epoch})")
    elif (
        checkpoint_path.exists()
        or unconstrained_checkpoint_path.exists()
        or latest_checkpoint_path.exists()
    ):
        raise FileExistsError(
            "Refusing to overwrite an existing fold checkpoint. Use --resume "
            f"or choose a new --output-root: {checkpoint_dir}"
        )

    feature_cache_metadata_by_split = None
    if cached_feature_mode:
        train_loader, val_loader, feature_cache_metadata_by_split = (
            cached_feature_loaders(
                model=model,
                train_dataset=classification_dataset,
                val_dataset=validation_dataset,
                config=config,
                output_root=args.output_root,
                initialization_path=initialization_path,
                fold=args.fold,
                model_name=model_name,
                device=device,
            )
        )

    print(f"Stage            : {args.stage}")
    print(f"Model            : {model_name}")
    print(f"Training config  : {config.__config_path__}")
    print(
        "Segmentation loss: "
        f"BCE mode={getattr(config, 'SEG_BCE_MODE', 'mean')}, "
        f"BCE weight={config.SEG_BCE_WEIGHT}, "
        f"Dice weight={config.SEG_DICE_WEIGHT}"
    )
    print(f"Dataset root     : {dataset_root}")
    print(f"Checkpoint output: {checkpoint_path}")
    print(f"Class counts     : {class_counts.tolist()}")
    print(
        "Classification features: "
        f"{getattr(config, 'CLASSIFICATION_FEATURE_MODE', 'direct')}"
    )
    print(
        "Optimizer/LR      : "
        f"{getattr(config, 'OPTIMIZER', 'adamw')} / "
        f"{getattr(config, 'LR_SCHEDULER', 'none')}"
    )

    for epoch in range(start_epoch, int(config.NUM_EPOCHS) + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        if cached_feature_mode:
            train_result = train_cached_classifier_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                criterion,
                config,
                device,
                args.fold,
                epoch,
            )
        else:
            train_result = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                criterion,
                config,
                args.stage,
                device,
                args.fold,
                epoch,
            )
        validation_interval = max(
            1,
            int(getattr(config, "VALIDATE_EVERY_N_EPOCHS", 1)),
        )
        should_validate = (
            epoch % validation_interval == 0
            or epoch == int(config.NUM_EPOCHS)
        )
        if not should_validate:
            print(
                f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
                f"lr={current_lr:.6g} | "
                f"train_loss={train_result['loss']:.4f} | "
                f"validation=skipped"
            )
            if lr_scheduler is not None:
                lr_scheduler.step()
            continue
        if cached_feature_mode:
            val_result = validate_cached_classifier(
                model,
                val_loader,
                criterion,
                config,
                device,
            )
        else:
            val_result = validate(
                model,
                val_loader,
                criterion,
                config,
                args.stage,
                device,
            )
        reference_met = segmentation_reference_met(val_result, config)
        reference_text = (
            "n/a" if reference_met is None else str(reference_met).lower()
        )
        print(
            f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
            f"lr={current_lr:.6g} | "
            f"train_loss={train_result['loss']:.4f} | "
            f"val_loss={val_result.loss:.4f} | "
            f"val_acc={val_result.accuracy:.4f} | "
            f"val_f1={val_result.macro_f1:.4f} | "
            f"val_meta_precision={val_result.metastasis_precision:.4f} | "
            f"val_meta_recall={val_result.metastasis_recall:.4f} | "
            f"val_meta_f{config.METASTASIS_F_BETA:g}="
            f"{val_result.metastasis_fbeta:.4f} | "
            f"positive_dice={val_result.positive_dice:.4f} | "
            f"nnunet_foreground_dice="
            f"{val_result.nnunet_foreground_mean_dice:.4f} | "
            f"positive_precision={val_result.positive_voxel_precision:.4f} | "
            f"positive_voxel_recall={val_result.positive_voxel_recall:.4f} | "
            f"positive_miss_rate={val_result.positive_complete_miss_rate:.4f} | "
            f"normal_mean_fp_ml="
            f"{val_result.normal_false_positive_volume_ml:.4f} | "
            f"normal_p95_fp_ml="
            f"{val_result.normal_p95_false_positive_volume_ml:.4f} | "
            f"nnunet_reference_met={reference_text}"
        )

        improved = is_better_result(
            val_result,
            best_result,
            args.stage,
            config,
        )
        if improved:
            best_result = val_result
            best_epoch = epoch
            patience_counter = 0
            model_to_save = unwrap_model(model)
            constraints_met = (
                True
                if args.stage == "segmentation"
                else classification_constraints_met(val_result, config)
            )
            selected_checkpoint_path = (
                checkpoint_path
                if constraints_met
                else unconstrained_checkpoint_path
            )
            torch.save(
                {
                    "model_state": model_to_save.state_dict(),
                    "model_name": model_name,
                    "model_capabilities": {
                        "classification": True,
                        "subtype": False,
                        "segmentation": True,
                    },
                    "segmentation_target_mode": "binary_lesion",
                    "stage": args.stage,
                    "fold": args.fold,
                    "epoch": epoch,
                    "validation": asdict(val_result),
                    "selection_constraints_met": constraints_met,
                    "initialization_path": initialization_path,
                    "train_config_path": str(config.__config_path__),
                    "train_config": config_snapshot(config),
                    "dataset_root": str(dataset_root),
                    "class_counts": class_counts.tolist(),
                    "classification_feature_cache": (
                        feature_cache_metadata_by_split
                    ),
                },
                selected_checkpoint_path,
            )
        should_stop = False
        if (
            not improved
            and args.stage != "segmentation"
            and best_result is not None
            and not classification_constraints_met(best_result, config)
        ):
            # 未出现满足验证约束的候选前完整训练，不消耗 early-stop patience。
            patience_counter = 0
        elif not improved and epoch > int(config.MIN_EPOCHS):
            patience_counter += 1
            if patience_counter >= int(config.PATIENCE):
                print(
                    f"[Early Stopping] Best epoch={best_epoch}, "
                    f"positive_dice={best_result.positive_dice:.4f}, "
                    f"macro_f1={best_result.macro_f1:.4f}"
                )
                should_stop = True
        elif not improved:
            patience_counter = 0

        if lr_scheduler is not None:
            lr_scheduler.step()

        torch.save(
            {
                "model_state": unwrap_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "lr_scheduler_state": (
                    lr_scheduler.state_dict()
                    if lr_scheduler is not None
                    else None
                ),
                "scaler_state": scaler.state_dict(),
                "model_name": model_name,
                "stage": args.stage,
                "fold": args.fold,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation": (
                    asdict(best_result) if best_result is not None else None
                ),
                "best_selection_constraints_met": (
                    args.stage == "segmentation"
                    or (
                        best_result is not None
                        and classification_constraints_met(best_result, config)
                    )
                ),
                "patience_counter": patience_counter,
                "train_config_path": str(config.__config_path__),
                "train_config": config_snapshot(config),
                "classification_feature_cache": (
                    feature_cache_metadata_by_split
                ),
            },
            latest_checkpoint_path,
        )
        if should_stop:
            break

    if args.stage == "segmentation" or (
        best_result is not None
        and classification_constraints_met(best_result, config)
    ):
        print(f"[Finished] Eligible best checkpoint: {checkpoint_path}")
    else:
        print(
            "[Finished] No validation epoch met all classification constraints; "
            f"best unconstrained checkpoint: {unconstrained_checkpoint_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, K_FOLDS + 1),
    )
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument(
        "--init-checkpoint-root",
        default=None,
        help="Previous-stage experiment root (or its checkpoints directory)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from this fold's latest checkpoint without overwriting it.",
    )
    main(parser.parse_args())
