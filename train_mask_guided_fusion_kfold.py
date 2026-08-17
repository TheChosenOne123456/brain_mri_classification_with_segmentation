"""重新训练三路分类器，并按异步 epoch 组合选择融合 validation 工作点。"""

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_utils import (
    TRAIN_CONFIG_FIELDS,
    infer_data_dir,
    load_python_config,
    resolve_input_artifact_dir,
    resolve_output_artifact_dir,
)
from configs.global_config import (
    ALL_SEQUENCES,
    CLASS_NAMES,
    K_FOLDS,
    NUM_CLASSES,
    SEED,
)
from models.NNUNetMaskGuidedClassifier import NNUNetMaskGuidedClassifier
from models.model_factory import create_model, forward_model, unwrap_model
import train_kfold as single_train
from train_foundation_nnunet_guided import (
    build_outer_test_fold_map,
    load_split_dataset,
    native_nnunet_checkpoint_path,
    normalize_case_id,
    sha256_file,
    source_folds_for_split,
)
from utils.nnunet_mask_cache import (
    extract_mask_cache,
    load_mask_guided_dataset,
)
from utils.train_and_test import load_pt_dataset, set_seed


MODEL_NAMES = (
    "FoundationModel_ori",
    "FoundationModel_ori",
    "NNUNetMaskGuidedClassifier",
)
REFERENCE_MODEL_NAMES = (
    "FoundationModel_ori",
    "FoundationModel_ori",
    "FoundationModel",
)
MODEL_NAME = MODEL_NAMES[2]

CONFIG_FIELDS = TRAIN_CONFIG_FIELDS + (
    "GRADIENT_CLIP_NORM",
    "LR_SCHEDULER",
    "LR_WARMUP_EPOCHS",
    "LR_WARMUP_START_FACTOR",
    "LR_DECAY_EPOCHS",
    "MIN_LEARNING_RATE",
    "SUBTYPE_LOSS_WEIGHT",
    "SUBTYPE_POS_WEIGHT_POWER",
    "GUIDANCE_STAGE_PROJECTION_DIM",
    "GUIDANCE_STATS_PROJECTION_DIM",
    "GUIDANCE_SUBTYPE_HIDDEN_DIM",
    "GUIDANCE_SUBTYPE_DROPOUT",
    "GUIDANCE_MASK_CACHE_SIZE",
    "GUIDANCE_VOXEL_VOLUME_ML",
    "MASK_CACHE_CHUNK_SIZE",
    "FEATURE_EXTRACTION_NUM_WORKERS",
    "NNUNET_MODEL_RELATIVE_DIR",
    "NNUNET_CHECKPOINT_NAME",
    "NNUNET_USE_DATAPARALLEL",
    "SEGMENTATION_VALIDATION_INFERENCE",
    "SEG_VALIDATION_ROI_SIZE",
    "SEG_VALIDATION_OVERLAP",
    "SEG_VALIDATION_SW_BATCH_SIZE",
    "SEG_VALIDATION_MIRROR_AXES",
    "SEG_VALIDATION_CROP_NONZERO",
    "SEG_VALIDATION_RENORMALIZE_NONZERO",
    "MIX_ALPHA_MIN",
    "MIX_ALPHA_MAX",
    "MIX_ALPHA_STEPS",
    "METASTASIS_F_BETA",
    "LOCAL_GUIDANCE_ACCURACY_TOLERANCE",
    "LOCAL_GUIDANCE_PRECISION_TOLERANCE",
    "LOCAL_MIN_METASTASIS_PRECISION",
    "CANDIDATES_PER_RANKING",
    "FUSION_REFERENCE_ACCURACY_TOLERANCE",
    "FUSION_REFERENCE_PRECISION_TOLERANCE",
    "FUSION_MIN_VAL_METASTASIS_PRECISION",
    "PARALLEL_MIN_FREE_GPU_MEMORY_GB",
)


def _device(config):
    device = torch.device(config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _amp_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _make_scaler(device):
    if device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _state_dict_cpu(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in unwrap_model(model).state_dict().items()
    }


def _save_torch_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_json_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _classification_metrics(labels, probabilities, beta):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    metastasis_index = CLASS_NAMES.index("metastasis")
    metastasis = single_train.compute_class_metrics(
        labels,
        predictions,
        metastasis_index,
        beta=beta,
    )
    true_probability = probabilities[np.arange(len(labels)), labels]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "nll": float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean()),
        "metastasis_precision": float(metastasis["precision"]),
        "metastasis_recall": float(metastasis["recall"]),
        "metastasis_fbeta": float(metastasis["fbeta"]),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=range(NUM_CLASSES),
        ).tolist(),
    }


def _corrected_probabilities(base_probabilities, expert_probabilities, alpha):
    base = np.asarray(base_probabilities, dtype=np.float64)
    expert = np.asarray(expert_probabilities, dtype=np.float64)
    abnormal = base[:, 1:].sum(axis=1)
    base_subtype = base[:, 2] / np.clip(abnormal, 1e-12, None)
    subtype = (1.0 - float(alpha)) * base_subtype + float(alpha) * expert
    corrected = np.stack(
        (
            base[:, 0],
            abnormal * (1.0 - subtype),
            abnormal * subtype,
        ),
        axis=1,
    )
    return corrected / corrected.sum(axis=1, keepdims=True)


def _alpha_values(config):
    minimum = float(config.MIX_ALPHA_MIN)
    maximum = float(config.MIX_ALPHA_MAX)
    steps = int(config.MIX_ALPHA_STEPS)
    if not 0.0 <= minimum <= maximum <= 1.0 or steps < 2:
        raise ValueError("Invalid MIX_ALPHA range")
    return np.linspace(minimum, maximum, steps)


def _select_local_guidance_workpoint(
    labels,
    base_probabilities,
    expert_probabilities,
    config,
    beta,
):
    baseline = _classification_metrics(labels, base_probabilities, beta)
    minimum_accuracy = max(
        0.0,
        baseline["accuracy"] - float(config.LOCAL_GUIDANCE_ACCURACY_TOLERANCE),
    )
    minimum_precision = max(
        float(config.LOCAL_MIN_METASTASIS_PRECISION),
        baseline["metastasis_precision"]
        - float(config.LOCAL_GUIDANCE_PRECISION_TOLERANCE),
    )
    workpoints = []
    for alpha in _alpha_values(config):
        probabilities = _corrected_probabilities(
            base_probabilities,
            expert_probabilities,
            alpha,
        )
        metrics = _classification_metrics(labels, probabilities, beta)
        metrics["alpha"] = float(alpha)
        metrics["constraints_met"] = bool(
            metrics["accuracy"] >= minimum_accuracy
            and metrics["metastasis_precision"] >= minimum_precision
        )
        workpoints.append(metrics)
    eligible = [item for item in workpoints if item["constraints_met"]]
    pool = eligible if eligible else workpoints
    selected = max(
        pool,
        key=lambda item: (
            item["metastasis_fbeta"],
            item["metastasis_recall"],
            item["accuracy"],
            item["macro_f1"],
            item["metastasis_precision"],
            -item["nll"],
            -item["alpha"],
        ),
    )
    return selected, baseline


def _ranking_key(candidate, ranking):
    metrics = candidate["metrics"]
    common = (
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["metastasis_fbeta"],
        metrics["metastasis_recall"],
        metrics["metastasis_precision"],
        -metrics["nll"],
        -int(candidate["epoch"]),
    )
    if ranking == "accuracy":
        return (metrics["accuracy"], *common[1:])
    if ranking == "macro_f1":
        return (metrics["macro_f1"], metrics["accuracy"], *common[2:])
    if ranking == "metastasis_fbeta":
        return (
            bool(metrics.get("constraints_met", True)),
            metrics["metastasis_fbeta"],
            metrics["metastasis_recall"],
            metrics["accuracy"],
            metrics["metastasis_precision"],
            -metrics["nll"],
            -int(candidate["epoch"]),
        )
    if ranking == "metastasis_recall":
        return (
            metrics["metastasis_recall"],
            metrics["metastasis_precision"],
            metrics["accuracy"],
            metrics["macro_f1"],
            -metrics["nll"],
            -int(candidate["epoch"]),
        )
    if ranking == "nll":
        return (-metrics["nll"], *common)
    raise ValueError(f"Unknown archive ranking: {ranking}")


def _archive_epochs(history, candidates_per_ranking):
    rankings = (
        "accuracy",
        "macro_f1",
        "metastasis_fbeta",
        "metastasis_recall",
        "nll",
    )
    selected = set()
    for ranking in rankings:
        ordered = sorted(
            history,
            key=lambda item: _ranking_key(item, ranking),
            reverse=True,
        )
        selected.update(
            int(item["epoch"])
            for item in ordered[: int(candidates_per_ranking)]
        )
    return selected


def _update_snapshots(model, history, snapshots, candidates_per_ranking):
    selected_epochs = _archive_epochs(history, candidates_per_ranking)
    current_epoch = int(history[-1]["epoch"])
    if current_epoch in selected_epochs and current_epoch not in snapshots:
        snapshots[current_epoch] = _state_dict_cpu(model)
    for epoch in tuple(snapshots):
        if epoch not in selected_epochs:
            del snapshots[epoch]
    missing = selected_epochs - set(snapshots)
    if missing:
        raise RuntimeError(
            "A previously evicted checkpoint unexpectedly re-entered the archive: "
            f"{sorted(missing)}"
        )


def _scheduler(optimizer, config):
    name = str(config.LR_SCHEDULER).lower()
    if name == "none":
        return None
    if name != "cosine_warmup":
        raise ValueError("LR_SCHEDULER must be none or cosine_warmup")
    warmup = int(config.LR_WARMUP_EPOCHS)
    decay_epochs = int(config.LR_DECAY_EPOCHS)
    start_factor = float(config.LR_WARMUP_START_FACTOR)
    minimum_factor = float(config.MIN_LEARNING_RATE) / float(
        config.LEARNING_RATE
    )
    if not 0.0 < start_factor <= 1.0:
        raise ValueError("LR_WARMUP_START_FACTOR must lie in (0, 1]")
    if decay_epochs <= warmup:
        raise ValueError("LR_DECAY_EPOCHS must be greater than LR_WARMUP_EPOCHS")

    def factor(epoch_index):
        if warmup > 0 and epoch_index < warmup:
            progress = epoch_index / max(warmup - 1, 1)
            return start_factor + progress * (1.0 - start_factor)
        denominator = max(decay_epochs - warmup - 1, 1)
        progress = min(
            max((epoch_index - warmup) / denominator, 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _standard_datasets(data_root, fold, sequence_id):
    datasets_root = resolve_input_artifact_dir(data_root, "datasets")
    sequence_name = ALL_SEQUENCES[sequence_id - 1]
    fold_dir = (
        datasets_root
        / f"seq{sequence_id}_{sequence_name}"
        / f"fold{int(fold)}"
    )
    data_dir = infer_data_dir(data_root)
    return (
        load_pt_dataset(fold_dir / "train.pt", data_root=data_dir),
        load_pt_dataset(fold_dir / "val.pt", data_root=data_dir),
    )


def _loaders(train_dataset, val_dataset, config):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    num_workers = int(config.NUM_WORKERS)
    common = {
        "batch_size": int(config.BATCH_SIZE),
        "num_workers": num_workers,
        "pin_memory": str(config.DEVICE).startswith("cuda"),
    }
    if num_workers > 0:
        # 训练模型已经位于 CUDA 后，fork worker 会继承并长期持有 CUDA
        # 上下文。spawn 隔离数据进程，persistent 避免每个 epoch 重建。
        common.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
        )
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **common,
        ),
        DataLoader(val_dataset, shuffle=False, **common),
    )


def _optimizer_step(loss, optimizer, scaler, model, clip_norm):
    if scaler is None:
        loss.backward()
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        return
    scaler.scale(loss).backward()
    if clip_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    scaler.step(optimizer)
    scaler.update()


def _disable_batch_progress():
    return os.environ.get("MASK_GUIDED_DISABLE_BATCH_PROGRESS", "0") == "1"


def _train_standard_epoch(model, loader, criterion, optimizer, scaler, device, desc, clip):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for inputs, labels, _, _, _ in tqdm(
        loader,
        desc=desc,
        leave=False,
        disable=_disable_batch_progress(),
    ):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device):
            logits = forward_model(model, inputs)["classification"]
            loss = criterion(logits, labels)
        _optimizer_step(loss, optimizer, scaler, model, clip)
        loss_sum += float(loss.item())
        correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        count += int(labels.numel())
    return {
        "loss": loss_sum / max(len(loader), 1),
        "accuracy": correct / max(count, 1),
    }


def _validate_standard(model, loader, criterion, device, beta):
    model.eval()
    labels_all = []
    probabilities_all = []
    case_ids = []
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, labels, _, _, batch_case_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = forward_model(model, inputs)["classification"].float()
            loss_sum += float(criterion(logits, labels_device).item())
            labels_all.append(labels.numpy())
            probabilities_all.append(F.softmax(logits, dim=1).cpu().numpy())
            case_ids.extend(normalize_case_id(value) for value in batch_case_ids)
    labels = np.concatenate(labels_all).astype(np.int64, copy=False)
    probabilities = np.concatenate(probabilities_all).astype(np.float64, copy=False)
    metrics = _classification_metrics(labels, probabilities, beta)
    metrics["loss"] = loss_sum / max(len(loader), 1)
    return {
        "case_ids": tuple(case_ids),
        "labels": labels,
        "probabilities": probabilities,
        "metrics": metrics,
    }


def _subtype_criterion(class_counts, config, device):
    inflammation = float(class_counts[1])
    metastasis = float(class_counts[2])
    if inflammation <= 0 or metastasis <= 0:
        raise ValueError("Both abnormal classes are required for subtype training")
    positive_weight = (inflammation / metastasis) ** float(
        config.SUBTYPE_POS_WEIGHT_POWER
    )
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )


def _mask_guided_loss(outputs, labels, classification_criterion, subtype_criterion, weight):
    classification_loss = classification_criterion(
        outputs["classification"],
        labels,
    )
    abnormal = labels > 0
    if abnormal.any():
        subtype_targets = (labels[abnormal] == 2).float()
        subtype_loss = subtype_criterion(
            outputs["subtype_logit"][abnormal],
            subtype_targets,
        )
    else:
        subtype_loss = outputs["subtype_logit"].sum() * 0.0
    return classification_loss + float(weight) * subtype_loss


def _train_mask_guided_epoch(
    model,
    loader,
    classification_criterion,
    subtype_criterion,
    optimizer,
    scaler,
    device,
    config,
    desc,
):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for inputs, labels, masks, statistics, _ in tqdm(
        loader,
        desc=desc,
        leave=False,
        disable=_disable_batch_progress(),
    ):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        statistics = statistics.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device):
            outputs = model(inputs, masks, statistics)
            loss = _mask_guided_loss(
                outputs,
                labels,
                classification_criterion,
                subtype_criterion,
                config.SUBTYPE_LOSS_WEIGHT,
            )
        _optimizer_step(
            loss,
            optimizer,
            scaler,
            model,
            float(config.GRADIENT_CLIP_NORM),
        )
        loss_sum += float(loss.item())
        correct += int(
            (outputs["classification"].detach().argmax(dim=1) == labels)
            .sum()
            .item()
        )
        count += int(labels.numel())
    return {
        "loss": loss_sum / max(len(loader), 1),
        "accuracy": correct / max(count, 1),
    }


def _validate_mask_guided(
    model,
    loader,
    classification_criterion,
    subtype_criterion,
    device,
    config,
    beta,
):
    model.eval()
    labels_all = []
    base_probabilities_all = []
    expert_probabilities_all = []
    case_ids = []
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, labels, masks, statistics, batch_case_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            statistics = statistics.to(device, non_blocking=True)
            outputs = model(inputs, masks, statistics)
            loss = _mask_guided_loss(
                outputs,
                labels_device,
                classification_criterion,
                subtype_criterion,
                config.SUBTYPE_LOSS_WEIGHT,
            )
            loss_sum += float(loss.item())
            labels_all.append(labels.numpy())
            base_probabilities_all.append(
                outputs["base_probabilities"].float().cpu().numpy()
            )
            expert_probabilities_all.append(
                outputs["expert_probability"].float().cpu().numpy()
            )
            case_ids.extend(normalize_case_id(value) for value in batch_case_ids)
    labels = np.concatenate(labels_all).astype(np.int64, copy=False)
    base_probabilities = np.concatenate(base_probabilities_all).astype(
        np.float64,
        copy=False,
    )
    expert_probabilities = np.concatenate(expert_probabilities_all).astype(
        np.float64,
        copy=False,
    )
    selected, baseline = _select_local_guidance_workpoint(
        labels,
        base_probabilities,
        expert_probabilities,
        config,
        beta,
    )
    selected = dict(selected)
    selected["loss"] = loss_sum / max(len(loader), 1)
    return {
        "case_ids": tuple(case_ids),
        "labels": labels,
        "base_probabilities": base_probabilities,
        "expert_probabilities": expert_probabilities,
        "metrics": selected,
        "base_metrics": baseline,
    }


def _local_rank(candidate, mask_guided):
    metrics = candidate["metrics"]
    if mask_guided:
        return (
            bool(metrics.get("constraints_met", False)),
            metrics["metastasis_fbeta"],
            metrics["metastasis_recall"],
            metrics["accuracy"],
            metrics["macro_f1"],
            -metrics["nll"],
        )
    return (
        metrics["macro_f1"],
        metrics["accuracy"],
        metrics["metastasis_fbeta"],
        -metrics["nll"],
    )


def _history_row(candidate, train_metrics, learning_rate):
    metrics = candidate["metrics"]
    return {
        "epoch": candidate["epoch"],
        "learning_rate": learning_rate,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": metrics["loss"],
        "val_nll": metrics["nll"],
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_metastasis_precision": metrics["metastasis_precision"],
        "val_metastasis_recall": metrics["metastasis_recall"],
        "val_metastasis_fbeta": metrics["metastasis_fbeta"],
        "guidance_alpha": metrics.get("alpha"),
        "guidance_constraints_met": metrics.get("constraints_met"),
    }


def _train_sequence(
    *,
    sequence_id,
    train_dataset,
    val_dataset,
    config,
    device,
    beta,
):
    train_loader, val_loader = _loaders(train_dataset, val_dataset, config)
    set_seed(SEED)
    mask_guided = sequence_id == 3
    if mask_guided:
        model = NNUNetMaskGuidedClassifier(
            num_classes=NUM_CLASSES,
            in_channels=1,
            stage_projection_dim=config.GUIDANCE_STAGE_PROJECTION_DIM,
            stats_projection_dim=config.GUIDANCE_STATS_PROJECTION_DIM,
            subtype_hidden_dim=config.GUIDANCE_SUBTYPE_HIDDEN_DIM,
            subtype_dropout=config.GUIDANCE_SUBTYPE_DROPOUT,
        )
    else:
        model = create_model(
            MODEL_NAMES[sequence_id - 1],
            num_classes=NUM_CLASSES,
            in_channels=1,
            sequence_id=sequence_id,
        )
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    class_counts = torch.bincount(
        torch.tensor(train_dataset.labels, dtype=torch.long),
        minlength=NUM_CLASSES,
    )
    classification_criterion = single_train.build_classification_criterion(
        class_counts,
        len(train_dataset),
    )
    subtype_criterion = (
        _subtype_criterion(class_counts, config, device)
        if mask_guided
        else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.LEARNING_RATE),
        weight_decay=float(config.WEIGHT_DECAY),
    )
    scheduler = _scheduler(optimizer, config)
    scaler = _make_scaler(device)
    history = []
    history_rows = []
    snapshots = {}
    best_rank = None
    stale_epochs = 0

    for epoch in range(1, int(config.NUM_EPOCHS) + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if mask_guided:
            train_metrics = _train_mask_guided_epoch(
                model,
                train_loader,
                classification_criterion,
                subtype_criterion,
                optimizer,
                scaler,
                device,
                config,
                f"Fold Seq3 epoch {epoch}",
            )
            validation = _validate_mask_guided(
                model,
                val_loader,
                classification_criterion,
                subtype_criterion,
                device,
                config,
                beta,
            )
        else:
            train_metrics = _train_standard_epoch(
                model,
                train_loader,
                classification_criterion,
                optimizer,
                scaler,
                device,
                f"Fold Seq{sequence_id} epoch {epoch}",
                float(config.GRADIENT_CLIP_NORM),
            )
            validation = _validate_standard(
                model,
                val_loader,
                classification_criterion,
                device,
                beta,
            )
        candidate = {
            "epoch": epoch,
            "case_ids": validation["case_ids"],
            "labels": validation["labels"],
            "metrics": validation["metrics"],
            "train_metrics": dict(train_metrics),
        }
        if mask_guided:
            candidate["base_probabilities"] = validation["base_probabilities"]
            candidate["expert_probabilities"] = validation[
                "expert_probabilities"
            ]
            candidate["base_metrics"] = validation["base_metrics"]
        else:
            candidate["probabilities"] = validation["probabilities"]
        history.append(candidate)
        _update_snapshots(
            model,
            history,
            snapshots,
            config.CANDIDATES_PER_RANKING,
        )
        history_rows.append(_history_row(candidate, train_metrics, learning_rate))
        rank = _local_rank(candidate, mask_guided)
        improved = best_rank is None or rank > best_rank
        if improved:
            best_rank = rank
            stale_epochs = 0
        elif epoch >= int(config.MIN_EPOCHS):
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step()

        metrics = candidate["metrics"]
        suffix = (
            f" | alpha={metrics['alpha']:.2f}"
            f" | guide_constraints={'yes' if metrics['constraints_met'] else 'no'}"
            if mask_guided
            else ""
        )
        print(
            f"[Seq{sequence_id}] epoch [{epoch}/{config.NUM_EPOCHS}] "
            f"lr={learning_rate:.2e} | train_loss={train_metrics['loss']:.4f} "
            f"| val_acc={metrics['accuracy']:.4f} "
            f"| val_f1={metrics['macro_f1']:.4f} "
            f"| meta_P/R/F{beta:g}="
            f"{metrics['metastasis_precision']:.4f}/"
            f"{metrics['metastasis_recall']:.4f}/"
            f"{metrics['metastasis_fbeta']:.4f}{suffix}",
            flush=True,
        )
        if (
            epoch >= int(config.MIN_EPOCHS)
            and stale_epochs >= int(config.PATIENCE)
        ):
            print(
                f"[Seq{sequence_id}] early stopping at epoch {epoch}",
                flush=True,
            )
            break

    selected_epochs = _archive_epochs(history, config.CANDIDATES_PER_RANKING)
    archived = [
        candidate
        for candidate in history
        if int(candidate["epoch"]) in selected_epochs
    ]
    if set(selected_epochs) != set(snapshots):
        raise RuntimeError("Final candidate archive and model snapshots differ")
    return {
        "model_name": MODEL_NAMES[sequence_id - 1],
        "class_counts": class_counts,
        "criterion": classification_criterion,
        "candidates": archived,
        "snapshots": snapshots,
        "history_rows": history_rows,
    }


def _save_candidate_archive(result, output_root, fold, sequence_id, config):
    sequence_name = ALL_SEQUENCES[sequence_id - 1]
    root = (
        Path(output_root).expanduser().resolve()
        / "checkpoints"
        / "candidates"
        / f"seq{sequence_id}_{sequence_name}"
        / result["model_name"]
        / f"fold{int(fold)}"
    )
    config_snapshot = {name: getattr(config, name) for name in CONFIG_FIELDS}
    descriptors = []
    loss_metadata = single_train.get_classification_loss_metadata(
        result["criterion"],
        result["class_counts"],
    )
    for candidate in result["candidates"]:
        epoch = int(candidate["epoch"])
        path = root / f"epoch{epoch:03d}.pth"
        payload = {
            "model_state": result["snapshots"][epoch],
            "model_name": result["model_name"],
            "model_capabilities": {
                "classification": True,
                "subtype": bool(sequence_id == 3),
                "segmentation": False,
            },
            "fold": int(fold),
            "sequence_id": int(sequence_id),
            "sequence_name": sequence_name,
            "epoch": epoch,
            "val_loss": float(candidate["metrics"]["loss"]),
            "val_acc": float(candidate["metrics"]["accuracy"]),
            "val_f1": float(candidate["metrics"]["macro_f1"]),
            "val_metastasis_precision": float(
                candidate["metrics"]["metastasis_precision"]
            ),
            "val_metastasis_recall": float(
                candidate["metrics"]["metastasis_recall"]
            ),
            "val_metastasis_fbeta": float(
                candidate["metrics"]["metastasis_fbeta"]
            ),
            "val_dice": None,
            "train_loss": float(candidate["train_metrics"]["loss"]),
            "train_acc": float(candidate["train_metrics"]["accuracy"]),
            "validation_metrics": candidate["metrics"],
            "train_config_path": str(config.__config_path__),
            "train_config": config_snapshot,
            "candidate_archive": True,
            **loss_metadata,
        }
        _save_torch_atomic(payload, path)
        descriptor = dict(candidate)
        descriptor["checkpoint_path"] = path
        descriptors.append(descriptor)
    return descriptors


def _worker_result_path(output_root, fold, run_id, sequence_id):
    return (
        Path(output_root).expanduser().resolve()
        / "reports"
        / "parallel_sequence_workers"
        / f"fold{int(fold)}"
        / str(run_id)
        / f"seq{int(sequence_id)}_candidates.pt"
    )


def _train_and_archive_sequence(
    *,
    args,
    config,
    device,
    beta,
    sequence_id,
):
    output_root = Path(args.output_root).expanduser().resolve()
    if sequence_id in (1, 2):
        train_dataset, val_dataset = _standard_datasets(
            args.data_root,
            args.fold,
            sequence_id,
        )
    elif sequence_id == 3:
        flair_train = load_split_dataset(args.data_root, args.fold, "train")
        flair_val = load_split_dataset(args.data_root, args.fold, "val")
        mask_cache_dir = output_root / "mask_cache"
        train_dataset = load_mask_guided_dataset(
            flair_train,
            mask_cache_dir / f"fold{int(args.fold)}_train.pt",
        )
        val_dataset = load_mask_guided_dataset(
            flair_val,
            mask_cache_dir / f"fold{int(args.fold)}_val.pt",
        )
    else:
        raise ValueError(f"Unsupported sequence worker: {sequence_id}")

    result = _train_sequence(
        sequence_id=sequence_id,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        beta=beta,
    )
    candidates = _save_candidate_archive(
        result,
        output_root,
        args.fold,
        sequence_id,
        config,
    )
    history_path = (
        output_root
        / "reports"
        / "sequence_validation_history"
        / f"fold{int(args.fold)}_seq{sequence_id}.csv"
    )
    _write_history(history_path, result["history_rows"])
    del result
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return candidates


def _run_sequence_worker(args, config, device, beta):
    sequence_id = int(args.sequence_worker)
    if not args.worker_run_id:
        raise ValueError("--worker-run-id is required with --sequence-worker")
    set_seed(SEED)
    print(
        f"[并行 Seq{sequence_id}] worker PID={os.getpid()} | "
        f"可见 GPU={torch.cuda.device_count() if device.type == 'cuda' else 0} | "
        f"全局 batch={config.BATCH_SIZE}",
        flush=True,
    )
    candidates = _train_and_archive_sequence(
        args=args,
        config=config,
        device=device,
        beta=beta,
        sequence_id=sequence_id,
    )
    result_path = _worker_result_path(
        args.output_root,
        args.fold,
        args.worker_run_id,
        sequence_id,
    )
    _save_torch_atomic(
        {
            "schema_version": 1,
            "run_id": str(args.worker_run_id),
            "fold": int(args.fold),
            "sequence_id": sequence_id,
            "candidates": candidates,
        },
        result_path,
    )
    print(
        f"[并行 Seq{sequence_id}] 训练完成，候选数={len(candidates)} | "
        f"结果={result_path}",
        flush=True,
    )


def _parallel_worker_command(args, sequence_id, run_id):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--data-root",
        str(Path(args.data_root).expanduser().resolve()),
        "--output-root",
        str(Path(args.output_root).expanduser().resolve()),
        "--reference-checkpoint-root",
        str(Path(args.reference_checkpoint_root).expanduser().resolve()),
        "--nnunet-results-root",
        str(Path(args.nnunet_results_root).expanduser().resolve()),
        "--fold",
        str(int(args.fold)),
        "--sequence-worker",
        str(int(sequence_id)),
        "--worker-run-id",
        str(run_id),
    ]
    return command


def _load_parallel_worker_result(args, sequence_id, run_id):
    path = _worker_result_path(
        args.output_root,
        args.fold,
        run_id,
        sequence_id,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Parallel worker result not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != 1
        or payload.get("run_id") != str(run_id)
        or int(payload.get("fold", -1)) != int(args.fold)
        or int(payload.get("sequence_id", -1)) != int(sequence_id)
    ):
        raise RuntimeError(f"Invalid parallel worker result: {path}")
    candidates = payload.get("candidates")
    if not candidates:
        raise RuntimeError(f"Parallel worker produced no candidates: {path}")
    return candidates


def _parallel_worker_log_path(output_root, fold, run_id, sequence_id):
    return (
        Path(output_root).expanduser().resolve()
        / "logs"
        / "parallel_workers"
        / f"fold{int(fold)}"
        / str(run_id)
        / f"seq{int(sequence_id)}.log"
    )


def _relay_worker_output(sequence_id, process, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if process.stdout is None:
        raise RuntimeError("Parallel worker stdout pipe was not created")
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        for line in process.stdout:
            stream.write(line)
            already_prefixed = line.startswith(
                (f"[Seq{sequence_id}]", f"[并行 Seq{sequence_id}]")
            )
            terminal_line = (
                line if already_prefixed else f"[Seq{sequence_id}] {line}"
            )
            print(terminal_line, end="", flush=True)
    process.stdout.close()


def _terminate_worker_group(process, force=False):
    group_signal = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(process.pid, group_signal)
    except ProcessLookupError:
        return
    except PermissionError:
        if process.poll() is None:
            if force:
                process.kill()
            else:
                process.terminate()


def _wait_after_termination(processes, timeout_seconds=15.0):
    for process in processes.values():
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_worker_group(process, force=True)
            process.wait()


def _validate_parallel_gpu_memory(config):
    if not torch.cuda.is_available():
        return
    minimum_gb = float(config.PARALLEL_MIN_FREE_GPU_MEMORY_GB)
    if minimum_gb <= 0:
        return
    insufficient = []
    for device_index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        if free_gb < minimum_gb:
            insufficient.append(
                f"GPU {device_index}: free={free_gb:.2f} GiB/"
                f"total={total_gb:.2f} GiB"
            )
    if insufficient:
        details = "; ".join(insufficient)
        raise RuntimeError(
            "并行训练启动前显存检查失败。三个序列都会使用全部可见 GPU，"
            f"每张卡至少需要 {minimum_gb:.2f} GiB 空闲显存；{details}。"
            "请先结束遗留/其他 GPU 进程，或设置 PARALLEL_SEQUENCES=0。"
        )


def _train_sequences_in_parallel(args, config):
    _validate_parallel_gpu_memory(config)
    run_id = uuid4().hex
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MASK_GUIDED_DISABLE_BATCH_PROGRESS"] = "1"
    processes = {}
    relay_threads = {}
    log_root = (
        Path(args.output_root).expanduser().resolve()
        / "logs"
        / "parallel_workers"
        / f"fold{int(args.fold)}"
        / run_id
    )
    print(
        "启动三序列并行训练：三个 worker 均使用全部可见 GPU，"
        "每路保持独立模型、优化器、学习率和早停状态。\n"
        f"三路独立日志目录：{log_root}",
        flush=True,
    )
    for sequence_id in (1, 2, 3):
        command = _parallel_worker_command(args, sequence_id, run_id)
        processes[sequence_id] = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        log_path = _parallel_worker_log_path(
            args.output_root,
            args.fold,
            run_id,
            sequence_id,
        )
        relay_threads[sequence_id] = threading.Thread(
            target=_relay_worker_output,
            args=(sequence_id, processes[sequence_id], log_path),
            name=f"seq{sequence_id}-log-relay",
            daemon=True,
        )
        relay_threads[sequence_id].start()
        print(
            f"已启动 Seq{sequence_id} worker，PID={processes[sequence_id].pid} "
            f"| 日志={log_path}",
            flush=True,
        )

    remaining = dict(processes)
    failure = None
    try:
        while remaining and failure is None:
            for sequence_id, process in tuple(remaining.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del remaining[sequence_id]
                if return_code != 0:
                    failure = (sequence_id, return_code)
                    break
            if remaining:
                time.sleep(1.0)
        if failure is not None:
            for process in processes.values():
                _terminate_worker_group(process)
            _wait_after_termination(processes)
        else:
            for process in processes.values():
                if process.poll() is None:
                    process.wait()
        # 主 worker 正常退出后也清理同进程组中可能尚未退出的数据进程。
        for process in processes.values():
            _terminate_worker_group(process)
        for thread in relay_threads.values():
            thread.join()
    except BaseException:
        for process in processes.values():
            _terminate_worker_group(process)
        _wait_after_termination(processes)
        for thread in relay_threads.values():
            thread.join(timeout=5.0)
        raise
    if failure is not None:
        sequence_id, return_code = failure
        raise RuntimeError(
            f"Seq{sequence_id} parallel worker failed with "
            f"exit code {return_code}; inspect {log_root / f'seq{sequence_id}.log'}"
        )

    return [
        _load_parallel_worker_result(args, sequence_id, run_id)
        for sequence_id in (1, 2, 3)
    ]


def _validate_alignment(candidates):
    reference = candidates[0]
    for candidate in candidates[1:]:
        if candidate["case_ids"] != reference["case_ids"]:
            raise RuntimeError("Validation case order differs across sequences")
        if not np.array_equal(candidate["labels"], reference["labels"]):
            raise RuntimeError("Validation labels differ across sequences")


def _reference_checkpoint_path(root, sequence_id, model_name, fold):
    checkpoint_root = resolve_input_artifact_dir(root, "checkpoints")
    sequence_name = ALL_SEQUENCES[sequence_id - 1]
    path = (
        checkpoint_root
        / f"seq{sequence_id}_{sequence_name}"
        / model_name
        / f"fold{int(fold)}_model_best.pth"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Reference checkpoint not found: {path}")
    return path


def _reference_fusion(data_root, checkpoint_root, fold, config, device, beta):
    results = []
    metadata = []
    for sequence_id, model_name in enumerate(REFERENCE_MODEL_NAMES, start=1):
        _, val_dataset = _standard_datasets(data_root, fold, sequence_id)
        _, val_loader = _loaders(val_dataset, val_dataset, config)
        model = create_model(
            model_name,
            num_classes=NUM_CLASSES,
            in_channels=1,
            sequence_id=sequence_id,
        ).to(device)
        path = _reference_checkpoint_path(
            checkpoint_root,
            sequence_id,
            model_name,
            fold,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        result = _validate_standard(
            model,
            val_loader,
            nn.CrossEntropyLoss().to(device),
            device,
            beta,
        )
        results.append(result)
        metadata.append(
            {
                "sequence_id": sequence_id,
                "model_name": model_name,
                "checkpoint": str(path),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _validate_alignment(results)
    probabilities = np.mean(
        np.stack([result["probabilities"] for result in results], axis=1),
        axis=1,
    )
    return _classification_metrics(results[0]["labels"], probabilities, beta), metadata


def _fusion_rank(workpoint):
    metrics = workpoint["metrics"]
    return (
        bool(workpoint["constraints_met"]),
        metrics["metastasis_fbeta"],
        metrics["metastasis_recall"],
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["metastasis_precision"],
        -metrics["nll"],
        -sum(workpoint["epochs"]),
        -workpoint["alpha"],
    )


def _search_fusion(
    candidate_sets,
    reference_metrics,
    config,
    beta,
):
    minimum_accuracy = max(
        0.0,
        reference_metrics["accuracy"]
        - float(config.FUSION_REFERENCE_ACCURACY_TOLERANCE),
    )
    minimum_precision = max(
        float(config.FUSION_MIN_VAL_METASTASIS_PRECISION),
        reference_metrics["metastasis_precision"]
        - float(config.FUSION_REFERENCE_PRECISION_TOLERANCE),
    )
    best = None
    scanned = 0
    for seq1 in candidate_sets[0]:
        for seq2 in candidate_sets[1]:
            for seq3 in candidate_sets[2]:
                _validate_alignment((seq1, seq2, seq3))
                for alpha in _alpha_values(config):
                    flair_probabilities = _corrected_probabilities(
                        seq3["base_probabilities"],
                        seq3["expert_probabilities"],
                        alpha,
                    )
                    probabilities = (
                        seq1["probabilities"]
                        + seq2["probabilities"]
                        + flair_probabilities
                    ) / 3.0
                    metrics = _classification_metrics(
                        seq1["labels"],
                        probabilities,
                        beta,
                    )
                    workpoint = {
                        "epochs": (
                            int(seq1["epoch"]),
                            int(seq2["epoch"]),
                            int(seq3["epoch"]),
                        ),
                        "alpha": float(alpha),
                        "metrics": metrics,
                        "constraints_met": bool(
                            metrics["accuracy"] >= minimum_accuracy
                            and metrics["metastasis_precision"] >= minimum_precision
                        ),
                        "minimum_accuracy": minimum_accuracy,
                        "minimum_metastasis_precision": minimum_precision,
                        "candidates": (seq1, seq2, seq3),
                    }
                    scanned += 1
                    if best is None or _fusion_rank(workpoint) > _fusion_rank(best):
                        best = workpoint
    if best is None:
        raise RuntimeError("No fusion candidate was evaluated")
    best["scanned_workpoints"] = scanned
    return best


def _publish_selection(
    best,
    output_root,
    fold,
    config,
    reference_metrics,
    reference_metadata,
    parallel_sequences,
):
    output_root = Path(output_root).expanduser().resolve()
    checkpoint_root = resolve_output_artifact_dir(output_root, "checkpoints")
    selection_id = f"fold{int(fold)}-{uuid4().hex}"
    status = (
        "constrained_async_fusion_f2"
        if best["constraints_met"]
        else "diagnostic_unconstrained"
    )
    selected_paths = []
    config_snapshot = {name: getattr(config, name) for name in CONFIG_FIELDS}
    selection = {
        "selection_id": selection_id,
        "selection_status": status,
        "fold": int(fold),
        "model_names": list(MODEL_NAMES),
        "selected_epochs": list(best["epochs"]),
        "guidance_alpha": float(best["alpha"]),
        "equal_probability_weights": [1 / 3, 1 / 3, 1 / 3],
        "validation_metrics": best["metrics"],
        "meets_selection_constraints": bool(best["constraints_met"]),
        "minimum_val_accuracy": float(best["minimum_accuracy"]),
        "minimum_val_metastasis_precision": float(
            best["minimum_metastasis_precision"]
        ),
        "reference_validation_metrics": reference_metrics,
        "reference_checkpoints": reference_metadata,
        "scanned_workpoints": int(best["scanned_workpoints"]),
        "test_parameters_locked": True,
        "sequence_training_mode": (
            "parallel_all_visible_gpus"
            if parallel_sequences
            else "sequential"
        ),
    }
    for sequence_id, candidate in enumerate(best["candidates"], start=1):
        source = torch.load(
            candidate["checkpoint_path"],
            map_location="cpu",
            weights_only=False,
        )
        model_state = dict(source["model_state"])
        if sequence_id == 3:
            if "guidance_mix_alpha" not in model_state:
                raise RuntimeError("Seq3 candidate lacks guidance_mix_alpha")
            model_state["guidance_mix_alpha"] = torch.tensor(
                float(best["alpha"]),
                dtype=model_state["guidance_mix_alpha"].dtype,
            )
        sequence_name = ALL_SEQUENCES[sequence_id - 1]
        destination = (
            checkpoint_root
            / f"seq{sequence_id}_{sequence_name}"
            / MODEL_NAMES[sequence_id - 1]
            / f"fold{int(fold)}_model_best.pth"
        )
        payload = {
            **source,
            "model_state": model_state,
            "selection_metric": "fusion_metastasis_f2",
            "fusion_selection_id": selection_id,
            "fusion_selection": selection,
            "meets_fusion_selection_constraints": bool(
                best["constraints_met"]
            ),
            "selected_guidance_alpha": (
                float(best["alpha"]) if sequence_id == 3 else None
            ),
            "train_config": config_snapshot,
        }
        _save_torch_atomic(payload, destination)
        selected_paths.append(destination)
    manifest = {
        **selection,
        "checkpoint_paths": [str(path) for path in selected_paths],
        "candidate_checkpoint_paths": [
            str(candidate["checkpoint_path"])
            for candidate in best["candidates"]
        ],
    }
    manifest_path = (
        checkpoint_root
        / "fusion_selection"
        / f"fold{int(fold)}_selection.json"
    )
    _save_json_atomic(manifest, manifest_path)
    return manifest_path


def main(args):
    config = load_python_config(args.config, CONFIG_FIELDS)
    single_train.apply_train_config(config, TRAIN_CONFIG_FIELDS)
    device = _device(config)
    beta = float(config.METASTASIS_F_BETA)
    if args.sequence_worker is not None:
        _run_sequence_worker(args, config, device, beta)
        return

    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = (
        output_root
        / "checkpoints"
        / "fusion_selection"
        / f"fold{int(args.fold)}_selection.json"
    )
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Fusion selection already exists: {manifest_path}; "
            "use --overwrite only for an intentional fresh run"
        )
    if args.parallel_sequences:
        # 在任何 nnU-Net/参考模型 CUDA 工作开始前先快速失败；正式启动三个
        # worker 前还会再次检查，覆盖缓存阶段中新出现的显存占用。
        _validate_parallel_gpu_memory(config)

    set_seed(SEED)
    flair_train = load_split_dataset(args.data_root, args.fold, "train")
    flair_val = load_split_dataset(args.data_root, args.fold, "val")
    outer_test_folds = build_outer_test_fold_map(args.data_root)
    train_source_folds = source_folds_for_split(
        flair_train,
        "train",
        args.fold,
        outer_test_folds,
    )
    val_source_folds = source_folds_for_split(
        flair_val,
        "val",
        args.fold,
        outer_test_folds,
    )
    required_source_folds = sorted(
        set(train_source_folds) | set(val_source_folds)
    )
    nnunet_paths = {
        fold: native_nnunet_checkpoint_path(
            args.nnunet_results_root,
            config,
            fold,
        )
        for fold in required_source_folds
    }
    nnunet_sha256 = {
        fold: sha256_file(path) for fold, path in nnunet_paths.items()
    }
    mask_cache_dir = output_root / "mask_cache"
    print(
        "训练集 OOF nnU-Net 来源："
        + ", ".join(
            f"fold{fold}={train_source_folds.count(fold)}"
            for fold in sorted(set(train_source_folds))
        )
    )
    print(f"验证集 nnU-Net 来源：fold{args.fold}")
    train_mask_cache = extract_mask_cache(
        split="train",
        dataset=flair_train,
        source_folds=train_source_folds,
        target_fold=args.fold,
        nnunet_paths=nnunet_paths,
        nnunet_sha256=nnunet_sha256,
        cache_dir=mask_cache_dir,
        config=config,
        device=device,
        rebuild=args.rebuild_mask_cache,
    )
    val_mask_cache = extract_mask_cache(
        split="val",
        dataset=flair_val,
        source_folds=val_source_folds,
        target_fold=args.fold,
        nnunet_paths=nnunet_paths,
        nnunet_sha256=nnunet_sha256,
        cache_dir=mask_cache_dir,
        config=config,
        device=device,
        rebuild=args.rebuild_mask_cache,
    )
    if args.cache_only:
        print("mask 缓存已完成；根据 --cache-only 停止训练")
        return

    reference_metrics, reference_metadata = _reference_fusion(
        args.data_root,
        args.reference_checkpoint_root,
        args.fold,
        config,
        device,
        beta,
    )
    print(
        "参考融合 validation："
        f"acc={reference_metrics['accuracy']:.4f} | "
        f"macro_f1={reference_metrics['macro_f1']:.4f} | "
        f"meta_P/R/F{beta:g}="
        f"{reference_metrics['metastasis_precision']:.4f}/"
        f"{reference_metrics['metastasis_recall']:.4f}/"
        f"{reference_metrics['metastasis_fbeta']:.4f}"
    )

    if args.parallel_sequences:
        candidate_sets = _train_sequences_in_parallel(args, config)
    else:
        candidate_sets = [
            _train_and_archive_sequence(
                args=args,
                config=config,
                device=device,
                beta=beta,
                sequence_id=sequence_id,
            )
            for sequence_id in (1, 2, 3)
        ]

    best = _search_fusion(
        candidate_sets,
        reference_metrics,
        config,
        beta,
    )
    manifest_path = _publish_selection(
        best,
        output_root,
        args.fold,
        config,
        reference_metrics,
        reference_metadata,
        args.parallel_sequences,
    )
    metrics = best["metrics"]
    print(
        f"Fold {args.fold} 异步融合选择完成 | "
        f"epochs={best['epochs']} | alpha={best['alpha']:.2f} | "
        f"acc={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f} | "
        f"meta_P/R/F{beta:g}="
        f"{metrics['metastasis_precision']:.4f}/"
        f"{metrics['metastasis_recall']:.4f}/"
        f"{metrics['metastasis_fbeta']:.4f} | "
        f"constraints={'yes' if best['constraints_met'] else 'no'}"
    )
    print(f"融合选择清单：{manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference-checkpoint-root", required=True)
    parser.add_argument("--nnunet-results-root", required=True)
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, K_FOLDS + 1),
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--rebuild-mask-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--parallel-sequences",
        action="store_true",
        help="同时启动三个训练进程；每个进程使用全部可见 GPU",
    )
    parser.add_argument(
        "--sequence-worker",
        type=int,
        choices=(1, 2, 3),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-run-id", help=argparse.SUPPRESS)
    main(parser.parse_args())
