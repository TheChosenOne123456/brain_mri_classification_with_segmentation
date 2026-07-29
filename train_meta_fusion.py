"""训练并评估受约束的三模态 OOF late-fusion。

为什么不用 train/val 上的自由权重矩阵
---------------------------------------
基础模型已经直接拟合过 ``train.pt``，并使用 ``val.pt`` 选择 checkpoint。
在这些预测上训练自由的 3x9 矩阵容易利用基础模型的过度自信；再用很小的
inner-val 同时选择 epoch 和阈值，也容易产生选择过拟合。

本脚本改用 cross-fitted OOF 协议。评估目标 fold N 时：

1. 使用另外四个 fold 各自 ``test.pt`` 上的基础模型 OOF 概率选择融合参数；
2. 参数只有 T1、T2、FLAIR 三个非负且和为 1 的权重，以及一个转移阈值；
3. normal/异常 gate 始终沿用三个模态等权融合，只允许改变
   inflammation/metastasis 之间的判定；
4. fold N 的标签完全不参与该 fold 的参数选择，只用于最终评估。

因此每个病例的最终 cross-fitted 预测都来自一个没有读取该病例标签的融合器。

用法
----
使用当前三个基准模型运行全部五折：

    python train_meta_fusion.py \
        --config output/runs-meta-linear/meta_fusion_config.py \
        --data-root output/data-hdbet \
        --checkpoint-roots \
            output/runs-cross-entropy \
            output/runs-cross-entropy \
            output/runs-cross-entropy \
        --output-root output/runs-meta-linear

只输出 fold 1 的 cross-fitted 结果：

    python train_meta_fusion.py \
        --config output/runs-meta-linear/meta_fusion_config.py \
        --data-root output/data-hdbet \
        --checkpoint-roots \
            output/runs-cross-entropy \
            output/runs-cross-entropy \
            output/runs-cross-entropy \
        --output-root output/runs-meta-linear \
        --fold 1

注意：即使只评估一个 fold，也必须有另外四折的 OOF 概率作为融合训练数据。
脚本会自动生成所有缺失的 fold test 概率；已有兼容缓存会直接复用。

三个模态来自不同实验时，按 T1、T2、FLAIR 顺序分别传入：

    --checkpoint-roots output/runs-t1 output/runs-t2 output/runs-flair

输出
----
``OUTPUT_ROOT/probability_cache/foldN_test.npz``
    基础模型 test OOF 概率，可在后续运行中复用。
``OUTPUT_ROOT/checkpoints/meta_fusion_constrained_oof/``
    每个目标 fold 的三个权重、转移阈值、训练来源与配置快照。
``OUTPUT_ROOT/reports/constrained_oof/``
    每折预测 CSV、每折 JSON 和 cross-fitted 汇总 JSON。

旧版自由线性矩阵产生的 ``reports/foldN_*`` 和
``checkpoints/meta_fusion/`` 不会被覆盖。
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.config_utils import (
    META_FUSION_CONFIG_FIELDS,
    infer_data_dir,
    load_python_config,
    resolve_input_artifact_dir,
)
from configs.global_config import (
    ALL_SEQUENCES,
    CLASS_NAMES,
    K_FOLDS,
    NUM_CLASSES,
    SEED,
)
from models.FoundationModel import FoundationModel
from models.FoundationModel_ori import FoundationModel as FoundationModel_ori
from utils.dataset import load_nii_as_tensor
from utils.train_and_test import load_pt_dataset, set_seed


warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
)


MODEL_REGISTRY = {
    "FoundationModel": FoundationModel,
    "FoundationModel_ori": FoundationModel_ori,
}
METHOD_NAME = "constrained_oof_fusion"


class AlignedProbabilityDataset(Dataset):
    """只加载三序列图像，并强制检查 case/label 对齐。"""

    def __init__(self, datasets):
        if len(datasets) != len(ALL_SEQUENCES):
            raise ValueError(
                f"Expected {len(ALL_SEQUENCES)} sequence datasets, "
                f"got {len(datasets)}"
            )

        self.cases_by_sequence = [dataset.cases for dataset in datasets]
        expected_length = len(self.cases_by_sequence[0])
        if any(
            len(cases) != expected_length
            for cases in self.cases_by_sequence
        ):
            raise ValueError("Sequence datasets have different sample counts")

        for idx in range(expected_length):
            ids = {
                str(cases[idx]["case_id"])
                for cases in self.cases_by_sequence
            }
            labels = {
                int(cases[idx]["label"])
                for cases in self.cases_by_sequence
            }
            if len(ids) != 1 or len(labels) != 1:
                raise ValueError(
                    f"Unaligned sequence datasets at index {idx}: "
                    f"case_ids={sorted(ids)}, labels={sorted(labels)}"
                )

    def __len__(self):
        return len(self.cases_by_sequence[0])

    def __getitem__(self, idx):
        xs = [
            load_nii_as_tensor(cases[idx]["nii_path"])
            for cases in self.cases_by_sequence
        ]
        reference = self.cases_by_sequence[0][idx]
        return (
            xs,
            torch.tensor(int(reference["label"]), dtype=torch.long),
            str(reference["case_id"]),
        )


def validate_config(config):
    if len(config.BASE_MODEL_NAMES) != len(ALL_SEQUENCES):
        raise ValueError(
            "BASE_MODEL_NAMES must contain exactly three entries in "
            "T1, T2, FLAIR order"
        )

    unknown_models = sorted(
        set(config.BASE_MODEL_NAMES) - set(MODEL_REGISTRY)
    )
    if unknown_models:
        raise ValueError(f"Unsupported base model names: {unknown_models}")

    if config.BASE_BATCH_SIZE <= 0:
        raise ValueError("BASE_BATCH_SIZE must be > 0")
    if config.NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS must be >= 0")
    if config.DEVICE not in {"cuda", "cpu"}:
        raise ValueError("DEVICE must be 'cuda' or 'cpu'")

    if not 0 < config.META_WEIGHT_STEP <= 1:
        raise ValueError("META_WEIGHT_STEP must be in (0, 1]")
    weight_units = round(1.0 / config.META_WEIGHT_STEP)
    if not np.isclose(
        weight_units * config.META_WEIGHT_STEP,
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "META_WEIGHT_STEP must divide 1 exactly, for example 0.1 or 0.05"
        )

    if config.META_SELECTION_METRIC not in {
        "macro_f1",
        "metastasis_f1",
        "metastasis_recall",
    }:
        raise ValueError(
            "META_SELECTION_METRIC must be macro_f1, metastasis_f1, "
            "or metastasis_recall"
        )
    if not 0 <= config.META_MIN_ACCURACY <= 1:
        raise ValueError("META_MIN_ACCURACY must be between 0 and 1")
    if not 0 <= config.META_ACCURACY_TOLERANCE <= 1:
        raise ValueError("META_ACCURACY_TOLERANCE must be between 0 and 1")
    if config.META_THRESHOLD_STEPS < 2:
        raise ValueError("META_THRESHOLD_STEPS must be at least 2")
    if not (
        0
        <= config.META_THRESHOLD_MIN
        < config.META_THRESHOLD_MAX
        <= 1
    ):
        raise ValueError(
            "Expected 0 <= META_THRESHOLD_MIN < "
            "META_THRESHOLD_MAX <= 1"
        )


def resolve_runtime_device(requested_device):
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested_device


def build_dataset_dirs(data_root):
    dataset_root = resolve_input_artifact_dir(data_root, "datasets")
    return dataset_root, [
        dataset_root / f"seq{seq_id}_{seq_name}"
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
    ]


def build_checkpoint_paths(checkpoint_roots, model_names, fold_idx):
    paths = []
    for seq_idx, (root, seq_name, model_name) in enumerate(
        zip(checkpoint_roots, ALL_SEQUENCES, model_names),
        start=1,
    ):
        checkpoint_dir = resolve_input_artifact_dir(root, "checkpoints")
        path = (
            checkpoint_dir
            / f"seq{seq_idx}_{seq_name}"
            / model_name
            / f"fold{fold_idx}_model_best.pth"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {seq_name} checkpoint for fold {fold_idx}: {path}"
            )
        paths.append(path)
    return paths


def load_base_models(checkpoint_paths, model_names, device):
    models = []
    for path, model_name in zip(checkpoint_paths, model_names):
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(
            num_classes=NUM_CLASSES,
            in_channels=1,
        ).to(device)
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        if device == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()
        models.append(model)
    return models


def load_aligned_test(dataset_dirs, fold_idx, processed_data_root):
    datasets = []
    split_paths = []
    for dataset_dir in dataset_dirs:
        split_path = dataset_dir / f"fold{fold_idx}" / "test.pt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing test dataset: {split_path}")
        datasets.append(
            load_pt_dataset(
                split_path,
                data_root=processed_data_root,
            )
        )
        split_paths.append(split_path)
    return AlignedProbabilityDataset(datasets), split_paths


def build_cache_signature(
    fold_idx,
    split_paths,
    checkpoint_paths,
    model_names,
):
    # 保持 schema_version=1 和旧版字段，允许复用已生成的 fold test 缓存。
    return {
        "schema_version": 1,
        "fold": int(fold_idx),
        "split": "test",
        "base_model_names": list(model_names),
        "datasets": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in split_paths
        ],
        "checkpoints": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in checkpoint_paths
        ],
    }


def load_probability_cache(cache_path, expected_signature):
    if not cache_path.is_file():
        return None

    with np.load(cache_path, allow_pickle=False) as cache:
        signature = json.loads(str(cache["signature"].item()))
        if signature != expected_signature:
            print(f"[Cache] Signature changed, recomputing: {cache_path}")
            return None
        return {
            "case_ids": cache["case_ids"].astype(str),
            "labels": cache["labels"].astype(np.int64),
            "probabilities": cache["probabilities"].astype(np.float32),
        }


def save_probability_cache(cache_path, signature, result):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        signature=json.dumps(signature, sort_keys=True),
        case_ids=np.asarray(result["case_ids"], dtype=str),
        labels=np.asarray(result["labels"], dtype=np.int64),
        probabilities=np.asarray(
            result["probabilities"],
            dtype=np.float32,
        ),
    )


def collect_base_probabilities(dataset, models, config, device, description):
    loader = DataLoader(
        dataset,
        batch_size=config.BASE_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    all_case_ids = []
    all_labels = []
    all_probabilities = []

    with torch.inference_mode():
        for xs, labels, case_ids in tqdm(loader, desc=description):
            sequence_probabilities = []
            for model, x in zip(models, xs):
                logits = model(x.to(device))
                sequence_probabilities.append(
                    F.softmax(logits, dim=1).cpu().numpy()
                )

            all_probabilities.append(
                np.stack(sequence_probabilities, axis=1)
            )
            all_labels.extend(labels.numpy().tolist())
            all_case_ids.extend(str(case_id) for case_id in case_ids)

    if not all_probabilities:
        raise ValueError(f"No probabilities collected for {description}")

    return {
        "case_ids": np.asarray(all_case_ids, dtype=str),
        "labels": np.asarray(all_labels, dtype=np.int64),
        "probabilities": np.concatenate(all_probabilities, axis=0),
    }


def prepare_fold_probabilities(
    fold_idx,
    config,
    dataset_dirs,
    processed_data_root,
    checkpoint_roots,
    output_root,
    device,
    force_recompute,
):
    dataset, split_paths = load_aligned_test(
        dataset_dirs,
        fold_idx,
        processed_data_root,
    )
    checkpoint_paths = build_checkpoint_paths(
        checkpoint_roots,
        config.BASE_MODEL_NAMES,
        fold_idx,
    )
    signature = build_cache_signature(
        fold_idx,
        split_paths,
        checkpoint_paths,
        config.BASE_MODEL_NAMES,
    )
    cache_path = (
        output_root
        / "probability_cache"
        / f"fold{fold_idx}_test.npz"
    )

    cached = None
    if not force_recompute:
        cached = load_probability_cache(cache_path, signature)
    if cached is not None:
        print(f"[Cache] Reusing {cache_path}")
        return cached

    print(f"\nLoading base models for fold {fold_idx}...")
    models = load_base_models(
        checkpoint_paths,
        config.BASE_MODEL_NAMES,
        device,
    )
    result = collect_base_probabilities(
        dataset,
        models,
        config,
        device,
        description=f"Fold {fold_idx} test OOF probabilities",
    )
    save_probability_cache(cache_path, signature, result)
    del models
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def validate_probability_array(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    expected_shape = (len(ALL_SEQUENCES), NUM_CLASSES)
    if (
        probabilities.ndim != 3
        or tuple(probabilities.shape[1:]) != expected_shape
    ):
        raise ValueError(
            f"Expected probabilities [N,{expected_shape[0]},"
            f"{expected_shape[1]}], got {probabilities.shape}"
        )
    return probabilities


def compute_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ValueError(
            f"Label/prediction shape mismatch: "
            f"{labels.shape} vs {predictions.shape}"
        )

    confusion = np.bincount(
        labels * NUM_CLASSES + predictions,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    label_count = confusion.sum(axis=1).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        label_count,
        out=np.zeros_like(true_positive),
        where=label_count > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    metastasis_idx = CLASS_NAMES.index("metastasis")

    return {
        "accuracy": float(true_positive.sum() / max(len(labels), 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "metastasis_precision": float(precision[metastasis_idx]),
        "metastasis_recall": float(recall[metastasis_idx]),
        "metastasis_f1": float(f1[metastasis_idx]),
        "confusion_matrix": confusion.tolist(),
    }


def weighted_probabilities(base_probabilities, weights):
    base_probabilities = validate_probability_array(base_probabilities)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(ALL_SEQUENCES),):
        raise ValueError(
            f"Expected {len(ALL_SEQUENCES)} weights, got {weights.shape}"
        )
    if np.any(weights < -1e-10) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(
            f"Weights must be non-negative and sum to 1, got {weights}"
        )
    return np.sum(
        base_probabilities * weights[None, :, None],
        axis=1,
    )


def predict_constrained_fusion(
    base_probabilities,
    weights,
    metastasis_threshold,
):
    """固定等权 normal gate，只调整异常病例的炎症/转移边界。"""
    base_probabilities = validate_probability_array(base_probabilities)
    equal_probabilities = base_probabilities.mean(axis=1)
    fused_probabilities = weighted_probabilities(
        base_probabilities,
        weights,
    )

    normal_idx = CLASS_NAMES.index("normal")
    inflammation_idx = CLASS_NAMES.index("inflammation")
    metastasis_idx = CLASS_NAMES.index("metastasis")

    normal_gate = equal_probabilities[:, normal_idx] >= np.maximum(
        equal_probabilities[:, inflammation_idx],
        equal_probabilities[:, metastasis_idx],
    )
    subtype_denominator = (
        fused_probabilities[:, inflammation_idx]
        + fused_probabilities[:, metastasis_idx]
    )
    metastasis_score = np.divide(
        fused_probabilities[:, metastasis_idx],
        np.clip(subtype_denominator, 1e-12, None),
    )

    predictions = np.full(
        len(base_probabilities),
        inflammation_idx,
        dtype=np.int64,
    )
    predictions[normal_gate] = normal_idx
    predictions[
        (~normal_gate)
        & (metastasis_score >= float(metastasis_threshold))
    ] = metastasis_idx
    return {
        "predictions": predictions,
        "equal_probabilities": equal_probabilities,
        "fused_probabilities": fused_probabilities,
        "metastasis_score": metastasis_score,
        "normal_gate": normal_gate,
    }


def generate_weight_grid(step):
    units = int(round(1.0 / step))
    weights = []
    for t1_units in range(units + 1):
        for t2_units in range(units - t1_units + 1):
            flair_units = units - t1_units - t2_units
            weights.append(
                np.asarray(
                    [t1_units, t2_units, flair_units],
                    dtype=np.float64,
                )
                / units
            )
    return weights


def candidate_selection_key(
    metrics,
    feasible,
    selection_metric,
    weights,
    threshold,
):
    primary = metrics[selection_metric]
    distance_from_equal = float(
        np.square(
            np.asarray(weights) - (1.0 / len(ALL_SEQUENCES))
        ).sum()
    )
    conservative_threshold_distance = abs(float(threshold) - 0.5)

    if feasible:
        return (
            1,
            primary,
            metrics["metastasis_f1"],
            metrics["macro_f1"],
            metrics["accuracy"],
            -distance_from_equal,
            -conservative_threshold_distance,
        )
    return (
        0,
        metrics["accuracy"],
        primary,
        metrics["metastasis_f1"],
        metrics["macro_f1"],
        -distance_from_equal,
        -conservative_threshold_distance,
    )


def select_constrained_parameters(
    base_probabilities,
    labels,
    config,
):
    base_probabilities = validate_probability_array(base_probabilities)
    equal_weights = np.full(
        len(ALL_SEQUENCES),
        1.0 / len(ALL_SEQUENCES),
        dtype=np.float64,
    )
    baseline_output = predict_constrained_fusion(
        base_probabilities,
        equal_weights,
        metastasis_threshold=0.5,
    )
    baseline_metrics = compute_metrics(
        labels,
        baseline_output["predictions"],
    )
    accuracy_floor = max(
        float(config.META_MIN_ACCURACY),
        baseline_metrics["accuracy"]
        - float(config.META_ACCURACY_TOLERANCE),
    )

    thresholds = np.linspace(
        config.META_THRESHOLD_MIN,
        config.META_THRESHOLD_MAX,
        config.META_THRESHOLD_STEPS,
    )
    weight_grid = generate_weight_grid(config.META_WEIGHT_STEP)
    best = None
    candidate_count = 0
    feasible_count = 0

    normal_idx = CLASS_NAMES.index("normal")
    inflammation_idx = CLASS_NAMES.index("inflammation")
    metastasis_idx = CLASS_NAMES.index("metastasis")
    equal_probabilities = base_probabilities.mean(axis=1)
    normal_gate = equal_probabilities[:, normal_idx] >= np.maximum(
        equal_probabilities[:, inflammation_idx],
        equal_probabilities[:, metastasis_idx],
    )

    for weights in weight_grid:
        fused_probabilities = weighted_probabilities(
            base_probabilities,
            weights,
        )
        subtype_denominator = (
            fused_probabilities[:, inflammation_idx]
            + fused_probabilities[:, metastasis_idx]
        )
        metastasis_score = np.divide(
            fused_probabilities[:, metastasis_idx],
            np.clip(subtype_denominator, 1e-12, None),
        )
        for threshold in thresholds:
            predictions = np.full(
                len(base_probabilities),
                inflammation_idx,
                dtype=np.int64,
            )
            predictions[normal_gate] = normal_idx
            predictions[
                (~normal_gate)
                & (metastasis_score >= float(threshold))
            ] = metastasis_idx
            metrics = compute_metrics(
                labels,
                predictions,
            )
            feasible = metrics["accuracy"] >= accuracy_floor
            if feasible:
                feasible_count += 1
            candidate_count += 1
            key = candidate_selection_key(
                metrics,
                feasible,
                config.META_SELECTION_METRIC,
                weights,
                threshold,
            )
            if best is None or key > best["key"]:
                best = {
                    "weights": [float(value) for value in weights],
                    "threshold": float(threshold),
                    "metrics": metrics,
                    "feasible": bool(feasible),
                    "key": key,
                }

    return {
        "best": best,
        "baseline_metrics": baseline_metrics,
        "accuracy_floor": float(accuracy_floor),
        "candidate_count": int(candidate_count),
        "feasible_candidate_count": int(feasible_count),
    }


def concatenate_oof_sources(fold_probabilities, source_folds):
    case_ids = np.concatenate(
        [fold_probabilities[fold]["case_ids"] for fold in source_folds]
    )
    labels = np.concatenate(
        [fold_probabilities[fold]["labels"] for fold in source_folds]
    )
    probabilities = np.concatenate(
        [
            fold_probabilities[fold]["probabilities"]
            for fold in source_folds
        ],
        axis=0,
    )
    if len(set(case_ids.tolist())) != len(case_ids):
        raise ValueError(
            "OOF source folds contain duplicate case IDs; "
            "cross-fitted training would be ambiguous"
        )
    return {
        "case_ids": case_ids,
        "labels": labels,
        "probabilities": probabilities,
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy              : {metrics['accuracy']:.4f}")
    print(f"Macro F1              : {metrics['macro_f1']:.4f}")
    print(f"Macro recall          : {metrics['macro_recall']:.4f}")
    print(f"Metastasis precision  : {metrics['metastasis_precision']:.4f}")
    print(f"Metastasis recall     : {metrics['metastasis_recall']:.4f}")
    print(f"Metastasis F1         : {metrics['metastasis_f1']:.4f}")
    print("Confusion matrix:")
    print(np.asarray(metrics["confusion_matrix"]))


def save_prediction_csv(
    output_path,
    case_ids,
    labels,
    base_probabilities,
    baseline_output,
    constrained_output,
):
    fields = ["case_id", "label", "label_name"]
    for seq_name in ALL_SEQUENCES:
        fields.extend(
            f"{seq_name}_prob_{class_name}"
            for class_name in CLASS_NAMES
        )
    fields.extend(
        f"equal_prob_{class_name}"
        for class_name in CLASS_NAMES
    )
    fields.extend(["equal_pred", "equal_pred_name"])
    fields.extend(
        f"constrained_prob_{class_name}"
        for class_name in CLASS_NAMES
    )
    fields.extend(
        [
            "normal_gate_equal",
            "metastasis_conditional_score",
            "constrained_pred",
            "constrained_pred_name",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for idx, case_id in enumerate(case_ids):
            equal_pred = int(baseline_output["predictions"][idx])
            constrained_pred = int(
                constrained_output["predictions"][idx]
            )
            row = {
                "case_id": case_id,
                "label": int(labels[idx]),
                "label_name": CLASS_NAMES[int(labels[idx])],
                "equal_pred": equal_pred,
                "equal_pred_name": CLASS_NAMES[equal_pred],
                "normal_gate_equal": int(
                    constrained_output["normal_gate"][idx]
                ),
                "metastasis_conditional_score": float(
                    constrained_output["metastasis_score"][idx]
                ),
                "constrained_pred": constrained_pred,
                "constrained_pred_name": CLASS_NAMES[
                    constrained_pred
                ],
            }
            for seq_idx, seq_name in enumerate(ALL_SEQUENCES):
                for class_idx, class_name in enumerate(CLASS_NAMES):
                    row[f"{seq_name}_prob_{class_name}"] = float(
                        base_probabilities[idx, seq_idx, class_idx]
                    )
            for class_idx, class_name in enumerate(CLASS_NAMES):
                row[f"equal_prob_{class_name}"] = float(
                    baseline_output["equal_probabilities"][
                        idx,
                        class_idx,
                    ]
                )
                row[f"constrained_prob_{class_name}"] = float(
                    constrained_output["fused_probabilities"][
                        idx,
                        class_idx,
                    ]
                )
            writer.writerow(row)


def config_snapshot(config):
    return {
        name: getattr(config, name)
        for name in META_FUSION_CONFIG_FIELDS
    }


def run_target_fold(
    target_fold,
    fold_probabilities,
    config,
    checkpoint_roots,
    processed_data_root,
    output_root,
):
    source_folds = [
        fold
        for fold in range(1, K_FOLDS + 1)
        if fold != target_fold
    ]
    training = concatenate_oof_sources(
        fold_probabilities,
        source_folds,
    )
    target = fold_probabilities[target_fold]
    overlap = set(training["case_ids"].tolist()) & set(
        target["case_ids"].tolist()
    )
    if overlap:
        raise ValueError(
            f"Target fold {target_fold} overlaps its OOF training pool: "
            f"{len(overlap)} cases"
        )

    print(
        f"\n{'=' * 18} Target Fold {target_fold} "
        f"{'=' * 18}"
    )
    print(
        f"Selecting constrained parameters on folds "
        f"{source_folds} ({len(training['labels'])} OOF cases)..."
    )
    selection = select_constrained_parameters(
        training["probabilities"],
        training["labels"],
        config,
    )
    best = selection["best"]
    weights = best["weights"]
    threshold = best["threshold"]
    constraint = "OK" if best["feasible"] else "UNMET"
    print(
        "Selected weights   : "
        + ", ".join(
            f"{seq_name}={weight:.3f}"
            for seq_name, weight in zip(ALL_SEQUENCES, weights)
        )
    )
    print(f"Selected threshold : {threshold:.3f}")
    print(
        f"OOF accuracy floor : {selection['accuracy_floor']:.4f} "
        f"({constraint})"
    )
    print_metrics(
        f"Fold {target_fold} Source OOF Equal Soft Voting",
        selection["baseline_metrics"],
    )
    print_metrics(
        f"Fold {target_fold} Source OOF Selected Fusion",
        best["metrics"],
    )

    equal_weights = np.full(
        len(ALL_SEQUENCES),
        1.0 / len(ALL_SEQUENCES),
    )
    baseline_output = predict_constrained_fusion(
        target["probabilities"],
        equal_weights,
        metastasis_threshold=0.5,
    )
    constrained_output = predict_constrained_fusion(
        target["probabilities"],
        weights,
        metastasis_threshold=threshold,
    )
    baseline_metrics = compute_metrics(
        target["labels"],
        baseline_output["predictions"],
    )
    constrained_metrics = compute_metrics(
        target["labels"],
        constrained_output["predictions"],
    )

    print_metrics(
        f"Fold {target_fold} Equal Soft Voting",
        baseline_metrics,
    )
    print_metrics(
        f"Fold {target_fold} Constrained OOF Fusion",
        constrained_metrics,
    )

    checkpoint_path = (
        output_root
        / "checkpoints"
        / "meta_fusion_constrained_oof"
        / f"fold{target_fold}_fusion.pth"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": METHOD_NAME,
            "target_fold": target_fold,
            "source_folds": source_folds,
            "weights": weights,
            "metastasis_threshold": threshold,
            "normal_gate": "equal_soft_voting_argmax",
            "selection_metric": config.META_SELECTION_METRIC,
            "accuracy_floor": selection["accuracy_floor"],
            "accuracy_constraint_met": best["feasible"],
            "source_oof_baseline_metrics": selection[
                "baseline_metrics"
            ],
            "source_oof_selected_metrics": best["metrics"],
            "meta_config_path": str(config.__config_path__),
            "meta_config": config_snapshot(config),
            "data_root": str(Path(processed_data_root).resolve()),
            "base_checkpoint_roots": [
                str(Path(root).resolve())
                for root in checkpoint_roots
            ],
        },
        checkpoint_path,
    )

    report_dir = output_root / "reports" / "constrained_oof"
    prediction_path = (
        report_dir
        / f"fold{target_fold}_test_predictions.csv"
    )
    save_prediction_csv(
        prediction_path,
        target["case_ids"],
        target["labels"],
        target["probabilities"],
        baseline_output,
        constrained_output,
    )

    fold_report = {
        "method": METHOD_NAME,
        "target_fold": target_fold,
        "source_folds": source_folds,
        "source_oof_samples": int(len(training["labels"])),
        "target_test_samples": int(len(target["labels"])),
        "selection": {
            "weights": weights,
            "metastasis_threshold": threshold,
            "normal_gate": "equal_soft_voting_argmax",
            "selection_metric": config.META_SELECTION_METRIC,
            "accuracy_floor": selection["accuracy_floor"],
            "accuracy_constraint_met": best["feasible"],
            "candidate_count": selection["candidate_count"],
            "feasible_candidate_count": selection[
                "feasible_candidate_count"
            ],
            "source_oof_baseline_metrics": selection[
                "baseline_metrics"
            ],
            "source_oof_selected_metrics": best["metrics"],
        },
        "equal_soft_voting": baseline_metrics,
        METHOD_NAME: constrained_metrics,
        "parameter_checkpoint": str(checkpoint_path.resolve()),
        "test_predictions": str(prediction_path.resolve()),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"fold{target_fold}_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(fold_report, file, indent=2, ensure_ascii=False)

    return {
        "report": fold_report,
        "labels": target["labels"],
        "equal_predictions": baseline_output["predictions"],
        "constrained_predictions": constrained_output["predictions"],
    }


def mean_metric(fold_results, method, metric):
    values = [
        result["report"][method][metric]
        for result in fold_results
    ]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def build_summary(fold_results):
    labels = np.concatenate(
        [result["labels"] for result in fold_results]
    )
    equal_predictions = np.concatenate(
        [result["equal_predictions"] for result in fold_results]
    )
    constrained_predictions = np.concatenate(
        [
            result["constrained_predictions"]
            for result in fold_results
        ]
    )
    tracked_metrics = (
        "accuracy",
        "macro_f1",
        "metastasis_precision",
        "metastasis_recall",
        "metastasis_f1",
    )
    methods = ("equal_soft_voting", METHOD_NAME)
    return {
        "protocol": (
            "For each target fold, select non-negative modality weights "
            "and a metastasis threshold on the other four folds' OOF-test "
            "probabilities; preserve the equal-voting normal gate."
        ),
        "folds": [
            result["report"]["target_fold"]
            for result in fold_results
        ],
        "mean_fold_metrics": {
            method: {
                metric: mean_metric(
                    fold_results,
                    method,
                    metric,
                )
                for metric in tracked_metrics
            }
            for method in methods
        },
        "pooled_cross_fitted_metrics": {
            "equal_soft_voting": compute_metrics(
                labels,
                equal_predictions,
            ),
            METHOD_NAME: compute_metrics(
                labels,
                constrained_predictions,
            ),
        },
        "fold_reports": [
            result["report"]
            for result in fold_results
        ],
    }


def main(args):
    config = load_python_config(
        args.config,
        META_FUSION_CONFIG_FIELDS,
    )
    validate_config(config)
    set_seed(SEED)
    np.random.seed(SEED)

    dataset_root, dataset_dirs = build_dataset_dirs(args.data_root)
    processed_data_root = infer_data_dir(args.data_root)
    if processed_data_root is None:
        raise FileNotFoundError(
            f"Could not infer preprocessed data directory "
            f"from {args.data_root}"
        )

    checkpoint_roots = [
        Path(root).expanduser().resolve()
        for root in args.checkpoint_roots
    ]
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_runtime_device(config.DEVICE)
    target_folds = (
        [args.fold]
        if args.fold is not None
        else list(range(1, K_FOLDS + 1))
    )

    print("\n>>> Constrained Cross-Fitted OOF Fusion <<<")
    print(f"Config          : {config.__config_path__}")
    print(f"Dataset root    : {dataset_root}")
    print(f"Processed data  : {processed_data_root}")
    for seq_name, model_name, root in zip(
        ALL_SEQUENCES,
        config.BASE_MODEL_NAMES,
        checkpoint_roots,
    ):
        print(f"{seq_name:<7} source : {root} ({model_name})")
    print(f"Output root     : {output_root}")
    print(f"Device          : {device}")
    print(f"Target folds    : {target_folds}")
    print(
        "Protocol        : each target fold is trained on the other "
        "four folds' OOF-test probabilities"
    )
    print(
        "Normal gate     : fixed equal soft voting; only "
        "inflammation/metastasis is adjusted"
    )

    fold_probabilities = {}
    for fold_idx in range(1, K_FOLDS + 1):
        fold_probabilities[fold_idx] = prepare_fold_probabilities(
            fold_idx=fold_idx,
            config=config,
            dataset_dirs=dataset_dirs,
            processed_data_root=processed_data_root,
            checkpoint_roots=checkpoint_roots,
            output_root=output_root,
            device=device,
            force_recompute=args.force_recompute,
        )

    fold_results = []
    for target_fold in target_folds:
        fold_results.append(
            run_target_fold(
                target_fold=target_fold,
                fold_probabilities=fold_probabilities,
                config=config,
                checkpoint_roots=checkpoint_roots,
                processed_data_root=processed_data_root,
                output_root=output_root,
            )
        )

    summary = build_summary(fold_results)
    summary_path = (
        output_root
        / "reports"
        / "constrained_oof"
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 58)
    print("Pooled Cross-Fitted Summary")
    print("=" * 58)
    print_metrics(
        "Equal Soft Voting",
        summary["pooled_cross_fitted_metrics"][
            "equal_soft_voting"
        ],
    )
    print_metrics(
        "Constrained OOF Fusion",
        summary["pooled_cross_fitted_metrics"][METHOD_NAME],
    )
    print(f"\nSummary JSON: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Select three non-negative modality weights and a metastasis "
            "threshold with cross-fitted OOF probabilities while preserving "
            "the equal-voting normal gate."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to meta_fusion_config.py.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Data experiment root containing datasets and data.",
    )
    parser.add_argument(
        "--checkpoint-roots",
        nargs=3,
        required=True,
        metavar=("T1_ROOT", "T2_ROOT", "FLAIR_ROOT"),
        help=(
            "Three training experiment/checkpoint roots in T1, T2, FLAIR "
            "order. The same root may be repeated three times."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help=(
            "Output experiment root for probability caches, constrained "
            "parameters, and reports."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, K_FOLDS + 1),
        default=None,
        help=(
            f"Report one target fold only; omit to report all {K_FOLDS}. "
            "All fold probability caches are still required for OOF training."
        ),
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Ignore compatible base probability caches and run all base "
            "model inference again."
        ),
    )
    main(parser.parse_args())
