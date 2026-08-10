"""锁定 validation 工作点后，评估 Foundation + nnU-Net 引导模型。

该入口只读取各 fold 的 test split。混合系数 alpha 和 expert epoch 均直接来自
validation 选择的 checkpoint；测试过程中不扫描阈值、不重新拟合任何参数。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from configs.config_utils import load_python_config, resolve_input_artifact_dir
from configs.global_config import CLASS_NAMES, K_FOLDS, NUM_CLASSES, SEED
from models.FoundationModelNNUNetGuided import FoundationModelNNUNetGuided
from train_foundation_nnunet_guided import (
    CONFIG_FIELDS,
    MODEL_NAME,
    SELECTION_CONSTRAINED,
    SELECTION_FALLBACK,
    GuidedFeatureDataset,
    classification_metrics,
    extract_feature_cache,
    foundation_checkpoint_path,
    load_foundation_model,
    load_split_dataset,
    native_nnunet_checkpoint_path,
    normalize_case_id,
    sha256_file,
)
from utils.train_and_test import set_seed


METHOD_NAMES = (
    "foundation_flair",
    "guided_flair",
    "baseline_fusion",
    "guided_fusion",
)


def guided_checkpoint_path(checkpoint_root, fold):
    checkpoint_dir = resolve_input_artifact_dir(checkpoint_root, "checkpoints")
    path = (
        checkpoint_dir
        / "seq3_FLAIR"
        / MODEL_NAME
        / f"fold{fold}_model_best.pth"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Guided checkpoint not found: {path}")
    return path


def validate_guided_checkpoint(checkpoint, checkpoint_path, fold, foundation_path):
    if checkpoint.get("model_name") != MODEL_NAME:
        raise RuntimeError(f"Unexpected model_name in {checkpoint_path}")
    if int(checkpoint.get("fold", -1)) != int(fold):
        raise RuntimeError(f"Fold mismatch in {checkpoint_path}")
    selection_status = checkpoint.get("selection_status")
    if selection_status is None and bool(
        checkpoint.get("selection_constraints_met", False)
    ):
        # 兼容加入显式 selection_status 之前生成的 Fold 1-4 checkpoint。
        selection_status = SELECTION_CONSTRAINED
    if selection_status not in (SELECTION_CONSTRAINED, SELECTION_FALLBACK):
        raise RuntimeError(
            f"Refusing unconstrained checkpoint for locked test evaluation: "
            f"{checkpoint_path}"
        )
    expected_sha256 = sha256_file(foundation_path)
    if checkpoint.get("foundation_sha256") != expected_sha256:
        raise RuntimeError(
            f"Foundation SHA-256 mismatch for guided Fold {fold}: "
            f"{checkpoint_path}"
        )
    validation = checkpoint.get("validation", {})
    if "mix_alpha" not in validation:
        raise RuntimeError(f"Validation-selected mix_alpha missing: {checkpoint_path}")
    if selection_status == SELECTION_CONSTRAINED:
        if not bool(checkpoint.get("selection_constraints_met", False)):
            raise RuntimeError(
                f"Constrained selection flag missing: {checkpoint_path}"
            )
    else:
        state_alpha = float(checkpoint["guidance_state"]["guidance_mix_alpha"])
        if not np.isclose(float(validation["mix_alpha"]), 0.0, atol=1e-8):
            raise RuntimeError(f"Fallback validation alpha is not zero: {checkpoint_path}")
        if not np.isclose(state_alpha, 0.0, atol=1e-8):
            raise RuntimeError(f"Fallback state alpha is not zero: {checkpoint_path}")
        if validation != checkpoint.get("baseline_validation"):
            raise RuntimeError(
                f"Fallback validation does not exactly reproduce baseline: "
                f"{checkpoint_path}"
            )
    return selection_status


def prepare_guided_test_outputs(
    *,
    fold,
    config,
    data_root,
    output_root,
    foundation_checkpoint_root,
    nnunet_results_root,
    device,
    rebuild_cache,
):
    foundation_path = foundation_checkpoint_path(foundation_checkpoint_root, fold)
    foundation_sha256 = sha256_file(foundation_path)
    model, _ = load_foundation_model(foundation_path, config, device)

    checkpoint_path = guided_checkpoint_path(output_root, fold)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selection_status = validate_guided_checkpoint(
        checkpoint,
        checkpoint_path,
        fold,
        foundation_path,
    )

    test_dataset = load_split_dataset(data_root, fold, "test")
    source_folds = (int(fold),) * len(test_dataset)
    nnunet_path = native_nnunet_checkpoint_path(
        nnunet_results_root,
        config,
        fold,
    )
    nnunet_paths = {int(fold): nnunet_path}
    nnunet_sha256 = {int(fold): sha256_file(nnunet_path)}
    cache_dir = Path(output_root).expanduser().resolve() / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = extract_feature_cache(
        split="test",
        dataset=test_dataset,
        source_folds=source_folds,
        target_fold=fold,
        model=model,
        foundation_path=foundation_path,
        foundation_sha256=foundation_sha256,
        nnunet_paths=nnunet_paths,
        nnunet_sha256=nnunet_sha256,
        cache_dir=cache_dir,
        config=config,
        device=device,
        rebuild=rebuild_cache,
    )

    model.load_guidance_state(checkpoint["guidance_state"])
    model.freeze_foundation()
    model.eval()
    model.to(device)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    feature_dataset = GuidedFeatureDataset(payload)
    loader = DataLoader(
        feature_dataset,
        batch_size=int(config.BATCH_SIZE),
        shuffle=False,
        num_workers=int(config.NUM_WORKERS),
        pin_memory=(device.type == "cuda"),
    )

    all_labels = []
    all_base_logits = []
    all_expert_logits = []
    with torch.no_grad():
        for features, base_logits, labels in loader:
            features = features.to(device, non_blocking=True)
            expert_logits = model.guidance_expert_logit(features)
            all_labels.append(labels.long())
            all_base_logits.append(base_logits.float())
            all_expert_logits.append(expert_logits.float().cpu())

    labels = torch.cat(all_labels)
    base_logits = torch.cat(all_base_logits)
    expert_logits = torch.cat(all_expert_logits)
    alpha = float(model.guidance_mix_alpha.item())
    checkpoint_alpha = float(checkpoint["validation"]["mix_alpha"])
    if not np.isclose(alpha, checkpoint_alpha, atol=1e-7):
        raise RuntimeError(
            f"Guidance alpha mismatch: state={alpha}, validation={checkpoint_alpha}"
        )
    metastasis_probability = model.mixed_subtype_probability(
        base_logits,
        expert_logits,
        alpha,
    )
    foundation_probabilities = F.softmax(base_logits, dim=1)
    guided_probabilities = model.corrected_probabilities(
        base_logits,
        metastasis_probability,
    )
    guided_predictions = model.hierarchical_predictions(
        base_logits,
        metastasis_probability,
    )
    result = {
        "case_ids": tuple(str(case_id) for case_id in payload["case_ids"]),
        "labels": labels.numpy(),
        "foundation_probabilities": foundation_probabilities.numpy(),
        "foundation_predictions": base_logits.argmax(dim=1).numpy(),
        "guided_probabilities": guided_probabilities.numpy(),
        "guided_predictions": guided_predictions.numpy(),
        "expert_probabilities": torch.sigmoid(expert_logits).numpy(),
        "alpha": alpha,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_status": selection_status,
        "validation": checkpoint["validation"],
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_locked_baseline_probabilities(
    reference_root,
    *,
    fold,
    expected_case_ids,
    expected_labels,
):
    """读取历史 FP32 基线概率，避免不一致的重新推理改变对照组。"""
    path = Path(reference_root).expanduser().resolve() / f"fold{fold}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Locked baseline probability table not found: {path}")
    rows_by_id = {}
    with open(path, "r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case_id = normalize_case_id(row["case_id"])
            if case_id in rows_by_id:
                raise RuntimeError(f"Duplicate case {case_id} in {path}")
            rows_by_id[case_id] = row
    expected_case_ids = tuple(expected_case_ids)
    if set(rows_by_id) != set(expected_case_ids):
        raise RuntimeError(f"Locked baseline cases do not match Fold {fold}: {path}")
    rows = [rows_by_id[case_id] for case_id in expected_case_ids]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    if not np.array_equal(labels, np.asarray(expected_labels)):
        raise RuntimeError(f"Locked baseline labels do not match Fold {fold}: {path}")

    expected_models = {
        "seq1_t1_model": "FoundationModel_ori",
        "seq2_t2_model": "FoundationModel_ori",
        "seq3_flair_model": "FoundationModel",
    }
    for column, expected_model in expected_models.items():
        observed = {row[column] for row in rows}
        if observed != {expected_model}:
            raise RuntimeError(
                f"Unexpected {column} in locked baseline: {observed} vs "
                f"{expected_model}"
            )

    class_suffixes = ("normal", "inflammation", "metastasis")

    def probability_columns(prefix):
        return np.asarray(
            [
                [float(row[f"{prefix}_prob_{suffix}"]) for suffix in class_suffixes]
                for row in rows
            ],
            dtype=np.float32,
        )

    probabilities = {
        "t1": probability_columns("seq1_t1"),
        "t2": probability_columns("seq2_t2"),
        "flair": probability_columns("seq3_flair"),
        "fusion": probability_columns("soft_vote"),
    }
    recomputed_fusion = (
        probabilities["t1"] + probabilities["t2"] + probabilities["flair"]
    ) / 3.0
    if not np.allclose(
        recomputed_fusion,
        probabilities["fusion"],
        rtol=0.0,
        atol=2e-7,
    ):
        raise RuntimeError(f"Locked soft vote is not an equal probability mean: {path}")
    stored_predictions = np.asarray(
        [int(row["soft_vote_pred"]) for row in rows],
        dtype=np.int64,
    )
    if not np.array_equal(stored_predictions, recomputed_fusion.argmax(axis=1)):
        raise RuntimeError(f"Locked soft-vote predictions are inconsistent: {path}")
    probabilities["fusion"] = recomputed_fusion
    probabilities["path"] = str(path)
    return probabilities


def guided_probabilities_on_locked_foundation(guided, foundation_probabilities):
    """在历史 FP32 Foundation 概率上应用 validation 锁定的 subtype expert。"""
    foundation_probabilities = np.asarray(foundation_probabilities, dtype=np.float32)
    abnormal_probability = foundation_probabilities[:, 1:].sum(axis=1)
    foundation_subtype_probability = np.divide(
        foundation_probabilities[:, 2],
        abnormal_probability,
        out=np.full_like(abnormal_probability, 0.5),
        where=abnormal_probability > 0,
    )
    alpha = float(guided["alpha"])
    metastasis_probability = (
        (1.0 - alpha) * foundation_subtype_probability
        + alpha * np.asarray(guided["expert_probabilities"], dtype=np.float32)
    )
    probabilities = np.stack(
        (
            foundation_probabilities[:, 0],
            abnormal_probability * (1.0 - metastasis_probability),
            abnormal_probability * metastasis_probability,
        ),
        axis=1,
    )
    predictions = foundation_probabilities.argmax(axis=1)
    abnormal = predictions != 0
    predictions = predictions.copy()
    predictions[abnormal] = (
        metastasis_probability[abnormal] >= 0.5
    ).astype(np.int64) + 1
    return probabilities, predictions


def metrics_for(labels, predictions, beta):
    metrics = classification_metrics(labels, predictions, beta)
    metrics["classification_report"] = classification_report(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return metrics


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(
        f"  accuracy={metrics['accuracy']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f} | "
        f"metastasis precision/recall/F2="
        f"{metrics['metastasis_precision']:.4f}/"
        f"{metrics['metastasis_recall']:.4f}/"
        f"{metrics['metastasis_fbeta']:.4f}"
    )
    print(f"  confusion={metrics['confusion_matrix']}")


def evaluate_fold(args, config, fold, device):
    print(f"\n{'=' * 24} Fold {fold} locked test {'=' * 24}")
    guided = prepare_guided_test_outputs(
        fold=fold,
        config=config,
        data_root=args.data_root,
        output_root=args.guided_checkpoint_root,
        foundation_checkpoint_root=args.foundation_checkpoint_root,
        nnunet_results_root=args.nnunet_results_root,
        device=device,
        rebuild_cache=args.rebuild_cache,
    )
    labels = guided["labels"]
    locked = load_locked_baseline_probabilities(
        args.baseline_probability_root,
        fold=fold,
        expected_case_ids=guided["case_ids"],
        expected_labels=labels,
    )
    cached_foundation_predictions = guided["foundation_predictions"]
    locked_foundation_predictions = locked["flair"].argmax(axis=1)
    if not np.array_equal(
        cached_foundation_predictions,
        locked_foundation_predictions,
    ):
        changed = int(
            np.sum(cached_foundation_predictions != locked_foundation_predictions)
        )
        raise RuntimeError(
            f"Cached and locked Foundation FLAIR predictions differ for "
            f"{changed} Fold {fold} cases"
        )
    guided_probabilities, guided_predictions = (
        guided_probabilities_on_locked_foundation(guided, locked["flair"])
    )
    predictions = {
        "foundation_flair": locked_foundation_predictions,
        "guided_flair": guided_predictions,
    }
    probabilities = {
        "foundation_flair": locked["flair"],
        "guided_flair": guided_probabilities,
    }

    if not args.skip_fusion:
        probabilities["baseline_fusion"] = locked["fusion"]
        probabilities["guided_fusion"] = (
            locked["t1"]
            + locked["t2"]
            + probabilities["guided_flair"]
        ) / 3.0
        predictions["baseline_fusion"] = probabilities["baseline_fusion"].argmax(1)
        predictions["guided_fusion"] = probabilities["guided_fusion"].argmax(1)

    metrics = {
        name: metrics_for(labels, value, config.METASTASIS_F_BETA)
        for name, value in predictions.items()
    }
    print(
        f"Guided checkpoint: selection={guided['selection_status']}, "
        f"epoch={guided['checkpoint_epoch']}, "
        f"validation-selected alpha={guided['alpha']:.2f}"
    )
    print(f"Locked FP32 baseline probabilities: {locked['path']}")
    for name in METHOD_NAMES:
        if name in metrics:
            print_metrics(name, metrics[name])
    return {
        "fold": int(fold),
        "case_ids": guided["case_ids"],
        "labels": labels,
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": metrics,
        "alpha": guided["alpha"],
        "checkpoint_epoch": guided["checkpoint_epoch"],
        "selection_status": guided["selection_status"],
        "baseline_probability_path": locked["path"],
        "validation": guided["validation"],
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_reports(report_root, fold_results, pooled_metrics):
    report_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "evaluation_policy": (
            "Locked test evaluation: checkpoint epoch and alpha were selected on "
            "each fold validation split; no test-time fitting or threshold scan."
        ),
        "folds": [
            {
                "fold": result["fold"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "selection_status": result["selection_status"],
                "baseline_probability_path": result["baseline_probability_path"],
                "alpha": result["alpha"],
                "validation": result["validation"],
                "test": result["metrics"],
            }
            for result in fold_results
        ],
        "pooled_test": pooled_metrics,
    }
    summary_path = report_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(json_ready(summary), stream, ensure_ascii=False, indent=2)

    table_path = report_root / "per_case_predictions.csv"
    probability_methods = [
        name for name in METHOD_NAMES if name in fold_results[0]["probabilities"]
    ]
    fieldnames = ["fold", "case_id", "label"]
    for name in probability_methods:
        fieldnames.extend(
            (
                f"{name}_p_normal",
                f"{name}_p_inflammation",
                f"{name}_p_metastasis",
                f"{name}_prediction",
            )
        )
    with open(table_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in fold_results:
            for index, case_id in enumerate(result["case_ids"]):
                row = {
                    "fold": result["fold"],
                    "case_id": case_id,
                    "label": int(result["labels"][index]),
                }
                for name in probability_methods:
                    probability = result["probabilities"][name][index]
                    row.update(
                        {
                            f"{name}_p_normal": float(probability[0]),
                            f"{name}_p_inflammation": float(probability[1]),
                            f"{name}_p_metastasis": float(probability[2]),
                            f"{name}_prediction": int(
                                result["predictions"][name][index]
                            ),
                        }
                    )
                writer.writerow(row)
    return summary_path, table_path


def main(args):
    config = load_python_config(args.config, CONFIG_FIELDS)
    set_seed(SEED)
    device = torch.device(args.device or config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    folds = tuple(int(fold) for fold in args.folds)
    if len(set(folds)) != len(folds):
        raise ValueError("--folds contains duplicate fold numbers")
    if any(fold < 1 or fold > int(K_FOLDS) for fold in folds):
        raise ValueError(f"folds must lie in [1, {K_FOLDS}]")
    print("Evaluation policy: locked test; no alpha/threshold fitting on test")
    print(
        "Baseline fusion: locked FP32 equal soft vote of "
        "seq1/T1 FoundationModel_ori + seq2/T2 FoundationModel_ori + "
        "seq3/FLAIR FoundationModel"
    )
    print(
        "Guided fusion  : same locked seq1/seq2 probabilities; only the "
        "seq3 inflammation/metastasis conditional probability is replaced"
    )
    print(f"Baseline probability root: {Path(args.baseline_probability_root).resolve()}")
    print(f"Folds: {', '.join(str(fold) for fold in folds)}")
    print(f"Device: {device}; visible GPUs: {torch.cuda.device_count()}")

    fold_results = [evaluate_fold(args, config, fold, device) for fold in folds]
    all_case_ids = [case_id for result in fold_results for case_id in result["case_ids"]]
    if len(all_case_ids) != len(set(all_case_ids)):
        raise RuntimeError("A case occurs in more than one evaluated test fold")

    labels = np.concatenate([result["labels"] for result in fold_results])
    available_methods = [
        name for name in METHOD_NAMES if name in fold_results[0]["predictions"]
    ]
    pooled_metrics = {}
    print(f"\n{'=' * 24} Pooled locked test {'=' * 24}")
    for name in available_methods:
        predictions = np.concatenate(
            [result["predictions"][name] for result in fold_results]
        )
        pooled_metrics[name] = metrics_for(
            labels,
            predictions,
            config.METASTASIS_F_BETA,
        )
        print_metrics(name, pooled_metrics[name])

    report_root = Path(args.report_root).expanduser().resolve()
    summary_path, table_path = write_reports(
        report_root,
        fold_results,
        pooled_metrics,
    )
    print(f"\nSummary: {summary_path}")
    print(f"Per-case predictions: {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--guided-checkpoint-root", required=True)
    parser.add_argument("--foundation-checkpoint-root", required=True)
    parser.add_argument("--nnunet-results-root", required=True)
    parser.add_argument("--baseline-probability-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(1, int(K_FOLDS) + 1)),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    main(parser.parse_args())
