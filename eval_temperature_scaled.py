"""
逐 fold 验证集温度缩放与三序列软投票评估。

每个序列各拟合一个标量温度，可分别最小化单序列 NLL，也可联合最小化
等权融合 NLL；测试集只用于最终评估，不参与温度或其他参数的选择。
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from configs.config_utils import (
    TRAIN_CONFIG_FIELDS,
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
from models.model_factory import (
    MODEL_CHOICES,
    create_model,
    forward_model,
    model_capabilities,
)
from utils.fusion_diagnostics import (
    probability_metrics,
    save_probability_diagnostics,
    write_probability_diagnostics_readme,
)
from utils.temperature_scaling import (
    fit_fusion_temperatures,
    fit_temperature,
    temperature_scale_probabilities,
)
from utils.train_and_test import load_pt_dataset, set_seed


DEFAULT_MODEL_NAMES = (
    "FoundationModel_ori",
    "FoundationModel_ori",
    "FoundationModelLesionAwareHierarchical",
)


class AlignedMultiSequenceDataset(Dataset):
    """返回同一病例的三个单序列张量，并在加载时检查顺序一致性。"""

    def __init__(self, datasets):
        if len(datasets) != len(ALL_SEQUENCES):
            raise ValueError("one dataset is required for each sequence")
        reference_cases = datasets[0].cases
        reference_keys = [
            (str(case["case_id"]), int(case["label"]))
            for case in reference_cases
        ]
        for seq_idx, dataset in enumerate(datasets[1:], start=2):
            keys = [
                (str(case["case_id"]), int(case["label"]))
                for case in dataset.cases
            ]
            if keys != reference_keys:
                raise ValueError(
                    f"seq{seq_idx} cases are not aligned with seq1"
                )
        self.datasets = tuple(datasets)

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        xs = []
        label = None
        case_id = None
        for seq_idx, dataset in enumerate(self.datasets):
            x, current_label, _, _, current_case_id = dataset[index]
            if seq_idx == 0:
                label = current_label
                case_id = current_case_id
            xs.append(x)
        return xs, label, case_id


def resolve_checkpoint_roots(shared_root, sequence_roots):
    if shared_root is not None and sequence_roots is not None:
        raise ValueError(
            "Specify either --checkpoint-root or --checkpoint-roots, not both"
        )
    if sequence_roots is not None:
        return tuple(
            resolve_input_artifact_dir(root, "checkpoints")
            for root in sequence_roots
        )
    if shared_root is None:
        raise ValueError(
            "One of --checkpoint-root or --checkpoint-roots is required"
        )
    checkpoint_root = resolve_input_artifact_dir(shared_root, "checkpoints")
    return (checkpoint_root,) * len(ALL_SEQUENCES)


def combine_hierarchical_probabilities(main_probabilities, subtype_probabilities):
    abnormal_probability = main_probabilities[:, 1:].sum(dim=1, keepdim=True)
    return torch.cat(
        (
            main_probabilities[:, :1],
            abnormal_probability * subtype_probabilities,
        ),
        dim=1,
    )


def effective_probabilities(model, x):
    capabilities = model_capabilities(model)
    outputs = forward_model(
        model,
        x,
        return_subtype=capabilities["subtype"],
    )
    main_probabilities = F.softmax(outputs["classification"], dim=1)
    if not capabilities["subtype"]:
        return main_probabilities
    subtype_probabilities = F.softmax(outputs["subtype"], dim=1)
    return combine_hierarchical_probabilities(
        main_probabilities,
        subtype_probabilities,
    )


def load_split(dataset_dirs, fold_idx, split_name, processed_data_root):
    datasets = []
    for dataset_dir in dataset_dirs:
        split_path = dataset_dir / f"fold{fold_idx}" / f"{split_name}.pt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {split_path}")
        datasets.append(
            load_pt_dataset(split_path, data_root=processed_data_root)
        )
    return AlignedMultiSequenceDataset(datasets)


def load_models(fold_idx, ckpt_dirs, model_names, device):
    models = []
    checkpoint_paths = []
    for seq_idx, model_name in enumerate(model_names):
        checkpoint_path = (
            ckpt_dirs[seq_idx]
            / model_name
            / f"fold{fold_idx}_model_best.pth"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}"
            )
        model = create_model(
            model_name,
            num_classes=NUM_CLASSES,
            in_channels=1,
            sequence_id=seq_idx + 1,
        ).to(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()
        models.append(model)
        checkpoint_paths.append(str(checkpoint_path))
    return models, checkpoint_paths


def infer_base_probabilities(loader, models, device):
    all_case_ids = []
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for xs, labels, case_ids in loader:
            sequence_probabilities = []
            for seq_idx, x in enumerate(xs):
                sequence_probabilities.append(
                    effective_probabilities(models[seq_idx], x.to(device))
                )
            all_probabilities.append(
                torch.stack(sequence_probabilities, dim=1).cpu().numpy()
            )
            all_labels.append(labels.numpy())
            all_case_ids.extend(str(case_id) for case_id in case_ids)
    if not all_probabilities:
        raise ValueError("dataset split contains no samples")
    return {
        "case_ids": np.asarray(all_case_ids, dtype=str),
        "labels": np.concatenate(all_labels).astype(np.int64, copy=False),
        "base_probabilities": np.concatenate(all_probabilities, axis=0),
    }


def apply_temperatures(base_probabilities, temperatures):
    return np.stack(
        [
            temperature_scale_probabilities(
                base_probabilities[:, seq_idx, :],
                temperature,
            )
            for seq_idx, temperature in enumerate(temperatures)
        ],
        axis=1,
    )


def fit_sequence_temperatures(
    validation_data,
    model_names,
    temperature_objective="independent",
):
    labels = validation_data["labels"]
    base_probabilities = validation_data["base_probabilities"]
    independent_fits = [
        fit_temperature(base_probabilities[:, seq_idx, :], labels)
        for seq_idx in range(len(ALL_SEQUENCES))
    ]
    independent_temperatures = tuple(
        fit["temperature"] for fit in independent_fits
    )
    joint_fusion_fit = None
    if temperature_objective == "independent":
        temperatures = independent_temperatures
        fit_objective = "independent_per_sequence_multiclass_nll"
    elif temperature_objective == "fusion":
        joint_fusion_fit = fit_fusion_temperatures(
            base_probabilities,
            labels,
            initial_temperatures=independent_temperatures,
        )
        temperatures = tuple(joint_fusion_fit["temperatures"])
        fit_objective = "joint_equal_soft_vote_multiclass_nll"
    else:
        raise ValueError(
            f"Unsupported temperature objective: {temperature_objective}"
        )

    sequence_reports = []
    for seq_idx, (sequence_name, model_name) in enumerate(
        zip(ALL_SEQUENCES, model_names)
    ):
        raw_probabilities = base_probabilities[:, seq_idx, :]
        temperature = temperatures[seq_idx]
        calibrated_probabilities = temperature_scale_probabilities(
            raw_probabilities,
            temperature,
        )
        sequence_report = {
            "sequence_id": seq_idx + 1,
            "sequence": sequence_name,
            "model": model_name,
            "temperature": float(temperature),
            "temperature_selection_objective": fit_objective,
            "independent_reference_temperature": float(
                independent_temperatures[seq_idx]
            ),
            "probability_metrics_before": probability_metrics(
                labels,
                raw_probabilities,
            ),
            "probability_metrics_after": probability_metrics(
                labels,
                calibrated_probabilities,
            ),
        }
        if temperature_objective == "independent":
            sequence_report.update(independent_fits[seq_idx])
        sequence_reports.append(sequence_report)

    calibrated_base = apply_temperatures(base_probabilities, temperatures)
    raw_fusion = base_probabilities.mean(axis=1)
    calibrated_fusion = calibrated_base.mean(axis=1)
    return (
        tuple(temperatures),
        calibrated_base,
        {
            "fit_split": "validation",
            "fit_objective": fit_objective,
            "sample_count": int(len(labels)),
            "sequences": sequence_reports,
            "joint_fusion_fit": joint_fusion_fit,
            "fusion_probability_metrics_before": probability_metrics(
                labels,
                raw_fusion,
            ),
            "fusion_probability_metrics_after": probability_metrics(
                labels,
                calibrated_fusion,
            ),
        },
    )


def classification_metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    metastasis_idx = CLASS_NAMES.index("metastasis")
    per_class_precision = precision_score(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    per_class_recall = recall_score(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )
    return {
        "sample_count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "metastasis_precision": float(per_class_precision[metastasis_idx]),
        "metastasis_recall": float(per_class_recall[metastasis_idx]),
        "metastasis_f1": float(per_class_f1[metastasis_idx]),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(range(NUM_CLASSES)),
        ).tolist(),
    }


def comparison_summary(labels, raw_probabilities, calibrated_probabilities):
    raw_predictions = raw_probabilities.argmax(axis=1)
    calibrated_predictions = calibrated_probabilities.argmax(axis=1)
    raw_correct = raw_predictions == labels
    calibrated_correct = calibrated_predictions == labels
    changed = raw_predictions != calibrated_predictions
    return {
        "changed_prediction_count": int(changed.sum()),
        "changed_prediction_rate": float(changed.mean()),
        "corrected_count": int((~raw_correct & calibrated_correct).sum()),
        "harmed_count": int((raw_correct & ~calibrated_correct).sum()),
        "both_correct_count": int((raw_correct & calibrated_correct).sum()),
        "both_wrong_count": int((~raw_correct & ~calibrated_correct).sum()),
    }


def save_comparison_csv(
    output_path,
    scope,
    case_ids,
    labels,
    raw_probabilities,
    calibrated_probabilities,
):
    raw_predictions = raw_probabilities.argmax(axis=1)
    calibrated_predictions = calibrated_probabilities.argmax(axis=1)
    rows = []
    for index, case_id in enumerate(case_ids):
        label = int(labels[index])
        raw_prediction = int(raw_predictions[index])
        calibrated_prediction = int(calibrated_predictions[index])
        row = {
            "scope": scope,
            "case_id": case_id,
            "label": label,
            "label_name": CLASS_NAMES[label],
        }
        for class_idx, class_name in enumerate(CLASS_NAMES):
            row[f"uncalibrated_prob_{class_name}"] = float(
                raw_probabilities[index, class_idx]
            )
            row[f"calibrated_prob_{class_name}"] = float(
                calibrated_probabilities[index, class_idx]
            )
        row.update(
            {
                "uncalibrated_pred": raw_prediction,
                "uncalibrated_pred_name": CLASS_NAMES[raw_prediction],
                "calibrated_pred": calibrated_prediction,
                "calibrated_pred_name": CLASS_NAMES[calibrated_prediction],
                "prediction_changed": int(
                    raw_prediction != calibrated_prediction
                ),
                "corrected_by_calibration": int(
                    raw_prediction != label and calibrated_prediction == label
                ),
                "harmed_by_calibration": int(
                    raw_prediction == label and calibrated_prediction != label
                ),
            }
        )
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(output_path, value):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_report_readme(report_root):
    report_root.mkdir(parents=True, exist_ok=True)
    readme_path = report_root / "README.md"
    readme_path.write_text(
        """# Temperature-scaled fusion evaluation

- 每个 fold 使用该 fold 的 `val.pt`，为 T1、T2、FLAIR 各拟合一个标量温度。
- `--temperature-objective independent` 分别最小化三个单序列 NLL；`fusion` 联合最小化等权融合 NLL。
- `test.pt` 不参与温度拟合。
- `foldN_temperature_scaling.json` 保存温度及验证集校准前后指标。
- `foldN_comparison.csv` 保存测试集融合预测在校准前后的逐病例变化。
- `uncalibrated/` 与 `calibrated/` 保存完整概率和校准诊断表。
- `summary.json` 保存五折温度、pooled 测试指标和校准前后变化。

标量温度缩放不改变任何单序列模型的类别排序。普通模型等价于
`softmax(logits / T)`；层级 FLAIR 对规范化的联合三分类概率执行
`softmax(log(probability) / T)`。`T > 1` 会软化概率，`T < 1` 会锐化概率。
""",
        encoding="utf-8",
    )
    return readme_path


def evaluate_fold(
    fold_idx,
    dataset_dirs,
    ckpt_dirs,
    processed_data_root,
    config,
    model_names,
    temperature_objective,
    report_root=None,
):
    print(f"\n{'=' * 20} Fold {fold_idx} {'=' * 20}")
    started = time.time()
    models, checkpoint_paths = load_models(
        fold_idx,
        ckpt_dirs,
        model_names,
        config.DEVICE,
    )
    validation_set = load_split(
        dataset_dirs,
        fold_idx,
        "val",
        processed_data_root,
    )
    test_set = load_split(
        dataset_dirs,
        fold_idx,
        "test",
        processed_data_root,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    validation_data = infer_base_probabilities(
        validation_loader,
        models,
        config.DEVICE,
    )
    temperatures, _, temperature_report = fit_sequence_temperatures(
        validation_data,
        model_names,
        temperature_objective=temperature_objective,
    )
    temperature_report.update(
        {
            "fold": fold_idx,
            "checkpoint_paths": checkpoint_paths,
        }
    )

    test_data = infer_base_probabilities(
        test_loader,
        models,
        config.DEVICE,
    )
    raw_base = test_data["base_probabilities"]
    calibrated_base = apply_temperatures(raw_base, temperatures)
    raw_fusion = raw_base.mean(axis=1)
    calibrated_fusion = calibrated_base.mean(axis=1)
    labels = test_data["labels"]
    case_ids = test_data["case_ids"]

    raw_metrics = classification_metrics(labels, raw_fusion)
    calibrated_metrics = classification_metrics(labels, calibrated_fusion)
    paired = comparison_summary(labels, raw_fusion, calibrated_fusion)
    temperature_report["test_uncalibrated_metrics"] = raw_metrics
    temperature_report["test_calibrated_metrics"] = calibrated_metrics
    temperature_report["test_comparison"] = paired

    if report_root is not None:
        uncalibrated_root = report_root / "uncalibrated"
        calibrated_root = report_root / "calibrated"
        save_probability_diagnostics(
            report_root=uncalibrated_root,
            fold_idx=fold_idx,
            case_ids=case_ids,
            labels=labels,
            base_probabilities=raw_base,
            fusion_probabilities=raw_fusion,
            model_names=model_names,
        )
        save_probability_diagnostics(
            report_root=calibrated_root,
            fold_idx=fold_idx,
            case_ids=case_ids,
            labels=labels,
            base_probabilities=calibrated_base,
            fusion_probabilities=calibrated_fusion,
            model_names=model_names,
        )
        write_json(
            report_root / f"fold{fold_idx}_temperature_scaling.json",
            temperature_report,
        )
        save_comparison_csv(
            report_root / f"fold{fold_idx}_comparison.csv",
            f"fold{fold_idx}",
            case_ids,
            labels,
            raw_fusion,
            calibrated_fusion,
        )

    print(
        f"Objective    : {temperature_report['fit_objective']}"
    )
    print(
        "Temperatures : "
        + ", ".join(
            f"{sequence}={temperature:.4f}"
            for sequence, temperature in zip(ALL_SEQUENCES, temperatures)
        )
    )
    print(
        "Accuracy     : "
        f"{raw_metrics['accuracy']:.4f} -> "
        f"{calibrated_metrics['accuracy']:.4f}"
    )
    print(
        "Macro F1    : "
        f"{raw_metrics['macro_f1']:.4f} -> "
        f"{calibrated_metrics['macro_f1']:.4f}"
    )
    print(
        "Meta P/R/F1 : "
        f"{raw_metrics['metastasis_precision']:.4f}/"
        f"{raw_metrics['metastasis_recall']:.4f}/"
        f"{raw_metrics['metastasis_f1']:.4f} -> "
        f"{calibrated_metrics['metastasis_precision']:.4f}/"
        f"{calibrated_metrics['metastasis_recall']:.4f}/"
        f"{calibrated_metrics['metastasis_f1']:.4f}"
    )
    print(
        "Changed      : "
        f"{paired['changed_prediction_count']} "
        f"(corrected {paired['corrected_count']}, "
        f"harmed {paired['harmed_count']})"
    )
    print("\n===== Calibrated Test Results =====")
    print(f"Sequence      : ALL (Temperature-scaled Soft Voting) (Fold {fold_idx})")
    print(f"Test samples  : {len(labels)}")
    print(f"Accuracy      : {calibrated_metrics['accuracy']:.4f}")
    print(f"Precision     : {calibrated_metrics['macro_precision']:.4f}")
    print(f"Recall        : {calibrated_metrics['macro_recall']:.4f}")
    print(f"F1-score      : {calibrated_metrics['macro_f1']:.4f}")
    print("\nConfusion Matrix:")
    print(np.asarray(calibrated_metrics["confusion_matrix"]))
    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            calibrated_fusion.argmax(axis=1),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    print(
        "Misclassified: "
        f"{int((calibrated_fusion.argmax(axis=1) != labels).sum())}"
    )
    print(f"Elapsed      : {time.time() - started:.1f}s")
    return {
        "fold": fold_idx,
        "temperatures": list(temperatures),
        "temperature_report": temperature_report,
        "test_data": test_data,
        "calibrated_base_probabilities": calibrated_base,
        "raw_fusion_probabilities": raw_fusion,
        "calibrated_fusion_probabilities": calibrated_fusion,
    }


def main(args):
    config = load_python_config(args.config, TRAIN_CONFIG_FIELDS)
    dataset_root = resolve_input_artifact_dir(args.data_root, "datasets")
    processed_data_root = infer_data_dir(args.data_root)
    checkpoint_roots = resolve_checkpoint_roots(
        args.checkpoint_root,
        args.checkpoint_roots,
    )
    model_names = tuple(args.model_names)
    report_root = (
        Path(args.report_root).expanduser().resolve()
        if args.report_root is not None
        else None
    )
    dataset_dirs = [
        dataset_root / f"seq{seq_idx}_{sequence_name}"
        for seq_idx, sequence_name in enumerate(ALL_SEQUENCES, start=1)
    ]
    ckpt_dirs = [
        checkpoint_roots[seq_idx - 1]
        / f"seq{seq_idx}_{sequence_name}"
        for seq_idx, sequence_name in enumerate(ALL_SEQUENCES, start=1)
    ]

    set_seed(SEED)
    if report_root is not None:
        write_report_readme(report_root)
        write_probability_diagnostics_readme(report_root / "uncalibrated")
        write_probability_diagnostics_readme(report_root / "calibrated")
    print("\n>>> Validation-fitted per-sequence temperature scaling <<<")
    print(f"Config      : {config.__config_path__}")
    print(f"Dataset     : {dataset_root}")
    print(f"Objective   : {args.temperature_objective}")
    print(
        f"Report      : {report_root}"
        if report_root is not None
        else "Report      : disabled (terminal output only)"
    )
    for seq_idx, (sequence_name, model_name, checkpoint_root) in enumerate(
        zip(ALL_SEQUENCES, model_names, checkpoint_roots),
        start=1,
    ):
        print(
            f"Seq{seq_idx} {sequence_name:<5}: {model_name} | "
            f"{checkpoint_root}"
        )

    folds = [args.fold] if args.fold is not None else range(1, K_FOLDS + 1)
    fold_results = [
        evaluate_fold(
            fold_idx,
            dataset_dirs,
            ckpt_dirs,
            processed_data_root,
            config,
            model_names,
            args.temperature_objective,
            report_root,
        )
        for fold_idx in folds
    ]

    case_ids = np.concatenate(
        [result["test_data"]["case_ids"] for result in fold_results]
    )
    labels = np.concatenate(
        [result["test_data"]["labels"] for result in fold_results]
    )
    raw_base = np.concatenate(
        [
            result["test_data"]["base_probabilities"]
            for result in fold_results
        ],
        axis=0,
    )
    calibrated_base = np.concatenate(
        [result["calibrated_base_probabilities"] for result in fold_results],
        axis=0,
    )
    raw_fusion = np.concatenate(
        [result["raw_fusion_probabilities"] for result in fold_results],
        axis=0,
    )
    calibrated_fusion = np.concatenate(
        [result["calibrated_fusion_probabilities"] for result in fold_results],
        axis=0,
    )

    if report_root is not None:
        save_probability_diagnostics(
            report_root=report_root / "uncalibrated",
            case_ids=case_ids,
            labels=labels,
            base_probabilities=raw_base,
            fusion_probabilities=raw_fusion,
            model_names=model_names,
        )
        save_probability_diagnostics(
            report_root=report_root / "calibrated",
            case_ids=case_ids,
            labels=labels,
            base_probabilities=calibrated_base,
            fusion_probabilities=calibrated_fusion,
            model_names=model_names,
        )
        save_comparison_csv(
            report_root / "pooled_comparison.csv",
            "pooled",
            case_ids,
            labels,
            raw_fusion,
            calibrated_fusion,
        )

    raw_metrics = classification_metrics(labels, raw_fusion)
    calibrated_metrics = classification_metrics(labels, calibrated_fusion)
    paired = comparison_summary(labels, raw_fusion, calibrated_fusion)
    temperatures_by_fold = {
        f"fold{result['fold']}": {
            sequence_name: result["temperatures"][seq_idx]
            for seq_idx, sequence_name in enumerate(ALL_SEQUENCES)
        }
        for result in fold_results
    }
    temperature_matrix = np.asarray(
        [result["temperatures"] for result in fold_results],
        dtype=np.float64,
    )
    summary = {
        "calibration_protocol": (
            "Each fold uses only its matching validation split to fit three "
            "temperatures. The selected objective is "
            f"{args.temperature_objective}."
        ),
        "temperature_objective": args.temperature_objective,
        "model_names": list(model_names),
        "temperatures_by_fold": temperatures_by_fold,
        "temperature_mean": {
            sequence_name: float(temperature_matrix[:, seq_idx].mean())
            for seq_idx, sequence_name in enumerate(ALL_SEQUENCES)
        },
        "temperature_std": {
            sequence_name: float(temperature_matrix[:, seq_idx].std())
            for seq_idx, sequence_name in enumerate(ALL_SEQUENCES)
        },
        "pooled_uncalibrated_classification": raw_metrics,
        "pooled_calibrated_classification": calibrated_metrics,
        "pooled_uncalibrated_probability_metrics": probability_metrics(
            labels,
            raw_fusion,
        ),
        "pooled_calibrated_probability_metrics": probability_metrics(
            labels,
            calibrated_fusion,
        ),
        "pooled_comparison": paired,
        "fold_reports": [
            result["temperature_report"] for result in fold_results
        ],
    }
    if report_root is not None:
        write_json(report_root / "summary.json", summary)

    print("\n" + "=" * 54)
    print("POOLED TEST COMPARISON")
    print("=" * 54)
    print(
        f"Accuracy      : {raw_metrics['accuracy']:.4f} -> "
        f"{calibrated_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1      : {raw_metrics['macro_f1']:.4f} -> "
        f"{calibrated_metrics['macro_f1']:.4f}"
    )
    print(
        "Metastasis P/R/F1: "
        f"{raw_metrics['metastasis_precision']:.4f}/"
        f"{raw_metrics['metastasis_recall']:.4f}/"
        f"{raw_metrics['metastasis_f1']:.4f} -> "
        f"{calibrated_metrics['metastasis_precision']:.4f}/"
        f"{calibrated_metrics['metastasis_recall']:.4f}/"
        f"{calibrated_metrics['metastasis_f1']:.4f}"
    )
    print(f"Confusion matrix before:\n{np.asarray(raw_metrics['confusion_matrix'])}")
    print(
        "Confusion matrix after:\n"
        f"{np.asarray(calibrated_metrics['confusion_matrix'])}"
    )
    print(
        "Prediction changes: "
        f"{paired['changed_prediction_count']} "
        f"(corrected {paired['corrected_count']}, "
        f"harmed {paired['harmed_count']})"
    )
    print("\nCalibrated classification report:")
    print(
        classification_report(
            labels,
            calibrated_fusion.argmax(axis=1),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    if report_root is not None:
        print(f"Summary: {report_root / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the train_config.py used for evaluation settings",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Experiment root containing datasets, or the datasets directory",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Shared training root containing checkpoints",
    )
    parser.add_argument(
        "--checkpoint-roots",
        nargs=len(ALL_SEQUENCES),
        metavar=("SEQ1_ROOT", "SEQ2_ROOT", "SEQ3_ROOT"),
        default=None,
        help="Per-sequence training roots; exclusive with --checkpoint-root",
    )
    parser.add_argument(
        "--model-names",
        nargs=len(ALL_SEQUENCES),
        choices=MODEL_CHOICES,
        metavar=("SEQ1_MODEL", "SEQ2_MODEL", "SEQ3_MODEL"),
        default=DEFAULT_MODEL_NAMES,
        help="Model class for seq1, seq2, and seq3",
    )
    parser.add_argument(
        "--temperature-objective",
        choices=("independent", "fusion"),
        default="independent",
        help=(
            "independent minimizes each sequence NLL separately; fusion jointly "
            "minimizes equal-soft-vote NLL"
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, K_FOLDS + 1),
        default=None,
        help=f"Evaluate one fold (1~{K_FOLDS}); default evaluates all folds",
    )
    parser.add_argument(
        "--report-root",
        default=None,
        help=(
            "Optional output directory for temperatures and paired diagnostics; "
            "omit it for terminal output only"
        ),
    )
    main(parser.parse_args())
