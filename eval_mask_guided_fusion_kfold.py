"""评估 validation 已锁定的异步 nnU-Net mask-guided 三序列融合。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from configs.config_utils import (
    infer_data_dir,
    load_python_config,
    resolve_input_artifact_dir,
)
from configs.global_config import ALL_SEQUENCES, CLASS_NAMES, K_FOLDS, NUM_CLASSES, SEED
from models.NNUNetMaskGuidedClassifier import NNUNetMaskGuidedClassifier
from models.model_factory import create_model, forward_model
from train_foundation_nnunet_guided import (
    load_split_dataset,
    native_nnunet_checkpoint_path,
    normalize_case_id,
    sha256_file,
)
from train_mask_guided_fusion_kfold import (
    CONFIG_FIELDS,
    MODEL_NAMES,
    _classification_metrics,
    _save_json_atomic,
)
from utils.nnunet_mask_cache import extract_mask_cache, load_mask_guided_dataset
from utils.train_and_test import load_pt_dataset, set_seed


def _device(config):
    device = torch.device(config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _checkpoint_path(root, fold, sequence_id, model_name):
    checkpoint_root = resolve_input_artifact_dir(root, "checkpoints")
    sequence_name = ALL_SEQUENCES[sequence_id - 1]
    path = (
        checkpoint_root
        / f"seq{sequence_id}_{sequence_name}"
        / model_name
        / f"fold{int(fold)}_model_best.pth"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Locked checkpoint not found: {path}")
    return path


def _manifest_path(root, fold):
    path = (
        resolve_input_artifact_dir(root, "checkpoints")
        / "fusion_selection"
        / f"fold{int(fold)}_selection.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Fusion selection manifest not found: {path}")
    return path


def _load_locked_checkpoints(root, fold):
    manifest_path = _manifest_path(root, fold)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_names") != list(MODEL_NAMES):
        raise RuntimeError(f"Unexpected model set in {manifest_path}")
    if int(manifest.get("fold", -1)) != int(fold):
        raise RuntimeError(f"Fold mismatch in {manifest_path}")
    if manifest.get("selection_status") != "constrained_async_fusion_f2":
        raise RuntimeError(
            f"Refusing non-constrained fusion selection: {manifest_path}"
        )
    if not bool(manifest.get("meets_selection_constraints", False)):
        raise RuntimeError(f"Fusion constraints were not met: {manifest_path}")
    if not bool(manifest.get("test_parameters_locked", False)):
        raise RuntimeError(f"Test-lock marker is missing: {manifest_path}")
    selection_id = manifest.get("selection_id")
    selected_epochs = tuple(int(x) for x in manifest["selected_epochs"])
    checkpoints = []
    for sequence_id, model_name in enumerate(MODEL_NAMES, start=1):
        path = _checkpoint_path(root, fold, sequence_id, model_name)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != model_name:
            raise RuntimeError(f"Model name mismatch in {path}")
        if int(checkpoint.get("fold", -1)) != int(fold):
            raise RuntimeError(f"Fold mismatch in {path}")
        if int(checkpoint.get("epoch", -1)) != selected_epochs[sequence_id - 1]:
            raise RuntimeError(f"Selected epoch mismatch in {path}")
        if checkpoint.get("fusion_selection_id") != selection_id:
            raise RuntimeError(f"Fusion selection ID mismatch in {path}")
        if not bool(checkpoint.get("meets_fusion_selection_constraints", False)):
            raise RuntimeError(f"Checkpoint is not constraint-approved: {path}")
        checkpoints.append((path, checkpoint))
    alpha = float(manifest["guidance_alpha"])
    state_alpha = float(
        checkpoints[2][1]["model_state"]["guidance_mix_alpha"].item()
    )
    if not np.isclose(alpha, state_alpha, atol=1e-8):
        raise RuntimeError(
            f"Guidance alpha mismatch: manifest={alpha}, state={state_alpha}"
        )
    return manifest, checkpoints


def _standard_test_dataset(data_root, fold, sequence_id):
    datasets_root = resolve_input_artifact_dir(data_root, "datasets")
    sequence_name = ALL_SEQUENCES[sequence_id - 1]
    path = (
        datasets_root
        / f"seq{sequence_id}_{sequence_name}"
        / f"fold{int(fold)}"
        / "test.pt"
    )
    return load_pt_dataset(path, data_root=infer_data_dir(data_root))


def _loader(dataset, config):
    num_workers = int(config.NUM_WORKERS)
    options = {
        "batch_size": int(config.BATCH_SIZE),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": str(config.DEVICE).startswith("cuda"),
    }
    if num_workers > 0:
        options.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
        )
    return DataLoader(
        dataset,
        **options,
    )


def _evaluate_standard(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    case_ids = []
    with torch.no_grad():
        for inputs, batch_labels, _, _, batch_case_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = forward_model(model, inputs)["classification"].float()
            probabilities.append(F.softmax(logits, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
            case_ids.extend(normalize_case_id(value) for value in batch_case_ids)
    return {
        "case_ids": tuple(case_ids),
        "labels": np.concatenate(labels).astype(np.int64, copy=False),
        "probabilities": np.concatenate(probabilities).astype(
            np.float64,
            copy=False,
        ),
    }


def _evaluate_mask_guided(model, loader, device):
    model.eval()
    probabilities = []
    base_probabilities = []
    expert_probabilities = []
    labels = []
    case_ids = []
    with torch.no_grad():
        for inputs, batch_labels, masks, statistics, batch_case_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            statistics = statistics.to(device, non_blocking=True)
            outputs = model(inputs, masks, statistics)
            probabilities.append(outputs["probabilities"].float().cpu().numpy())
            base_probabilities.append(
                outputs["base_probabilities"].float().cpu().numpy()
            )
            expert_probabilities.append(
                outputs["expert_probability"].float().cpu().numpy()
            )
            labels.append(batch_labels.numpy())
            case_ids.extend(normalize_case_id(value) for value in batch_case_ids)
    return {
        "case_ids": tuple(case_ids),
        "labels": np.concatenate(labels).astype(np.int64, copy=False),
        "probabilities": np.concatenate(probabilities).astype(
            np.float64,
            copy=False,
        ),
        "base_probabilities": np.concatenate(base_probabilities).astype(
            np.float64,
            copy=False,
        ),
        "expert_probabilities": np.concatenate(expert_probabilities).astype(
            np.float64,
            copy=False,
        ),
    }


def _validate_alignment(results):
    reference = results[0]
    for result in results[1:]:
        if result["case_ids"] != reference["case_ids"]:
            raise RuntimeError("Test case order differs across sequences")
        if not np.array_equal(result["labels"], reference["labels"]):
            raise RuntimeError("Test labels differ across sequences")


def _model_from_checkpoint(sequence_id, checkpoint, config, device):
    if sequence_id == 3:
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
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def _fold_result(
    *,
    fold,
    config,
    data_root,
    output_root,
    nnunet_results_root,
    device,
    rebuild_mask_cache,
):
    manifest, checkpoints = _load_locked_checkpoints(output_root, fold)
    test_datasets = [
        _standard_test_dataset(data_root, fold, sequence_id)
        for sequence_id in (1, 2)
    ]
    flair_test = load_split_dataset(data_root, fold, "test")
    nnunet_path = native_nnunet_checkpoint_path(
        nnunet_results_root,
        config,
        fold,
    )
    test_mask_path = extract_mask_cache(
        split="test",
        dataset=flair_test,
        source_folds=(int(fold),) * len(flair_test),
        target_fold=fold,
        nnunet_paths={int(fold): nnunet_path},
        nnunet_sha256={int(fold): sha256_file(nnunet_path)},
        cache_dir=Path(output_root).expanduser().resolve() / "mask_cache",
        config=config,
        device=device,
        rebuild=rebuild_mask_cache,
    )
    test_datasets.append(load_mask_guided_dataset(flair_test, test_mask_path))
    results = []
    for sequence_id, (path, checkpoint) in enumerate(checkpoints, start=1):
        model = _model_from_checkpoint(
            sequence_id,
            checkpoint,
            config,
            device,
        )
        if sequence_id == 3:
            result = _evaluate_mask_guided(
                model,
                _loader(test_datasets[sequence_id - 1], config),
                device,
            )
        else:
            result = _evaluate_standard(
                model,
                _loader(test_datasets[sequence_id - 1], config),
                device,
            )
        result["checkpoint_path"] = str(path)
        result["checkpoint_epoch"] = int(checkpoint["epoch"])
        results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _validate_alignment(results)
    fusion_probabilities = np.mean(
        np.stack([result["probabilities"] for result in results], axis=1),
        axis=1,
    )
    beta = float(config.METASTASIS_F_BETA)
    return {
        "fold": int(fold),
        "manifest": manifest,
        "case_ids": results[0]["case_ids"],
        "labels": results[0]["labels"],
        "sequence_results": results,
        "fusion_probabilities": fusion_probabilities,
        "sequence_metrics": [
            _classification_metrics(
                result["labels"],
                result["probabilities"],
                beta,
            )
            for result in results
        ],
        "fusion_metrics": _classification_metrics(
            results[0]["labels"],
            fusion_probabilities,
            beta,
        ),
    }


def _json_metrics(labels, probabilities, beta):
    metrics = _classification_metrics(labels, probabilities, beta)
    predictions = np.asarray(probabilities).argmax(axis=1)
    metrics["classification_report"] = classification_report(
        labels,
        predictions,
        labels=range(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    return metrics


def _save_reports(fold_results, report_root, beta):
    report_root = Path(report_root).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    rows = []
    pooled_labels = []
    pooled_sequence_probabilities = [[], [], []]
    pooled_fusion_probabilities = []
    class_suffixes = ("normal", "inflammation", "metastasis")
    for fold_result in fold_results:
        labels = fold_result["labels"]
        fusion = fold_result["fusion_probabilities"]
        pooled_labels.append(labels)
        pooled_fusion_probabilities.append(fusion)
        for sequence_id, result in enumerate(
            fold_result["sequence_results"],
            start=1,
        ):
            pooled_sequence_probabilities[sequence_id - 1].append(
                result["probabilities"]
            )
        for index, case_id in enumerate(fold_result["case_ids"]):
            row = {
                "fold": fold_result["fold"],
                "case_id": case_id,
                "label": int(labels[index]),
                "fusion_prediction": int(fusion[index].argmax()),
                "fusion_selection_id": fold_result["manifest"]["selection_id"],
                "seq1_epoch": fold_result["manifest"]["selected_epochs"][0],
                "seq2_epoch": fold_result["manifest"]["selected_epochs"][1],
                "seq3_epoch": fold_result["manifest"]["selected_epochs"][2],
                "guidance_alpha": fold_result["manifest"]["guidance_alpha"],
            }
            for class_index, suffix in enumerate(class_suffixes):
                row[f"fusion_prob_{suffix}"] = float(
                    fusion[index, class_index]
                )
                for sequence_id, result in enumerate(
                    fold_result["sequence_results"],
                    start=1,
                ):
                    row[f"seq{sequence_id}_prob_{suffix}"] = float(
                        result["probabilities"][index, class_index]
                    )
            rows.append(row)
    csv_path = report_root / "per_case_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = np.concatenate(pooled_labels)
    sequence_probabilities = [
        np.concatenate(parts) for parts in pooled_sequence_probabilities
    ]
    fusion_probabilities = np.concatenate(pooled_fusion_probabilities)
    summary = {
        "evaluation_policy": (
            "每折 epoch 与 guidance alpha 均由 validation 锁定；test 不扫描参数。"
        ),
        "model_names": list(MODEL_NAMES),
        "folds": [
            {
                "fold": result["fold"],
                "selection_id": result["manifest"]["selection_id"],
                "selected_epochs": result["manifest"]["selected_epochs"],
                "guidance_alpha": result["manifest"]["guidance_alpha"],
                "sequence_metrics": result["sequence_metrics"],
                "fusion_metrics": result["fusion_metrics"],
            }
            for result in fold_results
        ],
        "pooled_test": {
            "sequences": [
                _json_metrics(labels, probabilities, beta)
                for probabilities in sequence_probabilities
            ],
            "fusion": _json_metrics(labels, fusion_probabilities, beta),
        },
    }
    summary_path = report_root / "summary.json"
    _save_json_atomic(summary, summary_path)
    return summary, csv_path, summary_path


def main(args):
    config = load_python_config(args.config, CONFIG_FIELDS)
    device = _device(config)
    set_seed(SEED)
    fold_results = []
    for fold in args.folds:
        result = _fold_result(
            fold=fold,
            config=config,
            data_root=args.data_root,
            output_root=args.checkpoint_root,
            nnunet_results_root=args.nnunet_results_root,
            device=device,
            rebuild_mask_cache=args.rebuild_mask_cache,
        )
        fold_results.append(result)
        metrics = result["fusion_metrics"]
        print(
            f"Fold {fold} fusion | acc={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"meta_P/R/F{config.METASTASIS_F_BETA:g}="
            f"{metrics['metastasis_precision']:.4f}/"
            f"{metrics['metastasis_recall']:.4f}/"
            f"{metrics['metastasis_fbeta']:.4f}"
        )
    summary, csv_path, summary_path = _save_reports(
        fold_results,
        args.report_root,
        float(config.METASTASIS_F_BETA),
    )
    pooled = summary["pooled_test"]["fusion"]
    print("\n===== Pooled locked test =====")
    print(
        f"Accuracy={pooled['accuracy']:.4f} | "
        f"Macro-F1={pooled['macro_f1']:.4f} | "
        f"Metastasis P/R/F{config.METASTASIS_F_BETA:g}="
        f"{pooled['metastasis_precision']:.4f}/"
        f"{pooled['metastasis_recall']:.4f}/"
        f"{pooled['metastasis_fbeta']:.4f}"
    )
    print(f"逐病例结果：{csv_path}")
    print(f"汇总结果：{summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--nnunet-results-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(1, K_FOLDS + 1)),
        choices=range(1, K_FOLDS + 1),
    )
    parser.add_argument("--rebuild-mask-cache", action="store_true")
    main(parser.parse_args())
