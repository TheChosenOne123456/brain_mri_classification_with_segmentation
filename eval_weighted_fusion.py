"""
Weighted heterogeneous late-fusion evaluation.

Use this script to test alternative T1/T2/FLAIR soft-voting weights while
checking that the original internal k-fold performance does not drop too much.
It can also reuse external_eval.py diagnostic CSV files for fast external
analysis without rerunning MRI preprocessing or model inference.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from configs.global_config import ALL_SEQUENCES, CLASS_NAMES, K_FOLDS, NUM_CLASSES
from runtime_defaults import (
    BATCH_SIZE,
    CKPT_DIRS,
    DATASET_DIRS,
    DEVICE,
    INFERENCE_OUTPUT_DIR,
    NUM_WORKERS,
)
from models.FoundationModel import FoundationModel
from models.FoundationModel_ori import FoundationModel as FoundationModel_ori
from utils.train_and_test import load_pt_dataset, set_seed
from configs.global_config import SEED

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


DEFAULT_INTERNAL_BASELINE_ACC = 0.9112
RUNTIME_DEVICE = DEVICE


def resolve_runtime_device(requested_device):
    device = requested_device or DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA is configured but not available. Falling back to CPU for this evaluation.")
        return "cpu"
    return device


class MultiSequenceDataset(Dataset):
    def __init__(self, datasets_list):
        self.datasets = datasets_list
        self.labels = datasets_list[0].labels

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        xs = []
        for ds in self.datasets:
            x, y, _mask_tensor, _has_mask_flag, _case_id = ds[idx]
            xs.append(x)

        _, y, _, _, case_id = self.datasets[0][idx]
        return xs, y, case_id


def normalize_weights(weights):
    weights = np.array(weights, dtype=np.float32)
    if weights.shape != (3,):
        raise ValueError("--weights must contain exactly three numbers: T1 T2 FLAIR")
    if np.any(weights < 0):
        raise ValueError("--weights must be non-negative")

    total = float(weights.sum())
    if total <= 0:
        raise ValueError("--weights sum must be > 0")
    return weights / total


def format_weights(weights):
    return f"T1={weights[0]:.3f}, T2={weights[1]:.3f}, FLAIR={weights[2]:.3f}"


def load_models_for_fold(fold_idx):
    models = []
    for seq_idx, seq_name in enumerate(ALL_SEQUENCES):
        if seq_name == "FLAIR":
            model_name = "FoundationModel"
            model_class = FoundationModel
        else:
            model_name = "FoundationModel_ori"
            model_class = FoundationModel_ori

        ckpt_path = CKPT_DIRS[seq_idx] / model_name / f"fold{fold_idx}_model_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {seq_name}: {ckpt_path}")

        try:
            model = model_class(num_classes=NUM_CLASSES, in_channels=1)
        except TypeError:
            model = model_class(num_classes=NUM_CLASSES)
        model = model.to(RUNTIME_DEVICE)
        checkpoint = torch.load(ckpt_path, map_location=RUNTIME_DEVICE)
        model.load_state_dict(checkpoint["model_state"])

        if RUNTIME_DEVICE == "cuda" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.eval()
        models.append(model)

    return models


def load_test_dataset_for_fold(fold_idx):
    test_sets = []
    for seq_idx, seq_name in enumerate(ALL_SEQUENCES):
        dataset_dir = DATASET_DIRS[seq_idx] / f"fold{fold_idx}"
        test_path = dataset_dir / "test.pt"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test dataset for {seq_name}: {test_path}")
        test_sets.append(load_pt_dataset(test_path))

    return MultiSequenceDataset(test_sets)


def compute_metrics(labels, preds):
    labels = list(labels)
    preds = list(preds)
    return {
        "acc": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "cm": confusion_matrix(labels, preds, labels=range(NUM_CLASSES)),
        "per_class_recall": recall_score(labels, preds, labels=range(NUM_CLASSES), average=None, zero_division=0),
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy      : {metrics['acc']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1-score      : {metrics['f1']:.4f}")
    print("Per-class recall:")
    for class_name, recall_value in zip(CLASS_NAMES, metrics["per_class_recall"]):
        print(f"  {class_name:<15}: {recall_value:.4f}")
    print("Confusion Matrix:")
    print(metrics["cm"])


def evaluate_internal_fold(fold_idx, weights):
    print(f"\n{'=' * 20} Internal Fold {fold_idx} {'=' * 20}")
    print(f"Weights: {format_weights(weights)}")

    t0 = time.time()
    test_set = load_test_dataset_for_fold(fold_idx)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Loaded test data in {time.time() - t0:.1f}s")

    models = load_models_for_fold(fold_idx)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for xs, y, _case_ids in test_loader:
            y = y.to(RUNTIME_DEVICE)
            weighted_prob = None

            for seq_idx, x in enumerate(xs):
                x = x.to(RUNTIME_DEVICE)
                logits = models[seq_idx](x)
                prob = F.softmax(logits, dim=1)
                weighted = prob * float(weights[seq_idx])
                weighted_prob = weighted if weighted_prob is None else weighted_prob + weighted

            preds = weighted_prob.argmax(dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    del models
    if RUNTIME_DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return compute_metrics(all_labels, all_preds)


def evaluate_internal(args, weights):
    folds = [args.fold] if args.fold is not None else list(range(1, K_FOLDS + 1))
    fold_metrics = []

    for fold_idx in folds:
        fold_metric = evaluate_internal_fold(fold_idx, weights)
        fold_metrics.append(fold_metric)
        print_metrics(f"Internal Fold {fold_idx} Results", fold_metric)

    if len(fold_metrics) == 1:
        avg_acc = fold_metrics[0]["acc"]
        avg_f1 = fold_metrics[0]["f1"]
        print("\nSingle-fold internal check complete.")
    else:
        avg_acc = float(np.mean([m["acc"] for m in fold_metrics]))
        std_acc = float(np.std([m["acc"] for m in fold_metrics]))
        avg_precision = float(np.mean([m["precision"] for m in fold_metrics]))
        avg_recall = float(np.mean([m["recall"] for m in fold_metrics]))
        avg_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
        std_f1 = float(np.std([m["f1"] for m in fold_metrics]))

        print("\n" + "=" * 50)
        print(f"Internal {len(fold_metrics)}-Fold Weighted Fusion Report")
        print("=" * 50)
        print(f"Weights       : {format_weights(weights)}")
        print(f"Accuracy      : {avg_acc:.4f} ±{std_acc:.4f}")
        print(f"Precision     : {avg_precision:.4f}")
        print(f"Recall        : {avg_recall:.4f}")
        print(f"F1-score      : {avg_f1:.4f} ±{std_f1:.4f}")

    acc_drop = args.internal_baseline_acc - avg_acc
    print("\nInternal baseline guard:")
    print(f"  Baseline acc : {args.internal_baseline_acc:.4f}")
    print(f"  Weighted acc : {avg_acc:.4f}")
    print(f"  Acc drop     : {acc_drop:.4f}")
    if acc_drop > args.max_internal_acc_drop:
        print(f"  Status       : WARNING, drop exceeds {args.max_internal_acc_drop:.4f}")
    else:
        print(f"  Status       : OK, drop within {args.max_internal_acc_drop:.4f}")

    return {"acc": avg_acc, "f1": avg_f1, "acc_drop": acc_drop}


def discover_external_csvs():
    output_dir = INFERENCE_OUTPUT_DIR
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("external_eval_*_foldsall.csv"))


def read_external_sequence_probs(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_case = defaultdict(lambda: defaultdict(list))
    labels = {}

    for row in rows:
        if row["level"] != "sequence":
            continue

        case_name = row["case"]
        seq_name = row["seq_name"]
        probs = np.array([float(row[f"prob_{class_name}"]) for class_name in CLASS_NAMES], dtype=np.float32)
        by_case[case_name][seq_name].append(probs)
        labels[case_name] = int(row["gt_label"])

    cases = []
    for case_name, seq_probs in by_case.items():
        if not all(seq_name in seq_probs for seq_name in ALL_SEQUENCES):
            continue

        avg_seq_probs = {
            seq_name: np.mean(seq_probs[seq_name], axis=0)
            for seq_name in ALL_SEQUENCES
        }
        cases.append((case_name, labels[case_name], avg_seq_probs))

    return cases


def evaluate_external_csv(csv_path, weights):
    cases = read_external_sequence_probs(csv_path)
    if not cases:
        print(f"\n[Warning] No usable sequence rows found in {csv_path}")
        return None

    labels = []
    preds = []

    for _case_name, gt_label, seq_probs in cases:
        weighted_prob = sum(
            float(weights[seq_idx]) * seq_probs[seq_name]
            for seq_idx, seq_name in enumerate(ALL_SEQUENCES)
        )
        pred = int(np.argmax(weighted_prob))
        labels.append(gt_label)
        preds.append(pred)

    metrics = compute_metrics(labels, preds)
    metrics["n"] = len(cases)
    metrics["gt_counts"] = Counter(CLASS_NAMES[label] for label in labels)
    metrics["pred_counts"] = Counter(CLASS_NAMES[pred] for pred in preds)
    return metrics


def evaluate_external(args, weights):
    csv_paths = [Path(p) for p in args.external_csvs] if args.external_csvs else discover_external_csvs()
    if not csv_paths:
        print("\nNo external diagnostic CSV files found; skipping external CSV evaluation.")
        return []

    results = []
    all_labels = []
    all_preds = []

    print("\n" + "=" * 50)
    print("External CSV Weighted Fusion Report")
    print("=" * 50)
    print(f"Weights: {format_weights(weights)}")

    for csv_path in csv_paths:
        result = evaluate_external_csv(csv_path, weights)
        if result is None:
            continue

        print_metrics(f"External CSV: {csv_path.name}", result)
        print(f"Samples       : {result['n']}")
        print(f"GT counts     : {dict(result['gt_counts'])}")
        print(f"Pred counts   : {dict(result['pred_counts'])}")
        results.append((csv_path, result))

        cases = read_external_sequence_probs(csv_path)
        for _case_name, gt_label, seq_probs in cases:
            weighted_prob = sum(
                float(weights[seq_idx]) * seq_probs[seq_name]
                for seq_idx, seq_name in enumerate(ALL_SEQUENCES)
            )
            all_labels.append(gt_label)
            all_preds.append(int(np.argmax(weighted_prob)))

    if all_labels:
        combined = compute_metrics(all_labels, all_preds)
        print_metrics("Combined External CSV Results", combined)

    return results


def main(args):
    global RUNTIME_DEVICE
    set_seed(SEED)
    RUNTIME_DEVICE = resolve_runtime_device(args.device)
    weights = normalize_weights(args.weights)

    print("\n>>> Weighted Heterogeneous Late Fusion Evaluation <<<")
    print(f"Weights: {format_weights(weights)}")
    print(f"Device : {RUNTIME_DEVICE}")

    if not args.skip_internal:
        evaluate_internal(args, weights)

    if not args.skip_external:
        evaluate_external(args, weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate weighted heterogeneous late fusion.")
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        default=[0.1, 0.7, 0.2],
        metavar=("T1", "T2", "FLAIR"),
        help="Fusion weights for T1, T2, and FLAIR. Defaults to方案1: 0.1 0.7 0.2.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, K_FOLDS + 1),
        help=f"Specific internal fold to evaluate. If omitted, evaluate all {K_FOLDS} folds.",
    )
    parser.add_argument(
        "--external_csvs",
        nargs="*",
        default=None,
        help="Optional external diagnostic CSV files. If omitted, auto-discover them in the default training experiment's inference_outputs.",
    )
    parser.add_argument(
        "--skip_internal",
        action="store_true",
        help="Skip internal k-fold evaluation.",
    )
    parser.add_argument(
        "--skip_external",
        action="store_true",
        help="Skip external CSV evaluation.",
    )
    parser.add_argument(
        "--internal_baseline_acc",
        type=float,
        default=DEFAULT_INTERNAL_BASELINE_ACC,
        help="Baseline internal mean accuracy used for drop check.",
    )
    parser.add_argument(
        "--max_internal_acc_drop",
        type=float,
        default=0.02,
        help="Warn if weighted internal mean accuracy drops by more than this value.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Evaluation device. Defaults to the migrated training experiment config, with automatic CPU fallback if CUDA is unavailable.",
    )
    main(parser.parse_args())
