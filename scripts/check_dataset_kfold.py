"""检查 K-Fold 数据集的类别分布。"""

import argparse
import warnings
from collections import Counter
from pathlib import Path

import torch

from configs.config_utils import resolve_input_artifact_dir
from configs.global_config import (
    ALL_SEQUENCES,
    CLASS_NAMES,
    K_FOLDS,
    NUM_CLASSES,
)


def extract_labels(dataset_dict):
    """兼容当前轻量 cases schema 和历史 labels tensor schema。"""
    if "cases" in dataset_dict:
        return [int(case["label"]) for case in dataset_dict["cases"]]
    if "labels" in dataset_dict:
        labels = dataset_dict["labels"]
        return labels.tolist() if hasattr(labels, "tolist") else list(labels)
    raise KeyError("Neither 'cases' nor 'labels' exists in dataset file")


def check_sequence_distribution(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()

    print(f"\n{'=' * 60}")
    print(f"Checking sequence dataset: {dataset_dir.name}")
    print(f"Path: {dataset_dir}")
    print(f"Classes: {NUM_CLASSES} ({CLASS_NAMES})")
    print(f"{'=' * 60}")

    if not dataset_dir.is_dir():
        print(f"  [Warning] Directory not found: {dataset_dir}")
        return

    for fold_idx in range(1, K_FOLDS + 1):
        fold_name = f"fold{fold_idx}"
        fold_dir = dataset_dir / fold_name
        print(f"\n--- {fold_name} ---")
        if not fold_dir.is_dir():
            print(f"  [Warning] {fold_name} dir not found. Skipping.")
            continue

        stats = {name: Counter() for name in ("train", "val", "test")}
        for split_name in stats:
            pt_path = fold_dir / f"{split_name}.pt"
            if not pt_path.is_file():
                print(f"  [Warning] {split_name}.pt not found.")
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dataset_dict = torch.load(
                        pt_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                stats[split_name].update(extract_labels(dataset_dict))
            except Exception as exc:
                print(f"  [Error] Failed to load {pt_path.name}: {exc}")

        header_labels = [
            f"{class_name}({label_id})"
            for label_id, class_name in enumerate(CLASS_NAMES)
        ]
        header = (
            f"  {'Split':<8} | "
            + " | ".join(f"{label:<15}" for label in header_labels)
            + f" | {'Total':<8}"
        )
        print(header)
        print(f"  {'-' * len(header)}")

        for split_name in ("train", "val", "test"):
            counts = [
                stats[split_name].get(label_id, 0)
                for label_id in range(NUM_CLASSES)
            ]
            row = " | ".join(f"{count:<15}" for count in counts)
            print(f"  {split_name:<8} | {row} | {sum(counts):<8}")


def main():
    parser = argparse.ArgumentParser(
        description="Check class distributions in K-Fold .pt datasets"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Experiment root containing datasets, or the datasets directory itself",
    )
    parser.add_argument(
        "--seq",
        type=int,
        choices=range(1, len(ALL_SEQUENCES) + 1),
        default=None,
        help="Only check one sequence; default checks all sequences",
    )
    args = parser.parse_args()

    dataset_root = resolve_input_artifact_dir(args.data_root, "datasets")
    seq_ids = [args.seq] if args.seq is not None else range(1, len(ALL_SEQUENCES) + 1)

    print(f"Dataset root: {dataset_root}")
    for seq_id in seq_ids:
        seq_name = ALL_SEQUENCES[seq_id - 1]
        check_sequence_distribution(dataset_root / f"seq{seq_id}_{seq_name}")


if __name__ == "__main__":
    main()
