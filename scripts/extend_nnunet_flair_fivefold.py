"""把单 Fold Dataset501 安全扩展为与项目 Fold 1-5 对齐的 nnU-Net splits。"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from configs.config_utils import infer_data_dir, resolve_input_artifact_dir
from scripts.prepare_nnunet_flair_dataset import (
    SEQUENCE_ID,
    SEQUENCE_NAME,
    collect_supervised_cases,
    load_json,
    load_project_split,
    transfer_image,
    validate_dataset_identity,
    write_binary_label,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extend an existing Fold-1-only Dataset501 into five project-aligned "
            "nnU-Net splits without changing the existing fold 0 definition."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--nnunet-raw-root", required=True)
    parser.add_argument("--nnunet-preprocessed-root", required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="FLAIRLesion")
    parser.add_argument(
        "--image-transfer",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write missing raw cases and five-fold metadata. Omit for dry-run.",
    )
    return parser.parse_args()


def save_json_atomic(value, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def case_signature(case):
    return (
        int(case["class_label"]),
        str(case["case_type"]),
        str(Path(case["image_path"]).resolve()),
        (
            str(Path(case["mask_path"]).resolve())
            if case["mask_path"] is not None
            else None
        ),
    )


def collect_fivefold_cases(data_root):
    processed_data_root = infer_data_dir(data_root)
    if processed_data_root is None:
        raise FileNotFoundError(f"Processed data directory not found: {data_root}")
    dataset_root = resolve_input_artifact_dir(data_root, "datasets")
    mask_index_path = processed_data_root / "mask_index.json"
    if not mask_index_path.is_file():
        raise FileNotFoundError(f"Mask index not found: {mask_index_path}")
    mask_index = load_json(mask_index_path)

    cases_by_id = {}
    fold_splits = []
    fold_counts = []
    for project_fold in range(1, 6):
        fold_root = (
            dataset_root
            / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}"
            / f"fold{project_fold}"
        )
        split_cases = {}
        excluded_counts = {}
        for split_name in ("train", "val", "test"):
            included, excluded = collect_supervised_cases(
                load_project_split(fold_root, split_name),
                split_name,
                processed_data_root,
                mask_index,
            )
            split_cases[split_name] = included
            excluded_counts[split_name] = len(excluded)
            for case in included:
                case_id = case["nnunet_id"]
                existing = cases_by_id.get(case_id)
                if existing is not None and case_signature(existing) != case_signature(
                    case
                ):
                    raise ValueError(
                        f"Inconsistent metadata across folds for {case_id}"
                    )
                cases_by_id[case_id] = case

        train_ids = [case["nnunet_id"] for case in split_cases["train"]]
        val_ids = [case["nnunet_id"] for case in split_cases["val"]]
        test_ids = [case["nnunet_id"] for case in split_cases["test"]]
        if set(train_ids) & set(val_ids):
            raise ValueError(f"Project Fold {project_fold} train/val overlap")
        fold_splits.append({"train": train_ids, "val": val_ids})
        fold_counts.append(
            {
                "project_fold": project_fold,
                "nnunet_fold": project_fold - 1,
                "train": len(train_ids),
                "val": len(val_ids),
                "held_out_test": len(test_ids),
                "excluded_abnormal_without_mask": excluded_counts,
            }
        )

    expected_ids = set(cases_by_id)
    for project_fold, (split, counts) in enumerate(
        zip(fold_splits, fold_counts),
        start=1,
    ):
        fold_root = (
            dataset_root
            / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}"
            / f"fold{project_fold}"
        )
        test_cases, _ = collect_supervised_cases(
            load_project_split(fold_root, "test"),
            "test",
            processed_data_root,
            mask_index,
        )
        covered = set(split["train"]) | set(split["val"]) | {
            case["nnunet_id"] for case in test_cases
        }
        if covered != expected_ids:
            raise ValueError(
                f"Project Fold {project_fold} does not partition the common "
                f"supervised case set: {len(covered)} vs {len(expected_ids)}"
            )
        if len(split["train"]) + len(split["val"]) + counts["held_out_test"] != len(
            expected_ids
        ):
            raise ValueError(f"Project Fold {project_fold} split counts are invalid")
    return cases_by_id, fold_splits, fold_counts


def raw_case_ids(images_dir, labels_dir):
    image_suffix = "_0000.nii.gz"
    label_suffix = ".nii.gz"
    image_ids = {
        path.name[: -len(image_suffix)]
        for path in images_dir.glob(f"*{image_suffix}")
    }
    label_ids = {
        path.name[: -len(label_suffix)]
        for path in labels_dir.glob(f"*{label_suffix}")
    }
    if image_ids != label_ids:
        raise ValueError(
            "Existing nnU-Net raw images/labels differ: "
            f"images-only={len(image_ids - label_ids)}, "
            f"labels-only={len(label_ids - image_ids)}"
        )
    return image_ids


def backup_metadata(raw_dataset_root, preprocessed_dataset_root):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = raw_dataset_root.parent.parent / "metadata-backups" / timestamp
    backup_root.mkdir(parents=True, exist_ok=False)
    candidates = (
        raw_dataset_root / "dataset.json",
        raw_dataset_root / "splits_final.json",
        raw_dataset_root / "conversion_manifest.json",
        preprocessed_dataset_root / "dataset.json",
        preprocessed_dataset_root / "splits_final.json",
    )
    for path in candidates:
        if path.is_file():
            prefix = (
                "preprocessed" if path.parent == preprocessed_dataset_root else "raw"
            )
            relative_name = f"{prefix}_{path.name}"
            shutil.copy2(path, backup_root / relative_name)
    return backup_root


def extend_dataset(args):
    dataset_folder = validate_dataset_identity(args.dataset_id, args.dataset_name)
    raw_dataset_root = Path(args.nnunet_raw_root).resolve() / dataset_folder
    preprocessed_dataset_root = (
        Path(args.nnunet_preprocessed_root).resolve() / dataset_folder
    )
    images_dir = raw_dataset_root / "imagesTr"
    labels_dir = raw_dataset_root / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Existing raw Dataset501 not found: {raw_dataset_root}")
    if not preprocessed_dataset_root.is_dir():
        raise FileNotFoundError(
            f"Existing preprocessed Dataset501 not found: {preprocessed_dataset_root}"
        )

    cases_by_id, splits, fold_counts = collect_fivefold_cases(
        Path(args.data_root).resolve()
    )
    expected_ids = set(cases_by_id)
    existing_ids = raw_case_ids(images_dir, labels_dir)
    unexpected_ids = existing_ids - expected_ids
    if unexpected_ids:
        raise ValueError(
            f"Raw dataset has {len(unexpected_ids)} unexpected cases: "
            f"{sorted(unexpected_ids)[:10]}"
        )

    raw_split_path = raw_dataset_root / "splits_final.json"
    preprocessed_split_path = preprocessed_dataset_root / "splits_final.json"
    existing_splits = load_json(raw_split_path)
    if not existing_splits:
        raise ValueError("Existing raw splits_final.json is empty")
    if existing_splits[0] != splits[0]:
        raise ValueError(
            "Refusing extension because the existing nnU-Net fold 0 no longer "
            "matches project Fold 1"
        )
    if preprocessed_split_path.is_file():
        preprocessed_splits = load_json(preprocessed_split_path)
        if not preprocessed_splits or preprocessed_splits[0] != splits[0]:
            raise ValueError(
                "Existing preprocessed fold 0 does not match project Fold 1"
            )

    dataset_json_path = raw_dataset_root / "dataset.json"
    preprocessed_dataset_json_path = preprocessed_dataset_root / "dataset.json"
    dataset_json = load_json(dataset_json_path)
    if int(dataset_json.get("numTraining", -1)) != len(existing_ids):
        raise ValueError(
            "dataset.json numTraining does not match existing raw cases: "
            f"{dataset_json.get('numTraining')} vs {len(existing_ids)}"
        )
    preprocessed_dataset_json = load_json(preprocessed_dataset_json_path)
    if int(preprocessed_dataset_json.get("numTraining", -1)) != len(existing_ids):
        raise ValueError(
            "Preprocessed dataset.json numTraining does not match existing raw "
            f"cases: {preprocessed_dataset_json.get('numTraining')} vs "
            f"{len(existing_ids)}"
        )
    comparable_raw_json = dict(dataset_json)
    comparable_preprocessed_json = dict(preprocessed_dataset_json)
    comparable_raw_json.pop("numTraining", None)
    comparable_preprocessed_json.pop("numTraining", None)
    if comparable_raw_json != comparable_preprocessed_json:
        raise ValueError(
            "Raw and preprocessed dataset.json differ beyond numTraining; "
            "refusing to guess which metadata should be retained"
        )
    missing_ids = sorted(expected_ids - existing_ids)

    print(f"Dataset          : {raw_dataset_root}")
    print(f"Existing raw     : {len(existing_ids)} cases")
    print(f"Five-fold union  : {len(expected_ids)} cases")
    print(f"Missing raw      : {len(missing_ids)} cases")
    for counts in fold_counts:
        print(
            f"Project Fold {counts['project_fold']} -> nnU-Net fold "
            f"{counts['nnunet_fold']}: train={counts['train']}, "
            f"val={counts['val']}, held-out test={counts['held_out_test']}"
        )
    if not args.apply:
        print("Dry run complete; pass --apply to extend Dataset501.")
        return

    backup_root = backup_metadata(raw_dataset_root, preprocessed_dataset_root)
    print(f"Metadata backup  : {backup_root}")
    added_paths = []
    try:
        for index, case_id in enumerate(missing_ids, start=1):
            case = cases_by_id[case_id]
            image_destination = images_dir / f"{case_id}_0000.nii.gz"
            label_destination = labels_dir / f"{case_id}.nii.gz"
            added_paths.extend((image_destination, label_destination))
            transfer_image(
                case["image_path"],
                image_destination,
                args.image_transfer,
            )
            write_binary_label(case, label_destination)
            if index % 25 == 0 or index == len(missing_ids):
                print(f"Added {index}/{len(missing_ids)} missing cases")
    except Exception:
        for path in reversed(added_paths):
            if path.exists() or path.is_symlink():
                path.unlink()
        raise

    dataset_json["numTraining"] = len(expected_ids)
    manifest = {
        "dataset": dataset_folder,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_data_root": str(Path(args.data_root).resolve()),
        "project_folds": [1, 2, 3, 4, 5],
        "nnunet_folds": [0, 1, 2, 3, 4],
        "test_data_used_for_training_or_validation": False,
        "test_data_note": (
            "Test split IDs are inventoried only to verify each fold partitions "
            "the common supervised set; they are absent from that fold's "
            "nnU-Net train and val lists."
        ),
        "total_supervised_cases": len(expected_ids),
        "previous_raw_cases": len(existing_ids),
        "added_raw_cases": len(missing_ids),
        "fold_counts": fold_counts,
    }
    save_json_atomic(dataset_json, dataset_json_path)
    save_json_atomic(dataset_json, preprocessed_dataset_json_path)
    save_json_atomic(splits, raw_split_path)
    save_json_atomic(splits, preprocessed_split_path)
    save_json_atomic(manifest, raw_dataset_root / "fivefold_manifest.json")
    print("Five-fold raw extension complete.")
    print(
        "Required next step: rerun 3d_fullres preprocessing so the added cases "
        "receive preprocessed .b2nd/.pkl files before training folds 1-4."
    )


def main():
    extend_dataset(parse_args())


if __name__ == "__main__":
    main()
