"""将项目 Fold 的 FLAIR 分割监督转换为 nnU-Net v2 数据集。

只读取指定 Fold 的 train/val，不读取 test。正常病例生成全零标签；异常病例仅在
具有可靠 FLAIR mask 时纳入，并将原 mask 二值化为 lesion=1。脚本同时保存与项目
划分一致的 ``splits_final.json``；nnU-Net 中的 fold 0 对应该项目的指定 Fold。
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from configs.config_utils import infer_data_dir, resolve_input_artifact_dir


SEQUENCE_ID = 3
SEQUENCE_NAME = "FLAIR"
DEFAULT_DATASET_ID = 501
DEFAULT_DATASET_NAME = "FLAIRLesion"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a binary FLAIR lesion dataset for nnU-Net v2 while "
            "preserving one project train/validation split."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Project data experiment root containing data/ and datasets/.",
    )
    parser.add_argument(
        "--nnunet-raw-root",
        required=True,
        help="Target nnUNet_raw directory.",
    )
    parser.add_argument("--fold", type=int, default=1, help="Project fold number.")
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=DEFAULT_DATASET_ID,
        help="nnU-Net dataset ID (default: 501).",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Alphanumeric nnU-Net dataset name.",
    )
    parser.add_argument(
        "--image-transfer",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to place imagesTr files (default: hardlink).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources and report counts without writing output.",
    )
    return parser.parse_args()


def validate_dataset_identity(dataset_id, dataset_name):
    if not 1 <= int(dataset_id) <= 999:
        raise ValueError("--dataset-id must lie between 1 and 999")
    if re.fullmatch(r"[A-Za-z0-9]+", dataset_name) is None:
        raise ValueError("--dataset-name must contain only letters and numbers")
    return f"Dataset{int(dataset_id):03d}_{dataset_name}"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(value, path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def format_case_id(value):
    case_id = str(value)
    return case_id.zfill(4) if case_id.isdigit() else case_id


def resolve_case_path(path_value, processed_data_root):
    path = Path(path_value)
    if path.is_file():
        return path.resolve()

    if len(path.parts) >= 3:
        candidate = processed_data_root.joinpath(*path.parts[-3:])
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Case image not found: {path_value}")


def resolve_direct_mask_path(case, processed_data_root):
    path_value = case.get("mask_path")
    if path_value is None:
        return None
    try:
        return resolve_case_path(path_value, processed_data_root)
    except FileNotFoundError:
        return None


def resolve_flair_mask(case, image_path, mask_index, processed_data_root):
    case_id = format_case_id(case["case_id"])
    if case.get("has_mask"):
        direct_path = resolve_direct_mask_path(case, processed_data_root)
        if direct_path is not None:
            return direct_path

    valid_sequences = {
        int(sequence_id) for sequence_id in mask_index.get(case_id, ())
    }
    if SEQUENCE_ID not in valid_sequences:
        return None

    mask_path = image_path.parent / f"case_{case_id}_{SEQUENCE_ID}_mask.nii.gz"
    if not mask_path.is_file():
        raise FileNotFoundError(
            f"mask_index lists a FLAIR mask but the file is missing: {mask_path}"
        )
    return mask_path.resolve()


def load_project_split(fold_root, split_name):
    split_path = fold_root / f"{split_name}.pt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Project split not found: {split_path}")
    data = torch.load(split_path, map_location="cpu", weights_only=False)
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Expected a cases list in {split_path}")
    return cases


def collect_supervised_cases(
    cases,
    split_name,
    processed_data_root,
    mask_index,
):
    included = []
    excluded_positive_ids = []
    seen_ids = set()

    for case in cases:
        case_id = format_case_id(case["case_id"])
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case ID in {split_name}: {case_id}")
        seen_ids.add(case_id)

        image_path = resolve_case_path(case["nii_path"], processed_data_root)
        label = int(case["label"])
        if label == 0:
            mask_path = None
            case_type = "normal_zero"
        else:
            mask_path = resolve_flair_mask(
                case,
                image_path,
                mask_index,
                processed_data_root,
            )
            if mask_path is None:
                excluded_positive_ids.append(case_id)
                continue
            case_type = "positive"

        included.append(
            {
                "case_id": case_id,
                "nnunet_id": f"case_{case_id}",
                "split": split_name,
                "class_label": label,
                "case_type": case_type,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    return included, excluded_positive_ids


def transfer_image(source, destination, mode):
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported image transfer mode: {mode}")


def copy_geometry(source_image, data):
    header = source_image.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(data.astype(np.uint8, copy=False), source_image.affine, header)

    qform, qform_code = source_image.get_qform(coded=True)
    sform, sform_code = source_image.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qform_code))
    if sform is not None:
        output.set_sform(sform, int(sform_code))
    return output


def write_binary_label(case, destination):
    image = nib.load(str(case["image_path"]))
    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3D image for {case['case_id']}, got {image.shape}"
        )

    if case["mask_path"] is None:
        binary = np.zeros(image.shape, dtype=np.uint8)
    else:
        mask = nib.load(str(case["mask_path"]))
        if mask.shape != image.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {case['case_id']}: "
                f"{image.shape} vs {mask.shape}"
            )
        if not np.allclose(mask.affine, image.affine, rtol=0.0, atol=1e-4):
            raise ValueError(
                f"Image/mask affine mismatch for {case['case_id']}"
            )
        binary = np.asarray(mask.dataobj) > 0
        if not np.any(binary):
            raise ValueError(f"Positive mask is empty for {case['case_id']}")

    nib.save(copy_geometry(image, binary), str(destination))
    return int(np.count_nonzero(binary))


def count_case_types(cases):
    positive = sum(case["case_type"] == "positive" for case in cases)
    normal = sum(case["case_type"] == "normal_zero" for case in cases)
    return positive, normal


def prepare_dataset(args):
    dataset_folder_name = validate_dataset_identity(
        args.dataset_id,
        args.dataset_name,
    )
    data_root = Path(args.data_root).resolve()
    processed_data_root = infer_data_dir(data_root)
    project_dataset_root = resolve_input_artifact_dir(data_root, "datasets")
    fold_root = (
        project_dataset_root
        / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}"
        / f"fold{args.fold}"
    )
    if not fold_root.is_dir():
        raise FileNotFoundError(f"FLAIR fold directory not found: {fold_root}")

    mask_index_path = processed_data_root / "mask_index.json"
    if not mask_index_path.is_file():
        raise FileNotFoundError(f"Mask index not found: {mask_index_path}")
    mask_index = load_json(mask_index_path)

    train_cases, excluded_train = collect_supervised_cases(
        load_project_split(fold_root, "train"),
        "train",
        processed_data_root,
        mask_index,
    )
    val_cases, excluded_val = collect_supervised_cases(
        load_project_split(fold_root, "val"),
        "val",
        processed_data_root,
        mask_index,
    )
    train_ids = {case["case_id"] for case in train_cases}
    val_ids = {case["case_id"] for case in val_cases}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(f"Train/validation case overlap: {overlap[:10]}")

    train_positive, train_normal = count_case_types(train_cases)
    val_positive, val_normal = count_case_types(val_cases)
    print(
        f"Project Fold {args.fold} supervised train: "
        f"positive={train_positive}, normal-zero={train_normal}"
    )
    print(
        f"Project Fold {args.fold} supervised val  : "
        f"positive={val_positive}, normal-zero={val_normal}"
    )
    print(
        "Excluded abnormal cases without reliable FLAIR mask: "
        f"train={len(excluded_train)}, val={len(excluded_val)}"
    )

    all_cases = train_cases + val_cases
    nnunet_raw_root = Path(args.nnunet_raw_root).resolve()
    output_root = nnunet_raw_root / dataset_folder_name
    print(f"nnU-Net dataset: {output_root}")
    print(
        f"Project Fold {args.fold} maps to nnU-Net fold 0 "
        f"({len(train_cases)} train / {len(val_cases)} val)"
    )
    if args.dry_run:
        print("Dry run complete; no output was written.")
        return

    if output_root.exists():
        raise FileExistsError(
            f"Target dataset already exists: {output_root}. "
            "Use a new dataset ID/name or remove the incomplete generated "
            "directory after checking it."
        )

    images_tr = output_root / "imagesTr"
    labels_tr = output_root / "labelsTr"
    images_ts = output_root / "imagesTs"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir()
    images_ts.mkdir()

    manifest_cases = []
    for index, case in enumerate(all_cases, start=1):
        image_destination = images_tr / f"{case['nnunet_id']}_0000.nii.gz"
        label_destination = labels_tr / f"{case['nnunet_id']}.nii.gz"
        transfer_image(case["image_path"], image_destination, args.image_transfer)
        foreground_voxels = write_binary_label(case, label_destination)
        manifest_cases.append(
            {
                "case_id": case["case_id"],
                "nnunet_id": case["nnunet_id"],
                "split": case["split"],
                "class_label": case["class_label"],
                "case_type": case["case_type"],
                "foreground_voxels": foreground_voxels,
                "source_image": str(case["image_path"]),
                "source_mask": (
                    str(case["mask_path"])
                    if case["mask_path"] is not None
                    else None
                ),
            }
        )
        if index % 100 == 0 or index == len(all_cases):
            print(f"Prepared {index}/{len(all_cases)} cases")

    dataset_json = {
        "channel_names": {"0": "FLAIR"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(all_cases),
        "file_ending": ".nii.gz",
    }
    splits_final = [
        {
            "train": [case["nnunet_id"] for case in train_cases],
            "val": [case["nnunet_id"] for case in val_cases],
        }
    ]
    manifest = {
        "dataset": dataset_folder_name,
        "project_data_root": str(data_root),
        "project_fold": int(args.fold),
        "nnunet_fold": 0,
        "sequence_id": SEQUENCE_ID,
        "sequence_name": SEQUENCE_NAME,
        "image_transfer": args.image_transfer,
        "test_data_read": False,
        "counts": {
            "train_positive": train_positive,
            "train_normal_zero": train_normal,
            "val_positive": val_positive,
            "val_normal_zero": val_normal,
            "excluded_train_abnormal_without_flair_mask": len(excluded_train),
            "excluded_val_abnormal_without_flair_mask": len(excluded_val),
        },
        "excluded_train_case_ids": excluded_train,
        "excluded_val_case_ids": excluded_val,
        "cases": manifest_cases,
    }
    save_json(dataset_json, output_root / "dataset.json")
    save_json(splits_final, output_root / "splits_final.json")
    save_json(manifest, output_root / "conversion_manifest.json")
    print("Dataset conversion complete.")
    print(
        "After preprocessing, copy splits_final.json to: "
        f"$nnUNet_preprocessed/{dataset_folder_name}/splits_final.json"
    )


def main():
    prepare_dataset(parse_args())


if __name__ == "__main__":
    main()
