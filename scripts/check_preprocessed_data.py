"""
检查预处理后的 NIfTI 数据质量。

读取显式 --data-root 下的 data，只检查文件，不修改数据。
默认快速检查内容包括：
- NIfTI shape 是否等于 TARGET_SHAPE 对应的 (X, Y, Z) 顺序
- spacing 是否等于 TARGET_SPACING
- 文件大小是否异常
- 同一 case 是否缺少序列

使用 --full-stats 时额外检查：
- NaN/Inf
- 全零或近乎全零
- 非零区域 bbox 是否异常小、是否贴边

用法：
    python -m scripts.check_preprocessed_data \
        --config dataxxx/preprocessing_config.py --data-root dataxxx
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from configs.config_utils import (
    PREPROCESS_CONFIG_FIELDS,
    load_python_config,
    resolve_input_artifact_dir,
)
from configs.global_config import (
    ALL_SEQUENCES,
    CLASS_NAMES,
)


def target_shape_to_nifti_shape(target_shape):
    """
    项目 TARGET_SHAPE 使用 SimpleITK array / torch 体数据顺序 (D, H, W)。
    nibabel 读取 NIfTI 时返回 (X, Y, Z)，对应 (W, H, D)。
    """
    d, h, w = target_shape
    return (w, h, d)


def parse_case_file(path: Path):
    """
    解析文件名 case_{case_id}_{seq_id}.nii.gz。
    mask 文件默认不进入这里。
    """
    name = path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        return None

    parts = stem.split("_")
    if len(parts) != 3 or parts[0] != "case":
        return None

    case_id, seq_id = parts[1], parts[2]
    if not seq_id.isdigit():
        return None

    return case_id, int(seq_id)


def parse_label(path: Path, data_root: Path):
    try:
        class_dir = path.relative_to(data_root).parts[0]
    except ValueError:
        return None, None

    if "_" not in class_dir:
        return None, class_dir

    label_text, class_name = class_dir.split("_", 1)
    if not label_text.isdigit():
        return None, class_dir

    return int(label_text), class_name


def format_tuple(values):
    return "x".join(str(v) for v in values)


def bbox_from_nonzero(data):
    nonzero = np.argwhere(data != 0)
    if nonzero.size == 0:
        return None

    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0)
    bbox_shape = maxs - mins + 1
    return mins, maxs, bbox_shape


def inspect_file(path: Path, data_root: Path, expected_shape, expected_spacing, atol, full_stats):
    reasons = []
    row = {
        "path": str(path),
        "file_size_mb": path.stat().st_size / (1024 * 1024),
    }

    parsed = parse_case_file(path)
    if parsed is None:
        row.update({"case_id": "", "seq_id": "", "sequence": ""})
        reasons.append("unrecognized_filename")
    else:
        case_id, seq_id = parsed
        row.update(
            {
                "case_id": case_id,
                "seq_id": seq_id,
                "sequence": ALL_SEQUENCES[seq_id - 1]
                if 1 <= seq_id <= len(ALL_SEQUENCES)
                else "",
            }
        )

    label_id, class_name = parse_label(path, data_root)
    row.update({"label": label_id if label_id is not None else "", "class_name": class_name or ""})

    try:
        img = nib.load(str(path))
        shape = tuple(int(x) for x in img.shape[:3])
        spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
        dtype = str(img.get_data_dtype())
    except Exception as e:
        row.update(
            {
                "shape": "",
                "spacing": "",
                "dtype": "",
                "min": "",
                "max": "",
                "mean": "",
                "std": "",
                "zero_ratio": "",
                "nonzero_bbox_shape": "",
                "touches_border": "",
                "reasons": f"load_failed:{type(e).__name__}:{e}",
            }
        )
        return row

    row.update(
        {
            "shape": format_tuple(shape),
            "spacing": format_tuple(f"{x:.6g}" for x in spacing),
            "dtype": dtype,
        }
    )

    if shape != expected_shape:
        reasons.append(f"bad_shape:{shape}")

    if any(abs(a - b) > atol for a, b in zip(spacing, expected_spacing)):
        reasons.append(f"bad_spacing:{spacing}")

    if row["file_size_mb"] <= 0:
        reasons.append("empty_file")
    elif row["file_size_mb"] < PREPROCESS_MIN_FILE_SIZE_MB:
        reasons.append(f"small_file:{row['file_size_mb']:.6f}MB")

    if not full_stats:
        row.update(
            {
                "min": "",
                "max": "",
                "mean": "",
                "std": "",
                "zero_ratio": "",
                "nonzero_bbox_shape": "",
                "touches_border": "",
                "reasons": ";".join(reasons),
            }
        )
        return row

    data = img.get_fdata(dtype=np.float32)

    finite_mask = np.isfinite(data)
    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())
    if nan_count:
        reasons.append(f"nan:{nan_count}")
    if inf_count:
        reasons.append(f"inf:{inf_count}")

    if finite_mask.any():
        finite_data = data[finite_mask]
        row.update(
            {
                "min": float(finite_data.min()),
                "max": float(finite_data.max()),
                "mean": float(finite_data.mean()),
                "std": float(finite_data.std()),
            }
        )
    else:
        row.update({"min": "", "max": "", "mean": "", "std": ""})
        reasons.append("no_finite_voxels")

    zero_ratio = float(np.mean(data == 0))
    row["zero_ratio"] = zero_ratio
    if zero_ratio >= PREPROCESS_MAX_ZERO_RATIO:
        reasons.append(f"mostly_zero:{zero_ratio:.6f}")

    bbox = bbox_from_nonzero(data)
    if bbox is None:
        row.update({"nonzero_bbox_shape": "", "touches_border": ""})
        reasons.append("all_zero")
    else:
        mins, maxs, bbox_shape = bbox
        touches_border = bool(np.any(mins == 0) or np.any(maxs == (np.asarray(shape) - 1)))
        row["nonzero_bbox_shape"] = format_tuple(int(x) for x in bbox_shape)
        row["touches_border"] = touches_border

        min_bbox_fraction = np.min(bbox_shape / np.asarray(expected_shape))
        if min_bbox_fraction < PREPROCESS_MIN_NONZERO_BBOX_FRACTION:
            reasons.append(f"tiny_nonzero_bbox:{tuple(int(x) for x in bbox_shape)}")
        if touches_border:
            reasons.append("nonzero_touches_border")

    row["reasons"] = ";".join(reasons)
    return row


def collect_files(data_root: Path, include_masks: bool):
    files = []
    for path in sorted(data_root.glob("*/*/*.nii*")):
        if path.name.endswith("_mask.nii.gz") and not include_masks:
            continue
        if parse_case_file(path) is None:
            continue
        files.append(path)
    return files


def summarize_rows(rows):
    anomalies = [row for row in rows if row["reasons"]]
    shape_counts = Counter(row["shape"] for row in rows)
    class_seq_counts = Counter(
        (row["class_name"], row["seq_id"]) for row in rows if row["class_name"] and row["seq_id"]
    )

    print(f"\n[SUMMARY]")
    print(f"  Files checked : {len(rows)}")
    print(f"  Anomalies     : {len(anomalies)}")

    print("\n[SHAPES]")
    for shape, count in shape_counts.most_common():
        print(f"  {shape}: {count}")

    print("\n[CLASS x SEQ COUNTS]")
    for class_name in CLASS_NAMES:
        counts = []
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1):
            counts.append(f"{seq_name}:{class_seq_counts.get((class_name, seq_id), 0)}")
        print(f"  {class_name}: " + " | ".join(counts))

    if anomalies:
        reason_counts = Counter()
        for row in anomalies:
            for reason in row["reasons"].split(";"):
                reason_counts[reason.split(":")[0]] += 1

        print("\n[ANOMALY REASONS]")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")

        print("\n[FIRST 30 ANOMALIES]")
        for row in anomalies[:30]:
            print(f"  {row['path']} | shape={row['shape']} | reasons={row['reasons']}")

    return anomalies


def summarize_missing_sequences(rows):
    case_to_seqs = defaultdict(set)
    case_to_label = {}

    for row in rows:
        case_id = row["case_id"]
        seq_id = row["seq_id"]
        if not case_id or not seq_id:
            continue
        case_to_seqs[case_id].add(int(seq_id))
        case_to_label[case_id] = row["class_name"]

    expected_seqs = set(range(1, len(ALL_SEQUENCES) + 1))
    incomplete = {
        case_id: sorted(expected_seqs - seqs)
        for case_id, seqs in case_to_seqs.items()
        if seqs != expected_seqs
    }

    print("\n[CASE COMPLETENESS]")
    print(f"  Cases seen         : {len(case_to_seqs)}")
    print(f"  Complete cases     : {len(case_to_seqs) - len(incomplete)}")
    print(f"  Incomplete cases   : {len(incomplete)}")

    if incomplete:
        label_counts = Counter(case_to_label.get(case_id, "") for case_id in incomplete)
        print("  Incomplete by class: " + ", ".join(f"{k}:{v}" for k, v in label_counts.items()))
        print("\n[FIRST 30 INCOMPLETE CASES]")
        for case_id, missing in list(sorted(incomplete.items()))[:30]:
            missing_names = [ALL_SEQUENCES[i - 1] for i in missing]
            print(f"  case_{case_id} | class={case_to_label.get(case_id, '')} | missing={missing_names}")

    return incomplete, case_to_label


def write_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "case_id",
        "seq_id",
        "sequence",
        "label",
        "class_name",
        "file_size_mb",
        "shape",
        "spacing",
        "dtype",
        "min",
        "max",
        "mean",
        "std",
        "zero_ratio",
        "nonzero_bbox_shape",
        "touches_border",
        "reasons",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[CSV] Saved per-file QC table to: {csv_path}")


def write_incomplete_csv(incomplete, case_to_label, csv_path: Path):
    if not incomplete:
        return

    out_path = csv_path.with_name(f"{csv_path.stem}_incomplete_cases{csv_path.suffix}")
    fieldnames = ["case_id", "class_name", "missing_seq_ids", "missing_sequences"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case_id, missing in sorted(incomplete.items()):
            writer.writerow(
                {
                    "case_id": case_id,
                    "class_name": case_to_label.get(case_id, ""),
                    "missing_seq_ids": ";".join(str(i) for i in missing),
                    "missing_sequences": ";".join(ALL_SEQUENCES[i - 1] for i in missing),
                }
            )
    print(f"[CSV] Saved incomplete-case table to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Check preprocessed NIfTI data quality.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the preprocessing_config.py used to produce the data",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Experiment root containing data, or the data directory itself",
    )
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--include-masks", action="store_true")
    parser.add_argument("--spacing-atol", type=float, default=1e-3)
    parser.add_argument(
        "--expected-shape",
        type=int,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Expected NIfTI shape in nibabel order (X Y Z). "
            "Default is derived from TARGET_SHAPE(D,H,W) as (W,H,D)."
        ),
    )
    parser.add_argument(
        "--full-stats",
        action="store_true",
        help="Read voxel data to compute NaN/Inf, zero ratio, intensity stats and nonzero bbox. Slower.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if anomalies exist.")
    args = parser.parse_args()

    config = load_python_config(args.config, PREPROCESS_CONFIG_FIELDS)
    for name in PREPROCESS_CONFIG_FIELDS:
        globals()[name] = getattr(config, name)

    data_root = resolve_input_artifact_dir(args.data_root, "data")
    expected_shape = (
        tuple(args.expected_shape)
        if args.expected_shape is not None
        else target_shape_to_nifti_shape(tuple(TARGET_SHAPE))
    )
    expected_spacing = tuple(TARGET_SPACING)

    print(f"[CONFIG] TARGET_SHAPE(D,H,W): {tuple(TARGET_SHAPE)}")
    print(f"[CONFIG] Expected NIfTI shape(X,Y,Z): {expected_shape}")
    print(f"[CONFIG] Expected spacing(X,Y,Z): {expected_spacing}")
    print(f"[CONFIG] preprocessing config: {config.__config_path__}")
    print(f"[CONFIG] data root: {data_root}")

    files = collect_files(data_root, include_masks=args.include_masks)
    rows = [
        inspect_file(
            path,
            data_root=data_root,
            expected_shape=expected_shape,
            expected_spacing=expected_spacing,
            atol=args.spacing_atol,
            full_stats=args.full_stats,
        )
        for path in files
    ]

    anomalies = summarize_rows(rows)
    incomplete, case_to_label = summarize_missing_sequences(rows)

    if args.csv_path:
        csv_path = Path(args.csv_path)
        write_csv(rows, csv_path)
        write_incomplete_csv(incomplete, case_to_label, csv_path)

    if args.strict and (anomalies or incomplete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
