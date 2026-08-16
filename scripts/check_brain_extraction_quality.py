"""检查预处理后脑提取前景的几何质量，并生成少量原图叠加图。"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from configs.config_utils import (
    PREPROCESS_CONFIG_FIELDS,
    load_python_config,
    resolve_input_artifact_dir,
)
from configs.global_config import ALL_SEQUENCES
from scripts.check_preprocessed_data import parse_case_file, parse_label
from scripts.preprocess_data import get_preprocess_meta_path
from utils.resample import resample_image
from utils.spatial import apply_crop_or_pad_from_meta


IMAGE_FIELDS = [
    "case_id",
    "label",
    "class_name",
    "seq_id",
    "sequence",
    "path",
    "brain_extractor",
    "foreground_voxels",
    "foreground_fraction",
    "foreground_volume_ml",
    "largest_component_fraction",
    "component_count",
    "bbox_shape_zyx",
    "bbox_fraction_zyx",
    "centroid_fraction_zyx",
    "nonempty_slices_zyx",
    "touches_z_low",
    "touches_z_high",
    "touches_y_low",
    "touches_y_high",
    "touches_x_low",
    "touches_x_high",
    "lesion_voxels",
    "lesion_retention",
    "flags",
]

CASE_FIELDS = [
    "case_id",
    "label",
    "class_name",
    "sequence_count",
    "volume_ratio_max_min",
    "dice_t1_t2",
    "dice_t1_flair",
    "dice_t2_flair",
    "min_pairwise_dice",
    "flags",
]


def format_values(values, precision=6):
    return "x".join(f"{float(value):.{precision}g}" for value in values)


def load_meta(image_path: Path):
    meta_path = get_preprocess_meta_path(image_path)
    if not meta_path.is_file():
        return {}, meta_path
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f), meta_path


def lesion_mask_path(image_path: Path):
    name = image_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = image_path.stem
    return image_path.with_name(f"{stem}_mask.nii.gz")


def foreground_bbox(mask):
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    return minimum, maximum, maximum - minimum + 1, coordinates.mean(axis=0)


def connected_component_metrics(mask):
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0, 0.0
    sizes = np.bincount(labels.reshape(-1))[1:]
    return int(count), float(sizes.max() / max(mask.sum(), 1))


def inspect_image(image_path: Path, data_root: Path, thresholds):
    parsed = parse_case_file(image_path)
    if parsed is None:
        raise ValueError(f"Unrecognized preprocessed image filename: {image_path}")
    case_id, seq_id = parsed
    label, class_name = parse_label(image_path, data_root)
    image = sitk.ReadImage(str(image_path))
    data = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    foreground = np.isfinite(data) & (data != 0)
    foreground_voxels = int(foreground.sum())
    foreground_fraction = float(foreground.mean())
    voxel_volume = float(np.prod(image.GetSpacing()))
    foreground_volume_ml = foreground_voxels * voxel_volume / 1000.0
    component_count, largest_component_fraction = connected_component_metrics(foreground)
    meta, meta_path = load_meta(image_path)
    flags = set()

    if not meta:
        flags.add("missing_preprocess_meta")
    if foreground_voxels == 0:
        flags.add("empty_foreground")
    if largest_component_fraction < thresholds["min_largest_component_fraction"]:
        flags.add("fragmented_foreground")

    bbox = foreground_bbox(foreground)
    if bbox is None:
        minimum = maximum = bbox_shape = np.zeros(3, dtype=int)
        centroid = np.zeros(3, dtype=float)
    else:
        minimum, maximum, bbox_shape, centroid = bbox
    shape = np.asarray(data.shape, dtype=float)
    bbox_fraction = bbox_shape / np.maximum(shape, 1)
    centroid_fraction = centroid / np.maximum(shape - 1, 1)
    touches = {
        "z_low": bool(minimum[0] == 0) if bbox is not None else False,
        "z_high": bool(maximum[0] == data.shape[0] - 1) if bbox is not None else False,
        "y_low": bool(minimum[1] == 0) if bbox is not None else False,
        "y_high": bool(maximum[1] == data.shape[1] - 1) if bbox is not None else False,
        "x_low": bool(minimum[2] == 0) if bbox is not None else False,
        "x_high": bool(maximum[2] == data.shape[2] - 1) if bbox is not None else False,
    }
    if any(touches[key] for key in ("y_low", "y_high", "x_low", "x_high")):
        flags.add("foreground_touches_inplane_border")

    nonempty_slices = [
        int(np.count_nonzero(foreground.any(axis=tuple(i for i in range(3) if i != axis))))
        for axis in range(3)
    ]

    lesion_path = lesion_mask_path(image_path)
    lesion_voxels = 0
    lesion_retention = None
    if lesion_path.is_file():
        lesion = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_path))) > 0
        if lesion.shape != foreground.shape:
            flags.add("lesion_shape_mismatch")
        else:
            lesion_voxels = int(lesion.sum())
            if lesion_voxels > 0:
                lesion_retention = float((lesion & foreground).sum() / lesion_voxels)
                if lesion_retention < thresholds["min_lesion_retention"]:
                    flags.add("low_lesion_retention")

    row = {
        "case_id": case_id,
        "label": label if label is not None else "",
        "class_name": class_name or "",
        "seq_id": seq_id,
        "sequence": ALL_SEQUENCES[seq_id - 1] if 1 <= seq_id <= len(ALL_SEQUENCES) else "",
        "path": str(image_path),
        "brain_extractor": meta.get("brain_extractor", ""),
        "foreground_voxels": foreground_voxels,
        "foreground_fraction": foreground_fraction,
        "foreground_volume_ml": foreground_volume_ml,
        "largest_component_fraction": largest_component_fraction,
        "component_count": component_count,
        "bbox_shape_zyx": format_values(bbox_shape, precision=8),
        "bbox_fraction_zyx": format_values(bbox_fraction),
        "centroid_fraction_zyx": format_values(centroid_fraction),
        "nonempty_slices_zyx": format_values(nonempty_slices, precision=8),
        **{f"touches_{key}": value for key, value in touches.items()},
        "lesion_voxels": lesion_voxels,
        "lesion_retention": "" if lesion_retention is None else lesion_retention,
        "flags": flags,
        "_foreground": foreground,
        "_meta": meta,
        "_meta_path": meta_path,
    }
    return row


def dice(mask_a, mask_b):
    denominator = int(mask_a.sum() + mask_b.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(mask_a, mask_b).sum() / denominator)


def build_case_row(image_rows, thresholds):
    first = image_rows[0]
    by_seq = {int(row["seq_id"]): row for row in image_rows}
    volumes = [float(row["foreground_volume_ml"]) for row in image_rows]
    positive_volumes = [value for value in volumes if value > 0]
    volume_ratio = (
        max(positive_volumes) / min(positive_volumes)
        if len(positive_volumes) == len(volumes) and positive_volumes
        else float("inf")
    )
    pairs = ((1, 2, "dice_t1_t2"), (1, 3, "dice_t1_flair"), (2, 3, "dice_t2_flair"))
    pairwise = {}
    for seq_a, seq_b, name in pairs:
        if seq_a in by_seq and seq_b in by_seq:
            pairwise[name] = dice(
                by_seq[seq_a]["_foreground"],
                by_seq[seq_b]["_foreground"],
            )
        else:
            pairwise[name] = None
    valid_dice = [value for value in pairwise.values() if value is not None]
    min_pairwise_dice = min(valid_dice) if valid_dice else None
    flags = set()
    if len(by_seq) != len(ALL_SEQUENCES):
        flags.add("missing_sequence")
    if volume_ratio > thresholds["max_cross_sequence_volume_ratio"]:
        flags.add("high_cross_sequence_volume_ratio")
    if (
        min_pairwise_dice is not None
        and min_pairwise_dice < thresholds["min_pairwise_dice"]
    ):
        flags.add("low_cross_sequence_dice")

    return {
        "case_id": first["case_id"],
        "label": first["label"],
        "class_name": first["class_name"],
        "sequence_count": len(by_seq),
        "volume_ratio_max_min": volume_ratio,
        **{
            name: "" if value is None else value
            for name, value in pairwise.items()
        },
        "min_pairwise_dice": "" if min_pairwise_dice is None else min_pairwise_dice,
        "flags": flags,
    }


def inspect_case_task(task):
    """处理单病例并在返回父进程前释放完整三维前景数组。"""
    case_id, paths, data_root, thresholds = task
    rows = [
        inspect_image(path, data_root, thresholds)
        for path in paths
    ]
    case_row = build_case_row(rows, thresholds)
    for row in rows:
        row.pop("_foreground", None)
    return rows, case_row


def add_volume_outlier_flags(image_rows, robust_z_threshold):
    by_sequence = defaultdict(list)
    for row in image_rows:
        if float(row["foreground_volume_ml"]) > 0:
            by_sequence[int(row["seq_id"])].append(row)

    distribution = {}
    for seq_id, rows in by_sequence.items():
        log_volumes = np.log([float(row["foreground_volume_ml"]) for row in rows])
        median = float(np.median(log_volumes))
        mad = float(np.median(np.abs(log_volumes - median)))
        scale = 1.4826 * mad
        for row, value in zip(rows, log_volumes):
            robust_z = abs(float(value) - median) / scale if scale > 0 else 0.0
            if robust_z > robust_z_threshold:
                row["flags"].add("foreground_volume_outlier")
        raw_volumes = np.exp(log_volumes)
        distribution[str(seq_id)] = {
            "count": len(rows),
            "volume_ml_p01": float(np.quantile(raw_volumes, 0.01)),
            "volume_ml_median": float(np.quantile(raw_volumes, 0.5)),
            "volume_ml_p99": float(np.quantile(raw_volumes, 0.99)),
        }
    return distribution


def csv_safe_row(row, fields):
    converted = {}
    for field in fields:
        value = row.get(field, "")
        if isinstance(value, set):
            value = ";".join(sorted(value))
        converted[field] = value
    return converted


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_safe_row(row, fields) for row in rows)


def display_limits(data):
    values = data[np.isfinite(data) & (data != 0)]
    if values.size == 0:
        values = data[np.isfinite(data)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def slice_with_largest_mask(mask, axis):
    other_axes = tuple(index for index in range(3) if index != axis)
    areas = mask.sum(axis=other_axes)
    return int(np.argmax(areas)) if areas.size else 0


def take_slice(volume, axis, index):
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def save_overlay(row, target_spacing, overlay_root: Path):
    meta = row["_meta"]
    raw_path = meta.get("raw_image_path")
    crop_meta = meta.get("crop")
    if not raw_path or not crop_meta or not Path(raw_path).is_file():
        return None
    raw_resampled = resample_image(raw_path, target_spacing=target_spacing, is_label=False)
    if raw_resampled is None:
        return None
    raw_cropped = apply_crop_or_pad_from_meta(raw_resampled, crop_meta)
    raw = sitk.GetArrayFromImage(raw_cropped).astype(np.float32, copy=False)
    foreground = row.get("_foreground")
    if foreground is None:
        processed_path = Path(row["path"])
        if not processed_path.is_file():
            return None
        processed = sitk.ReadImage(str(processed_path))
        foreground = sitk.GetArrayFromImage(processed) != 0
    if raw.shape != foreground.shape:
        return None

    lesion = None
    lesion_path = lesion_mask_path(Path(row["path"]))
    if lesion_path.is_file():
        lesion_candidate = sitk.GetArrayFromImage(sitk.ReadImage(str(lesion_path))) > 0
        if lesion_candidate.shape == foreground.shape:
            lesion = lesion_candidate

    vmin, vmax = display_limits(raw)
    planes = ((0, "axial"), (1, "coronal"), (2, "sagittal"))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, (dimension, title) in zip(axes, planes):
        index = slice_with_largest_mask(foreground, dimension)
        image_slice = take_slice(raw, dimension, index)
        mask_slice = take_slice(foreground, dimension, index)
        axis.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        if mask_slice.any() and not mask_slice.all():
            axis.contour(mask_slice.astype(float), levels=[0.5], colors=["red"], linewidths=0.8)
        if lesion is not None:
            lesion_slice = take_slice(lesion, dimension, index)
            if lesion_slice.any() and not lesion_slice.all():
                axis.contour(
                    lesion_slice.astype(float),
                    levels=[0.5],
                    colors=["yellow"],
                    linewidths=0.8,
                )
        axis.set_title(f"{title} index={index}")
        axis.axis("off")

    flags = ";".join(sorted(row["flags"])) or "none"
    figure.suptitle(
        f"case {row['case_id']} | {row['sequence']} | red=foreground | "
        f"yellow=lesion\nflags: {flags}",
        fontsize=10,
    )
    figure.tight_layout()
    overlay_root.mkdir(parents=True, exist_ok=True)
    output_path = overlay_root / f"case_{row['case_id']}_seq{row['seq_id']}.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def select_overlay_rows(image_rows, case_rows, count):
    if count <= 0:
        return []
    case_flags = {row["case_id"]: len(row["flags"]) for row in case_rows}

    def priority(row):
        lesion_penalty = (
            1.0 - float(row["lesion_retention"])
            if row["lesion_retention"] != ""
            else 0.0
        )
        return (
            len(row["flags"]) + case_flags.get(row["case_id"], 0),
            lesion_penalty,
            1.0 - float(row["largest_component_fraction"]),
        )

    ranked = sorted(image_rows, key=priority, reverse=True)
    return ranked[: min(count, len(ranked))]


def main(args):
    config = load_python_config(args.config, PREPROCESS_CONFIG_FIELDS)
    if not bool(config.FOREGROUND_ZERO_OUTSIDE):
        raise ValueError(
            "Brain-extraction QC derives the final foreground from zeroed background; "
            "FOREGROUND_ZERO_OUTSIDE must be True."
        )
    data_root = resolve_input_artifact_dir(args.data_root, "data")
    experiment_root = data_root.parent
    report_root = (
        Path(args.report_root).expanduser().resolve()
        if args.report_root
        else experiment_root / "reports" / "brain_extraction_qc"
    )
    thresholds = {
        "min_largest_component_fraction": args.min_largest_component_fraction,
        "min_lesion_retention": args.min_lesion_retention,
        "min_pairwise_dice": args.min_pairwise_dice,
        "max_cross_sequence_volume_ratio": args.max_cross_sequence_volume_ratio,
    }

    grouped_paths = defaultdict(list)
    for path in sorted(data_root.glob("*/*/case_*_*.nii*")):
        if path.name.endswith("_mask.nii.gz"):
            continue
        parsed = parse_case_file(path)
        if parsed is not None:
            grouped_paths[parsed[0]].append(path)
    case_ids = sorted(grouped_paths)
    if args.max_cases is not None:
        case_ids = case_ids[: args.max_cases]
    if not case_ids:
        raise RuntimeError(f"No preprocessed images found under {data_root}")

    image_rows = []
    case_rows = []
    print(f"Checking {len(case_ids)} cases under {data_root}")
    tasks = [
        (case_id, sorted(grouped_paths[case_id]), data_root, thresholds)
        for case_id in case_ids
    ]
    if args.num_workers == 1:
        results = map(inspect_case_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.num_workers)
        results = executor.map(inspect_case_task, tasks, chunksize=1)

    try:
        for index, (rows, case_row) in enumerate(results, start=1):
            image_rows.extend(rows)
            case_rows.append(case_row)
            if index % 100 == 0 or index == len(case_ids):
                print(f"  {index}/{len(case_ids)} cases")
    finally:
        if executor is not None:
            executor.shutdown()

    distributions = add_volume_outlier_flags(image_rows, args.volume_outlier_z)
    image_flags = Counter(flag for row in image_rows for flag in row["flags"])
    case_flags = Counter(flag for row in case_rows for flag in row["flags"])

    image_csv = report_root / "image_metrics.csv"
    case_csv = report_root / "case_metrics.csv"
    write_csv(image_csv, IMAGE_FIELDS, image_rows)
    write_csv(case_csv, CASE_FIELDS, case_rows)

    overlay_rows = select_overlay_rows(image_rows, case_rows, args.num_overlays)
    saved_overlays = []
    for row in overlay_rows:
        output_path = save_overlay(
            row,
            target_spacing=tuple(config.TARGET_SPACING),
            overlay_root=report_root / "overlays",
        )
        if output_path is not None:
            saved_overlays.append(str(output_path))

    summary = {
        "config": str(config.__config_path__),
        "data_root": str(data_root),
        "brain_extractor": config.BRAIN_EXTRACTOR,
        "cases_checked": len(case_rows),
        "images_checked": len(image_rows),
        "images_with_flags": sum(bool(row["flags"]) for row in image_rows),
        "cases_with_flags": sum(bool(row["flags"]) for row in case_rows),
        "image_flag_counts": dict(sorted(image_flags.items())),
        "case_flag_counts": dict(sorted(case_flags.items())),
        "thresholds": {**thresholds, "volume_outlier_z": args.volume_outlier_z},
        "volume_distributions_by_seq": distributions,
        "overlays": saved_overlays,
        "notes": [
            "Foreground is derived from nonzero voxels after configured dilation and crop.",
            "Z-border contact is reported but not automatically flagged because thick-slice scans commonly touch both Z bounds.",
            "Cross-sequence Dice is a screening signal, not a registration-quality metric.",
        ],
    }
    summary_path = report_root / "summary.json"
    report_root.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[SUMMARY]")
    print(f"  Cases checked     : {len(case_rows)}")
    print(f"  Images checked    : {len(image_rows)}")
    print(f"  Cases with flags  : {summary['cases_with_flags']}")
    print(f"  Images with flags : {summary['images_with_flags']}")
    print(f"  Image metrics     : {image_csv}")
    print(f"  Case metrics      : {case_csv}")
    print(f"  Summary           : {summary_path}")
    print(f"  Overlays saved    : {len(saved_overlays)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check brain-extraction quality in preprocessed NIfTI data."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--report-root")
    parser.add_argument("--num-overlays", type=int, default=24)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-largest-component-fraction", type=float, default=0.98)
    parser.add_argument("--min-lesion-retention", type=float, default=0.999)
    parser.add_argument("--min-pairwise-dice", type=float, default=0.75)
    parser.add_argument("--max-cross-sequence-volume-ratio", type=float, default=1.5)
    parser.add_argument("--volume-outlier-z", type=float, default=4.5)
    args = parser.parse_args()
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
