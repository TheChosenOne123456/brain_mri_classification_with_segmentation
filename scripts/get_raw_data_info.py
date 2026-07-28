"""
统计 global_config 中原始 NIfTI 图像的空间信息。

该脚本只读取图像 header / metadata，不做重采样或预处理。输出包括：
- SimpleITK 原始轴顺序: size/spacing = (X, Y, Z)
- 项目数组轴顺序: shape/spacing = (D, H, W) = (Z, Y, X)
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from configs.global_config import ALL_SEQUENCES, CLASS_DATA_MAP, RAW_DATA_PATH
from utils.sequences import identify_sequence


try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

if sitk is not None:
    sitk.ProcessObject_SetGlobalWarningDisplay(False)


AXIS_XYZ = ("X", "Y", "Z")
AXIS_DHW = ("D", "H", "W")


def iter_configured_nii_files(raw_root):
    """遍历 CLASS_DATA_MAP 配置的原始目录，产出 NIfTI 文件及其类别/序列信息。"""
    seen = set()
    for class_name, subdirs in CLASS_DATA_MAP.items():
        for subdir in subdirs:
            src_dir = raw_root / subdir
            if not src_dir.exists():
                yield {
                    "class_name": class_name,
                    "source_dir": src_dir,
                    "path": None,
                    "error": "source_dir_not_found",
                }
                continue

            for nii_file in sorted(src_dir.rglob("*.nii*")):
                if not is_nifti_file(nii_file):
                    continue

                resolved = nii_file.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)

                sequence = get_sequence_name(nii_file)
                if sequence is None:
                    yield {
                        "class_name": class_name,
                        "source_dir": src_dir,
                        "path": nii_file,
                        "sequence": None,
                        "error": "unknown_sequence",
                    }
                    continue

                yield {
                    "class_name": class_name,
                    "source_dir": src_dir,
                    "path": nii_file,
                    "sequence": sequence,
                    "error": "",
                }


def is_nifti_file(path):
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def get_sequence_name(path):
    seq_id = identify_sequence(path)
    if seq_id is None:
        return None
    return ALL_SEQUENCES[seq_id - 1]


def read_image_info(path, class_name, source_dir, sequence):
    if sitk is None:
        raise RuntimeError(
            "SimpleITK is not installed in the active Python environment. "
            "Please activate the project conda environment first."
        )

    img = sitk.ReadImage(str(path))
    dimension = img.GetDimension()
    if dimension < 3:
        raise ValueError(f"Expected at least 3D image, got dimension={dimension}")

    size = tuple(int(v) for v in img.GetSize()[:3])
    spacing = tuple(float(v) for v in img.GetSpacing()[:3])
    physical_size = tuple(size[i] * spacing[i] for i in range(3))

    size_dhw = (size[2], size[1], size[0])
    spacing_dhw = (spacing[2], spacing[1], spacing[0])
    physical_size_dhw = (physical_size[2], physical_size[1], physical_size[0])

    return {
        "class_name": class_name,
        "sequence": sequence,
        "dimension": dimension,
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
        "spacing_x": spacing[0],
        "spacing_y": spacing[1],
        "spacing_z": spacing[2],
        "physical_x": physical_size[0],
        "physical_y": physical_size[1],
        "physical_z": physical_size[2],
        "shape_d": size_dhw[0],
        "shape_h": size_dhw[1],
        "shape_w": size_dhw[2],
        "spacing_d": spacing_dhw[0],
        "spacing_h": spacing_dhw[1],
        "spacing_w": spacing_dhw[2],
        "physical_d": physical_size_dhw[0],
        "physical_h": physical_size_dhw[1],
        "physical_w": physical_size_dhw[2],
        "source_dir": str(source_dir),
        "path": str(path),
    }


def percentile(values, q):
    if not values:
        return None
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def mean(values):
    return sum(values) / len(values) if values else None


def std(values):
    if len(values) < 2:
        return 0.0 if values else None
    avg = mean(values)
    variance = sum((v - avg) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def mode_values(values, ndigits=None, max_items=5):
    if not values:
        return []

    if ndigits is None:
        normalized = values
    else:
        normalized = [round(float(v), ndigits) for v in values]

    counter = Counter(normalized)
    return counter.most_common(max_items)


def summarize_metric(rows, keys, mode_precision):
    summary = {}
    for key, axis in keys:
        values = [float(row[key]) for row in rows]
        ndigits = None if key.startswith(("size_", "shape_")) else mode_precision
        modes = mode_values(values, ndigits=ndigits)
        summary[axis] = {
            "mean": mean(values),
            "std": std(values),
            "min": min(values),
            "p25": percentile(values, 25),
            "median": percentile(values, 50),
            "p75": percentile(values, 75),
            "max": max(values),
            "modes": modes,
        }
    return summary


def format_number(value, ndigits=3):
    if value is None:
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{ndigits}f}".rstrip("0").rstrip(".")


def format_modes(modes):
    return ", ".join(f"{format_number(value)}({count})" for value, count in modes)


def print_metric_block(title, rows, keys, mode_precision):
    print(f"\n{title}")
    print("axis | mean | std | min | p25 | median | p75 | max | top modes")
    print("-" * 82)
    summary = summarize_metric(rows, keys, mode_precision)
    for axis in [axis for _, axis in keys]:
        item = summary[axis]
        print(
            f"{axis:>4} | "
            f"{format_number(item['mean']):>7} | "
            f"{format_number(item['std']):>7} | "
            f"{format_number(item['min']):>7} | "
            f"{format_number(item['p25']):>7} | "
            f"{format_number(item['median']):>7} | "
            f"{format_number(item['p75']):>7} | "
            f"{format_number(item['max']):>7} | "
            f"{format_modes(item['modes'])}"
        )


def print_group_summary(group_name, rows, mode_precision):
    print(f"\n{'=' * 100}")
    print(f"{group_name} | n={len(rows)}")

    print_metric_block(
        "SimpleITK size (X, Y, Z)",
        rows,
        [(f"size_{axis.lower()}", axis) for axis in AXIS_XYZ],
        mode_precision,
    )
    print_metric_block(
        "SimpleITK spacing mm (X, Y, Z)",
        rows,
        [(f"spacing_{axis.lower()}", axis) for axis in AXIS_XYZ],
        mode_precision,
    )
    print_metric_block(
        "Physical size mm (X, Y, Z) = size * spacing",
        rows,
        [(f"physical_{axis.lower()}", axis) for axis in AXIS_XYZ],
        mode_precision,
    )
    print_metric_block(
        "Project shape (D, H, W) = array order (Z, Y, X)",
        rows,
        [(f"shape_{axis.lower()}", axis) for axis in AXIS_DHW],
        mode_precision,
    )
    print_metric_block(
        "Project spacing mm (D, H, W) = (Z, Y, X)",
        rows,
        [(f"spacing_{axis.lower()}", axis) for axis in AXIS_DHW],
        mode_precision,
    )


def write_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_name",
        "sequence",
        "dimension",
        "size_x",
        "size_y",
        "size_z",
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "physical_x",
        "physical_y",
        "physical_z",
        "shape_d",
        "shape_h",
        "shape_w",
        "spacing_d",
        "spacing_h",
        "spacing_w",
        "physical_d",
        "physical_h",
        "physical_w",
        "source_dir",
        "path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    if sitk is None:
        raise SystemExit(
            "SimpleITK is not installed in the active Python environment. "
            "Please run this inside the BrainMRIClassification conda environment."
        )

    raw_root = Path(args.raw_root).resolve()

    rows = []
    errors = []
    missing_dirs = []

    print(f"Scanning configured class dirs from: {raw_root}")
    print(f"Target sequences from global_config.ALL_SEQUENCES: {', '.join(ALL_SEQUENCES)}")
    scan_items = list(iter_configured_nii_files(raw_root))
    metadata_items = [item for item in scan_items if item["error"] == ""]
    skipped_unknown_sequence = [
        item for item in scan_items if item["error"] == "unknown_sequence"
    ]
    print(f"Found {len(metadata_items)} target-sequence NIfTI files to read.")

    progress_items = metadata_items
    if tqdm is not None:
        progress_items = tqdm(
            metadata_items,
            total=len(metadata_items),
            desc="Reading raw image metadata",
            unit="file",
        )

    for item in progress_items:
        path = item["path"]
        try:
            info = read_image_info(
                path,
                item["class_name"],
                item["source_dir"],
                item["sequence"],
            )
        except Exception as e:
            errors.append(
                {
                    "class_name": item["class_name"],
                    "source_dir": str(item["source_dir"]),
                    "path": str(path),
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue

        rows.append(info)

    missing_dirs = [item for item in scan_items if item["error"] == "source_dir_not_found"]

    print(f"Raw root: {raw_root}")
    print(f"Configured classes: {', '.join(CLASS_DATA_MAP.keys())}")
    print(f"Readable images: {len(rows)}")
    print(f"Missing configured source dirs: {len(missing_dirs)}")
    print(f"Skipped unknown-sequence NIfTI files: {len(skipped_unknown_sequence)}")
    print(f"Read errors: {len(errors)}")

    if not rows:
        print("No readable NIfTI files found.")
        return

    sequence_counts = Counter(row["sequence"] for row in rows)
    class_counts = Counter(row["class_name"] for row in rows)
    dimension_counts = Counter(row["dimension"] for row in rows)
    print(f"By sequence: {dict(sequence_counts)}")
    print(f"By class: {dict(class_counts)}")
    print(f"By image dimension: {dict(dimension_counts)}")

    print_group_summary("ALL", rows, args.mode_precision)

    grouped = defaultdict(list)
    for row in rows:
        grouped[("sequence", row["sequence"])].append(row)
        grouped[("class", row["class_name"])].append(row)
        grouped[("class+sequence", f"{row['class_name']} / {row['sequence']}")].append(row)

    for (group_type, group_value), group_rows in sorted(grouped.items()):
        if len(group_rows) < args.min_group_count:
            continue
        print_group_summary(f"{group_type}: {group_value}", group_rows, args.mode_precision)

    if missing_dirs and args.show_errors:
        print("\nMissing source dirs")
        for item in missing_dirs:
            print(f"- {item['class_name']}: {item['source_dir']}")

    if skipped_unknown_sequence and args.show_skipped:
        print("\nSkipped unknown-sequence NIfTI files")
        for item in skipped_unknown_sequence:
            print(f"- {item['class_name']}: {item['path']}")

    if errors and args.show_errors:
        print("\nRead errors")
        for item in errors:
            print(f"- {item['class_name']}: {item['path']} | {item['error']}")

    if args.csv_path:
        csv_path = Path(args.csv_path).resolve()
        write_csv(rows, csv_path)
        print(f"\nPer-image CSV saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize raw NIfTI size/spacing from configs.global_config."
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(RAW_DATA_PATH),
        help="Raw data root. Defaults to configs.global_config.RAW_DATA_PATH.",
    )
    parser.add_argument(
        "--mode-precision",
        type=int,
        default=3,
        help="Decimal places used before counting approximate modes for spacing/physical size.",
    )
    parser.add_argument(
        "--min-group-count",
        type=int,
        default=1,
        help="Only print grouped summaries with at least this many readable images.",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Print missing configured directories and unreadable files.",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print NIfTI files skipped because they are not one of ALL_SEQUENCES.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="",
        help="Optional path to save one metadata row per readable image.",
    )
    main(parser.parse_args())
