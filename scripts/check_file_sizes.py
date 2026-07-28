"""
按文件大小快速筛查预处理后的 NIfTI。

只检查文件大小，不读取体素内容；用于快速发现明显过小的输出文件。

示例：
    python -m scripts.check_file_sizes --data-root dataxxx
    python -m scripts.check_file_sizes --data-root dataxxx --threshold-mb 1
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from configs.config_utils import resolve_input_artifact_dir
from configs.global_config import ALL_SEQUENCES, CLASS_NAMES


def parse_file(path: Path, data_root: Path):
    try:
        class_dir, seq_text = path.relative_to(data_root).parts[:2]
    except ValueError:
        return None

    if "_" in class_dir:
        label_text, class_name = class_dir.split("_", 1)
    else:
        label_text, class_name = "", class_dir

    if not seq_text.isdigit():
        return None

    seq_id = int(seq_text)
    seq_name = ALL_SEQUENCES[seq_id - 1] if 1 <= seq_id <= len(ALL_SEQUENCES) else ""
    parts = path.name.split("_")
    case_id = parts[1] if len(parts) >= 3 else ""

    return {
        "path": str(path),
        "case_id": case_id,
        "label": label_text,
        "class_name": class_name,
        "seq_id": seq_id,
        "sequence": seq_name,
        "file_size_mb": path.stat().st_size / (1024 * 1024),
    }


def main():
    parser = argparse.ArgumentParser(description="Check preprocessed NIfTI file sizes.")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Experiment root containing data, or the data directory itself",
    )
    parser.add_argument("--threshold-mb", type=float, default=2.0)
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--include-masks", action="store_true")
    parser.add_argument("--limit-print", type=int, default=30)
    args = parser.parse_args()

    data_root = resolve_input_artifact_dir(args.data_root, "data")
    rows = []

    for path in sorted(data_root.glob("*/*/case_*.nii.gz")):
        if path.name.endswith("_mask.nii.gz") and not args.include_masks:
            continue
        row = parse_file(path, data_root)
        if row is None:
            continue
        if row["file_size_mb"] < args.threshold_mb:
            rows.append(row)

    rows.sort(key=lambda row: row["file_size_mb"])

    print(f"[CONFIG] data_root: {data_root}")
    print(f"[CONFIG] threshold_mb: {args.threshold_mb}")
    print(f"[SUMMARY] files smaller than threshold: {len(rows)}")

    by_class_seq = Counter((row["class_name"], row["sequence"]) for row in rows)
    if by_class_seq:
        print("\n[CLASS x SEQ]")
        for class_name in CLASS_NAMES:
            counts = []
            for seq_name in ALL_SEQUENCES:
                counts.append(f"{seq_name}:{by_class_seq.get((class_name, seq_name), 0)}")
            print(f"  {class_name}: " + " | ".join(counts))

    print(f"\n[FIRST {args.limit_print}]")
    for row in rows[: args.limit_print]:
        print(
            f"  {row['file_size_mb']:.3f} MB | "
            f"{row['class_name']} | {row['sequence']} | "
            f"case_{row['case_id']} | {row['path']}"
        )

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "path",
                "case_id",
                "label",
                "class_name",
                "seq_id",
                "sequence",
                "file_size_mb",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[CSV] Saved to: {csv_path}")


if __name__ == "__main__":
    main()
