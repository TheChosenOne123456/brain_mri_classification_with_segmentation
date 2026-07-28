"""
删除预处理数据中小于指定文件大小阈值的 case。

用途：
    当预处理结果中出现明显过小、疑似损坏的 NIfTI 图像时，按 case 清理。
    脚本只用“图像文件”判断是否触发删除，默认忽略 *_mask.nii.gz 作为触发条件；
    一旦某个 case 被判定需要删除，会删除该 case 的所有序列图像和相关 mask，
    并同步从 case_index.json 中移除注册。

默认是 dry-run，不会真正删除。确认输出无误后加 --apply 执行。

用法：
    # 查看小于 1MB 的 case，不删除
    python -m scripts.delete_small_preprocessed_cases \
        --data-root dataxxx --threshold-mb 1

    # 真正删除小于 1MB 的 case，并更新 case_index.json
    python -m scripts.delete_small_preprocessed_cases \
        --data-root dataxxx \
        --threshold-mb 1 \
        --apply

    # 保存删除候选清单
    python -m scripts.delete_small_preprocessed_cases \
        --data-root dataxxx \
        --threshold-mb 1 \
        --csv-path dataxxx/delete_small_cases_report.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from configs.config_utils import resolve_input_artifact_dir
from utils.io import INDEX_FILE_NAME


def parse_case_id(path: Path):
    name = path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        return None

    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "case":
        return None

    return parts[1]


def is_mask_file(path: Path):
    return path.name.endswith("_mask.nii.gz") or path.name.endswith("_mask.nii")


def collect_small_image_files(data_root: Path, threshold_mb: float):
    small_files = []

    for path in sorted(data_root.glob("*/*/case_*.nii*")):
        if is_mask_file(path):
            continue

        case_id = parse_case_id(path)
        if case_id is None:
            continue

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < threshold_mb:
            small_files.append(
                {
                    "case_id": case_id,
                    "path": path,
                    "file_size_mb": size_mb,
                }
            )

    return small_files


def collect_case_files(data_root: Path, case_ids):
    case_files = defaultdict(list)
    case_id_set = set(case_ids)

    for path in sorted(data_root.glob("*/*/case_*.nii*")):
        case_id = parse_case_id(path)
        if case_id in case_id_set:
            case_files[case_id].append(path)

    return case_files


def load_case_index(index_path: Path):
    if not index_path.exists():
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_case_index(index, index_path: Path):
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def remove_case_ids_from_index(case_index, case_ids):
    case_id_ints = {int(case_id) for case_id in case_ids}
    removed = {}

    for case_key, case_id in list(case_index.items()):
        if int(case_id) in case_id_ints:
            removed[case_key] = case_id
            del case_index[case_key]

    return removed


def write_report(csv_path: Path, small_files, case_files, removed_index_entries):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "trigger_file",
        "trigger_file_size_mb",
        "files_to_delete",
        "index_removed",
        "index_case_key",
    ]

    triggers_by_case = defaultdict(list)
    for item in small_files:
        triggers_by_case[item["case_id"]].append(item)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case_id in sorted(case_files.keys()):
            index_keys = [
                key for key, value in removed_index_entries.items() if int(value) == int(case_id)
            ]
            for trigger in triggers_by_case[case_id]:
                writer.writerow(
                    {
                        "case_id": case_id,
                        "trigger_file": str(trigger["path"]),
                        "trigger_file_size_mb": f"{trigger['file_size_mb']:.6f}",
                        "files_to_delete": ";".join(str(p) for p in case_files[case_id]),
                        "index_removed": bool(index_keys),
                        "index_case_key": ";".join(index_keys),
                    }
                )


def main():
    parser = argparse.ArgumentParser(
        description="Delete preprocessed cases whose image files are smaller than a threshold."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Experiment root containing data, or the data directory itself",
    )
    parser.add_argument("--threshold-mb", type=float, required=True)
    parser.add_argument("--apply", action="store_true", help="Actually delete files and update index.")
    parser.add_argument("--csv-path", type=str, default="")
    args = parser.parse_args()

    if args.threshold_mb <= 0:
        raise ValueError("--threshold-mb must be positive.")

    data_root = resolve_input_artifact_dir(args.data_root, "data")

    index_path = data_root / INDEX_FILE_NAME
    small_files = collect_small_image_files(data_root, args.threshold_mb)
    case_ids = sorted({item["case_id"] for item in small_files})
    case_files = collect_case_files(data_root, case_ids)

    case_index = load_case_index(index_path)
    removed_index_entries = remove_case_ids_from_index(dict(case_index), case_ids)

    print(f"[CONFIG] data_root: {data_root}")
    print(f"[CONFIG] threshold_mb: {args.threshold_mb}")
    print(f"[MODE] {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"[SUMMARY] trigger small image files: {len(small_files)}")
    print(f"[SUMMARY] cases to delete: {len(case_ids)}")
    print(f"[SUMMARY] files to delete including masks: {sum(len(v) for v in case_files.values())}")
    print(f"[SUMMARY] case_index entries to remove: {len(removed_index_entries)}")

    if case_ids:
        print("\n[FIRST 30 CASES]")
        for case_id in case_ids[:30]:
            trigger_sizes = [
                f"{item['file_size_mb']:.3f}MB"
                for item in small_files
                if item["case_id"] == case_id
            ]
            print(
                f"  case_{case_id} | triggers={','.join(trigger_sizes)} | "
                f"files={len(case_files.get(case_id, []))}"
            )

    if args.csv_path:
        report_index_entries = removed_index_entries
        write_report(Path(args.csv_path), small_files, case_files, report_index_entries)
        print(f"\n[CSV] Saved report to: {args.csv_path}")

    if not args.apply:
        print("\n[DRY-RUN] No files were deleted. Re-run with --apply to delete.")
        return

    for files in case_files.values():
        for path in files:
            if path.exists():
                path.unlink()

    case_index = load_case_index(index_path)
    removed_index_entries = remove_case_ids_from_index(case_index, case_ids)
    save_case_index(case_index, index_path)

    print("\n[DONE]")
    print(f"  Deleted cases: {len(case_ids)}")
    print(f"  Deleted files including masks: {sum(len(v) for v in case_files.values())}")
    print(f"  Removed case_index entries: {len(removed_index_entries)}")
    print(f"  Updated index: {index_path}")


if __name__ == "__main__":
    main()
