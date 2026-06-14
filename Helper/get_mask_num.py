'''
统计 Mask 数据量脚本
功能：
1. 统计 versionx/data 下实际存在的 mask 文件数量（按类别、按序列）
2. 统计有 mask 的唯一样本数（按序列、整体）
3. 读取 mask_index.json（若存在）并做一致性对照
4. 可选：按 K-Fold split 统计 train/val/test 中“有mask样本”数量
'''

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到 sys.path，确保能导入 config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from configs.global_config import *


MASK_NAME_PAT = re.compile(r"^case_(\d+)_(\d+)_mask\.nii(\.gz)?$")


def normalize_case_id(v):
    s = str(v).strip()
    if s.isdigit():
        return f"{int(s):04d}"
    return s


def load_json(json_path: Path, default):
    if not json_path.exists():
        return default
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_class_dirs(data_root: Path):
    """
    只返回形如 '0_normal'、'1_inflammation' 的类别目录
    """
    dirs = []
    if not data_root.exists():
        return dirs

    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        # 类别目录格式: 数字_类别名
        if re.match(r"^\d+_.+$", d.name):
            dirs.append(d)
    return dirs


def collect_mask_stats_from_files(data_root: Path):
    """
    扫描 data_root，统计实际 mask 文件。
    返回：
    - class_seq_count: {class_name: {seq_id: count}}
    - seq_case_sets: {seq_id: set(case_id)}
    - all_case_set: set(case_id)
    - total_mask_files: int
    """
    class_seq_count = defaultdict(lambda: defaultdict(int))
    seq_case_sets = {i: set() for i in range(1, NUM_SEQUENCES + 1)}
    all_case_set = set()
    total_mask_files = 0

    class_dirs = safe_class_dirs(data_root)

    for cdir in class_dirs:
        class_name = cdir.name
        for seq_id in range(1, NUM_SEQUENCES + 1):
            seq_dir = cdir / str(seq_id)
            if not seq_dir.exists():
                continue

            for f in seq_dir.glob("case_*_*_mask.nii*"):
                m = MASK_NAME_PAT.match(f.name)
                if m is None:
                    continue

                case_id = normalize_case_id(m.group(1))
                seq_in_name = int(m.group(2))

                # 双保险：文件名中的序列号应与所在目录一致
                if seq_in_name != seq_id:
                    continue

                total_mask_files += 1
                class_seq_count[class_name][seq_id] += 1
                seq_case_sets[seq_id].add(case_id)
                all_case_set.add(case_id)

    return class_seq_count, seq_case_sets, all_case_set, total_mask_files


def print_file_stats(class_seq_count, seq_case_sets, all_case_set, total_mask_files):
    print("\n" + "=" * 72)
    print("Mask File Statistics (from data directory)")
    print("=" * 72)
    print(f"Total mask files       : {total_mask_files}")
    print(f"Unique masked cases    : {len(all_case_set)}")

    print("\nPer-sequence unique case count:")
    for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1):
        print(f"  Seq{seq_id}_{seq_name:<8}: {len(seq_case_sets[seq_id])}")

    print("\nPer-class x per-sequence mask file count:")
    header = f"{'ClassDir':<24} | " + " | ".join([f"seq{i}" for i in range(1, NUM_SEQUENCES + 1)]) + " | total"
    print(header)
    print("-" * len(header))

    for class_dir_name in sorted(class_seq_count.keys()):
        row_vals = []
        row_total = 0
        for seq_id in range(1, NUM_SEQUENCES + 1):
            c = class_seq_count[class_dir_name].get(seq_id, 0)
            row_vals.append(c)
            row_total += c
        row = f"{class_dir_name:<24} | " + " | ".join([f"{v:<4d}" for v in row_vals]) + f" | {row_total}"
        print(row)


def build_mask_sets_from_index(mask_index_data):
    """
    mask_index 结构示例:
    {
      "0001": [3],
      "0045": [1,2,3]
    }
    """
    seq_case_sets = {i: set() for i in range(1, NUM_SEQUENCES + 1)}
    all_case_set = set()

    for cid_raw, seq_list in mask_index_data.items():
        cid = normalize_case_id(cid_raw)
        if not isinstance(seq_list, list):
            continue

        valid_seqs = []
        for s in seq_list:
            try:
                si = int(s)
            except Exception:
                continue
            if 1 <= si <= NUM_SEQUENCES:
                valid_seqs.append(si)

        if len(valid_seqs) == 0:
            continue

        all_case_set.add(cid)
        for si in valid_seqs:
            seq_case_sets[si].add(cid)

    return seq_case_sets, all_case_set


def print_index_stats(mask_index_path: Path, mask_index_data, file_seq_sets, file_all_cases):
    print("\n" + "=" * 72)
    print("Mask Index Statistics (from mask_index.json)")
    print("=" * 72)
    print(f"Index path             : {mask_index_path}")
    print(f"Cases in index         : {len(mask_index_data)}")

    idx_seq_sets, idx_all_cases = build_mask_sets_from_index(mask_index_data)

    print("\nPer-sequence unique case count in index:")
    for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1):
        print(f"  Seq{seq_id}_{seq_name:<8}: {len(idx_seq_sets[seq_id])}")

    # 一致性对照
    only_in_index = sorted(idx_all_cases - file_all_cases)
    only_in_files = sorted(file_all_cases - idx_all_cases)

    print("\nConsistency check (index vs files):")
    print(f"  Cases only in index   : {len(only_in_index)}")
    print(f"  Cases only in files   : {len(only_in_files)}")

    # 防止刷屏，只展示前 20 个
    show_n = 20
    if len(only_in_index) > 0:
        print(f"  Example only-in-index : {only_in_index[:show_n]}")
    if len(only_in_files) > 0:
        print(f"  Example only-in-files : {only_in_files[:show_n]}")


def load_split_ids(split_json_path: Path):
    split_data = load_json(split_json_path, default={})
    out = {}
    for split_name in ["train", "val", "test"]:
        ids = split_data.get(f"{split_name}_ids", [])
        out[split_name] = [normalize_case_id(v) for v in ids]
    return out


def print_fold_mask_coverage(dataset_root: Path, file_all_cases, file_seq_sets):
    """
    按 fold 统计 split 中有 mask 的样本数。
    注意：K-Fold 三个序列 split 一致，这里优先从 seq3_FLAIR 的 split.json 读取。
    """
    print("\n" + "=" * 72)
    print("K-Fold Mask Coverage")
    print("=" * 72)

    seq3_dir = dataset_root / f"seq3_{ALL_SEQUENCES[2]}"
    if not seq3_dir.exists():
        print(f"Dataset directory not found: {seq3_dir}")
        print("Skip fold coverage stats.")
        return

    found_any = False
    for k in range(1, K_FOLDS + 1):
        fold_dir = seq3_dir / f"fold{k}"
        split_json = fold_dir / "split.json"
        if not split_json.exists():
            continue
        found_any = True

        split_ids = load_split_ids(split_json)

        print(f"\n--- fold{k} ---")
        header = f"  {'Split':<8} | {'Total':<8} | {'HasMask(any)':<12} | {'HasMask(seq3)':<13}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for split_name in ["train", "val", "test"]:
            ids = split_ids[split_name]
            total = len(ids)
            has_any = sum((cid in file_all_cases) for cid in ids)
            has_seq3 = sum((cid in file_seq_sets[3]) for cid in ids)
            print(f"  {split_name:<8} | {total:<8} | {has_any:<12} | {has_seq3:<13}")

    if not found_any:
        print(f"No split.json found under: {seq3_dir}")


def main(args):
    data_root = Path(args.data_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    mask_index_path = data_root / "mask_index.json"

    print("\n" + "=" * 72)
    print("Mask Statistics Script")
    print("=" * 72)
    print(f"EXPERIMENT_VERSION      : {EXPERIMENT_VERSION}")
    print(f"Data root               : {data_root}")
    print(f"Dataset root            : {dataset_root}")
    print(f"Class names             : {CLASS_NAMES}")
    print(f"Sequences               : {ALL_SEQUENCES}")

    if not data_root.exists():
        print("\n[Warning] Data root does not exist yet.")
        print("You can rerun this script after preprocess_data + preprocess_mask.")
        return

    # 1) 文件层统计
    class_seq_count, file_seq_sets, file_all_cases, total_mask_files = collect_mask_stats_from_files(data_root)
    print_file_stats(class_seq_count, file_seq_sets, file_all_cases, total_mask_files)

    # 2) mask_index 统计与一致性检查
    if mask_index_path.exists():
        mask_index_data = load_json(mask_index_path, default={})
        print_index_stats(mask_index_path, mask_index_data, file_seq_sets, file_all_cases)
    else:
        print("\n[Info] mask_index.json not found. Skip index consistency check.")

    # 3) 可选：K-Fold 覆盖率统计
    if args.by_fold:
        print_fold_mask_coverage(dataset_root, file_all_cases, file_seq_sets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check mask data volume and coverage")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(PROCESSED_DATA_PATH),
        help="Path to processed data root (e.g. version1/data)"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DATASET_ROOT),
        help="Path to dataset root (e.g. version1/datasets)"
    )
    parser.add_argument(
        "--by_fold",
        action="store_true",
        help="If set, also report train/val/test mask coverage for each fold"
    )
    args = parser.parse_args()

    main(args)