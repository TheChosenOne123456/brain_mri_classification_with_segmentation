'''
K-Fold 数据集构建脚本：
无论原始分布如何，保证所有序列使用相同的Case划分。
生成的目录结构：
dataset/seq1_T1/fold1/train.pt
dataset/seq1_T1/fold1/split.json
...
'''
import json
import random
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

# 导入复用函数
from utils.dataset import build_dataset, collect_cases_by_seq
from configs.global_config import *


def get_aligned_labels(case_ids, seq_case_maps):
    """获取共有病例的标签，并校验各序列中的标签是否一致。"""
    labels = []

    for case_id in case_ids:
        labels_by_seq = {
            seq_id: seq_cases[case_id]["label"]
            for seq_id, seq_cases in seq_case_maps.items()
        }
        unique_labels = set(labels_by_seq.values())
        if len(unique_labels) != 1:
            raise ValueError(
                f"Case {case_id} has inconsistent labels across sequences: "
                f"{labels_by_seq}"
            )
        labels.append(unique_labels.pop())

    return np.asarray(labels, dtype=np.int64)


def get_class_counts(labels):
    """返回包含零样本类别在内的类别计数，便于日志和 split.json 诊断。"""
    counts = Counter(int(label) for label in labels)
    return {
        class_name: counts.get(label_id, 0)
        for label_id, class_name in enumerate(CLASS_NAMES)
    }


def print_split_counts(split_name, labels):
    counts = get_class_counts(labels)
    counts_text = " | ".join(
        f"{class_name}: {count}" for class_name, count in counts.items()
    )
    print(f"  {split_name:<5}: {len(labels):>4} | {counts_text}")


def main():
    # 保证划分可复现
    random.seed(SEED)
    np.random.seed(SEED)
    
    # 1. 收集并对齐所有序列的 Case
    print("\n[STEP 1] Collecting and aligning cases...")
    seq_case_maps = {}
    for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1):
        seq_case_maps[seq_id] = collect_cases_by_seq(seq_id)
        print(f"  Seq {seq_id} ({seq_name}): {len(seq_case_maps[seq_id])} cases")
    
    # 取交集
    all_case_ids = sorted(list(set.intersection(*[set(cases.keys()) for cases in seq_case_maps.values()])))
    print(f"  Total complete cases: {len(all_case_ids)}")
    
    if len(all_case_ids) == 0:
        print("[ERROR] No common cases found across sequences.")
        return

    # 转为 numpy array 方便用 sklearn 切分，并校验三序列标签一致
    all_case_ids_np = np.array(all_case_ids)
    all_labels_np = get_aligned_labels(all_case_ids, seq_case_maps)

    invalid_labels = sorted(set(all_labels_np.tolist()) - set(range(NUM_CLASSES)))
    if invalid_labels:
        raise ValueError(
            f"Found labels outside configured range 0~{NUM_CLASSES - 1}: "
            f"{invalid_labels}"
        )

    class_counts = Counter(all_labels_np.tolist())
    too_small_classes = {
        CLASS_NAMES[label_id]: class_counts.get(label_id, 0)
        for label_id in range(NUM_CLASSES)
        if class_counts.get(label_id, 0) < K_FOLDS
    }
    if too_small_classes:
        raise ValueError(
            f"Each class needs at least K_FOLDS={K_FOLDS} complete cases for "
            f"StratifiedKFold, but got: {too_small_classes}"
        )

    print_split_counts("All", all_labels_np)

    # 2. 初始化 K-Fold
    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    # 3. 开始切分并保存
    print(f"\n[STEP 2] Building {K_FOLDS}-Fold datasets...")

    # enumerate 从 1 开始，我们习惯用 fold1 ~ fold5
    for fold_idx, (train_val_idx, test_idx) in enumerate(
        kf.split(all_case_ids_np, all_labels_np), start=1
    ):
        print(f"\n--- Processing Fold {fold_idx}/{K_FOLDS} ---")
        
        # 此时有了 Train+Val 的索引 和 Test 的索引
        # 我们还需要从 Train+Val 中切出一部分作为 Validation 用于早停
        #  Val 占 (Train+Val) 的 K_FOLDS_VAL_RATIO 比例
        train_sub_idx, val_sub_idx = train_test_split(
            train_val_idx,
            test_size=K_FOLDS_VAL_RATIO,
            random_state=SEED,
            shuffle=True,
            stratify=all_labels_np[train_val_idx]
        )

        fold_train_ids = all_case_ids_np[train_sub_idx]
        fold_val_ids   = all_case_ids_np[val_sub_idx]
        fold_test_ids  = all_case_ids_np[test_idx]

        fold_train_labels = all_labels_np[train_sub_idx]
        fold_val_labels = all_labels_np[val_sub_idx]
        fold_test_labels = all_labels_np[test_idx]

        print_split_counts("Train", fold_train_labels)
        print_split_counts("Val", fold_val_labels)
        print_split_counts("Test", fold_test_labels)

        # 对每个序列分别构建 dataset
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1):
            seq_cases = seq_case_maps[seq_id]
            
            # 根据 ID 提取 Case 对象
            train_data = build_dataset([seq_cases[cid] for cid in fold_train_ids])
            val_data   = build_dataset([seq_cases[cid] for cid in fold_val_ids])
            test_data  = build_dataset([seq_cases[cid] for cid in fold_test_ids])

            # 保存路径： datasets/seq1_T1/fold1/
            fold_dir = DATASET_ROOT / (f"seq{seq_id}_{seq_name}") / f"fold{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            torch.save(train_data, fold_dir / "train.pt")
            torch.save(val_data,   fold_dir / "val.pt")
            torch.save(test_data,  fold_dir / "test.pt")
            
            # --- 保存 Split 信息 (关键步骤) ---
            split_info = {
                "fold": fold_idx,
                "sequence": seq_name,
                "split_strategy": "StratifiedKFold + stratified train/val split",
                "seed": SEED,
                "train_ids": fold_train_ids.tolist(),
                "val_ids": fold_val_ids.tolist(),
                "test_ids": fold_test_ids.tolist(),
                "class_counts": {
                    "train": get_class_counts(fold_train_labels),
                    "val": get_class_counts(fold_val_labels),
                    "test": get_class_counts(fold_test_labels),
                },
            }
            with open(fold_dir / "split.json", "w", encoding='utf-8') as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)
            
        print(f"  Saved fold {fold_idx} data for all sequences.")

    print("\n[SUCCESS] K-Fold datasets building finished.")

if __name__ == "__main__":
    main()
