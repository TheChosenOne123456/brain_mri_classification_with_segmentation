'''
MLP+ResNet 晚期融合（软投票）评估脚本
- 读取 dataset 中的 3 个序列的 3D 测试数据
- 加载 BLcheckpoints 中对应的 3 个最佳 MLPResNet 模型权重
- 通过前向传播 + Softmax 提取三个模型的预测概率分布
- 平均概率（软投票）后输出最终指标，实现与深度学习主项目相同的融合评估对比
'''

import argparse
import numpy as np
import time
import sys
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 屏蔽 torch.load 的安全警告
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# 将项目根目录加入环境变量
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from configs.global_config import ALL_SEQUENCES, NUM_CLASSES, CLASS_NAMES, K_FOLDS, DATASET_ROOT
from configs.train_config import BATCH_SIZE, NUM_WORKERS
from utils.train_and_test import load_pt_dataset

# 导入你在 MLPResNet 中定义好的模型
from baseline_scripts.MLPResNet import resnet10_mlp


# ================== [数据集专区] ==================
class MultiSequenceDataset(Dataset):
    """
    为晚期融合设计：返回三个独立的 X (各自为单通道)，而不是拼接好的。
    """
    def __init__(self, datasets_list):
        self.datasets = datasets_list
        self.labels = datasets_list[0].labels

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        xs = []
        for ds in self.datasets:
            # 兼容带有 mask_tensor 和 has_mask_flag 的解包结构
            x, y, mask_tensor, has_mask_flag, case_id = ds[idx]  
            xs.append(x) # 保持独立的 [1, D, H, W]
        
        _, y, _, _, case_id = self.datasets[0][idx]
        return xs, y, case_id
# ================================================


def evaluate_mlpresnet_vote_single_fold(fold_idx, device):
    """
    使用三个模型进行软投票的评估函数
    """
    print(f"\n{'='*20} Evaluating Fold {fold_idx} {'='*20}")
    print(f"Mode: Late Fusion (Soft Voting) | Model: MLP+ResNet3D")

    # ---------- 1. 加载数据 ----------
    test_sets_list = []
    print("  -> Loading 3D test sets for all 3 sequences... ", end="", flush=True)
    t0 = time.time()
    
    for seq_id in range(1, len(ALL_SEQUENCES) + 1):
        seq_name = ALL_SEQUENCES[seq_id - 1]
        dataset_dir = DATASET_ROOT / f"seq{seq_id}_{seq_name}" / f"fold{fold_idx}"
        # 兼容备用命名
        if not dataset_dir.exists():
            dataset_dir = DATASET_ROOT / f"seq{seq_id}" / f"fold{fold_idx}"
            
        if not dataset_dir.exists():
            print(f"\n[Warning] Dataset missing for {seq_name} at {dataset_dir}. Skipping fold {fold_idx}.")
            return None
        test_sets_list.append(load_pt_dataset(dataset_dir / "test.pt"))
        
    print(f"Done in {time.time()-t0:.1f}s")
    
    test_set = MultiSequenceDataset(test_sets_list)
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # ---------- 2. 初始化并加载 3 个模型 ----------
    models = []
    for seq_id in range(1, len(ALL_SEQUENCES) + 1):
        seq_name = ALL_SEQUENCES[seq_id - 1]
        
        weight_file = ROOT_DIR / "BLcheckpoints" / f"seq{seq_id}_{seq_name}" / "MLPResNet" / f"fold{fold_idx}_model_best.pth"
        if not weight_file.exists():
            print(f"\n[Warning] Model weights not found for {seq_name} at {weight_file}.")
            print(f"Please run 'python baseline_scripts/MLPResNet.py --seq {seq_id}' first.")
            return None
        
        # 实例化单通道模型
        model = resnet10_mlp(num_classes=NUM_CLASSES, in_channels=1)
        model = model.to(device)
        model.load_state_dict(torch.load(weight_file, map_location=device))
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        model.eval()
        models.append(model)
        
    print("  -> Successfully loaded 3 MLP+ResNet models (T1, T2, FLAIR).")

    # ---------- 3. 测试 (软投票机制) ----------
    all_preds = []
    all_labels = []
    misclassified_cases = []

    with torch.no_grad():
        for xs, y, case_ids in test_loader:
            y = y.to(device)
            
            probs = []
            for i, x in enumerate(xs):
                x = x.to(device)
                logits = models[i](x)
                prob = F.softmax(logits, dim=1)
                probs.append(prob)
            
            # 平均概率 (Soft Voting)
            avg_prob = sum(probs) / len(probs)
            
            preds = avg_prob.argmax(dim=1)
            preds_cpu = preds.cpu().numpy()
            labels_cpu = y.cpu().numpy()

            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)

            for cid, p, gt in zip(case_ids, preds_cpu, labels_cpu):
                if p != gt:
                    misclassified_cases.append({
                        "case_id": cid, "gt": int(gt), "pred": int(p)
                    })

    # ---------- 4. 计算指标 ----------
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # ---------- 5. 打印结果 ----------
    print("\n===== Test Results =====")
    print(f"Sequence      : ALL (Soft Voting) (Fold {fold_idx})")
    print(f"Test samples  : {len(test_set)}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

    # 简要打印误判 (可选)
    print(f"\nTotal misclassified: {len(misclassified_cases)}")

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


# ================== 主流程 ==================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Starting PyTorch MLP+ResNet K-Fold Evaluation for: Late Fusion Soft Voting on {device} <<<")

    folds_to_run = [args.fold] if args.fold is not None else list(range(1, K_FOLDS + 1))
    
    if args.fold is not None:
        print(f"Mode: Single Fold Evaluation (Fold {args.fold})")
    else:
        print(f"Mode: All {K_FOLDS} Folds Average")

    metrics_history = []

    for k in folds_to_run:
        res = evaluate_mlpresnet_vote_single_fold(k, device)
        if res:
            metrics_history.append(res)
    
    if len(metrics_history) > 1:
        print("\n" + "="*50)
        print(f"   K-FOLDS AVERAGE REPORT ({len(metrics_history)} folds)   ")
        print("="*50)

        avg_acc, std_acc = np.mean([r['acc'] for r in metrics_history]), np.std([r['acc'] for r in metrics_history])
        avg_f1, std_f1 = np.mean([r['f1'] for r in metrics_history]), np.std([r['f1'] for r in metrics_history])
        avg_prec, std_prec = np.mean([r['precision'] for r in metrics_history]), np.std([r['precision'] for r in metrics_history])
        avg_rec, std_rec = np.mean([r['recall'] for r in metrics_history]), np.std([r['recall'] for r in metrics_history])

        print(f"Method        : Late Fusion Soft Voting")
        print(f"Model         : MLP+ResNet3D (PyTorch)")
        print("-" * 40)
        print(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
        print("-" * 40)
        print(f"{'Accuracy':<15} | {avg_acc:.4f}     | ±{std_acc:.4f}")
        print(f"{'Precision':<15} | {avg_prec:.4f}     | ±{std_prec:.4f}")
        print(f"{'Recall':<15} | {avg_rec:.4f}     | ±{std_rec:.4f}")
        print(f"{'F1-Score':<15} | {avg_f1:.4f}     | ±{std_f1:.4f}")
        print("-" * 40)
    elif len(metrics_history) == 0:
        print("\n[Error] No folds were successfully evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, K_FOLDS + 1),
        help=f"Specific fold to evaluate (1~{K_FOLDS}). If not set, run all folds.",
    )
    args = parser.parse_args()
    main(args)