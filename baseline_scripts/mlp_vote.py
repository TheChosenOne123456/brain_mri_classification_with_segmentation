'''
PyTorch MLP 晚期融合（软投票）评估脚本
- 读取 baseline_cache 中预先提取的 3 个序列的测试特征数据
- 加载 BLcheckpoints 中对应的 3 个最佳 MLP 模型权重
- 通过前向传播 + Softmax 提取三个模型的预测概率分布
- 平均概率（软投票）后输出最终指标，实现与深度学习相同的融合评估对比
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

from configs.global_config import ALL_SEQUENCES, NUM_CLASSES, CLASS_NAMES, K_FOLDS

# === 必须提供模型结构才能加载权重 ===
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def evaluate_mlp_vote_single_fold(fold_idx, device):
    """
    使用三个序列特征过各自的 MLP 模型获取概率，然后用软投票评估
    """
    print(f"\n{'='*20} Evaluating Fold {fold_idx} {'='*20}")
    print(f"Mode: Late Fusion (Soft Voting) | Model: Simple MLP (PyTorch)")

    test_labels = None
    seq_probs = []

    t0 = time.time()
    
    # ---------- 1. 遍历 3 个序列，加载特征并推理出概率 ----------
    for seq_id in range(1, len(ALL_SEQUENCES) + 1):
        seq_name = ALL_SEQUENCES[seq_id - 1]
        
        # 1.1 定位缓存与权重路径
        cache_file = ROOT_DIR / "baseline_cache" / f"seq{seq_id}_{seq_name}" / f"fold{fold_idx}_features.npz"
        weight_file = ROOT_DIR / "BLcheckpoints" / f"seq{seq_id}_{seq_name}" / "MLP" / f"fold{fold_idx}_model_best.pth"

        if not cache_file.exists():
            print(f"\n[Error] Cached features not found for {seq_name} at {cache_file}.")
            print(f"Please run 'python baseline_scripts/mlp.py --seq {seq_id}' first.")
            return None
            
        if not weight_file.exists():
            print(f"\n[Error] Model weights not found for {seq_name} at {weight_file}.")
            print(f"Please run 'python baseline_scripts/mlp.py --seq {seq_id}' first to train the model.")
            return None

        # 1.2 加载测试数据 (只取 test 即可)
        data = np.load(cache_file)
        X_test, y_test = data['X_test'], data['y_test']

        if test_labels is None:
            test_labels = y_test

        input_dim = X_test.shape[1]
        
        # 1.3 初始化模型并加载权重
        model = SimpleMLP(input_dim=input_dim, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(weight_file, map_location=device))
        model.to(device)
        model.eval()

        # 1.4 模型推理提取概率
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_tensor)
            # 使用 Softmax 将输出转变为概率
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
        seq_probs.append(probs)

    print(f"  -> Successfully ran inference for 3 sequences. (Took {time.time()-t0:.2f}s)")

    # ---------- 2. 测试 (软投票机制) ----------
    # np.mean 会在 3 个 (N, 3) 形状的数组上取平均
    avg_probs = np.mean(seq_probs, axis=0)
    # 取平均后最大的概率为预测类别
    y_pred = np.argmax(avg_probs, axis=1)

    # ---------- 3. 计算指标 ----------
    acc = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average='macro', zero_division=0)
    recall = recall_score(test_labels, y_pred, average='macro', zero_division=0)
    f1 = f1_score(test_labels, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(test_labels, y_pred)

    # ---------- 4. 打印结果 ----------
    print("\n===== Test Results =====")
    print(f"Sequence      : ALL (Soft Voting) (Fold {fold_idx})")
    print(f"Test samples  : {len(test_labels)}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ================== 主流程 ==================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Starting PyTorch MLP K-Fold Evaluation for: Late Fusion Soft Voting on {device} <<<")

    if args.fold is not None:
        folds_to_run = [args.fold]
        print(f"Mode: Single Fold Evaluation (Fold {args.fold})")
    else:
        folds_to_run = range(1, K_FOLDS + 1)
        print(f"Mode: All {K_FOLDS} Folds Average")

    metrics_history = []

    for k in folds_to_run:
        res = evaluate_mlp_vote_single_fold(k, device)
        if res:
            metrics_history.append(res)
    
    # ---------- 打印综合平均报告 ----------
    if len(metrics_history) > 1:
        print("\n" + "="*50)
        print(f"   K-FOLDS AVERAGE REPORT ({len(metrics_history)} folds)   ")
        print("="*50)

        avg_acc = np.mean([r['acc'] for r in metrics_history])
        std_acc = np.std([r['acc'] for r in metrics_history])
        
        avg_f1 = np.mean([r['f1'] for r in metrics_history])
        std_f1 = np.std([r['f1'] for r in metrics_history])
        
        avg_prec = np.mean([r['precision'] for r in metrics_history])
        std_prec = np.std([r['precision'] for r in metrics_history])
        
        avg_rec = np.mean([r['recall'] for r in metrics_history])
        std_rec = np.std([r['recall'] for r in metrics_history])

        print(f"Method        : Late Fusion Soft Voting")
        print(f"Model         : Simple MLP (PyTorch)")
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

# ================== CLI ==================
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