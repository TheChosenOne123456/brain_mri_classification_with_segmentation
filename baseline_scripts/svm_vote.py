'''
SVM 晚期融合（软投票）评估脚本
- 直接读取 baseline_cache 中预先提取的 3 个序列的特征数据
- 分别进行 PCA 降维和 SVM (带 probability=True) 拟合
- 使用 predict_proba 提取三个模型的预测概率分布
- 平均概率（软投票）后输出最终指标，实现与深度学习相同的融合评估对比
'''

import argparse
import numpy as np
import time
import sys
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 将项目根目录加入环境变量，以便导入 configs
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from configs.global_config import ALL_SEQUENCES, NUM_CLASSES, CLASS_NAMES, K_FOLDS


def evaluate_svm_vote_single_fold(fold_idx):
    """
    使用三个序列的特征，分别训练带有概率输出的 SVM，然后用软投票评估
    """
    print(f"\n{'='*20} Evaluating Fold {fold_idx} {'='*20}")
    print(f"Mode: Late Fusion (Soft Voting) | Model: SVM (RBF)")

    test_labels = None
    seq_probs = []

    t0 = time.time()
    
    # ---------- 1. 遍历 3 个序列，加载特征并拟合得出概率 ----------
    for seq_id in range(1, len(ALL_SEQUENCES) + 1):
        seq_name = ALL_SEQUENCES[seq_id - 1]
        cache_dir = ROOT_DIR / "baseline_cache" / f"seq{seq_id}_{seq_name}"
        cache_file = cache_dir / f"fold{fold_idx}_features.npz"

        if not cache_file.exists():
            print(f"\n[Error] Cached features not found for {seq_name} at {cache_file}.")
            print(f"Please run 'python baseline_scripts/svm.py --seq {seq_id}' first to extract features.")
            return None

        # 加载缓存 (与 svm.py 完全兼容)
        data = np.load(cache_file)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

        # 初始化地基标签（因为三序列的切分顺序是一致的，使用任意一个即可）
        if test_labels is None:
            test_labels = y_test

        # PCA 降维 (保持跟单通道 svm.py 完全一样的降维逻辑)
        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # 初始化 SVM 训练
        # 【核心修改】：开启 probability=True 才能输出软概率！
        svm_model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        svm_model.fit(X_train_pca.astype(np.float64), y_train)

        # 核心：获取概率概率分布
        # prob_dist shape: [n_samples, num_classes]
        prob_dist = svm_model.predict_proba(X_test_pca.astype(np.float64))
        seq_probs.append(prob_dist)

    print(f"  -> Successfully fitted SVM equipped with probabilities for 3 sequences. (Took {time.time()-t0:.2f}s)")

    # ---------- 2. 测试 (软投票机制) ----------
    # seq_probs 是个 list，包含三个 [n_samples, num_classes] 的 array
    # 取平均作为软投票的最终依据
    avg_probs = np.mean(seq_probs, axis=0)
    
    # 按照平均概率最大的类别作为最终预测
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
    print(f"\n>>> Starting K-Fold Evaluation for: SVM Late Fusion Soft Voting <<<")

    if args.fold is not None:
        folds_to_run = [args.fold]
        print(f"Mode: Single Fold Evaluation (Fold {args.fold})")
    else:
        folds_to_run = range(1, K_FOLDS + 1)
        print(f"Mode: All {K_FOLDS} Folds Average")

    metrics_history = []

    for k in folds_to_run:
        res = evaluate_svm_vote_single_fold(k)
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
        print(f"Model         : SVM (RBF)")
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