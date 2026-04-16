'''
K-Fold 软投票评估脚本 (Late Fusion / Soft Voting)：
- 加载已经训练好的单通道模型 (T1, T2, FLAIR)。
- 对同一个样本，分别获取三个模型的预测概率，取平均值后进行最终决策。
- 如果指定 --fold N，则只评估第 N 折。
- 如果不指定 --fold，则自动评估所有 fold 并计算平均指标。
'''

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from configs.train_config import *
from configs.global_config import *

from models.cnn3d import Simple3DCNN
from models.ResNet import ResNet10
from models.FoundationModel import FoundationModel
# [新增] 引入旧版原始模型
from models.FoundationModel_ori import FoundationModel as FoundationModel_ori
from utils.train_and_test import set_seed, load_pt_dataset

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


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
            # 必须修改解包，加上 mask_tensor 和 has_mask_flag
            x, y, mask_tensor, has_mask_flag, case_id = ds[idx]  
            xs.append(x) # 保持独立的 [1, D, H, W]
        
        # 同样这里也要适配解包
        _, y, _, _, case_id = self.datasets[0][idx]
        return xs, y, case_id
# ================================================


def evaluate_vote_single_fold(fold_idx):
    """
    使用三个模型进行软投票的评估函数
    """
    print(f"\n{'='*20} Evaluating Fold {fold_idx} {'='*20}")
    print(f"Mode: Late Fusion (Soft Voting) | Model: Heterogeneous Ensemble")

    # ---------- 1. 加载数据 ----------
    test_sets_list = []
    print("  -> Loading test sets for all 3 sequences... ", end="", flush=True)
    t0 = time.time()
    for idx, s_name in enumerate(ALL_SEQUENCES):
        d_dir = DATASET_DIRS[idx] / f"fold{fold_idx}"
        if not d_dir.exists():
            print(f"\n[Warning] Dataset missing for {s_name} at {d_dir}. Skipping fold {fold_idx}.")
            return None
        test_sets_list.append(load_pt_dataset(d_dir / "test.pt"))
    print(f"Done in {time.time()-t0:.1f}s")
    
    test_set = MultiSequenceDataset(test_sets_list)
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # ---------- 2. 初始化并加载 3 个模型 ----------
    models = []
    for seq_idx, s_name in enumerate(ALL_SEQUENCES):
        # 根据通道决定使用旧版(单任务)还是新版(双头)
        if s_name == "FLAIR":
            target_model_name = "FoundationModel"
            ModelClass = FoundationModel
        else:
            target_model_name = "FoundationModel_ori"
            ModelClass = FoundationModel_ori

        ckpt_dir = CKPT_DIRS[seq_idx] / target_model_name
        ckpt_path = ckpt_dir / f"fold{fold_idx}_model_best.pth"
        
        if not ckpt_path.exists():
            print(f"\n[Warning] Model checkpoint missing for {s_name} at {ckpt_path}. Skipping fold {fold_idx}.")
            return None
        
        # 实例化单通道模型
        try:
            model = ModelClass(num_classes=NUM_CLASSES, in_channels=1)
        except TypeError:
            model = ModelClass(num_classes=NUM_CLASSES)
            
        model = model.to(DEVICE)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        model.eval()
        models.append(model)
        
    print("  -> Successfully loaded 3 heterogeneous models (T1_ori, T2_ori, FLAIR_mt).")

    # ---------- 3. 测试 (软投票机制) ----------
    all_preds = []
    all_labels = []
    misclassified_cases = []

    # 注意：投票模式下不计算 Loss，因为直接比较概率
    with torch.no_grad():
        for xs, y, case_ids in test_loader:
            y = y.to(DEVICE)
            
            # xs 包含 3 个 batch tensor (T1, T2, FLAIR)
            probs = []
            for i, x in enumerate(xs):
                x = x.to(DEVICE)
                
                logits = models[i](x)
                
                # 将 logits 转换为概率分布
                prob = F.softmax(logits, dim=1)
                probs.append(prob)
            
            # --- 核心：平均概率 (Soft Voting) ---
            # 也可以在这里改成加权平均，例如:
            # avg_prob = 0.5 * probs[0] + 0.25 * probs[1] + 0.25 * probs[2]
            avg_prob = (probs[0] + probs[1] + probs[2]) / 3.0
            
            # 最终预测结果
            preds = avg_prob.argmax(dim=1)

            preds_cpu = preds.cpu().numpy()
            labels_cpu = y.cpu().numpy()

            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)

            # 收集误判 case
            for cid, p, gt in zip(case_ids, preds_cpu, labels_cpu):
                if p != gt:
                    misclassified_cases.append({
                        "case_id": cid,
                        "gt": int(gt),
                        "pred": int(p),
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
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    # ---------- 6. 打印误判 case ----------
    print("\n===== Misclassified Cases =====")
    print(f"Total misclassified: {len(misclassified_cases)}")

    if len(misclassified_cases) > 0:
        for item in misclassified_cases:
            print(
                f"CaseID: {item['case_id']} | "
                f"GT: {item['gt']} | Pred: {item['pred']}"
            )
    else:
        print("None")

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ================== 主流程 ==================
def main(args):
    set_seed(SEED)

    print(f"\n>>> Starting K-Fold Evaluation for: Late Fusion Soft Voting (Heterogeneous) <<<")

    if args.fold is not None:
        folds_to_run = [args.fold]
        print(f"Mode: Single Fold Evaluation (Fold {args.fold})")
    else:
        folds_to_run = range(1, K_FOLDS + 1)
        print(f"Mode: All {K_FOLDS} Folds Average")

    metrics_history = []

    for k in folds_to_run:
        res = evaluate_vote_single_fold(k)
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

        print(f"Method        : Late Fusion Soft Voting (Heterogeneous Ensemble)")
        print(f"Models        : Seq1/Seq2=FoundationModel_ori, Seq3=FoundationModel")
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
    # 移除了 --model 参数配置
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, K_FOLDS + 1),
        help=f"Specific fold to evaluate (1~{K_FOLDS}). If not set, run all folds.",
    )
    args = parser.parse_args()

    main(args)