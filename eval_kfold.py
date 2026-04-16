'''
K-Fold 评估脚本：
功能与 eval.py 类似，但支持 K-Fold 交叉验证模型。
- 如果指定 --fold N，则只评估第 N 折。
- 如果不指定 --fold，则自动评估所有 fold 并计算平均指标。
- 支持单通道(指定--seq) 与 多通道(不指定--seq) 模型评估。
[适配新服务器：8x RTX 3080]
'''

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from configs.train_config import *
from configs.global_config import *

from models.cnn3d import Simple3DCNN
from models.ResNet import ResNet10, ResNet18
from models.FoundationModel import FoundationModel
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


# ================== [新增专区：Dice 计算] ==================
def compute_dice(pred_mask, gt_mask, num_classes=3, smooth=1e-5):
    """
    计算整个 batch 的每个样本的平均 Dice
    pred_mask: [B, D, H, W]
    gt_mask: [B, D, H, W]
    """
    dices = []
    # 忽略类别 0 (背景/正常)，仅计算异常类别（如 1:炎症, 2:转移瘤）的 Dice
    for c in range(1, num_classes):
        pred_c = (pred_mask == c).float()
        gt_c = (gt_mask == c).float()

        intersection = (pred_c * gt_c).sum(dim=(1, 2, 3))
        cardinality = pred_c.sum(dim=(1, 2, 3)) + gt_c.sum(dim=(1, 2, 3))

        dice = (2. * intersection + smooth) / (cardinality + smooth)
        dices.append(dice)
    
    # 返回每个样本的平均 Dice: shape [B]
    return torch.stack(dices, dim=0).mean(dim=0)
# =========================================================


class MultiChannelDataset(Dataset):
    """
    代理 Dataset：将多个单通道的 Dataset 在被调用时动态组合成多通道。
    与 train_kfold.py 保持完全一致。
    支持兼容新版带mask的数据结构。
    """
    def __init__(self, datasets_list):
        self.datasets = datasets_list
        self.labels = datasets_list[0].labels

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        # 取第一个序列看看结构是 3 元素还是 5 元素
        item0 = self.datasets[0][idx]
        has_seg_data = (len(item0) == 5)

        xs = []
        final_mask = None
        final_has_mask = None
        final_y = None
        final_case_id = None

        for i, ds in enumerate(self.datasets):
            item = ds[idx]
            xs.append(item[0]) # x 是第一个元素

            if i == 0:
                final_y = item[1]
                if has_seg_data:
                    final_mask = item[2]
                    final_has_mask = item[3]
                    final_case_id = item[4]
                else:
                    final_case_id = item[2]

            # 如果这批数据是带 Mask 的新数据，我们就要找出真正的 Mask 主人
            if has_seg_data:
                current_has_mask = item[3]
                if current_has_mask > 0.5:
                    final_mask = item[2]
                    final_has_mask = item[3]

        multi_x = torch.cat(xs, dim=0)

        if has_seg_data:
            return multi_x, final_y, final_mask, final_has_mask, final_case_id
        else:
            return multi_x, final_y, final_case_id


def evaluate_single_fold(args_seq, model_name, fold_idx, ModelClass):
    """
    评估单个 Fold 的核心函数
    """
    # ---------- 判断并加载数据 ----------
    if args_seq is not None:
        # 单通道模式
        seq_idx = args_seq - 1
        seq_name = ALL_SEQUENCES[seq_idx]
        in_channels = 1
        
        dataset_dir = DATASET_DIRS[seq_idx] / f"fold{fold_idx}"
        ckpt_dir = CKPT_DIRS[seq_idx] / model_name
        ckpt_path = ckpt_dir / f"fold{fold_idx}_model_best.pth"

        if not dataset_dir.exists():
            print(f"\n[Warning] Dataset for fold {fold_idx} not found at {dataset_dir}. Skipping.")
            return None
        if not ckpt_path.exists():
            print(f"\n[Warning] Checkpoint for fold {fold_idx} not found at {ckpt_path}. Skipping.")
            return None
            
        test_set = load_pt_dataset(dataset_dir / "test.pt")
        
    else:
        # 多通道模式
        in_channels = len(ALL_SEQUENCES)
        seq_name = f"Multi-Fusion ({in_channels} Channels)"
        
        base_ckpt_dir = CKPT_DIRS[0].parent  
        ckpt_dir = base_ckpt_dir / "multi_channel" / model_name
        ckpt_path = ckpt_dir / f"fold{fold_idx}_model_best.pth"
        
        if not ckpt_path.exists():
            print(f"\n[Warning] Checkpoint for fold {fold_idx} not found at {ckpt_path}. Skipping.")
            return None

        test_sets_list = []
        import time
        print(f"  -> Loading test sets for {seq_name}... ", end="", flush=True)
        t0 = time.time()
        for idx, s_name in enumerate(ALL_SEQUENCES):
            d_dir = DATASET_DIRS[idx] / f"fold{fold_idx}"
            if not d_dir.exists():
                print(f"\n[Warning] Dataset missing for sequence {s_name} at {d_dir}. Skipping.")
                return None
            test_sets_list.append(load_pt_dataset(d_dir / "test.pt"))
        print(f"Done in {time.time()-t0:.1f}s")
        
        test_set = MultiChannelDataset(test_sets_list)

    print(f"\n{'='*20} Evaluating Fold {fold_idx} {'='*20}")
    print(f"Mode: {seq_name} | Model: {model_name}")

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ---------- 初始化并加载模型 ----------
    try:
        model = ModelClass(num_classes=NUM_CLASSES, in_channels=in_channels)
    except TypeError:
        print(f"[Warning] {model_name} does not accept 'in_channels'. Using default.")
        model = ModelClass(num_classes=NUM_CLASSES)
        
    model = model.to(DEVICE)
    
    # 训练时我们保存的是 model.module (如果用了多卡)，所以这里直接加载字典是对应的
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    # [修改点] 加载完权重后再启用 DataParallel 进行推理加速
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation!")
        model = nn.DataParallel(model)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    # ---------- 测试 ----------
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    valid_seg_dices = []
    misclassified_cases = []

    with torch.no_grad():
        for batch_data in test_loader:
            has_seg_data = (len(batch_data) == 5)
            
            if has_seg_data:
                x, y, masks, has_masks, case_ids = batch_data
                masks = masks.to(DEVICE)
            else:
                x, y, case_ids = batch_data[:3]
                
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # 判断是否能展开分割头（仅 FoundationModel 支持且需要开启 return_seg）
            if model_name == "FoundationModel" and has_seg_data:
                logits, seg_logits = model(x, True)
            else:
                logits = model(x)
                seg_logits = None
                
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            
            # --- 收集分割 Dice (若存在) ---
            if seg_logits is not None and has_seg_data:
                pred_masks = seg_logits.argmax(dim=1)
                
                # 自动解除多余的维度，比如 [B, 1, D, H, W] -> [B, D, H, W]
                if masks.dim() == 5 and masks.size(1) == 1:
                    masks = masks.squeeze(1)
                
                batch_dices = compute_dice(pred_masks, masks, num_classes=NUM_CLASSES)
                
                for i in range(len(case_ids)):
                    y_cls = int(y[i])
                    h_m = bool(has_masks[i])
                    # 包含专家标注 Mask 的患者，或是本身健康的患者（天然全零Mask）
                    if h_m or (y_cls == 0):
                        valid_seg_dices.append(batch_dices[i].item())

            preds_cpu = preds.cpu().numpy()
            labels_cpu = y.cpu().numpy()

            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)

            # --------- 收集误判 case ---------
            for cid, p, gt in zip(case_ids, preds_cpu, labels_cpu):
                if p != gt:
                    misclassified_cases.append({
                        "case_id": cid,
                        "gt": int(gt),
                        "pred": int(p),
                    })

    avg_loss = total_loss / len(test_loader)
    avg_dice = np.mean(valid_seg_dices) if len(valid_seg_dices) > 0 else 0.0

    # ---------- 计算指标 ----------
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # ---------- 打印结果 ----------
    print("\n===== Test Results =====")
    print(f"Sequence      : {seq_name} (Fold {fold_idx})")
    print(f"Test samples  : {len(test_set)}")
    print(f"Test loss     : {avg_loss:.4f}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")
    if len(valid_seg_dices) > 0:
        print(f"Seg Dice      : {avg_dice:.4f}  (Evaluated on {len(valid_seg_dices)} samples)")

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

    # ---------- 打印误判 case ----------
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
        "f1": f1,
        "loss": avg_loss,
        "dice": avg_dice
    }


# ================== 主流程 ==================
def main(args):
    set_seed(SEED)

    model_name = args.model
    if model_name == "cnn3d":
        ModelClass = Simple3DCNN
    elif model_name == "ResNet":
        ModelClass = ResNet10
    elif model_name == "ResNet18":
        ModelClass = ResNet18
    elif model_name == "FoundationModel":
        ModelClass = FoundationModel
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if args.seq is not None:
        seq_name = ALL_SEQUENCES[args.seq - 1]
    else:
        seq_name = f"Multi-Fusion ({len(ALL_SEQUENCES)} Channels)"

    print(f"\n>>> Starting K-Fold Evaluation for: {seq_name} <<<")

    # ---------- 确定要评估的 fold 列表 ----------
    if args.fold is not None:
        folds_to_run = [args.fold]
        print(f"Mode: Single Fold Evaluation (Fold {args.fold})")
    else:
        folds_to_run = range(1, K_FOLDS + 1)
        print(f"Mode: All {K_FOLDS} Folds Average")

    metrics_history = []

    # ---------- 循环评估 ----------
    for k in folds_to_run:
        res = evaluate_single_fold(args.seq, model_name, k, ModelClass)
        if res:
            metrics_history.append(res)
    
    # ---------- 这里如果是多折评估，打印平均值 ----------
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
        
        has_dice = any(r['dice'] > 0 for r in metrics_history)
        avg_dice = np.mean([r['dice'] for r in metrics_history]) if has_dice else 0.0
        std_dice = np.std([r['dice'] for r in metrics_history]) if has_dice else 0.0

        print(f"Sequence      : {seq_name}")
        print(f"Model         : {model_name}")
        print("-" * 40)
        print(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
        print("-" * 40)
        print(f"{'Accuracy':<15} | {avg_acc:.4f}     | ±{std_acc:.4f}")
        print(f"{'Precision':<15} | {avg_prec:.4f}     | ±{std_prec:.4f}")
        print(f"{'Recall':<15} | {avg_rec:.4f}     | ±{std_rec:.4f}")
        print(f"{'F1-Score':<15} | {avg_f1:.4f}     | ±{std_f1:.4f}")
        if has_dice:
            print(f"{'Seg Dice':<15} | {avg_dice:.4f}     | ±{std_dice:.4f}")
        print("-" * 40)
    elif len(metrics_history) == 0:
        print("\n[Error] No folds were successfully evaluated.")

# ================== CLI ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [修改] required=False, 允许为空以触发多通道评估
    parser.add_argument(
        "--seq",
        type=int,
        required=False,
        default=None,
        help="Which MRI sequence to evaluate (1~3). Leave empty for Multi-Channel.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn3d", "ResNet", "ResNet18", "FoundationModel"],
        help="Which model architecture to use",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, K_FOLDS + 1),
        help=f"Specific fold to evaluate (1~{K_FOLDS}). If not set, run all folds.",
    )
    args = parser.parse_args()

    main(args)