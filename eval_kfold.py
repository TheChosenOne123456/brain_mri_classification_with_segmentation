'''
K-Fold 评估脚本：
功能与 eval.py 类似，但支持 K-Fold 交叉验证模型。
- 如果指定 --fold N，则只评估第 N 折。
- 如果不指定 --fold，则自动评估所有 fold 并计算平均指标。
- 支持单通道(指定--seq) 与 多通道(不指定--seq) 模型评估。
[适配新服务器：8x RTX 3080]

层级模型 Fold 1 单序列评估示例（只评估已经训练完成的 checkpoint）：
python eval_kfold.py \
  --config output/runs-hierarchical/train_config.py \
  --data-root output/data-hdbet \
  --checkpoint-root output/runs-hierarchical \
  --seq 1 --model FoundationModelHierarchical --fold 1

训练仍占用全部 GPU 时，可在另一终端限制评估只使用物理 GPU 7 和 batch 1：
CUDA_VISIBLE_DEVICES=7 python eval_kfold.py \
  --config output/runs-hierarchical/train_config.py \
  --data-root output/data-hdbet \
  --checkpoint-root output/runs-hierarchical \
  --seq 1 --model FoundationModelHierarchical --fold 1 \
  --batch-size 1 --num-workers 4

将 --seq 1 替换为 2 或 3，可分别评估 T2 或 FLAIR。省略 --fold 时会
评估该序列所有已存在的 fold；缺少 checkpoint 的 fold 会被跳过。
'''

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from configs.config_utils import (
    infer_data_dir,
    load_python_config,
    resolve_input_artifact_dir,
)
from configs.global_config import (
    ALL_SEQUENCES,
    CLASS_NAMES,
    K_FOLDS,
    NUM_CLASSES,
    SEED,
)
from runtime_defaults import DATA_EXPERIMENT_ROOT, TRAIN_EXPERIMENT_ROOT

from models.model_factory import (
    MODEL_CHOICES,
    create_model,
    forward_model,
    hierarchical_predictions,
    model_capabilities,
)
from utils.train_and_test import set_seed, load_pt_dataset
from utils.segmentation import (
    binary_dice_per_sample,
    binary_lesion_predictions,
)

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


EVAL_RUNTIME_CONFIG_FIELDS = ("BATCH_SIZE", "DEVICE", "NUM_WORKERS")


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


def calculate_classification_metrics(labels, predictions):
    return {
        "acc": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "cm": confusion_matrix(labels, predictions),
    }


def print_classification_metrics(title, metrics):
    print(f"\n===== {title} =====")
    print(f"Accuracy      : {metrics['acc']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1-score      : {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics["cm"])


def evaluate_single_fold(args_seq, model_name, fold_idx):
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
            
        test_set = load_pt_dataset(
            dataset_dir / "test.pt",
            data_root=EVAL_DATA_ROOT,
        )
        
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
            test_sets_list.append(
                load_pt_dataset(
                    d_dir / "test.pt",
                    data_root=EVAL_DATA_ROOT,
                )
            )
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
    model = create_model(
        model_name,
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        sequence_id=args_seq,
    )
    capabilities = model_capabilities(model)
    print(f"Model Capabilities: {capabilities}")
    model = model.to(DEVICE)
    
    # 训练时我们保存的是 model.module (如果用了多卡)，所以这里直接加载字典是对应的
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    # [修改点] 加载完权重后再启用 DataParallel 进行推理加速
    if str(DEVICE).startswith("cuda") and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation!")
        model = nn.DataParallel(model)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    # ---------- 测试 ----------
    all_preds = []
    all_main_preds = []
    all_labels = []
    all_subtype_preds = []
    all_subtype_labels = []
    total_loss = 0.0
    
    valid_seg_dices = []
    positive_lesion_dices = []
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
            
            outputs = forward_model(
                model,
                x,
                return_subtype=capabilities["subtype"],
                return_seg=capabilities["segmentation"] and has_seg_data,
            )
            logits = outputs["classification"]
            seg_logits = outputs.get("segmentation")
                
            loss = criterion(logits, y)

            total_loss += loss.item()
            main_preds = logits.argmax(dim=1)
            if capabilities["subtype"]:
                subtype_logits = outputs["subtype"]
                preds = hierarchical_predictions(logits, subtype_logits)
                abnormal = y > 0
                if abnormal.any():
                    all_subtype_preds.extend(
                        subtype_logits[abnormal].argmax(dim=1).cpu().numpy()
                    )
                    all_subtype_labels.extend((y[abnormal] - 1).cpu().numpy())
            else:
                preds = main_preds
            
            # --- 收集分割 Dice (若存在) ---
            if seg_logits is not None and has_seg_data:
                # 自动解除多余的维度，比如 [B, 1, D, H, W] -> [B, D, H, W]
                if masks.dim() == 5 and masks.size(1) == 1:
                    masks = masks.squeeze(1)

                if seg_logits.size(1) == 1:
                    pred_masks = binary_lesion_predictions(seg_logits)
                    batch_dices = binary_dice_per_sample(pred_masks, masks)
                else:
                    pred_masks = seg_logits.argmax(dim=1)
                    batch_dices = compute_dice(
                        pred_masks,
                        masks,
                        num_classes=NUM_CLASSES,
                    )
                batch_binary_dices = binary_dice_per_sample(
                    pred_masks > 0,
                    masks > 0,
                )
                
                for i in range(len(case_ids)):
                    y_cls = int(y[i])
                    h_m = bool(has_masks[i])
                    # 包含专家标注 Mask 的患者，或是本身健康的患者（天然全零Mask）
                    if h_m or (y_cls == 0):
                        valid_seg_dices.append(batch_dices[i].item())
                    if h_m and y_cls > 0:
                        positive_lesion_dices.append(
                            batch_binary_dices[i].item()
                        )

            preds_cpu = preds.cpu().numpy()
            main_preds_cpu = main_preds.cpu().numpy()
            labels_cpu = y.cpu().numpy()

            all_preds.extend(preds_cpu)
            all_main_preds.extend(main_preds_cpu)
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
    avg_positive_lesion_dice = (
        np.mean(positive_lesion_dices)
        if len(positive_lesion_dices) > 0
        else 0.0
    )

    # ---------- 计算指标 ----------
    metrics = calculate_classification_metrics(all_labels, all_preds)
    main_metrics = calculate_classification_metrics(all_labels, all_main_preds)

    # ---------- 打印结果 ----------
    print("\n===== Test Results =====")
    print(f"Sequence      : {seq_name} (Fold {fold_idx})")
    print(f"Test samples  : {len(test_set)}")
    print(f"Test loss     : {avg_loss:.4f}")
    if capabilities["subtype"]:
        print_classification_metrics("Main Head Results", main_metrics)
        print_classification_metrics("Hierarchical Results", metrics)
        subtype_metrics = calculate_classification_metrics(
            all_subtype_labels,
            all_subtype_preds,
        )
        print_classification_metrics(
            "Subtype Head on Ground-Truth Abnormal Cases",
            subtype_metrics,
        )
    else:
        print(f"Accuracy      : {metrics['acc']:.4f}")
        print(f"Precision     : {metrics['precision']:.4f}")
        print(f"Recall        : {metrics['recall']:.4f}")
        print(f"F1-score      : {metrics['f1']:.4f}")
    if len(valid_seg_dices) > 0:
        print(f"Seg Dice      : {avg_dice:.4f}  (Evaluated on {len(valid_seg_dices)} samples)")
    if len(positive_lesion_dices) > 0:
        print(
            "Positive Lesion Dice: "
            f"{avg_positive_lesion_dice:.4f}  "
            f"(Evaluated on {len(positive_lesion_dices)} masked abnormal samples)"
        )

    if not capabilities["subtype"]:
        print("\nConfusion Matrix:")
        print(metrics["cm"])

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
        "acc": metrics["acc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "main_acc": main_metrics["acc"],
        "main_f1": main_metrics["f1"],
        "has_subtype": capabilities["subtype"],
        "loss": avg_loss,
        "dice": avg_dice,
        "positive_lesion_dice": avg_positive_lesion_dice,
    }


# ================== 主流程 ==================
def main(args):
    config = load_python_config(args.config, EVAL_RUNTIME_CONFIG_FIELDS)
    for name in EVAL_RUNTIME_CONFIG_FIELDS:
        globals()[name] = getattr(config, name)
    if args.device is not None:
        globals()["DEVICE"] = args.device
    if args.batch_size is not None:
        globals()["BATCH_SIZE"] = args.batch_size
    if args.num_workers is not None:
        globals()["NUM_WORKERS"] = args.num_workers

    dataset_root = resolve_input_artifact_dir(args.data_root, "datasets")
    ckpt_root = resolve_input_artifact_dir(args.checkpoint_root, "checkpoints")
    globals()["EVAL_DATA_ROOT"] = infer_data_dir(args.data_root)
    globals()["DATASET_DIRS"] = [
        dataset_root / f"seq{seq_id}_{seq_name}"
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
    ]
    globals()["CKPT_DIRS"] = [
        ckpt_root / f"seq{seq_id}_{seq_name}"
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
    ]

    set_seed(SEED)

    model_name = args.model

    if args.seq is not None:
        seq_name = ALL_SEQUENCES[args.seq - 1]
    else:
        seq_name = f"Multi-Fusion ({len(ALL_SEQUENCES)} Channels)"

    print(f"\n>>> Starting K-Fold Evaluation for: {seq_name} <<<")
    print(f"Evaluation config: {config.__config_path__}")
    print(f"Dataset input   : {dataset_root}")
    print(f"Checkpoint input: {ckpt_root}")

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
        res = evaluate_single_fold(args.seq, model_name, k)
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
        if any(r["has_subtype"] for r in metrics_history):
            avg_main_acc = np.mean([r["main_acc"] for r in metrics_history])
            avg_main_f1 = np.mean([r["main_f1"] for r in metrics_history])
            print(f"{'Main-head Acc':<15} | {avg_main_acc:.4f}     |")
            print(f"{'Main-head F1':<15} | {avg_main_f1:.4f}     |")
        if has_dice:
            print(f"{'Seg Dice':<15} | {avg_dice:.4f}     | ±{std_dice:.4f}")
        positive_dices = [
            r["positive_lesion_dice"]
            for r in metrics_history
            if r["positive_lesion_dice"] > 0
        ]
        if positive_dices:
            print(
                f"{'Positive Dice':<15} | {np.mean(positive_dices):.4f}     | "
                f"±{np.std(positive_dices):.4f}"
            )
        print("-" * 40)
    elif len(metrics_history) == 0:
        print("\n[Error] No folds were successfully evaluated.")

# ================== CLI ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(TRAIN_EXPERIMENT_ROOT / "train_config.py"),
        help="Training config used for runtime device and DataLoader settings",
    )
    parser.add_argument(
        "--data-root",
        default=str(DATA_EXPERIMENT_ROOT),
        help="Experiment root containing datasets (default: output/data-hdbet)",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=str(TRAIN_EXPERIMENT_ROOT),
        help=(
            "Training output root containing checkpoints "
            "(default: output/runs-cross-entropy)"
        ),
    )
    # [修改] required=False, 允许为空以触发多通道评估
    parser.add_argument(
        "--seq",
        type=int,
        choices=range(1, len(ALL_SEQUENCES) + 1),
        required=False,
        default=None,
        help="Which MRI sequence to evaluate (1~3). Leave empty for Multi-Channel.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_CHOICES,
        help="Which model architecture to use",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, K_FOLDS + 1),
        help=f"Specific fold to evaluate (1~{K_FOLDS}). If not set, run all folds.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional runtime override, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluation batch-size override.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker-count override.",
    )
    args = parser.parse_args()

    main(args)
