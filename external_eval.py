'''
外部独立数据集验证脚本 (End-to-End)
- 遍历外部数据集，提取带有 T1, T2, FLAIR 三种序列的患者目录
- 即时自动应用重采样、归一化和裁剪等预处理流程
- 使用加载的训练好的单通道异构模型进行软投票 (Heterogeneous Late Fusion & Soft Voting)
- 直接输出该数据类别的预测分布以及整体判定准确率
'''

# /home/ailab/data/brainMRI/脑膜转移_外院测试/
# /home/ailab/data/brainMRI/脑炎_外院测试/
# 运行示例：
# python external_eval.py --data_root <外部数据目录> --label <类别编号>

import argparse
import csv
import sys
import os
import tempfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import SimpleITK as sitk
# 关闭 SimpleITK 的底层警告输出，防止控制台刷屏
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import nibabel as nib

from sklearn.metrics import accuracy_score, confusion_matrix

# 导入项目中已有的组件
from runtime_defaults import *
from utils.resample import resample_image
from utils.intensity import normalize_intensity
from utils.spatial import center_crop_or_pad
from utils.sequences import identify_sequence

# 导入异构模型
from models.FoundationModel import FoundationModel
from models.FoundationModel_ori import FoundationModel as FoundationModel_ori

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


DIAGNOSTIC_FIELDS = [
    "case",
    "gt_label",
    "gt_class",
    "fold",
    "level",
    "seq_id",
    "seq_name",
    "model",
    *[f"prob_{class_name}" for class_name in CLASS_NAMES],
    "pred_label",
    "pred_class",
    "is_correct",
]


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def probs_to_row(
    case_name,
    gt_label,
    gt_class_name,
    fold_idx,
    level,
    seq_id,
    seq_name,
    model_name,
    probs,
):
    probs = probs.squeeze(0).detach().cpu().tolist()
    pred_label = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_label] if pred_label < len(CLASS_NAMES) else f"Class {pred_label}"

    row = {
        "case": case_name,
        "gt_label": gt_label,
        "gt_class": gt_class_name,
        "fold": fold_idx,
        "level": level,
        "seq_id": seq_id if seq_id is not None else "",
        "seq_name": seq_name if seq_name is not None else "",
        "model": model_name,
        "pred_label": pred_label,
        "pred_class": pred_class,
        "is_correct": int(pred_label == gt_label),
    }

    for class_idx, class_name in enumerate(CLASS_NAMES):
        row[f"prob_{class_name}"] = probs[class_idx] if class_idx < len(probs) else ""

    return row


def resolve_csv_path(args, data_root, target_folds, gt_label):
    if args.csv_path:
        return Path(args.csv_path).resolve()

    folds_name = "all" if args.fold is None else str(target_folds[0])
    filename = f"external_eval_{safe_name(data_root.name)}_label{gt_label}_folds{folds_name}.csv"
    return (INFERENCE_OUTPUT_DIR / filename).resolve()


def write_diagnostics_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_nii_to_tensor(nii_path):
    """
    接收单个 NIfTI 文件的路径，进行预处理并返回模型可以直接推断的 Tensor
    """
    # 1. 自动读取并重采样
    img = resample_image(nii_path, target_spacing=TARGET_SPACING, is_label=False)
    if img is None:
        return None
    
    # 2. 强度归一化
    img = normalize_intensity(img)
    
    # 3. 中心裁剪或填充
    img = center_crop_or_pad(img, TARGET_SHAPE)
    
    # [核心修复区]
    # SITK与Nibabel的解析坐标轴完全相反。模型训练集是通过 SITK 保存然后由 Nibabel 解析读取的。
    # 必须通过存临时文件并由nib加载，以强制转换出绝不歪曲的严格一致阵列！
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sitk.WriteImage(img, tmp_path)
        nii = nib.load(tmp_path)
        # 获取与训练集统一的 [X, Y, Z] 阵列
        data = nii.get_fdata(dtype=np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    # 转换至 Tensor 并补充 Batch(1) 和 Channel(1) 维度 -> [1, 1, D, H, W]
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    
    return tensor


def load_models_for_fold(fold_idx):
    models = []
    print(f"\nLoading Heterogeneous Models for fold {fold_idx}...")
    for seq_idx, s_name in enumerate(ALL_SEQUENCES, start=1):
        if s_name == "FLAIR":
            ModelClass = FoundationModel
            target_model_name = "FoundationModel"
        else:
            ModelClass = FoundationModel_ori
            target_model_name = "FoundationModel_ori"

        ckpt_dir = CKPT_DIRS[seq_idx - 1] / target_model_name
        ckpt_path = ckpt_dir / f"fold{fold_idx}_model_best.pth"

        if not ckpt_path.exists():
            print(f"[Error] Checkpoint missing for Sequence {s_name}: {ckpt_path}")
            sys.exit(1)

        try:
            model = ModelClass(num_classes=NUM_CLASSES, in_channels=1)
        except TypeError:
            model = ModelClass(num_classes=NUM_CLASSES)

        model = model.to(DEVICE)

        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)

        print(f"  -> Successfully loaded {s_name:<5} from {ckpt_path.name}")

    return models


def main(args):
    data_root = Path(args.data_root).resolve()
    gt_label = args.label
    target_folds = [args.fold] if args.fold is not None else list(range(1, K_FOLDS + 1))
    csv_path = resolve_csv_path(args, data_root, target_folds, gt_label)

    if not data_root.exists() or not data_root.is_dir():
        print(f"[Error] Data root {data_root} does not exist or is not a directory.")
        sys.exit(1)

    if gt_label < 0 or gt_label >= len(CLASS_NAMES):
        print(f"[Error] Label must be in [0, {len(CLASS_NAMES) - 1}], got {gt_label}.")
        sys.exit(1)

    gt_class_name = CLASS_NAMES[gt_label]
    print(f"\n{'='*20} External Dataset Evaluation {'='*20}")
    print(f"Dataset root : {data_root}")
    print(f"Ground Truth : {gt_label} ({gt_class_name})")
    print(f"Using Folds  : {target_folds}")
    print(f"Device       : {DEVICE}")
    print(f"CSV Output   : {csv_path}")
    print(f"{'='*69}")

    # ================== 1. 扫描有效数据 ==================
    cases = []
    print("\nScanning dataset for complete sequences...")
    for item in data_root.iterdir():
        if item.is_dir():
            files = list(item.rglob("*.nii*"))
            seq_maps = {}
            for f in files:
                seq_id = identify_sequence(f)
                if seq_id is not None:
                    seq_maps[seq_id] = f
            
            # 只提取 1, 2, 3 序列齐全的病例
            if 1 in seq_maps and 2 in seq_maps and 3 in seq_maps:
                cases.append((item.name, seq_maps))
                
    if not cases:
        print("[Error] No valid cases with all 3 required sequences (T1, T2, FLAIR) were found.")
        sys.exit(1)
        
    print(f"Found {len(cases)} complete evaluation cases.")

    # ================== 2. 按 fold 加载模型并累加预测概率 ==================
    prob_sums = {}
    prob_counts = {}
    diagnostic_rows = []

    print("\nStarting preprocessing and inference...")
    for fold_idx in target_folds:
        models = load_models_for_fold(fold_idx)
        pbar = tqdm(cases, desc=f"Evaluating fold {fold_idx}")

        for case_name, seq_maps in pbar:
            tensors = []
            valid = True

            for seq_idx in range(1, 4):
                tensor = preprocess_nii_to_tensor(seq_maps[seq_idx])
                if tensor is None:
                    valid = False
                    break
                tensors.append(tensor.to(DEVICE))

            if not valid:
                pbar.write(f"[Warning] Failed to preprocess components for {case_name}, skipping.")
                continue

            models_prob = []
            with torch.no_grad():
                for i in range(3):
                    logits = models[i](tensors[i])
                    prob = F.softmax(logits, dim=1)
                    models_prob.append(prob)
                    seq_id = i + 1
                    seq_name = ALL_SEQUENCES[i]
                    model_name = "FoundationModel" if seq_name == "FLAIR" else "FoundationModel_ori"
                    diagnostic_rows.append(
                        probs_to_row(
                            case_name=case_name,
                            gt_label=gt_label,
                            gt_class_name=gt_class_name,
                            fold_idx=fold_idx,
                            level="sequence",
                            seq_id=seq_id,
                            seq_name=seq_name,
                            model_name=model_name,
                            probs=prob,
                        )
                    )

            fold_prob = (sum(models_prob) / 3.0).detach().cpu()
            diagnostic_rows.append(
                probs_to_row(
                    case_name=case_name,
                    gt_label=gt_label,
                    gt_class_name=gt_class_name,
                    fold_idx=fold_idx,
                    level="fold_vote",
                    seq_id=None,
                    seq_name="ALL",
                    model_name="HeterogeneousSoftVoting",
                    probs=fold_prob,
                )
            )
            prob_sums[case_name] = fold_prob if case_name not in prob_sums else prob_sums[case_name] + fold_prob
            prob_counts[case_name] = prob_counts.get(case_name, 0) + 1

        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================== 3. 汇总预测结果 ==================
    all_preds = []
    incomplete_cases = []

    for case_name in sorted(prob_sums):
        if prob_counts[case_name] != len(target_folds):
            incomplete_cases.append(case_name)
            continue

        avg_prob = prob_sums[case_name] / prob_counts[case_name]
        diagnostic_rows.append(
            probs_to_row(
                case_name=case_name,
                gt_label=gt_label,
                gt_class_name=gt_class_name,
                fold_idx="ALL",
                level="final",
                seq_id=None,
                seq_name="ALL",
                model_name="HeterogeneousSoftVoting",
                probs=avg_prob,
            )
        )
        pred = avg_prob.argmax(dim=1).item()
        all_preds.append(pred)

    processed_count = len(all_preds)

    if processed_count == 0:
        print("\n[Error] No cases were successfully processed. Terminating.")
        return

    if incomplete_cases:
        print(f"\n[Warning] Skipped {len(incomplete_cases)} cases that were not processed by every requested fold.")

    write_diagnostics_csv(csv_path, diagnostic_rows)

    # ================== 4. 统计与报告 ==================
    all_labels = [gt_label] * processed_count
    
    acc = accuracy_score(all_labels, all_preds)
    
    print("\n" + "="*50)
    print(f"            FINAL EXTERNAL REPORT            ")
    print("="*50)
    print(f"Method          : Heterogeneous Late Fusion Soft Voting")
    print(f"Folds Used      : {target_folds}")
    print(f"Target GT Label : {gt_class_name} ({gt_label})")
    print(f"Cases Evaluated : {processed_count}")
    print(f"Accuracy        : {acc * 100:.2f} %")
    print(f"Diagnostics CSV : {csv_path}")
    print("-" * 50)
    
    unique_preds, counts = np.unique(all_preds, return_counts=True)
    print("Prediction Distributions:")
    for p, c in zip(unique_preds, counts):
        pred_class_name = CLASS_NAMES[p] if p < len(CLASS_NAMES) else f"Class {p}"
        print(f"  -> {pred_class_name:<15}: {c:>4}  ({(c / processed_count) * 100:.2f}%)")
        
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
    print(cm)

    final_rows = [row for row in diagnostic_rows if row["level"] == "final"]
    if final_rows:
        target_prob_key = f"prob_{gt_class_name}"
        target_probs = [float(row[target_prob_key]) for row in final_rows if row[target_prob_key] != ""]
        metastasis_probs = []
        if "metastasis" in CLASS_NAMES:
            metastasis_probs = [float(row["prob_metastasis"]) for row in final_rows if row["prob_metastasis"] != ""]
        if target_probs:
            print("\nFinal target-class probability summary:")
            print(
                f"  {gt_class_name:<15}: "
                f"mean={np.mean(target_probs):.4f}, "
                f"std={np.std(target_probs):.4f}, "
                f"min={np.min(target_probs):.4f}, "
                f"max={np.max(target_probs):.4f}"
            )
        if metastasis_probs and gt_class_name != "metastasis":
            print(
                f"  {'metastasis':<15}: "
                f"mean={np.mean(metastasis_probs):.4f}, "
                f"std={np.std(metastasis_probs):.4f}, "
                f"min={np.min(metastasis_probs):.4f}, "
                f"max={np.max(metastasis_probs):.4f}"
            )
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External independent dataset evaluation without global_config binding.")
    parser.add_argument(
        "--data_root", 
        type=str, 
        required=True, 
        help="Path to external dataset root. Expecting subdirectories for each patient containing .nii/.nii.gz files."
    )
    parser.add_argument(
        "--label", 
        type=int, 
        required=True, 
        help=f"Ground truth label integer for this batch of data. (e.g., 0, 1, or 2 based on your logic)."
    )
    parser.add_argument(
        "--fold", 
        type=int, 
        default=None, 
        choices=range(1, K_FOLDS + 1),
        help=f"Which fold checkpoints to load. If not set, ensemble all {K_FOLDS} folds."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Optional path for the per-case/per-sequence probability diagnostics CSV."
    )
    args = parser.parse_args()
    
    main(args)
