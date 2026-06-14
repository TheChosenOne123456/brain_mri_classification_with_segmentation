'''
外部独立数据集验证脚本 (End-to-End)
- 遍历外部数据集，提取带有 T1, T2, FLAIR 三种序列的患者目录
- 即时自动应用重采样、归一化和裁剪等预处理流程
- 使用加载的训练好的单通道异构模型进行软投票 (Heterogeneous Late Fusion & Soft Voting)
- 直接输出该数据类别的预测分布以及整体判定准确率
'''

import argparse
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
from configs.train_config import *
from configs.global_config import *
from utils.resample import resample_image
from utils.intensity import normalize_intensity
from utils.spatial import center_crop_or_pad
from utils.sequences import identify_sequence

# 导入异构模型
from models.FoundationModel import FoundationModel
from models.FoundationModel_ori import FoundationModel as FoundationModel_ori

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


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


def main(args):
    data_root = Path(args.data_root).resolve()
    gt_label = args.label
    fold_idx = args.fold

    if not data_root.exists() or not data_root.is_dir():
        print(f"[Error] Data root {data_root} does not exist or is not a directory.")
        sys.exit(1)

    gt_class_name = CLASS_NAMES[gt_label] if gt_label < len(CLASS_NAMES) else f"Class {gt_label}"
    print(f"\n{'='*20} External Dataset Evaluation {'='*20}")
    print(f"Dataset root : {data_root}")
    print(f"Ground Truth : {gt_label} ({gt_class_name})")
    print(f"Using Fold   : {fold_idx}")
    print(f"Device       : {DEVICE}")
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

    # ================== 2. 加载选折模型 ==================
    models = []
    print("\nLoading Heterogeneous Models...")
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

    # ================== 3. 执行评估与软投票 ==================
    all_preds = []
    processed_count = 0
    
    print("\nStarting preprocessing and inference...")
    pbar = tqdm(cases, desc=f"Evaluating out of {len(cases)} cases")
    
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
        
        avg_prob = sum(models_prob) / 3.0
        
        pred = avg_prob.argmax(dim=1).item()
        all_preds.append(pred)
        processed_count += 1

    if processed_count == 0:
        print("\n[Error] No cases were successfully processed. Terminating.")
        return

    # ================== 4. 统计与报告 ==================
    all_labels = [gt_label] * processed_count
    
    acc = accuracy_score(all_labels, all_preds)
    
    print("\n" + "="*50)
    print(f"            FINAL EXTERNAL REPORT            ")
    print("="*50)
    print(f"Method          : Late Fusion Soft Voting")
    print(f"Target GT Label : {gt_class_name} ({gt_label})")
    print(f"Cases Evaluated : {processed_count}")
    print(f"Accuracy        : {acc * 100:.2f} %")
    print("-" * 50)
    
    unique_preds, counts = np.unique(all_preds, return_counts=True)
    print("Prediction Distributions:")
    for p, c in zip(unique_preds, counts):
        pred_class_name = CLASS_NAMES[p] if p < len(CLASS_NAMES) else f"Class {p}"
        print(f"  -> {pred_class_name:<15}: {c:>4}  ({(c / processed_count) * 100:.2f}%)")
        
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
    print(cm)
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
        default=1, 
        choices=range(1, K_FOLDS + 1),
        help="Which fold checkpoints to load (default: 1)."
    )
    args = parser.parse_args()
    
    main(args)