'''
临床 K-Fold 异构集成推理脚本 (Clinical Inference via Heterogeneous Late Fusion)：
输入一个已经被预处理过的 Case ID（如 "0001"），脚本将：
- 自动在当前实验版本目录(version1/data)下检索其 Seq1, Seq2, Seq3 的图像。
- Seq1 和 Seq2 使用原版无分割模型 (FoundationModel_ori) 提取概率。
- Seq3 使用多任务模型 (FoundationModel) 提取概率并输出病灶分割 Mask。
- 最终打印三模态的融合投票结果，并将 Mask 保存到当前目录的 /infer_output 下。
'''

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from pathlib import Path

from configs.train_config import *
from configs.global_config import *

from models.FoundationModel import FoundationModel
from models.FoundationModel_ori import FoundationModel as FoundationModel_ori

from utils.train_and_test import set_seed

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

# 定义最终 Mask 的输出目录
OUTPUT_DIR = Path(EXPERIMENT_VERSION) / "infer_output"

def find_case_files(case_id, data_root):
    """
    根据给定的 Case ID，在 data 目录下遍历所有类别子目录，
    找出这个人的 Seq1, Seq2, Seq3 三个通道的预处理 NIfTI 绝对路径。
    同时返回解析到的实际地被类别名称 (groundtruth)。
    """
    nii_paths = {1: None, 2: None, 3: None}
    groundtruth_class = None
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")
        
    for label_dir in data_root.iterdir():
        if not label_dir.is_dir() or label_dir.name.endswith('.json'):
            continue
            
        for seq_idx in [1, 2, 3]:
            potential_file = label_dir / str(seq_idx) / f"case_{case_id}_{seq_idx}.nii.gz"
            if potential_file.exists():
                nii_paths[seq_idx] = potential_file
                # 解析诸如 "0_normal", "1_inflammation" 等格式的字符串，提取字母部分
                groundtruth_class = label_dir.name.split('_', 1)[-1]
                
    return nii_paths, groundtruth_class

def get_original_case_id(case_str_id, data_root):
    """
    从 case_index.json 中根据预处理后的自增 int 编号搜索出原始挂载数据时的医院原始号。
    case_str_id: '0001' 等格式的字符串。
    """
    index_path = data_root / "case_index.json"
    if not index_path.exists():
        return "Unknown (case_index.json not found)"
    try:
        case_int_id = int(case_str_id)
        with open(index_path, "r", encoding="utf-8") as f:
            case_index_data = json.load(f)
            # JSON 中的结构：{ "32302165664": 1, ... }
            for original_id, val in case_index_data.items():
                if val == case_int_id:
                    return original_id
    except ValueError:
        pass
    
    return "Unknown"


def load_tensor_from_nii(nii_path):
    """
    直接加载预处理好的等大小 NIfTI，转为模型所需的 [1, 1, D, H, W]
    并返回 affine 矩阵，以便后续保存为相同物理空间的图像。
    """
    nii = nib.load(str(nii_path))
    final_np = nii.get_fdata(dtype=np.float32)
    affine = nii.affine
    
    # 转 Tensor [D, H, W] -> [1, 1, D, H, W]
    tensor = torch.from_numpy(final_np).unsqueeze(0).unsqueeze(0)
    return tensor, affine


def get_model_instance(seq_idx):
    if seq_idx == 3:
        model = FoundationModel(num_classes=NUM_CLASSES, in_channels=1)
        model_name = "FoundationModel"
    else:
        model = FoundationModel_ori(num_classes=NUM_CLASSES, in_channels=1)
        model_name = "FoundationModel_ori"
    return model, model_name


def main(args):
    set_seed(SEED)
    
    case_id = args.id
    target_folds = [args.fold] if args.fold is not None else range(1, K_FOLDS + 1)
    
    data_root = PROCESSED_DATA_PATH # e.g. version1/data
    
    # ---------- 1. 查找并加载患者数据 ----------
    nii_paths, groundtruth = find_case_files(case_id, data_root)
    original_id = get_original_case_id(case_id, data_root)
    
    print(f"\n{'='*40}")
    print(f"=== Clinical Inference (Heterogeneous) ===")
    print(f"Preprocessed ID  : {case_id}")
    print(f"Original ID      : {original_id}")
    print(f"Ground Truth     : {groundtruth.upper() if groundtruth else 'Unknown'}")
    print(f"Mode             : {'Single Fold (Fold ' + str(args.fold) + ')' if args.fold else 'Ensemble (' + str(K_FOLDS) + '-Fold Averaging)'}")
    print(f"{'='*40}")
    
    missing_seqs = [seq for seq, p in nii_paths.items() if p is None]
    if len(missing_seqs) > 0:
        print(f"[Error] The following sequences are missing for case {case_id}: {missing_seqs}")
        print("Ensure the case ID exists in PROCESSED_DATA_PATH.")
        return

    print("  -> Found Sequence Data:")
    for seq, p in nii_paths.items():
        print(f"     - Seq {seq}: {p.parent.parent.name}/{p.name}")

    # 载入 Tensor ([1, 1, D, H, W])
    # T1, T2 用于分类；Seq3(FLAIR) 用于分类 + 分割。
    tensor_seq1, _ = load_tensor_from_nii(nii_paths[1])
    tensor_seq2, _ = load_tensor_from_nii(nii_paths[2])
    tensor_seq3, affine_seq3 = load_tensor_from_nii(nii_paths[3])
    
    tensor_seq1 = tensor_seq1.to(DEVICE)
    tensor_seq2 = tensor_seq2.to(DEVICE)
    tensor_seq3 = tensor_seq3.to(DEVICE)

    # 存储最终每个序列/融合的预测和掩码
    all_fold_probs = {1: [], 2: [], 3: [], "fused": []}  # 用于保存每个 Fold 的概率分布
    all_fold_masks = []  # 仅来自于 Seq3
    valid_folds = 0

    print("\n--- Running Inference ---")
    
    with torch.no_grad():
        for k in target_folds:
            fold_probs = {}
            seq3_mask_fold = None
            fold_success = True

            # 遍历三个序列提取特征
            for seq_idx, tensor_input in zip([1, 2, 3], [tensor_seq1, tensor_seq2, tensor_seq3]):
                model, model_name = get_model_instance(seq_idx)
                ckpt_path = CKPT_DIRS[seq_idx - 1] / model_name / f"fold{k}_model_best.pth"

                if not ckpt_path.exists():
                    print(f"[Warning] Checkpoint missing for Seq {seq_idx} at Fold {k}. Skipping this fold.")
                    fold_success = False
                    break
                
                model = model.to(DEVICE)
                checkpoint = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(checkpoint["model_state"], strict=False)
                
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                    
                model.eval()

                # 推理
                if seq_idx == 3:
                     # Seq3: 既要概率，也要分割图
                    # 对于 FoundationModel, 当 return_seg=True 时，返回 (cls_logits, seg_logits)
                    if hasattr(model, "module"):
                        logits, seg_logits = model.module(tensor_input, return_seg=True)
                    else:
                        logits, seg_logits = model(tensor_input, return_seg=True)
                    
                    # 提取概率
                    prob_np = F.softmax(logits, dim=1).cpu().numpy()[0]
                    # 提取掩码 (argmax 得到纯类别标记 [D, H, W])
                    mask_pred = seg_logits.argmax(dim=1).cpu().numpy()[0]  
                    seq3_mask_fold = mask_pred
                else:
                    # Seq1, Seq2: 纯分类
                    logits = model(tensor_input)
                    prob_np = F.softmax(logits, dim=1).cpu().numpy()[0]

                fold_probs[seq_idx] = prob_np

            if not fold_success:
                continue

            # 对于这个 Fold，计算 Late Fusion (简单平均软投票)
            fused_prob = np.mean([fold_probs[1], fold_probs[2], fold_probs[3]], axis=0)

            # 记录当前 fold 产生的结果
            all_fold_probs[1].append(fold_probs[1])
            all_fold_probs[2].append(fold_probs[2])
            all_fold_probs[3].append(fold_probs[3])
            all_fold_probs["fused"].append(fused_prob)
            all_fold_masks.append(seq3_mask_fold)
            
            valid_folds += 1
            print(f"  [Fold {k}] Integration Complete. Pred = {CLASS_NAMES[fused_prob.argmax()]} (Conf: {fused_prob.max():.4f})")

    if valid_folds == 0:
        print("\n[Error] No valid models were loaded across any fold. Exiting.")
        return

    # ---------- 3. 汇总 K-Fold 结果 (分类) ----------
    print("\n===== Final Probability Distribution =====")
    # 取全部有结果 Fold 的平局值
    avg_probs = {
        1: np.mean(all_fold_probs[1], axis=0),
        2: np.mean(all_fold_probs[2], axis=0),
        3: np.mean(all_fold_probs[3], axis=0),
        "fused": np.mean(all_fold_probs["fused"], axis=0),
    }

    print("-" * 75)
    print(f"{'Class Name':<15} | {'Seq1 (T1_ori)':<12} | {'Seq2 (T2_ori)':<12} | {'Seq3 (FLAIR_mt)':<14} || {'FUSED (VOTE)':<12}")
    print("-" * 75)
    
    fused_pred_idx = avg_probs["fused"].argmax()
    
    for idx, class_name in enumerate(CLASS_NAMES):
        p1 = avg_probs[1][idx]
        p2 = avg_probs[2][idx]
        p3 = avg_probs[3][idx]
        pf = avg_probs["fused"][idx]
        
        marker = " <*" if idx == fused_pred_idx else ""
        print(f"{class_name:<15} | {p1:.4f}       | {p2:.4f}       | {p3:.4f}         || {pf:.4f}{marker}")
    print("-" * 75)
    
    final_pred_class = CLASS_NAMES[fused_pred_idx]
    print(f"\n>>> FINAL DIAGNOSIS : {final_pred_class.upper()} <<<")


    # ---------- 4. 汇总 K-Fold 结果 (Mask 保存) ----------
    # 对于多折算出的多个 Mask，通过多数投票（Majority Voting）进行合并生成最终强鲁棒性掩码
    if len(all_fold_masks) > 0:
        # 堆叠所有的 mask [N_folds, D, H, W]
        stacked_masks = np.stack(all_fold_masks, axis=0) 
        final_mask = np.zeros_like(stacked_masks[0], dtype=np.uint8)
        
        # 遍历所有可能的标签值 (背景 0, 炎症 1, 转移瘤 2 等)
        # 求包含像素最多的那个类别作为这个像素点的最终分类
        for label_val in range(NUM_CLASSES): # 也可以换成从stacked里找unique
            # 统计有多少个 Fold 把这个像素指派给了 label_val
            votes_for_label = np.sum(stacked_masks == label_val, axis=0)
            # 如果支持这个 label 的 fold 数量超过了半数，就把对应像素赋值过去 
            # (简化的多数投票。如果是偶数 fold，可能会有并列情况，目前策略是谁数字大被谁覆盖)
            final_mask[votes_for_label > (valid_folds / 2)] = label_val
            
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_nii_path = OUTPUT_DIR / f"case_{case_id}_FLAIR_mask_pred.nii.gz"
        
        # 将 Numpy 保存为带有正确物理信息的 NIfTI
        pred_nii = nib.Nifti1Image(final_mask, affine_seq3)
        nib.save(pred_nii, out_nii_path)
        
        print(f"\n[Artifact Output] Predicted Seq3 Mask saved to:")
        print(f"  -> {out_nii_path.absolute()}")
        print("  (You can open this .nii.gz file via ITK-SNAP or 3D Slicer over the FLAIR scan)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Heterogeneous Late Fusion Inference")
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="The Preprocessed Case ID (e.g. 0001) to infer on.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, K_FOLDS + 1),
        help=f"Specific fold to use (1~{K_FOLDS}). If not set, use all {K_FOLDS} folds and average.",
    )
    
    args = parser.parse_args()
    main(args)