'''
临床 K-Fold 集成推理脚本：
输入一个已经被预处理过的 Case ID（如 "0001"），脚本将：
- 自动在当前默认数据实验目录(output/data-hdbet/data)下检索其 Seq1, Seq2, Seq3 的图像。
- baseline 模式保持原异构模型组合。
- hierarchical 模式使用三个独立训练的层级模型。
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

from runtime_defaults import *

from configs.config_utils import (
    infer_data_dir,
    resolve_input_artifact_dir,
)
from models.model_factory import (
    create_model,
    forward_model,
    model_capabilities,
)

from utils.dataset import load_nii_as_tensor
from utils.train_and_test import set_seed

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

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
    affine = nii.affine

    # 复用训练数据加载器的 X,Y,Z -> Z,Y,X 转置，保证输入轴顺序一致。
    tensor = load_nii_as_tensor(nii_path).unsqueeze(0)
    return tensor, affine


def get_model_instance(seq_idx, model_set):
    if model_set == "hierarchical":
        model_name = "FoundationModelHierarchical"
    elif seq_idx == 3:
        model_name = "FoundationModel"
    else:
        model_name = "FoundationModel_ori"
    model = create_model(
        model_name,
        num_classes=NUM_CLASSES,
        in_channels=1,
        sequence_id=seq_idx,
    )
    return model, model_name


def main(args):
    set_seed(SEED)
    
    case_id = args.id
    target_folds = [args.fold] if args.fold is not None else range(1, K_FOLDS + 1)
    
    data_root = infer_data_dir(args.data_root)
    if data_root is None:
        raise FileNotFoundError(
            f"Could not resolve a preprocessed data directory from {args.data_root}"
        )
    ckpt_root = resolve_input_artifact_dir(
        args.checkpoint_root,
        "checkpoints",
    )
    ckpt_dirs = [
        ckpt_root / f"seq{seq_id}_{seq_name}"
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
    ]
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = (
        output_root
        if output_root.name == "infer_output"
        else output_root / "infer_output"
    )
    
    # ---------- 1. 查找并加载患者数据 ----------
    nii_paths, groundtruth = find_case_files(case_id, data_root)
    original_id = get_original_case_id(case_id, data_root)
    
    print(f"\n{'='*40}")
    print(f"=== Clinical Inference ({args.model_set}) ===")
    print(f"Preprocessed ID  : {case_id}")
    print(f"Original ID      : {original_id}")
    print(f"Ground Truth     : {groundtruth.upper() if groundtruth else 'Unknown'}")
    print(f"Checkpoint Root  : {ckpt_root}")
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
    all_fold_probs = {
        1: [],
        2: [],
        3: [],
        "fused": [],
        "main_fused": [],
    }
    all_fold_subtype_probs = []
    all_fold_masks = []  # 仅来自于 Seq3
    valid_folds = 0

    print("\n--- Running Inference ---")
    
    with torch.no_grad():
        for k in target_folds:
            fold_probs = {}
            fold_subtype_probs = {}
            seq3_mask_fold = None
            fold_success = True

            # 遍历三个序列提取特征
            for seq_idx, tensor_input in zip([1, 2, 3], [tensor_seq1, tensor_seq2, tensor_seq3]):
                model, model_name = get_model_instance(
                    seq_idx,
                    args.model_set,
                )
                ckpt_path = ckpt_dirs[seq_idx - 1] / model_name / f"fold{k}_model_best.pth"

                if not ckpt_path.exists():
                    print(f"[Warning] Checkpoint missing for Seq {seq_idx} at Fold {k}. Skipping this fold.")
                    fold_success = False
                    break
                
                model = model.to(DEVICE)
                checkpoint = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(checkpoint["model_state"])
                
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)
                    
                model.eval()

                capabilities = model_capabilities(model)
                outputs = forward_model(
                    model,
                    tensor_input,
                    return_subtype=capabilities["subtype"],
                    return_seg=capabilities["segmentation"],
                )
                logits = outputs["classification"]
                prob_np = F.softmax(logits, dim=1).cpu().numpy()[0]

                if capabilities["subtype"]:
                    fold_subtype_probs[seq_idx] = (
                        F.softmax(outputs["subtype"], dim=1).cpu().numpy()[0]
                    )
                if capabilities["segmentation"]:
                    segmentation_logits = outputs["segmentation"]
                    if segmentation_logits.size(1) == 1:
                        seq3_mask_fold = (
                            torch.sigmoid(segmentation_logits)
                            .ge(0.5)
                            .squeeze(1)
                            .cpu()
                            .numpy()[0]
                        )
                    else:
                        seq3_mask_fold = (
                            segmentation_logits.argmax(dim=1).cpu().numpy()[0]
                        )

                fold_probs[seq_idx] = prob_np

            if not fold_success:
                continue

            # 对于这个 Fold，计算 Late Fusion (简单平均软投票)
            fused_main_prob = np.mean(
                [fold_probs[1], fold_probs[2], fold_probs[3]],
                axis=0,
            )
            if args.model_set == "hierarchical":
                fused_subtype_prob = np.mean(
                    [
                        fold_subtype_probs[1],
                        fold_subtype_probs[2],
                        fold_subtype_probs[3],
                    ],
                    axis=0,
                )
                abnormal_mass = 1.0 - fused_main_prob[0]
                fused_prob = np.array(
                    [
                        fused_main_prob[0],
                        abnormal_mass * fused_subtype_prob[0],
                        abnormal_mass * fused_subtype_prob[1],
                    ]
                )
                fold_pred_idx = (
                    0
                    if fused_main_prob.argmax() == 0
                    else int(fused_subtype_prob.argmax()) + 1
                )
                all_fold_subtype_probs.append(fused_subtype_prob)
            else:
                fused_prob = fused_main_prob
                fold_pred_idx = int(fused_prob.argmax())

            # 记录当前 fold 产生的结果
            all_fold_probs[1].append(fold_probs[1])
            all_fold_probs[2].append(fold_probs[2])
            all_fold_probs[3].append(fold_probs[3])
            all_fold_probs["fused"].append(fused_prob)
            all_fold_probs["main_fused"].append(fused_main_prob)
            all_fold_masks.append(seq3_mask_fold)
            
            valid_folds += 1
            print(
                f"  [Fold {k}] Integration Complete. "
                f"Pred = {CLASS_NAMES[fold_pred_idx]} "
                f"(display score: {fused_prob[fold_pred_idx]:.4f})"
            )

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
    seq_suffix = "hier" if args.model_set == "hierarchical" else "baseline"
    print(
        f"{'Class Name':<15} | {'Seq1 (' + seq_suffix + ')':<16} | "
        f"{'Seq2 (' + seq_suffix + ')':<16} | "
        f"{'Seq3 (' + seq_suffix + ')':<16} || {'FUSED':<12}"
    )
    print("-" * 75)
    
    if args.model_set == "hierarchical":
        avg_main_prob = np.mean(all_fold_probs["main_fused"], axis=0)
        avg_subtype_prob = np.mean(all_fold_subtype_probs, axis=0)
        fused_pred_idx = (
            0
            if avg_main_prob.argmax() == 0
            else int(avg_subtype_prob.argmax()) + 1
        )
    else:
        fused_pred_idx = int(avg_probs["fused"].argmax())
    
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
        for label_val in np.unique(stacked_masks):
            # 统计有多少个 Fold 把这个像素指派给了 label_val
            votes_for_label = np.sum(stacked_masks == label_val, axis=0)
            # 如果支持这个 label 的 fold 数量超过了半数，就把对应像素赋值过去 
            # (简化的多数投票。如果是偶数 fold，可能会有并列情况，目前策略是谁数字大被谁覆盖)
            final_mask[votes_for_label > (valid_folds / 2)] = label_val
            
        output_dir.mkdir(parents=True, exist_ok=True)
        out_nii_path = output_dir / f"case_{case_id}_FLAIR_mask_pred.nii.gz"
        
        # 模型内部统一使用 [Z, Y, X]；写回 NIfTI 时恢复 nibabel 的 [X, Y, Z]。
        final_mask_nifti = np.transpose(final_mask, (2, 1, 0))
        pred_nii = nib.Nifti1Image(final_mask_nifti, affine_seq3)
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
    parser.add_argument(
        "--model-set",
        choices=("baseline", "hierarchical"),
        default="baseline",
        help="Choose the existing heterogeneous ensemble or the hierarchical models.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DATA_EXPERIMENT_ROOT),
        help="Data experiment root containing preprocessed data.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=str(TRAIN_EXPERIMENT_ROOT),
        help="Training output root containing checkpoints.",
    )
    parser.add_argument(
        "--output-root",
        default=str(TRAIN_EXPERIMENT_ROOT),
        help="Output root; masks are written under infer_output.",
    )
    
    args = parser.parse_args()
    main(args)
