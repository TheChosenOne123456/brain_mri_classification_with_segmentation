'''
K-Fold 推理脚本：
输入一个 NIfTI 文件，利用 K 折模型进行综合判断（集成预测）或单折预测
功能：
- 如果指定 --fold N，则只使用该 Fold 的模型进行预测。
- 如果不指定 --fold，则加载所有 K 个模型，取平均概率作为最终结果。
'''

import argparse
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from pathlib import Path

from runtime_defaults import *

from models.cnn3d import Simple3DCNN
from models.ResNet import ResNet10
from utils.train_and_test import set_seed
from utils.resample import resample_image
from utils.spatial import center_crop_or_pad
from utils.intensity import normalize_intensity

# 忽略特定警告
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)


def preprocess_single_case(nii_path, do_preprocess=False):
    """
    对单个 NIfTI 文件进行预处理 (复用 infer.py 的逻辑)
    """
    if do_preprocess:
        print("  [Preprocess] Executing full preprocessing pipeline...")
        # 1. 重采样
        sitk_img = resample_image(nii_path, target_spacing=TARGET_SPACING, is_label=False)
        # 2. 归一化
        norm_np = normalize_intensity(sitk_img)
        # 3. 裁剪/填充
        final_np = center_crop_or_pad(norm_np, TARGET_SHAPE)
    else:
        print("  [Preprocess] Loading preprocessed data directly...")
        nii = nib.load(str(nii_path))
        final_np = nii.get_fdata(dtype=np.float32)

    # 转 Tensor [D, H, W] -> [1, 1, D, H, W]
    tensor = torch.from_numpy(final_np).unsqueeze(0).unsqueeze(0)
    return tensor


def load_model(model_name, ckpt_path):
    """
    加载单个模型权重
    """
    if model_name == "cnn3d":
        model = Simple3DCNN(num_classes=NUM_CLASSES)
    elif model_name == "ResNet":
        model = ResNet10(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(DEVICE)
    
    if not ckpt_path.exists():
        print(f"[Warning] Checkpoint not found: {ckpt_path}")
        return None

    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model
    except Exception as e:
        print(f"[Error] Failed to load checkpoint {ckpt_path}: {e}")
        return None


def main(args):
    set_seed(SEED)
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # ---------- 1. 准备配置 ----------
    seq_id = args.seq
    seq_idx = seq_id - 1
    seq_name = ALL_SEQUENCES[seq_idx]
    model_name = args.model

    print(f"\n=== K-Fold Inference Mode ===")
    print(f"Input File    : {input_path.name}")
    print(f"Sequence      : {seq_name}")
    print(f"Model Type    : {model_name}")
    
    # 确定要使用的 Folds
    if args.fold is not None:
        target_folds = [args.fold]
        print(f"Mode          : Single Fold (Fold {args.fold})")
    else:
        target_folds = range(1, K_FOLDS + 1)
        print(f"Mode          : Ensemble ({K_FOLDS}-Fold Averaging)")

    # ---------- 2. 预处理输入 ----------
    input_tensor = preprocess_single_case(input_path, do_preprocess=args.pre)
    input_tensor = input_tensor.to(DEVICE)

    # ---------- 3. 加载模型并推理 ----------
    fold_probs = []
    valid_folds = 0

    print("\n--- Running Inference ---")
    
    ckpt_base_dir = CKPT_DIRS[seq_idx] / model_name

    with torch.no_grad():
        for k in target_folds:
            ckpt_path = ckpt_base_dir / f"fold{k}_model_best.pth"
            
            # 加载模型
            model = load_model(model_name, ckpt_path)
            if model is None:
                continue

            # 推理
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1) # [1, NUM_CLASSES]
            prob_np = probs.cpu().numpy()[0]
            
            fold_probs.append(prob_np)
            valid_folds += 1
            
            # 打印单折结果 (可选)
            pred_idx = prob_np.argmax()
            print(f"  Fold {k}: Predicted {CLASS_NAMES[pred_idx]} "
                  f"(Conf: {prob_np[pred_idx]:.4f})")

    if valid_folds == 0:
        print("\n[Error] No valid models were loaded. Exiting.")
        return

    # ---------- 4. 汇总结果 (取平均) ----------
    avg_probs = np.mean(fold_probs, axis=0)
    final_pred_idx = avg_probs.argmax()
    final_pred_class = CLASS_NAMES[final_pred_idx]
    final_confidence = avg_probs[final_pred_idx]

    print("\n===== Final Ensemble Prediction =====")
    print(f"Result Class : {final_pred_class.upper()} (Label: {final_pred_idx})")
    print(f"Confidence   : {final_confidence:.4f}")
    
    print("\n[Class Probability Distribution]")
    print("-" * 40)
    print(f"{'Class Name':<20} | {'Probability':<10}")
    print("-" * 40)
    for idx, class_name in enumerate(CLASS_NAMES):
        prob = avg_probs[idx]
        marker = " <--" if idx == final_pred_idx else ""
        print(f"{class_name:<20} | {prob:.4f}{marker}")
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Fold Inference on a single MRI case")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input .nii.gz file",
    )
    parser.add_argument(
        "--seq",
        type=int,
        required=True,
        choices=range(1, NUM_SEQUENCES + 1),
        help=f"Which MRI sequence is this? (1~{NUM_SEQUENCES})",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn3d", "ResNet"],
        help="Which model architecture to use",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, K_FOLDS + 1),
        help=f"Specific fold to use (1~{K_FOLDS}). If not set, use all folds and average.",
    )
    parser.add_argument(
        "--pre",
        action="store_true",
        help="Enable full preprocessing (resample, normalize, crop) if input is raw data",
    )
    
    args = parser.parse_args()
    main(args)
