"""外部独立数据集端到端验证。

外部原始 NIfTI 会严格复用内部 ``preprocess_data.py`` 的预处理函数：
resample -> HD-BET -> mask 内归一化 -> mask 外置零 -> 前景中心裁剪/填充。
SimpleITK 数组直接以 [Z, Y, X] 转为 tensor，与内部 ``load_nii_as_tensor``
最终得到的轴顺序完全一致。

用法（当前 data-hdbet / runs-cross-entropy）：

    python external_eval.py \
        --preprocess-config output/data-hdbet/preprocessing_config.py \
        --checkpoint-root output/runs-cross-entropy \
        --output-root output/runs-cross-entropy \
        --data-root /path/to/external_cases \
        --label 2

只评估一个 fold：

    python external_eval.py \
        --preprocess-config output/data-hdbet/preprocessing_config.py \
        --checkpoint-root output/runs-cross-entropy \
        --output-root output/runs-cross-entropy \
        --data-root /path/to/external_cases \
        --label 2 --fold 1

每个病例目录必须包含可识别的 T1、T2、FLAIR。该脚本只负责外部验证；
内部 meta-fusion 的训练见 ``train_meta_fusion.py``。
"""

import argparse
import csv
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import SimpleITK as sitk
# 关闭 SimpleITK 的底层警告输出，防止控制台刷屏
sitk.ProcessObject_SetGlobalWarningDisplay(False)

from sklearn.metrics import accuracy_score, confusion_matrix

from configs.config_utils import (
    PREPROCESS_CONFIG_FIELDS,
    load_python_config,
    resolve_input_artifact_dir,
)
from configs.global_config import (
    ALL_SEQUENCES,
    CLASS_NAMES,
    K_FOLDS,
    NUM_CLASSES,
)
from scripts.preprocess_data import (
    apply_preprocessing_config,
    preprocess_image,
    select_sequence_files,
    validate_preprocess_setup,
)

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


def resolve_csv_path(args, data_root, target_folds, gt_label, output_root):
    if args.csv_path:
        return Path(args.csv_path).expanduser().resolve()

    folds_name = "all" if args.fold is None else str(target_folds[0])
    filename = f"external_eval_{safe_name(data_root.name)}_label{gt_label}_folds{folds_name}.csv"
    return (output_root / "inference_outputs" / filename).resolve()


def write_diagnostics_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def sitk_image_to_model_tensor(img, expected_shape):
    """把 SimpleITK [Z,Y,X] 数组转换成内部模型使用的 [1,1,D,H,W]。"""
    data = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
    if tuple(data.shape) != tuple(expected_shape):
        raise ValueError(
            f"Preprocessed array shape {tuple(data.shape)} does not match "
            f"configured TARGET_SHAPE {tuple(expected_shape)}"
        )
    data = np.ascontiguousarray(data)
    return torch.from_numpy(data).unsqueeze(0).unsqueeze(0)


def preprocess_nii_to_tensor(nii_path, expected_shape):
    """使用与内部数据完全相同的预处理函数处理一个外部 NIfTI。"""
    img, _meta = preprocess_image(nii_path)
    if img is None:
        return None
    return sitk_image_to_model_tensor(img, expected_shape)


def load_models_for_fold(fold_idx, ckpt_dirs, device):
    models = []
    print(f"\nLoading Heterogeneous Models for fold {fold_idx}...")
    for seq_idx, s_name in enumerate(ALL_SEQUENCES, start=1):
        if s_name == "FLAIR":
            ModelClass = FoundationModel
            target_model_name = "FoundationModel"
        else:
            ModelClass = FoundationModel_ori
            target_model_name = "FoundationModel_ori"

        ckpt_dir = ckpt_dirs[seq_idx - 1] / target_model_name
        ckpt_path = ckpt_dir / f"fold{fold_idx}_model_best.pth"

        if not ckpt_path.exists():
            print(f"[Error] Checkpoint missing for Sequence {s_name}: {ckpt_path}")
            sys.exit(1)

        try:
            model = ModelClass(num_classes=NUM_CLASSES, in_channels=1)
        except TypeError:
            model = ModelClass(num_classes=NUM_CLASSES)

        model = model.to(device)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)

        print(f"  -> Successfully loaded {s_name:<5} from {ckpt_path.name}")

    return models


def resolve_runtime_device(requested_device):
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested_device


def main(args):
    preprocess_config = load_python_config(
        args.preprocess_config,
        PREPROCESS_CONFIG_FIELDS,
    )
    apply_preprocessing_config(preprocess_config)
    validate_preprocess_setup()

    data_root = Path(args.data_root).expanduser().resolve()
    checkpoint_dir = resolve_input_artifact_dir(
        args.checkpoint_root,
        "checkpoints",
    )
    output_root = Path(args.output_root).expanduser().resolve()
    device = resolve_runtime_device(args.device)
    ckpt_dirs = [
        checkpoint_dir / f"seq{seq_id}_{seq_name}"
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
    ]

    gt_label = args.label
    target_folds = [args.fold] if args.fold is not None else list(range(1, K_FOLDS + 1))
    csv_path = resolve_csv_path(
        args,
        data_root,
        target_folds,
        gt_label,
        output_root,
    )

    if not data_root.exists() or not data_root.is_dir():
        print(f"[Error] Data root {data_root} does not exist or is not a directory.")
        sys.exit(1)

    if gt_label < 0 or gt_label >= len(CLASS_NAMES):
        print(f"[Error] Label must be in [0, {len(CLASS_NAMES) - 1}], got {gt_label}.")
        sys.exit(1)

    gt_class_name = CLASS_NAMES[gt_label]
    print(f"\n{'='*20} External Dataset Evaluation {'='*20}")
    print(f"Dataset root : {data_root}")
    print(f"Preprocess   : {preprocess_config.__config_path__}")
    print(f"Checkpoints  : {checkpoint_dir}")
    print(f"Ground Truth : {gt_label} ({gt_class_name})")
    print(f"Using Folds  : {target_folds}")
    print(f"Tensor shape : [1, 1, {', '.join(str(x) for x in preprocess_config.TARGET_SHAPE)}]")
    print(f"Device       : {device}")
    print(f"CSV Output   : {csv_path}")
    print(f"{'='*69}")

    # ================== 1. 扫描有效数据 ==================
    cases = []
    print("\nScanning dataset for complete sequences...")
    for item in sorted(data_root.iterdir()):
        if item.is_dir():
            selected_files, _candidates, missing_seq_ids = select_sequence_files(item)
            if not missing_seq_ids:
                cases.append((item.name, selected_files))
                
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
        models = load_models_for_fold(fold_idx, ckpt_dirs, device)
        pbar = tqdm(cases, desc=f"Evaluating fold {fold_idx}")

        for case_name, seq_maps in pbar:
            tensors = []
            valid = True

            for seq_idx in range(1, 4):
                tensor = preprocess_nii_to_tensor(
                    seq_maps[seq_idx],
                    expected_shape=preprocess_config.TARGET_SHAPE,
                )
                if tensor is None:
                    valid = False
                    break
                tensors.append(tensor.to(device))

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
    parser = argparse.ArgumentParser(
        description="External evaluation using the exact internal preprocessing pipeline."
    )
    parser.add_argument(
        "--preprocess-config",
        required=True,
        help="Path to preprocessing_config.py used to build the internal data experiment.",
    )
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        help="Training experiment root containing checkpoints, or checkpoints itself.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output experiment root; diagnostic CSV defaults to OUTPUT_ROOT/inference_outputs.",
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        required=True,
        help="External root containing one subdirectory per case.",
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
        "--csv-path",
        dest="csv_path",
        type=str,
        default=None,
        help="Optional path for the per-case/per-sequence probability diagnostics CSV."
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Inference device (default: cuda, with automatic CPU fallback).",
    )
    args = parser.parse_args()
    
    main(args)
