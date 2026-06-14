'''
K-Fold seq3 分割导出脚本：
- 遍历所有 fold 的测试集（或指定 fold）。
- 使用 seq3 的 FoundationModel 分割头生成预测 Mask。
- 将预测结果保存到 SEG_OUTPUT_DIR/foldx/case_xxxx_FLAIR_mask_pred.nii.gz。
- 若在 mask_index.json 中检测到该 case 存在分割 GT，则复制 GT 到同目录，方便对比。
'''

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.train_config import *
from configs.global_config import *
from models.FoundationModel import FoundationModel
from utils.train_and_test import set_seed, load_pt_dataset

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)


def normalize_case_id(case_id):
    """
    将 case_id 统一成 4 位字符串（如 7 -> 0007）
    """
    s = str(case_id).strip()
    if s.isdigit():
        return f"{int(s):04d}"
    return s


def load_mask_index(mask_index_path: Path):
    """
    读取 mask_index.json
    期望结构示例:
    {
      "0001": [3],
      "0045": [1, 2, 3]
    }
    """
    if not mask_index_path.exists():
        print(f"[Warning] mask_index not found: {mask_index_path}")
        return {}

    with open(mask_index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    norm = {}
    for k, v in data.items():
        ck = normalize_case_id(k)
        seqs = []
        for x in v:
            try:
                seqs.append(int(x))
            except Exception:
                pass
        norm[ck] = seqs
    return norm


def build_case_to_nii_map(pt_dataset):
    """
    从 PTDataset.cases 建立 case_id -> flair_nii_path 映射
    """
    m = {}
    for c in pt_dataset.cases:
        cid = normalize_case_id(c["case_id"])
        m[cid] = Path(c["nii_path"])
    return m


def save_pred_mask(pred_mask_np: np.ndarray, src_nii_path: Path, out_path: Path):
    """
    以源 FLAIR 的 affine/header 保存预测 mask
    pred_mask_np: [D, H, W], dtype uint8/整型
    """
    src_nii = nib.load(str(src_nii_path))
    hdr = src_nii.header.copy()
    hdr.set_data_dtype(np.uint8)

    pred_img = nib.Nifti1Image(pred_mask_np.astype(np.uint8), src_nii.affine, hdr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(pred_img, str(out_path))


def pick_gt_mask_path(case_id: str, flair_nii_path: Path, mask_index: dict):
    """
    基于 mask_index 判断 GT 是否存在，并返回可复制的 GT 文件路径（若有）。
    优先 seq3；若 seq3 不存在则退化到该 case 记录中的第一个序列（因为三序列已对齐，仍可用于对比）。
    """
    if case_id not in mask_index:
        return None, None

    valid_seqs = mask_index.get(case_id, [])
    if not valid_seqs:
        return None, None

    # 优先 FLAIR(seq3)
    if 3 in valid_seqs:
        gt_src = flair_nii_path.parent / f"case_{case_id}_3_mask.nii.gz"
        if gt_src.exists():
            return gt_src, 3

    # 回退：找任意可用 seq
    for seq_id in valid_seqs:
        gt_src = flair_nii_path.parent.parent / str(seq_id) / f"case_{case_id}_{seq_id}_mask.nii.gz"
        if gt_src.exists():
            return gt_src, seq_id

    return None, None


def load_seq3_model(ckpt_path: Path, device: str):
    model = FoundationModel(num_classes=NUM_CLASSES, in_channels=1)
    model = model.to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    msg = model.load_state_dict(state, strict=False)

    if len(getattr(msg, "missing_keys", [])) > 0 or len(getattr(msg, "unexpected_keys", [])) > 0:
        print(f"[Warning] Non-strict load at {ckpt_path.name}")
        if len(msg.missing_keys) > 0:
            print(f"  missing_keys: {msg.missing_keys}")
        if len(msg.unexpected_keys) > 0:
            print(f"  unexpected_keys: {msg.unexpected_keys}")

    if torch.cuda.device_count() > 1 and str(device).startswith("cuda"):
        model = nn.DataParallel(model)

    model.eval()
    return model


def export_one_fold(
    fold_idx: int,
    args,
    mask_index: dict,
):
    print(f"\n{'='*20} Exporting Fold {fold_idx} {'='*20}")

    seq3_dataset_dir = DATASET_DIRS[2] / f"fold{fold_idx}"
    ckpt_path = CKPT_DIRS[2] / "FoundationModel" / f"fold{fold_idx}_model_best.pth"

    if not seq3_dataset_dir.exists():
        print(f"[Warning] Missing dataset dir: {seq3_dataset_dir}")
        return {"ok": 0, "pred_saved": 0, "gt_copied": 0}

    if not ckpt_path.exists():
        print(f"[Warning] Missing checkpoint: {ckpt_path}")
        return {"ok": 0, "pred_saved": 0, "gt_copied": 0}

    test_pt = seq3_dataset_dir / "test.pt"
    if not test_pt.exists():
        print(f"[Warning] Missing test.pt: {test_pt}")
        return {"ok": 0, "pred_saved": 0, "gt_copied": 0}

    print("  -> Loading test set...")
    test_set = load_pt_dataset(test_pt)
    case_to_nii = build_case_to_nii_map(test_set)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(str(args.device).startswith("cuda")),
    )

    print("  -> Loading seq3 FoundationModel checkpoint...")
    model = load_seq3_model(ckpt_path, args.device)

    out_fold_dir = SEG_OUTPUT_DIR / f"fold{fold_idx}"
    out_fold_dir.mkdir(parents=True, exist_ok=True)

    pred_saved = 0
    gt_copied = 0
    total = 0

    print("  -> Running segmentation inference and exporting NIfTI...")
    with torch.inference_mode():
        for batch in test_loader:
            # 兼容 PTDataset 返回: x, y, mask, has_mask, case_id
            # 以及旧结构: x, y, case_id
            if len(batch) == 5:
                x, _, _, _, case_ids = batch
            else:
                x, _, case_ids = batch

            x = x.to(args.device, non_blocking=True)

            # seq3 FoundationModel 分割头
            if hasattr(model, "module"):
                _, seg_logits = model.module(x, return_seg=True)
            else:
                _, seg_logits = model(x, return_seg=True)

            # [B, C, D, H, W] -> [B, D, H, W]
            pred_masks = seg_logits.argmax(dim=1).cpu().numpy()

            for i in range(len(case_ids)):
                total += 1
                case_id = normalize_case_id(case_ids[i])
                if case_id not in case_to_nii:
                    print(f"[Warning] case_id not found in mapping: {case_id}")
                    continue

                flair_nii_path = case_to_nii[case_id]

                # 1) 保存预测
                pred_name = f"case_{case_id}_FLAIR_mask_pred.nii.gz"
                pred_out = out_fold_dir / pred_name
                if pred_out.exists() and (not args.overwrite):
                    pass
                else:
                    save_pred_mask(pred_masks[i], flair_nii_path, pred_out)
                pred_saved += 1

                # 2) 复制 GT（若存在）
                gt_src, gt_seq = pick_gt_mask_path(case_id, flair_nii_path, mask_index)
                if gt_src is not None:
                    if gt_seq == 3:
                        gt_name = f"case_{case_id}_FLAIR_mask_gt.nii.gz"
                    else:
                        gt_name = f"case_{case_id}_FLAIR_mask_gt_from_seq{gt_seq}.nii.gz"

                    gt_out = out_fold_dir / gt_name
                    if gt_out.exists() and (not args.overwrite):
                        pass
                    else:
                        shutil.copy2(str(gt_src), str(gt_out))
                    gt_copied += 1

    print(f"  -> Fold {fold_idx} done. Samples={total}, PredSaved={pred_saved}, GTCopied={gt_copied}")
    return {"ok": 1, "pred_saved": pred_saved, "gt_copied": gt_copied}


def main(args):
    set_seed(SEED)

    if args.device is None:
        args.device = DEVICE
    if args.batch_size is None:
        args.batch_size = BATCH_SIZE
    if args.num_workers is None:
        args.num_workers = NUM_WORKERS

    print("\n>>> Start exporting seq3 segmentation masks for test sets <<<")
    print(f"Device       : {args.device}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Num workers  : {args.num_workers}")
    print(f"Output root  : {SEG_OUTPUT_DIR.resolve()}")

    mask_index_path = PROCESSED_DATA_PATH / "mask_index.json"
    mask_index = load_mask_index(mask_index_path)
    print(f"Mask index   : {mask_index_path} (cases={len(mask_index)})")

    if args.fold is not None:
        folds = [args.fold]
        print(f"Mode         : Single fold ({args.fold})")
    else:
        folds = list(range(1, K_FOLDS + 1))
        print(f"Mode         : All {K_FOLDS} folds")

    SEG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sum_pred = 0
    sum_gt = 0
    ok_folds = 0

    for k in folds:
        res = export_one_fold(k, args, mask_index)
        if res["ok"] == 1:
            ok_folds += 1
            sum_pred += res["pred_saved"]
            sum_gt += res["gt_copied"]

    print("\n" + "=" * 52)
    print("Export Summary")
    print("=" * 52)
    print(f"Successful folds : {ok_folds}/{len(folds)}")
    print(f"Pred masks saved : {sum_pred}")
    print(f"GT masks copied  : {sum_gt}")
    print("=" * 52)

    if ok_folds == 0:
        print("[Error] No fold exported successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export seq3 FoundationModel segmentation masks for all fold test sets"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, K_FOLDS + 1),
        help=f"Specific fold to export (1~{K_FOLDS}). If not set, export all folds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to use, e.g. "cuda" or "cpu". Default from train_config.DEVICE',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size. Default from train_config.BATCH_SIZE",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override num_workers. Default from train_config.NUM_WORKERS",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if set.",
    )
    args = parser.parse_args()
    main(args)