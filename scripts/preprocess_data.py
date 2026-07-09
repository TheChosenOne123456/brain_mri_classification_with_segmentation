'''
数据预处理，实现重采样、归一化和裁剪/填充
[重构版 - 功能完全复刻]
'''
import argparse
import csv
import importlib.util
import re
from pathlib import Path
from tqdm import tqdm

import SimpleITK as sitk
# [新增] 关闭 SimpleITK 的底层警告输出，防止刷屏
sitk.ProcessObject_SetGlobalWarningDisplay(False)

from configs.global_config import *

from utils.sequences import identify_sequence
from utils.data_scan import collect_cases
from utils.io import load_index, save_index, INDEX_FILE_NAME
from utils.resample import resample_image, save_image
from utils.intensity import normalize_intensity, zero_outside_mask
from utils.spatial import center_crop_or_pad, crop_or_pad_around_mask
from utils.brain_extraction import dilate_mask, extract_foreground_mask_hd_bet
from utils.preprocess_qc import validate_preprocessed_image, validate_saved_file_size


ERROR_LOG_FIELDS = ["case_id", "case_key", "seq_id", "nii_file", "stage", "error"]


def append_error_log(log_path, row):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def validate_preprocess_setup():
    if BRAIN_EXTRACTOR == "hd-bet" and importlib.util.find_spec("HD_BET") is None:
        raise RuntimeError(
            "HD-BET is selected but not installed in the active Python environment. "
            "Install it with `pip install hd-bet`, or set `BRAIN_EXTRACTOR = 'none'` "
            "in configs/global_config.py to use the legacy preprocessing flow."
        )

    if BRAIN_EXTRACTOR not in {"hd-bet", "none"}:
        raise ValueError(f"Unsupported brain extractor: {BRAIN_EXTRACTOR}")

    if INTENSITY_CLIP_PERCENTILES is not None:
        low, high = INTENSITY_CLIP_PERCENTILES
        if not (0 <= low < high <= 100):
            raise ValueError(
                f"Invalid clip percentiles: {INTENSITY_CLIP_PERCENTILES}. "
                "Expected 0 <= LOW < HIGH <= 100."
            )

    if not (0 <= PREPROCESS_MAX_ZERO_RATIO <= 1):
        raise ValueError(
            f"Invalid PREPROCESS_MAX_ZERO_RATIO: {PREPROCESS_MAX_ZERO_RATIO}"
        )

    if not (0 <= PREPROCESS_MIN_NONZERO_BBOX_FRACTION <= 1):
        raise ValueError(
            "Invalid PREPROCESS_MIN_NONZERO_BBOX_FRACTION: "
            f"{PREPROCESS_MIN_NONZERO_BBOX_FRACTION}"
        )


def preprocess_image(nii_file):
    """
    单个 NIfTI 的预处理入口。

    新流程：
    resample -> HD-BET foreground mask -> mask 内归一化 -> mask 外置 0 -> 以 mask bbox 中心裁剪/填充。
    旧流程可通过 global_config.py 中的 BRAIN_EXTRACTOR = "none" 回退。
    """
    resampled_img = resample_image(
        nii_file,
        target_spacing=TARGET_SPACING,
        is_label=False,
    )
    if resampled_img is None:
        return None

    if BRAIN_EXTRACTOR == "none":
        normalized_img = normalize_intensity(resampled_img)
        return center_crop_or_pad(normalized_img, TARGET_SHAPE)

    foreground_mask = extract_foreground_mask_hd_bet(
        resampled_img,
        device=HD_BET_DEVICE,
        mode=HD_BET_MODE,
        do_tta=HD_BET_TTA,
        postprocess=True,
        target_orientation=HD_BET_TARGET_ORIENTATION,
        verbose=HD_BET_VERBOSE,
    )
    foreground_mask = dilate_mask(foreground_mask, FOREGROUND_DILATION_MM)

    normalized_img = normalize_intensity(
        resampled_img,
        mask_img=foreground_mask,
        clip_percentiles=INTENSITY_CLIP_PERCENTILES,
        robust=INTENSITY_ROBUST_ZSCORE,
    )

    if FOREGROUND_ZERO_OUTSIDE:
        normalized_img = zero_outside_mask(normalized_img, foreground_mask)

    return crop_or_pad_around_mask(normalized_img, foreground_mask, TARGET_SHAPE)


def main(args):
    validate_preprocess_setup()

    # 路径解析
    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    index_path = out_root / INDEX_FILE_NAME
    error_log_path = out_root / "preprocess_errors.csv"

    # ===== 读取 / 初始化 index =====
    case_index = load_index(index_path)

    # 恢复计数器状态
    if case_index:
        try:
            # 兼容旧索引结构：Key是原始路径/文件名，Value是int ID
            max_case_id = max(case_index.values())
        except ValueError:
            max_case_id = 0
    else:
        max_case_id = 0

    print(f"Starting preprocessing for {NUM_CLASSES} classes: {CLASS_NAMES}")
    print(f"Current Max Case ID: {max_case_id}")

    # ===== [核心修改] 动态遍历所有类别 =====
    # 逻辑：遍历 global_config 中定义的所有类别，而不是写死 normal/meningitis
    for label_id, label_name in enumerate(CLASS_NAMES):
        subdirs = CLASS_DATA_MAP[label_name]

        # 构造源目录列表
        target_dirs = []
        for subdir in subdirs:
            # 支持相对路径拼接
            d = raw_root / subdir
            if d.exists():
                target_dirs.append(d)
        
        if not target_dirs:
            print(f"No valid source directories for class {label_name}, skipping.")
            continue

        # 1. 收集该类别的所有原始 Case
        cases = collect_cases(target_dirs)
        
        # 2. 预创建输出目录结构
        # 格式：out_root / "{label_name}" / "{seq_id}" (e.g. 0_normal/1)
        # 注意：这里 label_name 已经包含了 label_id 前缀? 
        # 不，global_config 的 key 是 pure name (e.g. "normal")
        # 旧代码的输出目录名是 "0_normal", "1_meningitis"
        # 这里需要手动拼接 label_id 和 label_name 以保持一致
        
        # 构造符合旧代码习惯的文件夹名：id_name
        # e.g. 0_normal
        output_class_dir = f"{label_id}_{label_name}"

        for seq_id in range(1, NUM_SEQUENCES + 1):
            (out_root / output_class_dir / str(seq_id)).mkdir(parents=True, exist_ok=True)

        # 3. 处理该类别下的所有 Case
        desc = f"Processing {label_name} cases"
        
        for case_dir in tqdm(cases, desc=desc):
            # --- 提取唯一 Case Key (保持原逻辑) ---
            folder_name = case_dir.name
            match = re.findall(r'\d+', folder_name)
            if match:
                case_key = match[-1]
            else:
                case_key = folder_name

            # --- 增量检查 ---
            if case_key in case_index:
                continue

            # --- 分配新 ID ---
            max_case_id += 1
            # 格式化为 0001 (str)
            case_id_str = f"{max_case_id:04d}"
            # 记录索引
            case_index[case_key] = max_case_id

            # --- 扫描并处理 NIfTI 文件 ---
            # 使用 rglob 递归查找，与旧代码一致
            for nii_file in case_dir.rglob("*.nii*"):
                # [关键修正] 传入 Path 对象
                seq_id = identify_sequence(nii_file)
                
                if seq_id is None:
                    continue

                try:
                    fixed_img = preprocess_image(nii_file)
                    if fixed_img is None:
                        tqdm.write(f"[Warning] Failed to load/resample file: {nii_file.resolve()} | ID: {case_id_str}")
                        append_error_log(
                            error_log_path,
                            {
                                "case_id": case_id_str,
                                "case_key": case_key,
                                "seq_id": seq_id,
                                "nii_file": str(nii_file.resolve()),
                                "stage": "load_or_resample",
                                "error": "resample_image returned None",
                            },
                        )
                        continue

                    # 保存模型输入图像；foreground mask 只在预处理过程中临时使用
                    out_name = f"case_{case_id_str}_{seq_id}.nii.gz"
                    out_path = out_root / output_class_dir / str(seq_id) / out_name

                    ok, reasons = validate_preprocessed_image(
                        fixed_img,
                        target_shape=TARGET_SHAPE,
                        target_spacing=TARGET_SPACING,
                        max_zero_ratio=PREPROCESS_MAX_ZERO_RATIO,
                        min_nonzero_bbox_fraction=PREPROCESS_MIN_NONZERO_BBOX_FRACTION,
                    )
                    if not ok:
                        tqdm.write(
                            f"[Warning] QC failed before save: {nii_file.resolve()} | "
                            f"ID: {case_id_str} | reasons: {';'.join(reasons)}"
                        )
                        append_error_log(
                            error_log_path,
                            {
                                "case_id": case_id_str,
                                "case_key": case_key,
                                "seq_id": seq_id,
                                "nii_file": str(nii_file.resolve()),
                                "stage": "pre_save_qc",
                                "error": ";".join(reasons),
                            },
                        )
                        continue

                    if not out_path.exists():
                        save_image(fixed_img, out_path)

                    ok, reasons = validate_saved_file_size(
                        out_path,
                        min_file_size_mb=PREPROCESS_MIN_FILE_SIZE_MB,
                    )
                    if not ok:
                        if out_path.exists():
                            out_path.unlink()
                        tqdm.write(
                            f"[Warning] QC failed after save: {out_path.resolve()} | "
                            f"ID: {case_id_str} | reasons: {';'.join(reasons)}"
                        )
                        append_error_log(
                            error_log_path,
                            {
                                "case_id": case_id_str,
                                "case_key": case_key,
                                "seq_id": seq_id,
                                "nii_file": str(nii_file.resolve()),
                                "stage": "post_save_qc",
                                "error": ";".join(reasons),
                            },
                        )
                    elif reasons:
                        append_error_log(
                            error_log_path,
                            {
                                "case_id": case_id_str,
                                "case_key": case_key,
                                "seq_id": seq_id,
                                "nii_file": str(nii_file.resolve()),
                                "stage": "post_save_qc_warning",
                                "error": ";".join(reasons),
                            },
                        )
                
                except Exception as e:
                    # 其他 Python 错误
                    tqdm.write(f"\n[Error] Unknown error processing {nii_file}: {e}")
                    append_error_log(
                        error_log_path,
                        {
                            "case_id": case_id_str,
                            "case_key": case_key,
                            "seq_id": seq_id,
                            "nii_file": str(nii_file.resolve()),
                            "stage": "exception",
                            "error": f"{type(e).__name__}: {e}",
                        },
                    )
                    continue

    # ===== 保存索引 =====
    save_index(case_index, index_path)

    print(f"Finished. Total cases indexed: {max_case_id}")
    print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess brain MRI data")
    parser.add_argument(
        "--raw_root",
        type=str,
        default=str(RAW_DATA_PATH), # 适配 Path 对象转 str
        help="Path to raw brainMRI directory"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(PROCESSED_DATA_PATH), # 适配 Path 对象转 str
        help="Output processed data directory"
    )
    args = parser.parse_args()

    main(args)
