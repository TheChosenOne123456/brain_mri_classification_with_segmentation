'''
数据预处理，实现重采样、归一化和裁剪/填充
[重构版 - 功能完全复刻]
'''
import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk
# [新增] 关闭 SimpleITK 的底层警告输出，防止刷屏
sitk.ProcessObject_SetGlobalWarningDisplay(False)

from configs.global_config import *

from utils.sequences import identify_sequence
from utils.data_scan import collect_cases
from utils.io import load_index, save_index, INDEX_FILE_NAME
from utils.resample import resample_image, save_image
from utils.intensity import normalize_intensity, zero_outside_mask
from utils.spatial import center_crop_or_pad_with_meta, crop_or_pad_around_mask_with_meta
from utils.brain_extraction import dilate_mask, extract_foreground_mask_hd_bet
from utils.preprocess_qc import validate_preprocessed_image, validate_saved_file_size


ERROR_LOG_FIELDS = ["case_id", "case_key", "seq_id", "nii_file", "stage", "error"]


def load_excluded_case_ids(path):
    """读取每行一个病例编号的排除名单，忽略空行和 # 注释。"""
    path = Path(path)
    if not path.exists():
        return set()

    excluded = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            case_id = line.split("#", 1)[0].strip()
            if case_id:
                excluded.add(case_id)
    return excluded


def append_error_log(log_path, row):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def nii_stem(path: Path):
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def exact_sequence_names(seq_name):
    names = {seq_name.upper()}
    if seq_name.upper() in {"T1", "T2"}:
        names.add(f"{seq_name.upper()}WI")
    return names


def sequence_candidate_rank(nii_file, seq_id):
    seq_name = ALL_SEQUENCES[seq_id - 1]
    stem = nii_stem(nii_file).upper()
    exact_rank = 0 if stem in exact_sequence_names(seq_name) else 1
    return exact_rank, len(stem), str(nii_file).upper()


def select_sequence_files(case_dir):
    seq_candidates = {seq_id: [] for seq_id in range(1, NUM_SEQUENCES + 1)}

    for nii_file in sorted(case_dir.rglob("*.nii*")):
        seq_id = identify_sequence(nii_file)
        if seq_id is None:
            continue
        seq_candidates[seq_id].append(nii_file)

    selected = {}
    for seq_id, candidates in seq_candidates.items():
        if not candidates:
            continue
        selected[seq_id] = sorted(
            candidates,
            key=lambda path: sequence_candidate_rank(path, seq_id),
        )[0]

    missing_seq_ids = [
        seq_id for seq_id in range(1, NUM_SEQUENCES + 1) if seq_id not in selected
    ]
    return selected, seq_candidates, missing_seq_ids


def cleanup_outputs(output_paths):
    for out_path in output_paths:
        if out_path.exists():
            out_path.unlink()
        meta_path = get_preprocess_meta_path(out_path)
        if meta_path.exists():
            meta_path.unlink()
        # 兼容清理由旧流程生成的 foreground 中间产物；新流程不再保存它。
        foreground_path = get_preprocess_foreground_path(out_path)
        if foreground_path.exists():
            foreground_path.unlink()


def get_preprocess_meta_path(image_path: Path):
    name = image_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = image_path.stem
    return image_path.with_name(f"{stem}_preprocess.json")


def get_preprocess_foreground_path(image_path: Path):
    name = image_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = image_path.stem
    return image_path.with_name(f"preprocess_{stem}_foreground.nii.gz")


def get_case_output_paths(out_root, output_class_dir, case_id_str):
    return [
        out_root
        / output_class_dir
        / str(seq_id)
        / f"case_{case_id_str}_{seq_id}.nii.gz"
        for seq_id in range(1, NUM_SEQUENCES + 1)
    ]


def find_case_output_class_dirs(out_root, case_id_str):
    """查找磁盘上已经包含该 case 任一图像或 meta 的类别目录。"""
    class_dirs = []
    for class_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        output_paths = get_case_output_paths(out_root, class_dir.name, case_id_str)
        if any(
            out_path.exists() or get_preprocess_meta_path(out_path).exists()
            for out_path in output_paths
        ):
            class_dirs.append(class_dir.name)
    return class_dirs


def get_max_case_id_on_disk(out_root):
    """索引之外也可能存在历史遗留文件，分配新 ID 时一并避开。"""
    max_case_id = 0
    pattern = re.compile(r"^case_(\d+)_\d+\.nii(?:\.gz)?$")
    for path in out_root.glob("*/*/case_*_*.nii*"):
        match = pattern.match(path.name)
        if match:
            max_case_id = max(max_case_id, int(match.group(1)))
    return max_case_id


def existing_case_outputs_are_valid(output_paths):
    """
    已经登记到 case_index 的 case，只有三序列文件都存在且通过 QC 才允许跳过。
    否则认为是历史部分失败/污染输出，需要重新预处理并追加新 case_id。
    """
    for out_path in output_paths:
        if not out_path.exists():
            return False, [f"missing_output:{out_path}"]
        meta_path = get_preprocess_meta_path(out_path)
        if not meta_path.exists():
            return False, [f"missing_preprocess_meta:{meta_path}"]

        try:
            img = sitk.ReadImage(str(out_path))
        except Exception as e:
            return False, [f"read_existing_failed:{out_path}:{type(e).__name__}:{e}"]

        ok, reasons = validate_preprocessed_image(
            img,
            target_shape=TARGET_SHAPE,
            target_spacing=TARGET_SPACING,
            max_zero_ratio=PREPROCESS_MAX_ZERO_RATIO,
            min_nonzero_bbox_fraction=PREPROCESS_MIN_NONZERO_BBOX_FRACTION,
        )
        if not ok:
            return False, [f"{out_path}:{';'.join(reasons)}"]

        ok, reasons = validate_saved_file_size(
            out_path,
            min_file_size_mb=PREPROCESS_MIN_FILE_SIZE_MB,
        )
        if not ok:
            return False, [f"{out_path}:{';'.join(reasons)}"]

    return True, []


def image_geometry_to_dict(img):
    return {
        "size_xyz": [int(x) for x in img.GetSize()],
        "spacing_xyz": [float(x) for x in img.GetSpacing()],
        "origin_xyz": [float(x) for x in img.GetOrigin()],
        "direction": [float(x) for x in img.GetDirection()],
    }


def make_preprocess_meta(nii_file, resampled_img, crop_meta, foreground_mask=None):
    foreground_meta = None
    if foreground_mask is not None:
        fg_arr = sitk.GetArrayFromImage(foreground_mask) > 0
        coords = np.argwhere(fg_arr)
        if coords.size > 0:
            foreground_meta = {
                "bbox_min_zyx": [int(x) for x in coords.min(axis=0)],
                "bbox_max_zyx": [int(x) for x in coords.max(axis=0)],
                "voxel_count": int(coords.shape[0]),
            }

    return {
        "schema_version": 2,
        "raw_image_path": str(Path(nii_file).resolve()),
        "target_spacing_xyz": [float(x) for x in TARGET_SPACING],
        "target_shape_zyx": [int(x) for x in TARGET_SHAPE],
        "brain_extractor": BRAIN_EXTRACTOR,
        "foreground_zero_outside": bool(FOREGROUND_ZERO_OUTSIDE),
        "foreground_dilation_mm": float(FOREGROUND_DILATION_MM),
        "resampled_reference": image_geometry_to_dict(resampled_img),
        "crop": crop_meta,
        "foreground": foreground_meta,
    }


def save_preprocess_meta(meta, meta_path: Path):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


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
        return None, None

    if BRAIN_EXTRACTOR == "none":
        normalized_img = normalize_intensity(resampled_img)
        fixed_img, crop_meta = center_crop_or_pad_with_meta(normalized_img, TARGET_SHAPE)
        meta = make_preprocess_meta(nii_file, resampled_img, crop_meta, foreground_mask=None)
        return fixed_img, meta

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

    fixed_img, crop_meta = crop_or_pad_around_mask_with_meta(
        normalized_img,
        foreground_mask,
        TARGET_SHAPE,
    )
    meta = make_preprocess_meta(nii_file, resampled_img, crop_meta, foreground_mask)
    return fixed_img, meta


def main(args):
    validate_preprocess_setup()

    # 路径解析
    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    index_path = out_root / INDEX_FILE_NAME
    error_log_path = out_root / "preprocess_errors.csv"
    excluded_case_ids = load_excluded_case_ids(EXCLUDED_CASE_IDS_PATH)

    # ===== 读取 / 初始化 index =====
    case_index = load_index(index_path)

    # 恢复计数器状态；同时扫描磁盘，避免与未登记的历史遗留文件撞号。
    if case_index:
        try:
            # 兼容旧索引结构：Key是原始路径/文件名，Value是int ID
            max_case_id = max(case_index.values())
        except ValueError:
            max_case_id = 0
    else:
        max_case_id = 0
    max_case_id = max(max_case_id, get_max_case_id_on_disk(out_root))

    print(f"Starting preprocessing for {NUM_CLASSES} classes: {CLASS_NAMES}")
    print(f"Current Max Case ID: {max_case_id}")
    print(
        f"Excluded Case IDs: {len(excluded_case_ids)} "
        f"({Path(EXCLUDED_CASE_IDS_PATH).resolve()})"
    )

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

            if case_key in excluded_case_ids:
                continue

            # --- 增量检查 ---
            # case_index 是跨类别的全局索引，因此先在所有类别目录中查找旧 ID。
            # 同一个病例号可能重复出现在不同诊断源目录，已有任一完整输出即跳过。
            retry_case_id = None
            if case_key in case_index:
                existing_case_id = int(case_index[case_key])
                existing_case_id_str = f"{existing_case_id:04d}"
                existing_class_dirs = find_case_output_class_dirs(
                    out_root,
                    existing_case_id_str,
                )

                valid_existing = False
                invalid_reasons_by_class = {}
                for existing_class_dir in existing_class_dirs:
                    existing_paths = get_case_output_paths(
                        out_root,
                        existing_class_dir,
                        existing_case_id_str,
                    )
                    valid, reasons = existing_case_outputs_are_valid(existing_paths)
                    if valid:
                        valid_existing = True
                        break
                    invalid_reasons_by_class[existing_class_dir] = reasons

                if valid_existing:
                    continue

                # 旧 ID 的残留文件属于另一个类别时，留给遍历到原类别时修复。
                # 这样不会用当前这个跨类别重复目录覆盖原标签。
                if existing_class_dirs and output_class_dir not in existing_class_dirs:
                    append_error_log(
                        error_log_path,
                        {
                            "case_id": existing_case_id_str,
                            "case_key": case_key,
                            "seq_id": "",
                            "nii_file": str(case_dir.resolve()),
                            "stage": "duplicate_case_other_class_deferred",
                            "error": ",".join(existing_class_dirs),
                        },
                    )
                    continue

                existing_output_paths = get_case_output_paths(
                    out_root,
                    output_class_dir,
                    existing_case_id_str,
                )
                reasons = invalid_reasons_by_class.get(
                    output_class_dir,
                    [f"missing_output:{existing_output_paths[0]}"],
                )
                tqdm.write(
                    f"[Warning] Existing indexed case is incomplete/invalid, retry with same ID: "
                    f"{case_dir.resolve()} | ID: {existing_case_id_str} | "
                    f"reasons: {';'.join(reasons)}"
                )
                append_error_log(
                    error_log_path,
                    {
                        "case_id": existing_case_id_str,
                        "case_key": case_key,
                        "seq_id": "",
                        "nii_file": str(case_dir.resolve()),
                        "stage": "existing_output_invalid",
                        "error": ";".join(reasons),
                    },
                )
                retry_case_id = existing_case_id

            selected_files, _, missing_seq_ids = select_sequence_files(case_dir)
            if missing_seq_ids:
                missing_names = [ALL_SEQUENCES[seq_id - 1] for seq_id in missing_seq_ids]
                # 仅有 4/5 等非目标序列的目录会三序列全缺，数量较多，不在终端刷屏。
                if len(missing_seq_ids) < NUM_SEQUENCES:
                    tqdm.write(
                        f"[Warning] Incomplete sequences, skip case: {case_dir.resolve()} | "
                        f"missing: {','.join(missing_names)}"
                    )
                append_error_log(
                    error_log_path,
                    {
                        "case_id": "",
                        "case_key": case_key,
                        "seq_id": "",
                        "nii_file": str(case_dir.resolve()),
                        "stage": "missing_sequences",
                        "error": ",".join(missing_names),
                    },
                )
                continue

            # 旧 case 修复沿用原 ID；全新 case 才分配递增 ID。
            case_id = retry_case_id if retry_case_id is not None else max_case_id + 1
            case_id_str = f"{case_id:04d}"
            output_paths = get_case_output_paths(out_root, output_class_dir, case_id_str)
            cleanup_outputs(output_paths)

            processed_images = {}
            preprocess_metas = {}
            case_ok = True

            for seq_id in range(1, NUM_SEQUENCES + 1):
                nii_file = selected_files[seq_id]

                try:
                    fixed_img, preprocess_meta = preprocess_image(nii_file)
                    if fixed_img is None:
                        tqdm.write(
                            f"[Warning] Failed to load/resample file: {nii_file.resolve()} | "
                            f"ID: {case_id_str}"
                        )
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
                        case_ok = False
                        break

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
                        case_ok = False
                        break

                    processed_images[seq_id] = fixed_img
                    preprocess_metas[seq_id] = preprocess_meta

                except Exception as e:
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
                    case_ok = False
                    break

            if not case_ok:
                cleanup_outputs(output_paths)
                continue

            for seq_id in range(1, NUM_SEQUENCES + 1):
                nii_file = selected_files[seq_id]
                out_path = (
                    out_root
                    / output_class_dir
                    / str(seq_id)
                    / f"case_{case_id_str}_{seq_id}.nii.gz"
                )
                try:
                    save_image(processed_images[seq_id], out_path)
                    save_preprocess_meta(
                        preprocess_metas[seq_id],
                        get_preprocess_meta_path(out_path),
                    )

                    ok, reasons = validate_saved_file_size(
                        out_path,
                        min_file_size_mb=PREPROCESS_MIN_FILE_SIZE_MB,
                    )
                    if not ok:
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
                        case_ok = False
                        break
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
                    tqdm.write(f"\n[Error] Failed to save {out_path}: {e}")
                    append_error_log(
                        error_log_path,
                        {
                            "case_id": case_id_str,
                            "case_key": case_key,
                            "seq_id": seq_id,
                            "nii_file": str(nii_file.resolve()),
                            "stage": "save_exception",
                            "error": f"{type(e).__name__}: {e}",
                        },
                    )
                    case_ok = False
                    break

            if not case_ok:
                cleanup_outputs(output_paths)
                continue

            max_case_id = max(max_case_id, case_id)
            case_index[case_key] = case_id
            save_index(case_index, index_path)

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
