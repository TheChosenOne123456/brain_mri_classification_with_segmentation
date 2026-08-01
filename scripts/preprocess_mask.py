"""
预处理医生标注的病灶 mask，并保证它们和已预处理 MRI 图像逐体素对齐。

当前图像预处理会为每个合格序列保存：
    case_xxxx_seq_preprocess.json

mask 预处理只读取 JSON metadata，不重新运行 HD-BET：
    1. 读取同 case、同序列的 preprocess JSON；
    2. 用 JSON 中的 resampled reference geometry 构造 reference；
    3. 将医生 mask 最近邻重采样到该 reference；
    4. 用 JSON 中的 crop/pad 参数做完全相同的裁剪/填充；
    5. 保存到对应已预处理图像旁边。

用法：
    python -m scripts.preprocess_mask \
        --config dataxxx/preprocessing_config.py \
        --data-root dataxxx

    # 默认会清理旧的 *_mask.nii.gz 和 mask_index.json 后重建；
    # 如果只想增量追加，可加 --no-clear-existing-masks。增量模式会：
    # - 保留并跳过已有 mask 文件；
    # - 载入并合并已有 mask_index.json；
    # - 只处理尚不存在的 case/sequence mask。
    python -m scripts.preprocess_mask ... --no-clear-existing-masks
"""

import argparse
import json
import re
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

sitk.ProcessObject_SetGlobalWarningDisplay(False)

from configs.config_utils import (
    PREPROCESS_CONFIG_FIELDS,
    load_python_config,
    resolve_input_artifact_dir,
)
from configs.global_config import MASK_ROOTS
from utils.data_scan import collect_cases
from utils.io import INDEX_FILE_NAME, MASK_INDEX_FILE_NAME, load_index, save_index
from utils.resample import resample_to_reference, save_image
from utils.sequences import identify_sequence
from utils.spatial import apply_crop_or_pad_from_meta


def case_key_from_dir(case_dir: Path):
    folder_name = case_dir.name
    match = re.findall(r"\d+", folder_name)
    return match[-1] if match else folder_name


def clear_existing_masks(out_root: Path):
    removed = 0
    for mask_path in out_root.glob("*/*/case_*_*_mask.nii*"):
        mask_path.unlink()
        removed += 1

    mask_index_path = out_root / MASK_INDEX_FILE_NAME
    if mask_index_path.exists():
        mask_index_path.unlink()

    return removed


def add_mask_index_entry(mask_index, case_id_str, seq_id):
    sequences = mask_index.setdefault(case_id_str, [])
    normalized_sequences = sorted({int(seq) for seq in sequences} | {int(seq_id)})
    mask_index[case_id_str] = normalized_sequences


def get_preprocess_meta_path(image_path: Path):
    name = image_path.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = image_path.stem
    return image_path.with_name(f"{stem}_preprocess.json")


def load_preprocess_meta(image_path: Path):
    meta_path = get_preprocess_meta_path(image_path)
    if not meta_path.exists():
        return None, meta_path
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f), meta_path


def make_reference_from_meta(meta):
    ref = meta["resampled_reference"]
    reference = sitk.Image([int(x) for x in ref["size_xyz"]], sitk.sitkUInt8)
    reference.SetSpacing([float(x) for x in ref["spacing_xyz"]])
    reference.SetOrigin([float(x) for x in ref["origin_xyz"]])
    reference.SetDirection([float(x) for x in ref["direction"]])
    return reference


def preprocess_one_mask(mask_file: Path, base_img_path: Path, meta):
    """
    返回 fixed_mask 和错误信息。
    fixed_mask 的 voxel grid 与对应预处理图像保持一致。
    """
    reference_img = make_reference_from_meta(meta)

    resampled_mask = resample_to_reference(
        mask_file,
        reference_img=reference_img,
        is_label=True,
    )
    if resampled_mask is None:
        return None, "failed_to_resample_mask"

    resampled_mask = sitk.Cast(resampled_mask > 0, sitk.sitkUInt8)
    resampled_mask.CopyInformation(reference_img)
    fixed_mask = apply_crop_or_pad_from_meta(
        resampled_mask,
        meta["crop"],
    )
    fixed_mask = sitk.Cast(fixed_mask > 0, sitk.sitkUInt8)
    base_img = sitk.ReadImage(str(base_img_path))
    fixed_mask.CopyInformation(base_img)
    return fixed_mask, ""


def image_geometry_matches(img_a, img_b, atol=1e-5):
    if sitk.GetArrayFromImage(img_a).shape != sitk.GetArrayFromImage(img_b).shape:
        return False

    for vals_a, vals_b in (
        (img_a.GetSpacing(), img_b.GetSpacing()),
        (img_a.GetOrigin(), img_b.GetOrigin()),
        (img_a.GetDirection(), img_b.GetDirection()),
    ):
        if any(abs(float(a) - float(b)) > atol for a, b in zip(vals_a, vals_b)):
            return False

    return True


def main(args):
    config = load_python_config(args.config, PREPROCESS_CONFIG_FIELDS)
    out_root = resolve_input_artifact_dir(args.data_root, "data")
    mask_roots = [Path(p).resolve() for p in args.mask_roots]
    mask_roots = [p for p in mask_roots if p.exists()]

    if not mask_roots:
        print("未找到任何有效 Mask 原始路径，请检查路径是否正确。")
        return

    index_path = out_root / INDEX_FILE_NAME
    if not index_path.exists():
        print(f"找不到预处理索引文件 {index_path}，请先执行 preprocess_data。")
        return

    print("=== 开始处理 Mask 数据 ===")
    print(f"Preprocessing config: {config.__config_path__}")
    print(f"Data out root : {out_root}")
    print(f"Mask roots    : {[str(p) for p in mask_roots]}")

    if args.clear_existing_masks:
        removed = clear_existing_masks(out_root)
        print(f"已清理旧 mask 文件: {removed}")

    case_index = load_index(index_path)
    mask_index_path = out_root / MASK_INDEX_FILE_NAME
    mask_index = (
        {}
        if args.clear_existing_masks
        else load_index(mask_index_path)
    )
    cases = collect_cases(mask_roots)
    print(f"在源路径共扫描到 {len(cases)} 个样本文件夹。")

    processed_count = 0
    existing_count = 0
    skipped_count = 0

    for case_dir in tqdm(cases, desc="Processing masks"):
        case_key = case_key_from_dir(case_dir)
        if case_key not in case_index:
            skipped_count += 1
            continue

        case_id = int(case_index[case_key])
        case_id_str = f"{case_id:04d}"

        mask_files = [
            p for p in sorted(case_dir.rglob("*.nii*")) if "mask" in p.name.lower()
        ]
        mask_files = sorted(
            mask_files,
            key=lambda p: (identify_sequence(p) or 999, str(p)),
        )

        for mask_file in mask_files:
            seq_id = identify_sequence(mask_file)
            if seq_id is None:
                continue

            base_imgs = list(out_root.rglob(f"case_{case_id_str}_{seq_id}.nii.gz"))
            if not base_imgs:
                # 对应底图被 QC 排除时，不单独保存 mask
                continue

            base_img_path = base_imgs[0]
            out_mask_path = (
                base_img_path.parent / f"case_{case_id_str}_{seq_id}_mask.nii.gz"
            )

            if not args.clear_existing_masks and out_mask_path.exists():
                add_mask_index_entry(mask_index, case_id_str, seq_id)
                existing_count += 1
                continue

            try:
                meta, meta_path = load_preprocess_meta(base_img_path)
                if meta is None:
                    tqdm.write(
                        f"[Warning] 找不到 preprocess meta，跳过 mask: "
                        f"{mask_file} | expected={meta_path}"
                    )
                    skipped_count += 1
                    continue

                fixed_mask, error = preprocess_one_mask(
                    mask_file,
                    base_img_path,
                    meta,
                )
                if fixed_mask is None:
                    tqdm.write(f"[Warning] mask 预处理失败: {mask_file} | {error}")
                    skipped_count += 1
                    continue

                base_img = sitk.ReadImage(str(base_img_path))
                if not image_geometry_matches(fixed_mask, base_img):
                    tqdm.write(
                        f"[Warning] mask 与底图 geometry 不一致，跳过: {mask_file} | "
                        f"mask={sitk.GetArrayFromImage(fixed_mask).shape}, "
                        f"image={sitk.GetArrayFromImage(base_img).shape}"
                    )
                    skipped_count += 1
                    continue

                save_image(fixed_mask, out_mask_path)

                add_mask_index_entry(mask_index, case_id_str, seq_id)

                processed_count += 1

            except Exception as e:
                tqdm.write(f"\n[Error] Error processing mask {mask_file}: {e}")
                skipped_count += 1
                continue

    save_index(mask_index, mask_index_path)

    print("\n=== Mask 预处理完成 ===")
    print(f"新增保存 mask 文件数: {processed_count}")
    print(f"保留并跳过已有 mask 文件数: {existing_count}")
    print(f"跳过/失败数量: {skipped_count}")
    print(f"共录入了 {len(mask_index)} 个主样本（Case）的 Mask。")
    print(f"Mask 序列索引保存至 {mask_index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess brain MRI lesion masks")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the preprocessing_config.py used for the image data",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Experiment root containing data, or the data directory itself",
    )
    parser.add_argument(
        "--mask-roots",
        nargs="+",
        default=[str(p) for p in MASK_ROOTS],
        help="提供包含 MASK 文件的原始路径列表，可接收多个",
    )
    parser.add_argument(
        "--clear-existing-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "默认清理旧 mask 后重建；--no-clear-existing-masks 会保留已有 "
            "mask、合并旧 mask_index，并只补充缺失项"
        ),
    )
    args = parser.parse_args()

    main(args)
