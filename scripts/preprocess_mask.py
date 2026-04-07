'''
单独预处理Mask数据的脚本
扫描包含mask的原始文件夹，并向已经处理好的data目录中添加对应的预处理掩码。
'''

import argparse
import re
from pathlib import Path
from tqdm import tqdm

import SimpleITK as sitk
# 关闭 SimpleITK 的底层警告输出，防止刷屏
sitk.ProcessObject_SetGlobalWarningDisplay(False)

from configs.global_config import *
from utils.sequences import identify_sequence
from utils.data_scan import collect_cases
from utils.io import load_index, save_index, INDEX_FILE_NAME, MASK_INDEX_FILE_NAME
from utils.resample import resample_image, save_image
from utils.spatial import center_crop_or_pad



def main(args):
    out_root = Path(args.out_root).resolve()
    
    # 原始mask路径
    mask_roots = [Path(p).resolve() for p in args.mask_roots]
    mask_roots = [p for p in mask_roots if p.exists()]
    
    if not mask_roots:
        print("未找到任何有效 Mask 原始路径，请检查路径是否正确。")
        return

    print("=== 开始处理 Mask 数据 ===")
    print(f"Mask 源路径包含: {[str(p) for p in mask_roots]}")
    
    # ===== 1. 读取原有的 case_index =====
    index_path = out_root / INDEX_FILE_NAME
    if not index_path.exists():
        print(f"找不到预处理索引文件 {index_path}，请先执行原版预处理！")
        return
        
    case_index = load_index(index_path)
    
    # ===== 2. 初始化 mask_index =====
    # 结构: { "0001": [3], "0045": [1, 2, 3], ... } (记录某个 ID 的哪些序列有 Mask)
    mask_index_path = out_root / MASK_INDEX_FILE_NAME
    mask_index = load_index(mask_index_path) if mask_index_path.exists() else {}
    
    # 收集全部 case 文件夹
    cases = collect_cases(mask_roots)
    print(f"在源路径共扫描到 {len(cases)} 个样本文件夹。")
    
    # ===== 3. 处理每一个含有 Mask 的 Case =====
    for case_dir in tqdm(cases, desc="Processing masks"):
        # 匹配对应原图的唯一标识ID (和原脚本提取逻辑完全一致)
        folder_name = case_dir.name
        match = re.findall(r'\d+', folder_name)
        case_key = match[-1] if match else folder_name
            
        # 若原图没有被预处理过（比如不属于我们的目标类别），则丢弃该Mask
        if case_key not in case_index:
            continue
            
        case_id = case_index[case_key]
        case_id_str = f"{case_id:04d}"
        
        # 扫描该病例下所有的 nifty 文件
        for nii_file in case_dir.rglob("*.nii*"):
            # 筛除诸如 FLAIR.nii 这样的原图，只拿 FLAIR_mask.nii.gz
            if "mask" not in nii_file.name.lower():
                continue
                
            # 识别是不是我们要的三大序列 (T1, T2, FLAIR 分别被识别成 1, 2, 3)
            seq_id = identify_sequence(nii_file)
            if seq_id is None: 
                continue
                
            # 寻找对应的已经处理好的底图所在路径
            # 直接在 data/ 目录下 rglob 搜索其对应的处理好的文件名，无需关心类别文件夹叫什么
            base_imgs = list(out_root.rglob(f"case_{case_id_str}_{seq_id}.nii.gz"))
            if not base_imgs:
                # 没找到对应的底图（可能底图损坏被抛弃了），安全起见不要单独存Mask
                continue
                
            base_img_path = base_imgs[0]
            
            # 准备输出的 Mask 存放路径，和对应的原图放在同一个文件夹
            out_mask_name = f"case_{case_id_str}_{seq_id}_mask.nii.gz"
            out_mask_path = base_img_path.parent / out_mask_name

            try:
                # 1. 重采样: 特别注意，这里设 is_label=True -> 使用"最近邻插值"，这样保证Mask的 0 和 1 等整型类别不变
                resampled_mask = resample_image(nii_file, target_spacing=TARGET_SPACING, is_label=True)
                if resampled_mask is None:
                    continue
                
                # 2. (跳过!) Mask绝对不能像原始MRI那样进行归一化(normalize)，需要直接切到下一步
                
                # 3. 裁剪/填充: 调用的原函数里 pad 默认为 0，符合背景定义
                fixed_mask = center_crop_or_pad(resampled_mask, TARGET_SHAPE)
                
                # 4. 保存
                save_image(fixed_mask, out_mask_path)
                
                # 5. 更新索引
                if case_id_str not in mask_index:
                    mask_index[case_id_str] = []
                if seq_id not in mask_index[case_id_str]:
                    mask_index[case_id_str].append(seq_id)
                    # 排序一下，保持美观
                    mask_index[case_id_str].sort()

            except Exception as e:
                tqdm.write(f"\n[Error] Error processing mask {nii_file}: {e}")
                continue
                
    # ===== 4. 保存 mask_index 文件 =====
    save_index(mask_index, mask_index_path)
    
    print("\n=== Mask 预处理完成 ===")
    print(f"共录入了 {len(mask_index)} 个主样本（Case）的 Mask。")
    print(f"Mask 序列索引保存至 {mask_index_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess brain MRI mask data")
    parser.add_argument(
        "--mask_roots",
        nargs="+",
        default=[str(p) for p in MASK_ROOTS],
        help="提供包含 MASK 文件的原始路径列表 (可接收多个)"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(PROCESSED_DATA_PATH), 
        help="现有的预处理数据输出路径"
    )
    args = parser.parse_args()

    main(args)