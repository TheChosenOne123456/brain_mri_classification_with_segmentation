import json
import torch
import pandas as pd
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

def main():
    # --- 1. 配置相关路径 ---
    base_proj_dir = Path("/home/ailab/projects/brain_mri_classification")
    excel_dir = Path("/home/ailab/Desktop") # Excel 表格存放路径
    index_json_path = base_proj_dir / "version1" / "data" / "case_index.json"
    dataset_base_dir = base_proj_dir / "version1" / "datasets"
    
    # 新测试集保存路径
    output_dir = dataset_base_dir / "easy_inflammation_set"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seq_names = ["seq1_T1", "seq2_T2", "seq3_FLAIR"]
    splits = ["train.pt", "val.pt", "test.pt"]
    
    # --- 2. 提取 Excel 中的困难炎症原始 ID ---
    print("[1] 正在解析 Excel 获取困难病变原始 ID...")
    hard_raw_ids = set()
    excel_files = list(excel_dir.expanduser().glob("*.xlsx"))
    
    for excel_path in excel_files:
        try:
            df = pd.read_excel(excel_path, header=None)
            # 根据你提供的逻辑，读取第 3 列（index=2）
            raw_ids = df.iloc[:, 2].dropna().astype(str).tolist()
            for item in raw_ids:
                for p in item.split('/'):
                    clean_id = p.strip()
                    if clean_id:
                        hard_raw_ids.add(clean_id)
        except Exception as e:
            print(f"读取 {excel_path.name} 失败: {e}")
            
    print(f"  -> 从 Excel中共提取到 {len(hard_raw_ids)} 个困难炎症的原始编号。")

    # --- 3. 将原始 ID 转化为数据集中的 internal Case ID ---
    print("\n[2] 正在根据 case_index.json 映射 Data ID...")
    with open(index_json_path, 'r', encoding='utf-8') as f:
        case_index_map = json.load(f)
        
    hard_internal_ids = set()
    mapped_count = 0
    
    for raw_id in hard_raw_ids:
        if raw_id in case_index_map:
            internal_int = case_index_map[raw_id]
            # 为了兼容性，把 int 型 (如 1) 和 四位零填充型 (如 '0001') 同时加入过滤集合
            hard_internal_ids.add(internal_int)
            hard_internal_ids.add(str(internal_int))
            hard_internal_ids.add(f"{internal_int:04d}")
            mapped_count += 1
            
    print(f"  -> 成功映射了 {mapped_count} 个案例到数据集 ID。")

    # --- 4. 遍历 fold1 重建简单炎症数据集 ---
    print("\n[3] 开始提取各自 sequence 的简单炎症数据集...")
    
    for seq_name in seq_names:
        seq_dir = dataset_base_dir / seq_name / "fold1"
        if not seq_dir.exists():
            print(f"  [警告] 找不到对应的目录: {seq_dir}")
            continue
            
        print(f"\n  处理序列: {seq_name}")
        easy_images = []
        easy_labels = []
        easy_case_ids = []
        
        # 将 train.pt, val.pt, test.pt 中的数据合在一起扫一遍
        total_inf_count_in_seq = 0
        
        for split in splits:
            pt_path = seq_dir / split
            if not pt_path.exists():
                continue
                
            data = torch.load(pt_path)
            images = data["images"]
            labels = data["labels"]
            case_ids = data["case_ids"]
            
            for i in range(len(case_ids)):
                label = int(labels[i].item())
                c_id = case_ids[i]
                
                # 标签 1 代表炎症 (0 是正常, 2 是脑膜转移)
                if label == 1:
                    total_inf_count_in_seq += 1
                    
                    # 检查是否属于困难样本
                    # c_id 可能是 '0001' 或者 1
                    if c_id not in hard_internal_ids:
                        # 属于简单炎症，予以保留！
                        easy_images.append(images[i])
                        easy_labels.append(labels[i])
                        easy_case_ids.append(c_id)
                        
        # 组装并保存新的 .pt
        if len(easy_images) == 0:
            print(f"    没有找到符合条件的简单炎症数据！")
            continue
            
        new_data = {
            "images": torch.stack(easy_images, dim=0),
            "labels": torch.tensor(easy_labels, dtype=torch.long),
            "case_ids": easy_case_ids,
            "meta": {
                "description": f"Easy inflammation (label=1) cases filtered from {seq_name} version1",
                "num_samples": len(easy_images)
            }
        }
        
        save_path = output_dir / f"{seq_name}_easy_inflammation.pt"
        torch.save(new_data, save_path)
        
        print(f"    原 fold1 中总体炎症数量: {total_inf_count_in_seq}")
        print(f"    提取的简单炎症数量   : {len(easy_images)}")
        print(f"    数据集已保存至       : {save_path}")

    print("\n[顺利完成] 所有简单炎症数据集已成功生成！")

if __name__ == "__main__":
    main()