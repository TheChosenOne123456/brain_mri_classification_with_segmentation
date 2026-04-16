# import random
# import torch
# from torch.utils.data import Dataset
# from utils.dataset import load_nii_as_tensor  # 新增 NIfTI 读取方法引入

# # ================== 一些工具函数 ==================
# def set_seed(seed):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# # utils/train_and_test.py (部分替换)

# class PTDataset(Dataset):
#     def __init__(self, cases):
#         self.cases = cases
#         self.labels = [c["label"] for c in cases]

#     def __len__(self):
#         return len(self.cases)

#     def __getitem__(self, idx):
#         # 核心：真正到取数据时才依据路径去拿取和解码大文件
#         case_info = self.cases[idx]
#         image_tensor = load_nii_as_tensor(case_info["nii_path"])
#         label_val = case_info["label"]
#         label = torch.tensor(label_val, dtype=torch.long)
        
#         # 处理 Mask 标签
#         has_mask_bool = case_info.get("has_mask", False)
        
#         if has_mask_bool and case_info.get("mask_path") is not None:
#             # 1. 读取由于各种软件生成的可能带有 255 或其他非1值的原始标注
#             raw_mask_tensor = load_nii_as_tensor(case_info["mask_path"])
            
#             # 2. 彻底二值化清洗：凡是大于 0 的像素都归一化为纯净的 1 (病灶区域)
#             base_binary_mask = (raw_mask_tensor > 0).long()
            
#             # 3. 映射多类别：根据当前样本的分类 (label_val 即 1 或者 2)，为其赋予专属类别色彩！
#             # 比如类别 2 的转移瘤，其 Mask 像素此时全变成了 2
#             mask_tensor = base_binary_mask * label_val
            
#             has_mask_flag = torch.tensor(1.0, dtype=torch.float32)
            
#         elif label_val == 0:
#             # 2. 正常样本 (label=0) -> 全0属于真实的 Ground Truth，因此置 flag = 1.0!
#             mask_tensor = torch.zeros_like(image_tensor, dtype=torch.long)
#             has_mask_flag = torch.tensor(1.0, dtype=torch.float32)
            
#         else:
#             # 3. 阳性样本但缺少 Mask（例如你的绝大部分不明显炎症和部分早期数据） -> flag = 0.0 跳过分割 Loss
#             mask_tensor = torch.zeros_like(image_tensor, dtype=torch.long)
#             has_mask_flag = torch.tensor(0.0, dtype=torch.float32)
        
#         return image_tensor, label, mask_tensor, has_mask_flag, case_info["case_id"]


# def load_pt_dataset(pt_path):
#     # 此处读取到的只是个几十K的文本字典
#     data = torch.load(pt_path, weights_only=False)
#     # 将字典中的路径和索引喂给代理 Dataset
#     return PTDataset(data["cases"])

import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from utils.dataset import load_nii_as_tensor  # 新增 NIfTI 读取方法引入

# ================== 一些工具函数 ==================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# utils/train_and_test.py (部分替换)

class PTDataset(Dataset):
    def __init__(self, cases):
        self.cases = cases
        self.labels = [c["label"] for c in cases]
        
        # 加载 mask_index.json
        self.mask_index = {}
        if len(self.cases) > 0:
            sample_path = Path(self.cases[0]["nii_path"])
            # 根据 nii_path 反推 PROCESSED_DATA_PATH (比如 .../version1/data)
            # 路径结构: data / 类别目录(2_metastasis) / 序列号(1) / case_xxxx_1.nii.gz
            mask_index_path = sample_path.parent.parent.parent / "mask_index.json"
            if mask_index_path.exists():
                with open(mask_index_path, "r", encoding="utf-8") as f:
                    self.mask_index = json.load(f)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        # 核心：真正到取数据时才依据路径去拿取和解码大文件
        case_info = self.cases[idx]
        nii_path = Path(case_info["nii_path"])
        image_tensor = load_nii_as_tensor(nii_path)
        label_val = case_info["label"]
        label = torch.tensor(label_val, dtype=torch.long)
        case_id = case_info["case_id"]
        
        # 处理 Mask 标签
        has_mask_bool = case_info.get("has_mask", False)
        mask_path_to_load = None
        
        # 如果当前序列自带 mask，直接使用
        if has_mask_bool and case_info.get("mask_path") is not None:
            mask_path_to_load = Path(case_info["mask_path"])
        # 如果当前序列没有 mask，但样本是阳性，去 mask_index 跨序列查找并借用
        elif label_val != 0:
            if case_id in self.mask_index:
                valid_seqs = self.mask_index[case_id]
                # 优先选用 seq 3 (FLAIR) 的 mask，如果存在的话
                if 3 in valid_seqs or len(valid_seqs) > 0:
                    target_seq = 3 if 3 in valid_seqs else valid_seqs[0]
                    # 根据目标序列动态重构正确的 mask 绝对路径
                    target_dir = nii_path.parent.parent / str(target_seq)
                    mask_name = f"case_{case_id}_{target_seq}_mask.nii.gz"
                    potential_mask_path = target_dir / mask_name
                    if potential_mask_path.exists():
                        mask_path_to_load = potential_mask_path

        if mask_path_to_load is not None:
            # 1. 读取由于各种软件生成的可能带有 255 或其他非1值的原始标注
            raw_mask_tensor = load_nii_as_tensor(mask_path_to_load)
            
            # 2. 彻底二值化清洗：凡是大于 0 的像素都归一化为纯净的 1 (病灶区域)
            base_binary_mask = (raw_mask_tensor > 0).long()
            
            # 3. 映射多类别：根据当前样本的分类 (label_val 即 1 或者 2)，为其赋予专属类别色彩！
            mask_tensor = base_binary_mask * label_val
            has_mask_flag = torch.tensor(1.0, dtype=torch.float32)
            
        elif label_val == 0:
            # 2. 正常样本 (label=0) -> 全0属于真实的 Ground Truth，因此置 flag = 1.0!
            mask_tensor = torch.zeros_like(image_tensor, dtype=torch.long)
            has_mask_flag = torch.tensor(1.0, dtype=torch.float32)
            
        else:
            # 3. 阳性样本但缺少所有 Mask -> flag = 0.0 跳过分割 Loss
            mask_tensor = torch.zeros_like(image_tensor, dtype=torch.long)
            has_mask_flag = torch.tensor(0.0, dtype=torch.float32)
        
        return image_tensor, label, mask_tensor, has_mask_flag, case_id


def load_pt_dataset(pt_path):
    # 此处读取到的只是个几十K的文本字典
    data = torch.load(pt_path, weights_only=False)
    # 将字典中的路径和索引喂给代理 Dataset
    return PTDataset(data["cases"])