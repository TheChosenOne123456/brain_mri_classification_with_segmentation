import random
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

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        # 核心：真正到取数据时才依据路径去拿取和解码大文件
        case_info = self.cases[idx]
        image_tensor = load_nii_as_tensor(case_info["nii_path"])
        label_val = case_info["label"]
        label = torch.tensor(label_val, dtype=torch.long)
        
        # 处理 Mask 标签
        has_mask_bool = case_info.get("has_mask", False)
        
        if has_mask_bool and case_info.get("mask_path") is not None:
            # 1. 阳性样本，且拥有真实的 Mask 文件
            mask_tensor = load_nii_as_tensor(case_info["mask_path"]).long()
            has_mask_flag = torch.tensor(1.0, dtype=torch.float32)
            
        elif label_val == 0:
            # 2. 正常样本 (label=0) -> 全0属于真实的 Ground Truth，因此置 flag = 1.0!
            mask_tensor = torch.zeros_like(image_tensor, dtype=torch.long)
            has_mask_flag = torch.tensor(1.0, dtype=torch.float32)
            
        else:
            # 3. 阳性样本但缺少 Mask（例如你的绝大部分不明显炎症和部分早期数据） -> flag = 0.0 跳过分割 Loss
            mask_tensor = torch.zeros_like(image_tensor, dtype=torch.long)
            has_mask_flag = torch.tensor(0.0, dtype=torch.float32)
        
        return image_tensor, label, mask_tensor, has_mask_flag, case_info["case_id"]


def load_pt_dataset(pt_path):
    # 此处读取到的只是个几十K的文本字典
    data = torch.load(pt_path, weights_only=False)
    # 将字典中的路径和索引喂给代理 Dataset
    return PTDataset(data["cases"])
