from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from configs.global_config import CLASS_NAMES, SEED


# ================== 工具函数 ==================
def load_nii_as_tensor(nii_path: Path) -> torch.Tensor:
    """
    读取【已预处理】后的 nii.gz
    返回: torch.Tensor [1, D, H, W]
    注意：不做任何空间处理，只读 + 转 tensor
    """
    nii = nib.load(str(nii_path))
    data = nii.get_fdata(dtype=np.float32)
    # nibabel 返回 NIfTI 轴顺序 (X, Y, Z)；模型和预处理配置使用 (D, H, W) = (Z, Y, X)
    data = np.transpose(data, (2, 1, 0))
    data = np.ascontiguousarray(data)
    return torch.from_numpy(data.copy()).unsqueeze(0)


# 目前已弃用
def collect_cases(seq_id: int, label: int, data_root=None):
    """
    从 data/processed/{label}/{seq_id}/ 下收集该序列的所有 case
    """
    if data_root is None:
        from runtime_defaults import PROCESSED_DATA_PATH

        data_root = PROCESSED_DATA_PATH
    seq_dir = Path(data_root) / ("1_meningitis" if label == 1 else "0_normal") / str(seq_id)
    cases = []

    for nii_file in sorted(seq_dir.glob(f"case_*_{seq_id}.nii.gz")):
        case_id = nii_file.name.split("_")[1]
        cases.append({
            "case_id": case_id,
            "nii_path": nii_file,
            "label": label
        })

    return cases


# 按序列收集 case
def collect_cases_by_seq(seq_id: int, data_root=None, class_names=None):
    """
    按序列收集 case，返回 dict：
    {
        case_id: {
            "case_id": str,
            "nii_path": Path,
            "label": int
        }
    }
    """
    cases = {}
    if data_root is None:
        from runtime_defaults import PROCESSED_DATA_PATH

        data_root = PROCESSED_DATA_PATH
    data_root = Path(data_root)
    class_names = CLASS_NAMES if class_names is None else list(class_names)

    for label_id, label_name in enumerate(class_names):
        dir_name = f"{label_id}_{label_name}"
        
        seq_dir = data_root / dir_name / str(seq_id)
        
        if not seq_dir.exists():
            continue

        # 遍历该目录下的所有 case 文件
        # 文件名格式: case_{case_id}_{seq_id}.nii.gz
        for nii_file in seq_dir.glob(f"case_*_{seq_id}.nii.gz"):
            # 解析 case_id (文件名格式为 case_0001_1.nii.gz)
            parts = nii_file.name.split("_")
            if len(parts) >= 2:
                case_id = parts[1]
                
                # 探查是否存在对应的 Mask
                mask_file = nii_file.parent / f"case_{case_id}_{seq_id}_mask.nii.gz"
                has_mask = mask_file.exists()
                
                cases[case_id] = {
                    "case_id": case_id,
                    "nii_path": nii_file,
                    "label": label_id,
                    "has_mask": has_mask,                     # <--- 新增
                    "mask_path": mask_file if has_mask else None  # <--- 新增
                }

    return cases


def build_dataset(cases, seed=SEED):
    """
    即时加载改造版：不再完整载入并拼接图像张量。
    而是仅返回轻量级的 case 信息（包含 NIfTI 的路径）。
    随后外层的 torch.save 只会生成几十 KB 的记录小文件。
    """
    return {
        "cases": cases,
        "meta": {
            "num_samples": len(cases),
            "created_time": datetime.now().isoformat(),
            "seed": seed,
        }
    }
