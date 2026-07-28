"""尚未完全参数化的入口所使用的默认实验。

阶段配置本身属于实验产物，分别保存在数据实验和训练实验目录中。新建实验时
应显式传入 config/data/checkpoint 路径；这里仅为仍使用默认值的评估、推理和
基线脚本集中声明迁移后历史实验的位置。
"""

from configs.config_utils import (
    PREPROCESS_CONFIG_FIELDS,
    TRAIN_CONFIG_FIELDS,
    load_python_config,
)
from configs.global_config import *


DATA_EXPERIMENT_ROOT = PROJECT_ROOT / "output" / "data-hdbet"
TRAIN_EXPERIMENT_ROOT = PROJECT_ROOT / "output" / "runs-cross-entropy"

PREPROCESS_CONFIG_PATH = DATA_EXPERIMENT_ROOT / "preprocessing_config.py"
TRAIN_CONFIG_PATH = TRAIN_EXPERIMENT_ROOT / "train_config.py"

_preprocess_config = load_python_config(
    PREPROCESS_CONFIG_PATH,
    PREPROCESS_CONFIG_FIELDS,
)
_train_config = load_python_config(
    TRAIN_CONFIG_PATH,
    TRAIN_CONFIG_FIELDS,
)

for _field in PREPROCESS_CONFIG_FIELDS:
    globals()[_field] = getattr(_preprocess_config, _field)
for _field in TRAIN_CONFIG_FIELDS:
    globals()[_field] = getattr(_train_config, _field)

PROCESSED_DATA_PATH = DATA_EXPERIMENT_ROOT / "data"
DATASET_ROOT = DATA_EXPERIMENT_ROOT / "datasets"
CKPT_ROOT = TRAIN_EXPERIMENT_ROOT / "checkpoints"
INFERENCE_OUTPUT_DIR = TRAIN_EXPERIMENT_ROOT / "inference_outputs"
SEG_OUTPUT_DIR = TRAIN_EXPERIMENT_ROOT / "seg_outputs"

DATASET_DIRS = [
    DATASET_ROOT / f"seq{seq_id}_{seq_name}"
    for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
]
CKPT_DIRS = [
    CKPT_ROOT / f"seq{seq_id}_{seq_name}"
    for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
]
