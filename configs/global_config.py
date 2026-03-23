'''
项目的全局设置，用户可以根据需要进行修改
[重构] 
- 支持动态扩展类别
- 集中管理 "类别名称" 与 "数据源文件夹" 的映射
'''
from pathlib import Path

# EXPERIMENT_VERSION = "version1"   # 原版 
EXPERIMENT_VERSION = "version2"     # 采用加强版炎症数据

SEED = 42

# ========== Task & Data Source ==========
# [核心修改] 
# 定义类别及其对应的原始数据子目录列表
# 格式: "ClassName": ["SubDir1", "SubDir2", ...]
# 注意：Key 的顺序决定了 label id (0, 1, 2...)
# 如果需要添加新类别，只需在这里添加新的 key-value 对即可，预处理脚本会自动识别并处理
CLASS_DATA_MAP = {
    "normal": [
        "正常头颅MRI",
    ],
    # "meningitis": [
    #     "脑膜病变图像/脑膜炎主诊",
    #     "脑膜病变图像/脑膜炎次诊",
    # ],
    # "encephalitis": [
    #     "脑膜病变图像/脑炎",
    #     "脑膜病变图像/脑炎次诊",
    # ],
    # 脑膜炎和脑炎暂时合并为炎症类，后续可以根据需要拆分
    # "inflammation": [
    #     "脑膜病变图像/脑膜炎主诊",
    #     "脑膜病变图像/脑膜炎次诊",
    #     "脑膜病变图像/脑炎",
    #     "脑膜病变图像/脑炎次诊",
    # ],
    "inflammation": [
        "脑膜病变图像/不明显炎症",
    ],
    "metastasis": [
        "脑膜病变图像/脑膜转移",
    ],
    # 未来扩展示例:
    # "Metastasis": ["脑膜病变图像/脑膜转移"],
    # "Encephalitis": ["脑膜病变图像/脑炎"],
}

# 自动生成类别列表和数量
CLASS_NAMES = list(CLASS_DATA_MAP.keys())
NUM_CLASSES = len(CLASS_NAMES)

# 为了兼容旧代码，保留单独的列表（可选，但建议在预处理脚本中改用 CLASS_DATA_MAP）
# MENINGITIS_SUBDIRS = CLASS_DATA_MAP["Meningitis"] 
# NORMAL_SUBDIRS = CLASS_DATA_MAP["Normal"]

# ========== Sequences ==========
ALL_SEQUENCES = ["T1", "T2", "FLAIR"]
NUM_SEQUENCES = len(ALL_SEQUENCES)

# ========== Paths ==========
RAW_DATA_PATH = Path("/home/ailab/data/brainMRI/脑膜病变")    # 原始数据根目录
# PROCESSED_DATA_PATH = "data/processed"
PROCESSED_DATA_PATH = Path(f"{EXPERIMENT_VERSION}/data")

# ========== Preprocessing ==========
TARGET_SPACING = (1.0, 1.0, 1.0)  # (D, H, W)
TARGET_SHAPE = (160, 192, 160)

# ========== Dataset ==========
DATASET_ROOT = Path(f"{EXPERIMENT_VERSION}/datasets")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

K_FOLDS = 5 # 用于交叉验证的折数
K_FOLDS_VAL_RATIO = 0.15  # val在 train+val 中的比例

# ========== Inference ==========
INFERENCE_OUTPUT_DIR = Path(f"{EXPERIMENT_VERSION}/inference_outputs")