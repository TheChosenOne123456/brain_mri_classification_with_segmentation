"""项目级全局设置。

这里只放跨预处理、数据集构建、训练都必须保持一致的定义。各阶段容易变化的
参数随实验保存在 output/data-* 或 output/runs-* 下的独立配置文件中。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_CASE_IDS_PATH = PROJECT_ROOT / "excluded_case_ids.txt"

# 可复现性与交叉验证设置
SEED = 42
K_FOLDS = 5

# Key 的顺序决定 label id，修改后必须同步检查已有数据集和 checkpoint。
CLASS_DATA_MAP = {
    "normal": [
        "正常头颅MRI",
    ],
    "inflammation": [
        "脑膜病变图像/脑膜炎主诊",
        "脑膜病变图像/脑膜炎次诊",
        "脑膜病变图像/脑炎",
        "脑膜病变图像/脑炎次诊",
    ],
    "metastasis": [
        "脑膜病变图像/脑膜转移",
    ],
}
CLASS_NAMES = list(CLASS_DATA_MAP)
NUM_CLASSES = len(CLASS_NAMES)

# 固定序列顺序：seq1=T1、seq2=T2、seq3=FLAIR。
ALL_SEQUENCES = ["T1", "T2", "FLAIR"]
NUM_SEQUENCES = len(ALL_SEQUENCES)

# 原始数据路径属于项目级数据源定义，不随单次实验输出目录变化。
RAW_DATA_PATH = Path("/home/ailab/data/brainMRI/脑膜病变")
MASK_ROOTS = [
    Path("/home/ailab/data/brainMRI/脑炎mask"),
    Path("/home/ailab/data/brainMRI/脑膜炎mask"),
    Path("/home/ailab/data/brainMRI/脑膜转移mask"),
    Path("/home/ailab/data/brainMRI/脑炎mask1"),
    Path("/home/ailab/data/brainMRI/脑炎mask2"),
    Path("/home/ailab/data/brainMRI/脑炎mask3"),
    Path("/home/ailab/data/brainMRI/脑炎mask4"),
    Path("/home/ailab/data/brainMRI/脑炎mask5"),
    Path("/home/ailab/data/brainMRI/脑炎mask6"),
    Path("/home/ailab/data/brainMRI/脑炎mask7"),
    Path("/home/ailab/data/brainMRI/脑炎mask8"),
]
