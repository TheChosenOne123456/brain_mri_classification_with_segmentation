# configs/train_config.py
from pathlib import Path

from configs.global_config import *

# ================== 基本训练配置 ==================
SEED = 42

NUM_EPOCHS = 100
# MIN_EPOCHS = 40  # 最少训练轮数，保护期内不触发早停
MIN_EPOCHS = 25  # FoundationModel 训练更快，保护期适当缩短
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ================== 分类损失 ==================
CLASS_BALANCE_BETA = 0.9999
FOCAL_GAMMA = 2.0

DEVICE = "cuda"   # "cuda" or "cpu"
# 直接从 NIfTI 懒加载 3D 体数据时，多进程 DataLoader 容易出现共享内存/
# storage 问题，先用单进程保证训练可跑通。后续如需提速，建议先做张量缓存。
NUM_WORKERS = 32

# PATIENCE = 20  # 早停耐心值
PATIENCE = 5  # FoundationModel 训练更快，早停耐心值适当缩短

# ================== 序列信息（固定顺序！） ==================
# SEQ_IDS = [1, 2, 3]
# SEQ_NAMES = ["T1", "T2", "FLAIR"]

# ================== 数据集路径（与 SEQ_IDS 一一对应） ==================
DATASET_DIRS = [
    Path(f"{EXPERIMENT_VERSION}/datasets/seq1_T1"),
    Path(f"{EXPERIMENT_VERSION}/datasets/seq2_T2"),
    Path(f"{EXPERIMENT_VERSION}/datasets/seq3_FLAIR"),
    # Path(f"{EXPERIMENT_VERSION}/datasets/seq4_DWI"),
    # Path(f"{EXPERIMENT_VERSION}/datasets/seq5_+C"),
]

# ================== 模型保存路径 ==================
CKPT_ROOT = Path(f"{EXPERIMENT_VERSION}/checkpoints")

# 每个序列单独一个目录
CKPT_DIRS = [
    CKPT_ROOT / "seq1_T1",
    CKPT_ROOT / "seq2_T2",
    CKPT_ROOT / "seq3_FLAIR",
    # CKPT_ROOT / "seq4_DWI",
    # CKPT_ROOT / "seq5_+C",
]

# ================== 分割相关配置 ==================
SEG_ALPHA = 0.5  # 分割损失的权重占比
SEG_CLASS_WEIGHTS = [0.5, 2.0, 2.0]  # 背景和病灶的权重
