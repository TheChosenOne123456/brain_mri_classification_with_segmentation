"""阶段配置加载与标准产物目录解析。"""

import importlib.util
from pathlib import Path
from types import ModuleType
from uuid import uuid4


ARTIFACT_DIR_NAMES = {"data", "datasets", "checkpoints"}

PREPROCESS_CONFIG_FIELDS = (
    "TARGET_SPACING",
    "TARGET_SHAPE",
    "BRAIN_EXTRACTOR",
    "HD_BET_DEVICE",
    "HD_BET_MODE",
    "HD_BET_TTA",
    "HD_BET_VERBOSE",
    "HD_BET_TARGET_ORIENTATION",
    "FOREGROUND_DILATION_MM",
    "FOREGROUND_ZERO_OUTSIDE",
    "INTENSITY_CLIP_PERCENTILES",
    "INTENSITY_ROBUST_ZSCORE",
    "PREPROCESS_MIN_FILE_SIZE_MB",
    "PREPROCESS_MAX_ZERO_RATIO",
    "PREPROCESS_MIN_NONZERO_BBOX_FRACTION",
)

DATASET_CONFIG_FIELDS = ("K_FOLDS_VAL_RATIO",)

TRAIN_CONFIG_FIELDS = (
    "NUM_EPOCHS",
    "MIN_EPOCHS",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "DEVICE",
    "NUM_WORKERS",
    "PATIENCE",
    "CLASSIFICATION_LOSS",
    "CLASS_WEIGHT_POWER",
    "CLASS_BALANCE_BETA",
    "FOCAL_GAMMA",
    "SEG_ALPHA",
    "SEG_CLASS_WEIGHTS",
)

META_FUSION_CONFIG_FIELDS = (
    "BASE_MODEL_NAMES",
    "BASE_BATCH_SIZE",
    "NUM_WORKERS",
    "DEVICE",
    "META_WEIGHT_STEP",
    "META_SELECTION_METRIC",
    "META_MIN_ACCURACY",
    "META_ACCURACY_TOLERANCE",
    "META_THRESHOLD_MIN",
    "META_THRESHOLD_MAX",
    "META_THRESHOLD_STEPS",
)


def load_python_config(config_path, required_names=()):
    """从任意路径加载 Python 配置，并检查必需字段。"""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    module_name = f"_stage_config_{path.stem}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _validate_required_names(module, path, required_names)
    module.__config_path__ = path
    return module


def _validate_required_names(config: ModuleType, path: Path, required_names):
    missing = [name for name in required_names if not hasattr(config, name)]
    if missing:
        raise ValueError(
            f"Config {path} is missing required fields: {', '.join(missing)}"
        )


def resolve_input_artifact_dir(root, artifact_name):
    """将实验根目录或产物目录本身解析为已存在的标准产物目录。"""
    if artifact_name not in ARTIFACT_DIR_NAMES:
        raise ValueError(f"Unknown artifact directory: {artifact_name}")

    root = Path(root).expanduser().resolve()
    artifact_dir = root if root.name == artifact_name else root / artifact_name
    if not artifact_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {artifact_name} directory not found: {artifact_dir}"
        )
    return artifact_dir


def resolve_output_artifact_dir(root, artifact_name):
    """将输出根目录或产物目录本身解析为标准产物目录。"""
    if artifact_name not in ARTIFACT_DIR_NAMES:
        raise ValueError(f"Unknown artifact directory: {artifact_name}")

    root = Path(root).expanduser().resolve()
    return root if root.name == artifact_name else root / artifact_name


def infer_data_dir(data_root):
    """从实验根或 datasets 目录推断对应 data 目录，供旧 .pt 路径重定位。"""
    root = Path(data_root).expanduser().resolve()
    if root.name == "data":
        candidate = root
    elif root.name == "datasets":
        candidate = root.parent / "data"
    else:
        candidate = root / "data"
    return candidate if candidate.is_dir() else None
