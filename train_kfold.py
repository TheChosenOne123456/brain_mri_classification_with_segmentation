'''
K-Fold 训练脚本：指定 --fold 参数 (1~5) 进行训练
模型将保存为 fold{k}_model_best.pth
[适配新服务器：8x RTX 3080]

层级模型示例（seq1/seq2 无分割头，seq3 自动启用分割头）：
python train_kfold.py --config output/runs-hierarchical/train_config.py \
  --data-root output/data-hdbet --output-root output/runs-hierarchical \
  --seq 1 --model FoundationModelHierarchical --fold 1
'''
import argparse
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
# 引入 AMP 模块 (兼容新旧版本写法)
try:
    from torch.amp import GradScaler, autocast
    scaler_args = {'device': 'cuda'}
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    scaler_args = {}

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.config_utils import (
    TRAIN_CONFIG_FIELDS,
    infer_data_dir,
    load_python_config,
    resolve_input_artifact_dir,
    resolve_output_artifact_dir,
)
from configs.global_config import (
    ALL_SEQUENCES,
    CLASS_NAMES,
    K_FOLDS,
    NUM_CLASSES,
    SEED,
)
from models.model_factory import (
    MODEL_CHOICES,
    create_model,
    forward_model,
    hierarchical_predictions,
    model_capabilities,
    required_train_config_fields,
)
from utils.class_aware_sampling import (
    ClassAwareEpochSampler,
    parse_deferred_resampling_config,
)
from utils.losses import ClassBalancedFocalLoss
from utils.train_and_test import set_seed, load_pt_dataset
from utils.volume_augmentation import (
    parse_volume_augmentation_config,
    wrap_training_dataset,
)

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


def apply_train_config(config, field_names):
    """让训练循环继续使用原有同名常量，避免改动核心训练流程。"""
    for name in field_names:
        globals()[name] = getattr(config, name)


def get_classification_loss_metadata(criterion, class_counts):
    """生成与实际分类损失一致的 checkpoint 元数据。"""
    metadata = {
        "class_counts": class_counts.detach().cpu().tolist(),
    }

    if isinstance(criterion, nn.CrossEntropyLoss):
        weights = criterion.weight
        metadata.update({
            "classification_loss": "weighted_cross_entropy"
            if weights is not None else "cross_entropy",
            "class_weights": weights.detach().cpu().tolist()
            if weights is not None else None,
        })
        return metadata

    if isinstance(criterion, ClassBalancedFocalLoss):
        metadata.update({
            "classification_loss": "class_balanced_focal",
            "class_weights": criterion.class_weights.detach().cpu().tolist(),
            "class_balance_beta": CLASS_BALANCE_BETA,
            "focal_gamma": criterion.gamma,
        })
        return metadata

    metadata["classification_loss"] = criterion.__class__.__name__
    return metadata


def build_classification_criterion(class_counts, total_samples):
    """根据训练配置创建分类损失；默认分支完全复刻原加权交叉熵。"""
    if torch.any(class_counts <= 0):
        raise ValueError(
            f"Every class needs training samples, got {class_counts.tolist()}"
        )

    if CLASSIFICATION_LOSS == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
        print("Classification Loss: CrossEntropyLoss")
        return criterion.to(DEVICE)

    if CLASSIFICATION_LOSS == "weighted_cross_entropy":
        raw_weights = total_samples / (NUM_CLASSES * class_counts.float() + 1e-6)
        class_weights = torch.pow(raw_weights, CLASS_WEIGHT_POWER).to(DEVICE)
        print(f"Class Weights: {class_weights.tolist()}")
        print(
            "Classification Loss: weighted CrossEntropyLoss "
            f"(power={CLASS_WEIGHT_POWER})"
        )
        return nn.CrossEntropyLoss(weight=class_weights)

    if CLASSIFICATION_LOSS == "class_balanced_focal":
        criterion = ClassBalancedFocalLoss(
            samples_per_class=class_counts,
            beta=CLASS_BALANCE_BETA,
            gamma=FOCAL_GAMMA,
        ).to(DEVICE)
        print(f"Class-Balanced Weights: {criterion.class_weights.tolist()}")
        print(
            "Classification Loss: ClassBalancedFocalLoss "
            f"(beta={CLASS_BALANCE_BETA}, gamma={FOCAL_GAMMA})"
        )
        return criterion

    raise ValueError(
        f"Unsupported CLASSIFICATION_LOSS: {CLASSIFICATION_LOSS}. "
        "Expected cross_entropy, weighted_cross_entropy, or class_balanced_focal."
    )


def build_subtype_criterion(class_counts):
    """仅针对 inflammation/metastasis 构建配置化加权的二分类损失。"""
    subtype_counts = class_counts[1:].float()
    if len(subtype_counts) != 2 or torch.any(subtype_counts <= 0):
        raise ValueError(
            "The subtype head needs both inflammation and metastasis samples, "
            f"got class counts {class_counts.tolist()}"
        )
    raw_weights = subtype_counts.sum() / (2 * subtype_counts + 1e-6)
    subtype_weights = torch.pow(
        raw_weights,
        SUBTYPE_CLASS_WEIGHT_POWER,
    ).to(DEVICE)
    print(f"Subtype Class Weights: {subtype_weights.tolist()}")
    print(
        "Subtype Loss: weighted CrossEntropyLoss "
        f"(alpha={SUBTYPE_ALPHA}, power={SUBTYPE_CLASS_WEIGHT_POWER})"
    )
    return nn.CrossEntropyLoss(weight=subtype_weights)


def compute_subtype_loss(subtype_logits, labels, subtype_criterion):
    """正常样本不参与子类头损失；异常标签 1/2 映射为子类标签 0/1。"""
    abnormal = labels > 0
    if abnormal.any():
        return subtype_criterion(
            subtype_logits[abnormal],
            labels[abnormal] - 1,
        )
    return subtype_logits.sum() * 0.0


def compute_class_metrics(targets, predictions, class_index, beta=1.0):
    """计算指定类别的 precision、recall 和 F-beta。"""
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    true_positive = int(
        np.sum((targets == class_index) & (predictions == class_index))
    )
    predicted_positive = int(np.sum(predictions == class_index))
    actual_positive = int(np.sum(targets == class_index))

    precision = (
        true_positive / predicted_positive
        if predicted_positive > 0
        else 0.0
    )
    recall = (
        true_positive / actual_positive
        if actual_positive > 0
        else 0.0
    )
    beta_squared = beta ** 2
    denominator = beta_squared * precision + recall
    fbeta = (
        (1 + beta_squared) * precision * recall / denominator
        if denominator > 0
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "fbeta": float(fbeta),
    }


@dataclass(frozen=True)
class CheckpointCandidate:
    """一个 epoch 的验证表现及其 checkpoint 选择优先级。"""

    epoch: int
    primary_metric_name: str
    primary_metric_value: float
    macro_f1: float
    val_loss: float
    val_accuracy: float
    metastasis_precision: float
    metastasis_recall: float
    metastasis_fbeta: float
    accuracy_constraint_met: bool
    precision_constraint_met: bool

    @property
    def constraints_met(self):
        return (
            self.accuracy_constraint_met
            and self.precision_constraint_met
        )


def build_checkpoint_candidate(
    *,
    epoch,
    val_loss,
    val_accuracy,
    macro_f1,
    metastasis_metrics,
    selection_metric,
    min_accuracy,
    min_metastasis_precision,
):
    """将验证指标转换为清晰、可比较的 checkpoint 候选。"""
    if selection_metric == "metastasis_fbeta":
        primary_metric_value = metastasis_metrics["fbeta"]
    elif selection_metric == "macro_f1":
        primary_metric_value = macro_f1
    else:
        raise ValueError(
            f"Unsupported CHECKPOINT_SELECTION_METRIC: {selection_metric}. "
            "Expected metastasis_fbeta or macro_f1."
        )

    return CheckpointCandidate(
        epoch=epoch,
        primary_metric_name=selection_metric,
        primary_metric_value=float(primary_metric_value),
        macro_f1=float(macro_f1),
        val_loss=float(val_loss),
        val_accuracy=float(val_accuracy),
        metastasis_precision=float(metastasis_metrics["precision"]),
        metastasis_recall=float(metastasis_metrics["recall"]),
        metastasis_fbeta=float(metastasis_metrics["fbeta"]),
        accuracy_constraint_met=bool(val_accuracy >= min_accuracy),
        precision_constraint_met=bool(
            metastasis_metrics["precision"] >= min_metastasis_precision
        ),
    )


def is_better_checkpoint(candidate, best_candidate, metric_tolerance):
    """
    约束优先，其次目标指标；近似持平时偏好 macro-F1、低 loss 和早期 epoch。
    """
    if best_candidate is None:
        return True
    if candidate.constraints_met != best_candidate.constraints_met:
        return candidate.constraints_met

    primary_delta = (
        candidate.primary_metric_value
        - best_candidate.primary_metric_value
    )
    if abs(primary_delta) > metric_tolerance:
        return primary_delta > 0

    macro_f1_delta = candidate.macro_f1 - best_candidate.macro_f1
    if abs(macro_f1_delta) > metric_tolerance:
        return macro_f1_delta > 0

    if candidate.val_loss != best_candidate.val_loss:
        return candidate.val_loss < best_candidate.val_loss
    return candidate.epoch < best_candidate.epoch


def compute_segmentation_loss(
    seg_logits,
    mask,
    mask_flag,
    seg_criterion,
):
    gt_mask = mask.squeeze(1) if mask.dim() == 5 else mask
    unreduced_seg_loss = seg_criterion(seg_logits, gt_mask)
    per_sample_ce_loss = unreduced_seg_loss.mean(dim=(1, 2, 3))
    per_sample_dice_loss = compute_dice_loss(
        seg_logits,
        gt_mask,
        num_classes=NUM_CLASSES,
    )
    per_sample_total = per_sample_ce_loss + per_sample_dice_loss
    valid_mask_count = mask_flag.sum()
    if valid_mask_count > 0:
        return (per_sample_total * mask_flag).sum() / valid_mask_count
    return seg_logits.sum() * 0.0


def compute_dice(pred_mask, gt_mask, num_classes=3, smooth=1e-5):
    dices = []
    # 忽略类别 0 (背景/正常)，仅计算异常类别（如 1:炎症, 2:转移瘤）的 Dice
    for c in range(1, num_classes):
        pred_c = (pred_mask == c).float()
        gt_c = (gt_mask == c).float()

        intersection = (pred_c * gt_c).sum(dim=(1, 2, 3))
        cardinality = pred_c.sum(dim=(1, 2, 3)) + gt_c.sum(dim=(1, 2, 3))

        dice = (2. * intersection + smooth) / (cardinality + smooth)
        dices.append(dice)
    
    return torch.stack(dices, dim=0).mean(dim=0)

def compute_dice_loss(pred_logits, gt_mask, num_classes=3, smooth=1e-5):
    # 将模型输出的通道维映射为概率分布 [B, C, D, H, W]
    pred_probs = F.softmax(pred_logits, dim=1)
    
    # 将真实的标记也转为 One-hot 形式以便对应计算 [B, C, D, H, W]
    with torch.no_grad():
        gt_one_hot = F.one_hot(gt_mask, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    
    dice_loss = 0.0
    # 忽略类别 0 (背景)，仅去优化类别 1 和 2
    for c in range(1, num_classes):
        pred_c = pred_probs[:, c, ...]
        gt_c   = gt_one_hot[:, c, ...]
        
        intersection = (pred_c * gt_c).sum(dim=(1, 2, 3))
        cardinality  = pred_c.sum(dim=(1, 2, 3)) + gt_c.sum(dim=(1, 2, 3))
        
        dice = (2. * intersection + smooth) / (cardinality + smooth)
        dice_loss += (1.0 - dice)  # 转化为 Loss，1 减去 Dice
        
    return dice_loss / (num_classes - 1)

# ================== [新增专区] ==================
class MultiChannelDataset(Dataset):
    """
    代理 Dataset：将多个单通道的 Dataset 在被调用时动态组合成多通道。
    极其省内存且不需要关心底层具体的数据结构。
    """
    def __init__(self, datasets_list):
        self.datasets = datasets_list
        # 将第一个序列的 labels 暴露出去，供外部计算 loss 权重使用
        self.labels = datasets_list[0].labels

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        xs = []
        final_mask = None
        final_mask_flag = None
        final_y = None
        final_case_id = None
        
        # 只需要遍历一次！绝不重复调用 ds[idx] 导致硬盘多次读取 3D 影像！
        for i, ds in enumerate(self.datasets):
            x, y, mask, mask_flag, case_id = ds[idx]  
            xs.append(x)
            
            # 初始化基础标签 (用第一个序列的数据垫底)
            if i == 0:
                final_y = y
                final_case_id = case_id
                final_mask = mask
                final_mask_flag = mask_flag
            
            # 【核心修复】一旦发现当前序列（比如 FLAIR）有医生画的真实病灶 Mask，就覆盖上去！
            if mask_flag > 0.5:
                final_mask = mask
                final_mask_flag = mask_flag
        
        # 在通道维度(dim=0)进行拼接：3个 [1, D, H, W] -> [3, D, H, W]
        multi_x = torch.cat(xs, dim=0)
        
        return multi_x, final_y, final_mask, final_mask_flag, final_case_id
# ================================================

# 辅助函数：计算准确率
def calculate_accuracy(loader, model, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y, mask, mask_flag, _ in loader:
            x, y = x.to(device), y.to(device)
            # 验证时也开启 autocast 以节省显存
            with autocast(**({'device_type': 'cuda'} if 'device' in scaler_args else {})):
                outputs = forward_model(model, x)["classification"]
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


def _initial_model_name_for_sequence(initial_model_names, sequence_id):
    if isinstance(initial_model_names, dict):
        model_name = initial_model_names.get(
            sequence_id,
            initial_model_names.get(str(sequence_id)),
        )
    elif isinstance(initial_model_names, (list, tuple)):
        model_name = (
            initial_model_names[sequence_id - 1]
            if len(initial_model_names) >= sequence_id
            else None
        )
    else:
        model_name = None

    if not model_name:
        raise ValueError(
            "INITIAL_MODEL_NAMES must provide a source model for "
            f"sequence {sequence_id}"
        )
    return model_name


def initialize_from_baseline_checkpoint(
    model,
    *,
    config,
    init_checkpoint_root,
    sequence_id,
    sequence_name,
    fold,
    in_channels,
):
    """加载同序列、同 fold 的历史基线，并复制共享模块。"""
    initial_model_names = getattr(config, "INITIAL_MODEL_NAMES", None)
    source_model_name = _initial_model_name_for_sequence(
        initial_model_names,
        sequence_id,
    )
    source_checkpoint_root = resolve_input_artifact_dir(
        init_checkpoint_root,
        "checkpoints",
    )
    source_checkpoint_path = (
        source_checkpoint_root
        / f"seq{sequence_id}_{sequence_name}"
        / source_model_name
        / f"fold{fold}_model_best.pth"
    )
    if not source_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {source_checkpoint_path}"
        )

    source_model = create_model(
        source_model_name,
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        sequence_id=sequence_id,
    )
    checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    source_state = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint
    )
    source_model.load_state_dict(source_state, strict=True)

    initialize_method = getattr(model, "initialize_from_baseline", None)
    if initialize_method is None:
        raise TypeError(
            f"{model.__class__.__name__} does not support baseline initialization"
        )
    copied_parts = initialize_method(source_model)
    print(f"Initialized from: {source_checkpoint_path}")
    print(f"Copied model parts: {', '.join(copied_parts)}")
    return {
        "source_model_name": source_model_name,
        "source_checkpoint_path": str(source_checkpoint_path),
        "copied_model_parts": copied_parts,
    }


def configure_trainable_model_parts(model, trainable_model_parts):
    """冻结整个模型后，仅重新启用配置中列出的模块。"""
    if trainable_model_parts is None:
        return None
    if isinstance(trainable_model_parts, str):
        trainable_model_parts = (trainable_model_parts,)
    trainable_model_parts = tuple(trainable_model_parts)
    if not trainable_model_parts:
        raise ValueError("TRAINABLE_MODEL_PARTS cannot be empty")

    for parameter in model.parameters():
        parameter.requires_grad = False

    for part_name in trainable_model_parts:
        try:
            module = model.get_submodule(part_name)
        except AttributeError as error:
            available = ", ".join(name for name, _ in model.named_children())
            raise ValueError(
                f"Unknown trainable model part: {part_name}. "
                f"Top-level parts: {available}"
            ) from error
        for parameter in module.parameters():
            parameter.requires_grad = True

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters == 0:
        raise ValueError(
            f"TRAINABLE_MODEL_PARTS {trainable_model_parts} contain no parameters"
        )
    print(f"Trainable Model Parts: {', '.join(trainable_model_parts)}")
    print(
        "Trainable Parameters: "
        f"{trainable_parameters:,}/{total_parameters:,} "
        f"({trainable_parameters / total_parameters:.4%})"
    )
    return trainable_model_parts


def keep_frozen_modules_in_eval_mode(model):
    """model.train() 后保持完全冻结的顶层模块处于 eval 模式。"""
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    for module in base_model.children():
        parameters = tuple(module.parameters())
        if parameters and not any(
            parameter.requires_grad for parameter in parameters
        ):
            module.eval()


def model_part_has_trainable_parameters(model, part_name):
    try:
        module = model.get_submodule(part_name)
    except AttributeError:
        return False
    return any(parameter.requires_grad for parameter in module.parameters())


def main(args):
    model_name = args.model
    if model_name == "FLAIRUNet3D":
        raise ValueError(
            "FLAIRUNet3D uses binary lesion masks, deep supervision, staged "
            "freezing, and parameter-group learning rates. Train it with "
            "train_flair_unet.py instead of train_kfold.py."
        )
    train_config_fields = (
        TRAIN_CONFIG_FIELDS + required_train_config_fields(model_name)
    )
    config = load_python_config(args.config, train_config_fields)
    apply_train_config(config, train_config_fields)
    classification_alpha = float(
        getattr(config, "CLASSIFICATION_ALPHA", 1.0)
    )
    initialize_from_baseline = bool(
        getattr(config, "INITIALIZE_FROM_BASELINE", False)
        or args.init_checkpoint_root is not None
    )
    trainable_model_parts = getattr(
        config,
        "TRAINABLE_MODEL_PARTS",
        None,
    )
    checkpoint_selection_metric = getattr(
        config,
        "CHECKPOINT_SELECTION_METRIC",
        "macro_f1",
    )
    metastasis_f_beta = float(
        getattr(config, "METASTASIS_F_BETA", 2.0)
    )
    min_val_accuracy_by_fold = getattr(
        config,
        "MIN_VAL_ACCURACY_BY_FOLD",
        None,
    )
    if min_val_accuracy_by_fold is not None:
        fold_value = min_val_accuracy_by_fold.get(
            args.fold,
            min_val_accuracy_by_fold.get(str(args.fold)),
        )
        if fold_value is None:
            raise ValueError(
                "MIN_VAL_ACCURACY_BY_FOLD must provide a value for "
                f"fold {args.fold}"
            )
        min_val_accuracy = float(fold_value)
    else:
        min_val_accuracy = float(
            getattr(
                config,
                "MIN_VAL_ACCURACY",
                getattr(config, "HIERARCHICAL_MIN_VAL_ACCURACY", 0.0),
            )
        )
    min_val_metastasis_precision = float(
        getattr(
            config,
            "MIN_VAL_METASTASIS_PRECISION",
            getattr(
                config,
                "HIERARCHICAL_MIN_VAL_METASTASIS_PRECISION",
                0.0,
            ),
        )
    )
    checkpoint_metric_tolerance = float(
        getattr(config, "CHECKPOINT_METRIC_TOLERANCE", 1e-4)
    )
    deferred_resampling = parse_deferred_resampling_config(
        getattr(config, "DEFERRED_CLASS_AWARE_RESAMPLING", None),
        NUM_CLASSES,
    )
    train_augmentation = parse_volume_augmentation_config(
        getattr(config, "TRAIN_AUGMENTATION", None)
    )
    dataset_root = resolve_input_artifact_dir(args.data_root, "datasets")
    processed_data_root = infer_data_dir(args.data_root)
    ckpt_root = resolve_output_artifact_dir(args.output_root, "checkpoints")
    dataset_dirs = [
        dataset_root / f"seq{seq_id}_{seq_name}"
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
    ]
    ckpt_dirs = [
        ckpt_root / f"seq{seq_id}_{seq_name}"
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
    ]

    set_seed(SEED)
    
    current_fold = args.fold
    print(f"Training config : {config.__config_path__}")
    print(f"Dataset input   : {dataset_root}")
    print(f"Checkpoint root : {ckpt_root}")
    train_config_snapshot = {
        name: getattr(config, name)
        for name in train_config_fields
    }
    train_config_snapshot.update({
        "CLASSIFICATION_ALPHA": classification_alpha,
        "INITIALIZE_FROM_BASELINE": initialize_from_baseline,
        "INITIAL_MODEL_NAMES": getattr(config, "INITIAL_MODEL_NAMES", None),
        "TRAINABLE_MODEL_PARTS": trainable_model_parts,
        "CHECKPOINT_SELECTION_METRIC": checkpoint_selection_metric,
        "METASTASIS_F_BETA": metastasis_f_beta,
        "MIN_VAL_ACCURACY": min_val_accuracy,
        "MIN_VAL_ACCURACY_BY_FOLD": min_val_accuracy_by_fold,
        "MIN_VAL_METASTASIS_PRECISION": min_val_metastasis_precision,
        "HIERARCHICAL_MIN_VAL_METASTASIS_PRECISION": (
            min_val_metastasis_precision
        ),
        "CHECKPOINT_METRIC_TOLERANCE": checkpoint_metric_tolerance,
        "DEFERRED_CLASS_AWARE_RESAMPLING": (
            deferred_resampling.as_dict()
            if deferred_resampling is not None
            else None
        ),
        "TRAIN_AUGMENTATION": train_augmentation.as_dict(),
    })
    
    if args.seq is not None:
        # ---- 单通道模式 ----
        seq_id = args.seq
        seq_idx = seq_id - 1
        seq_name = ALL_SEQUENCES[seq_idx]
        in_channels = 1
        print(f"\n=== Training Fold {current_fold}/{K_FOLDS} | Seq: {seq_name} | Model: {model_name} (1-Channel) ===")

        dataset_dir = dataset_dirs[seq_idx] / f"fold{current_fold}"
        ckpt_dir = ckpt_dirs[seq_idx] / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if not dataset_dir.exists():
            print(f"Error: Dataset for fold {current_fold} not found at {dataset_dir}")
            print("Please run 'python -m scripts.build_dataset_kfold' first.")
            sys.exit(1)

        # 加载数据
        train_set = load_pt_dataset(
            dataset_dir / "train.pt",
            data_root=processed_data_root,
        )
        val_set = load_pt_dataset(
            dataset_dir / "val.pt",
            data_root=processed_data_root,
        )

    else:
        # ---- 多通道模式 ----
        in_channels = len(ALL_SEQUENCES)
        seq_name = f"Multi-Fusion ({in_channels} Channels)"
        print(f"\n=== Training Fold {current_fold}/{K_FOLDS} | Mode: {seq_name} | Model: {model_name} ===")

        train_sets_list = []
        val_sets_list = []
        
        for idx, s_name in enumerate(ALL_SEQUENCES):
            dataset_dir = dataset_dirs[idx] / f"fold{current_fold}"
            if not dataset_dir.exists():
                print(f"Error: Dataset missing for sequence {s_name} at {dataset_dir}")
                sys.exit(1)
            train_sets_list.append(
                load_pt_dataset(
                    dataset_dir / "train.pt",
                    data_root=processed_data_root,
                )
            )
            val_sets_list.append(
                load_pt_dataset(
                    dataset_dir / "val.pt",
                    data_root=processed_data_root,
                )
            )

        # 使用我们自定义的代理类包装
        train_set = MultiChannelDataset(train_sets_list)
        val_set   = MultiChannelDataset(val_sets_list)

        # 把多通道模型统一保存在独立路径中
        base_ckpt_dir = ckpt_root
        ckpt_dir = base_ckpt_dir / "multi_channel" / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)


    if train_augmentation.enabled:
        train_set = wrap_training_dataset(train_set, train_augmentation)
        print(f"Training Augmentation: {train_augmentation.as_dict()}")
        print("Validation Augmentation: disabled")


    all_labels = (
        train_set.labels.tolist()
        if isinstance(train_set.labels, torch.Tensor)
        else list(train_set.labels)
    )
    class_counts = torch.bincount(
        torch.tensor(all_labels),
        minlength=NUM_CLASSES,
    )
    total_samples = len(all_labels)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    class_aware_sampler = None
    resampled_train_loader = None
    if deferred_resampling is not None:
        class_aware_sampler = ClassAwareEpochSampler(
            all_labels,
            deferred_resampling.target_class_probabilities,
            num_samples=total_samples,
            seed=SEED + current_fold * 100_000,
        )
        resampled_train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=class_aware_sampler,
            num_workers=NUM_WORKERS,
        )
        print(
            "Deferred Class-Aware Resampling: natural sampling through "
            f"epoch {deferred_resampling.start_epoch - 1}; then target "
            f"counts {list(class_aware_sampler.target_class_counts)} "
            f"with {deferred_resampling.post_switch_loss}"
        )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = create_model(
        model_name,
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        sequence_id=args.seq,
    )
    capabilities = model_capabilities(model)
    print(f"Model Capabilities: {capabilities}")

    initialization_metadata = None
    if initialize_from_baseline:
        if args.seq is None:
            raise ValueError(
                "Baseline initialization requires an explicit --seq"
            )
        if args.init_checkpoint_root is None:
            raise ValueError(
                "This config enables baseline initialization; pass "
                "--init-checkpoint-root"
            )
        initialization_metadata = initialize_from_baseline_checkpoint(
            model,
            config=config,
            init_checkpoint_root=args.init_checkpoint_root,
            sequence_id=args.seq,
            sequence_name=seq_name,
            fold=current_fold,
            in_channels=in_channels,
        )

    configured_trainable_parts = configure_trainable_model_parts(
        model,
        trainable_model_parts,
    )
    use_subtype_loss = (
        capabilities["subtype"]
        and SUBTYPE_ALPHA > 0
        and model_part_has_trainable_parameters(model, "subtype_head")
    )
    use_segmentation_loss = (
        capabilities["segmentation"]
        and SEG_ALPHA > 0
        and model_part_has_trainable_parameters(model, "aux_heads")
    )
    if capabilities["subtype"] and not use_subtype_loss:
        print("Subtype Loss: disabled")
    if capabilities["segmentation"] and not use_segmentation_loss:
        print("Segmentation Loss: disabled (head is frozen or SEG_ALPHA <= 0)")
    if classification_alpha <= 0:
        print("Classification Loss Contribution: disabled (alpha=0)")
    model = model.to(DEVICE)

    # 启用多卡 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    natural_criterion = build_classification_criterion(
        class_counts,
        total_samples,
    )
    resampled_criterion = (
        nn.CrossEntropyLoss().to(DEVICE)
        if deferred_resampling is not None
        else None
    )
    subtype_criterion = (
        build_subtype_criterion(class_counts)
        if use_subtype_loss
        else None
    )

    seg_criterion = None
    if use_segmentation_loss:
        seg_class_weights = torch.tensor(
            SEG_CLASS_WEIGHTS,
            dtype=torch.float32,
            device=DEVICE,
        )
        seg_criterion = nn.CrossEntropyLoss(
            weight=seg_class_weights,
            reduction="none",
        )
    
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("The selected training configuration has no trainable parameters")
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 初始化混合精度 Scaler
    scaler = GradScaler(**scaler_args)

    # 训练循环 &早停
    best_candidate = None
    patience_counter = 0
    best_epoch = 0

    # 模型保存文件名区分 fold
    best_model_path = ckpt_dir / f"fold{current_fold}_model_best.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        resampling_active = (
            deferred_resampling is not None
            and epoch >= deferred_resampling.start_epoch
        )
        if resampling_active:
            balanced_epoch = epoch - deferred_resampling.start_epoch
            class_aware_sampler.set_epoch(balanced_epoch)
            epoch_train_loader = resampled_train_loader
            epoch_criterion = resampled_criterion
            sampling_stage = "class_aware"
        else:
            epoch_train_loader = train_loader
            epoch_criterion = natural_criterion
            sampling_stage = "natural"

        if (
            deferred_resampling is not None
            and epoch == deferred_resampling.start_epoch
        ):
            # Warm-up checkpoints are not valid candidates for the resampling
            # experiment; restart selection so the final file necessarily
            # reflects at least one class-aware epoch.
            best_candidate = None
            best_epoch = 0
            patience_counter = 0
            print(
                "\n[Sampling Stage] Switched to class-aware resampling "
                f"with epoch counts {list(class_aware_sampler.target_class_counts)}; "
                "classification loss is now unweighted CrossEntropyLoss; "
                "checkpoint selection restarted"
            )

        # --- Train ---
        model.train()
        keep_frozen_modules_in_eval_mode(model)
        total_loss = 0.0
        train_correct = 0
        train_total = 0 
        epoch_sample_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        
        # 进度条
        pbar = tqdm(
            epoch_train_loader,
            desc=f"Fold {current_fold} Ep {epoch}",
            leave=False,
        )
        for x, y, mask, mask_flag, _ in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mask, mask_flag = mask.to(DEVICE), mask_flag.to(DEVICE)
            optimizer.zero_grad()
            
            # AMP 前向传播
            # device_type='cuda' 用于新版 torch.amp.autocast，旧版不需要参数但兼容性不同
            if 'device' in scaler_args:
                actx = autocast(device_type='cuda')
            else:
                actx = autocast()

            with actx:
                outputs = forward_model(
                    model,
                    x,
                    return_subtype=capabilities["subtype"],
                    return_seg=use_segmentation_loss,
                )
                logits = outputs["classification"]
                loss = logits.sum() * 0.0
                if classification_alpha > 0:
                    loss = (
                        loss
                        + classification_alpha * epoch_criterion(logits, y)
                    )

                if use_subtype_loss:
                    loss_subtype = compute_subtype_loss(
                        outputs["subtype"],
                        y,
                        subtype_criterion,
                    )
                    loss = loss + SUBTYPE_ALPHA * loss_subtype

                if use_segmentation_loss:
                    loss_seg = compute_segmentation_loss(
                        outputs["segmentation"],
                        mask,
                        mask_flag,
                        seg_criterion,
                    )
                    loss = loss + SEG_ALPHA * loss_seg
            
            # AMP 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            epoch_sample_counts += torch.bincount(
                y.detach().cpu(),
                minlength=NUM_CLASSES,
            )
        
        train_loss = total_loss / len(epoch_train_loader)
        train_acc = train_correct / train_total

                # --- Val ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_preds = []
        all_hierarchical_preds = []
        all_subtype_preds = []
        all_subtype_targets = []
        all_targets = []
        
        valid_seg_dices = []  # <--- [新增] 收集每个 Batch 的有效 Dice

        with torch.no_grad():
            for x, y, mask, mask_flag, case_ids in val_loader:  # <--- [修改] 别忘了接出 case_ids 
                x, y = x.to(DEVICE), y.to(DEVICE)
                mask, mask_flag = mask.to(DEVICE), mask_flag.to(DEVICE) # <--- [新增]

                if 'device' in scaler_args:
                    actx = autocast(device_type='cuda')
                else:
                    actx = autocast()
                
                with actx:
                    outputs = forward_model(
                        model,
                        x,
                        return_subtype=capabilities["subtype"],
                        return_seg=use_segmentation_loss,
                    )
                    logits = outputs["classification"]
                    seg_logits = outputs.get("segmentation")
                    loss = logits.sum() * 0.0
                    if classification_alpha > 0:
                        loss = (
                            loss
                            + classification_alpha * epoch_criterion(logits, y)
                        )
                    if use_subtype_loss:
                        loss = loss + SUBTYPE_ALPHA * compute_subtype_loss(
                            outputs["subtype"],
                            y,
                            subtype_criterion,
                        )
                    if use_segmentation_loss:
                        loss = loss + SEG_ALPHA * compute_segmentation_loss(
                            outputs["segmentation"],
                            mask,
                            mask_flag,
                            seg_criterion,
                        )

                # ----- 计算和记录验证集 Dice -----
                if seg_logits is not None:
                    pred_masks = seg_logits.argmax(dim=1)
                    gt_masks = mask.squeeze(1) if mask.dim() == 5 else mask
                    batch_dices = compute_dice(pred_masks, gt_masks, num_classes=NUM_CLASSES)
                    
                    for i in range(len(y)):
                        if mask_flag[i] > 0.5: # 或者是 y[i] == 0 (因为你在 __getitem__ 里已经把正常的设为 1.0 了)
                            valid_seg_dices.append(batch_dices[i].item())
                # -------------------------------
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)


                all_preds.extend(predicted.cpu().numpy())
                if capabilities["subtype"]:
                    hierarchical_pred = hierarchical_predictions(
                        logits,
                        outputs["subtype"],
                    )
                    all_hierarchical_preds.extend(
                        hierarchical_pred.cpu().numpy()
                    )
                    abnormal = y > 0
                    if abnormal.any():
                        subtype_pred = outputs["subtype"].argmax(dim=1)
                        all_subtype_preds.extend(
                            subtype_pred[abnormal].cpu().numpy()
                        )
                        all_subtype_targets.extend(
                            (y[abnormal] - 1).cpu().numpy()
                        )
                all_targets.extend(y.cpu().numpy())

                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        metastasis_class_index = CLASS_NAMES.index("metastasis")
        val_metastasis_metrics = compute_class_metrics(
            all_targets,
            all_preds,
            metastasis_class_index,
            beta=metastasis_f_beta,
        )
        val_hierarchical_f1 = (
            f1_score(
                all_targets,
                all_hierarchical_preds,
                average="macro",
                zero_division=0,
            )
            if capabilities["subtype"]
            else None
        )
        val_hierarchical_acc = (
            float(
                np.mean(
                    np.asarray(all_hierarchical_preds)
                    == np.asarray(all_targets)
                )
            )
            if capabilities["subtype"]
            else None
        )
        val_hierarchical_metastasis_metrics = (
            compute_class_metrics(
                all_targets,
                all_hierarchical_preds,
                metastasis_class_index,
                beta=metastasis_f_beta,
            )
            if capabilities["subtype"]
            else None
        )
        val_subtype_metastasis_metrics = (
            compute_class_metrics(
                all_subtype_targets,
                all_subtype_preds,
                metastasis_class_index - 1,
                beta=metastasis_f_beta,
            )
            if capabilities["subtype"]
            else None
        )
        selection_accuracy = (
            val_hierarchical_acc
            if val_hierarchical_acc is not None
            else val_acc
        )
        selection_macro_f1 = (
            val_hierarchical_f1
            if val_hierarchical_f1 is not None
            else val_f1
        )
        selection_metastasis_metrics = (
            val_hierarchical_metastasis_metrics
            if val_hierarchical_metastasis_metrics is not None
            else val_metastasis_metrics
        )
        effective_selection_metric = checkpoint_selection_metric
        candidate = build_checkpoint_candidate(
            epoch=epoch,
            val_loss=val_loss,
            val_accuracy=selection_accuracy,
            macro_f1=selection_macro_f1,
            metastasis_metrics=selection_metastasis_metrics,
            selection_metric=effective_selection_metric,
            min_accuracy=min_val_accuracy,
            min_metastasis_precision=min_val_metastasis_precision,
        )

        val_dice = np.mean(valid_seg_dices) if len(valid_seg_dices) > 0 else 0.0

        # --- 打印格式 ---
        log_line = (
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}   "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
            f"val_f1: {val_f1:.4f} | "
            f"val_meta_recall: {val_metastasis_metrics['recall']:.4f} | "
            f"sampling: {sampling_stage} "
            f"{epoch_sample_counts.tolist()}"
        )
        if val_hierarchical_f1 is not None:
            constraint_status = "OK" if candidate.constraints_met else "BELOW"
            log_line += (
                f" | val_hier_acc: {val_hierarchical_acc:.4f}"
                f" | val_hier_f1: {val_hierarchical_f1:.4f}"
                f" | val_subtype_meta_recall: "
                f"{val_subtype_metastasis_metrics['recall']:.4f}"
                f" | val_hier_meta_precision: "
                f"{val_hierarchical_metastasis_metrics['precision']:.4f}"
                f" | val_hier_meta_recall: "
                f"{val_hierarchical_metastasis_metrics['recall']:.4f}"
                f" | val_hier_meta_f{metastasis_f_beta:g}: "
                f"{val_hierarchical_metastasis_metrics['fbeta']:.4f}"
                f" | selection_constraint: {constraint_status}"
            )
        elif (
            effective_selection_metric != "macro_f1"
            or min_val_accuracy > 0
            or min_val_metastasis_precision > 0
        ):
            constraint_status = "OK" if candidate.constraints_met else "BELOW"
            log_line += (
                f" | val_meta_precision: "
                f"{val_metastasis_metrics['precision']:.4f}"
                f" | val_meta_f{metastasis_f_beta:g}: "
                f"{val_metastasis_metrics['fbeta']:.4f}"
                f" | selection_constraint: {constraint_status}"
            )
        if use_segmentation_loss:
            log_line += f" | val_dice: {val_dice:.4f}"
        print(log_line)

        # --- Early Stopping check with MIN_EPOCHS ---
        is_improvement = is_better_checkpoint(
            candidate,
            best_candidate,
            checkpoint_metric_tolerance,
        )

        if is_improvement:
            best_candidate = candidate
            patience_counter = 0
            best_epoch = epoch
            
            # 保存时剥离 DataParallel 包装，否则以后加载会报错
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            
            torch.save({
                "model_state": model_to_save.state_dict(),
                "fold": current_fold,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_metastasis_precision": (
                    val_metastasis_metrics["precision"]
                ),
                "val_metastasis_recall": val_metastasis_metrics["recall"],
                "val_metastasis_fbeta": val_metastasis_metrics["fbeta"],
                "val_hierarchical_f1": val_hierarchical_f1,
                "val_hierarchical_acc": val_hierarchical_acc,
                "val_hierarchical_metastasis_precision": (
                    val_hierarchical_metastasis_metrics["precision"]
                    if val_hierarchical_metastasis_metrics is not None
                    else None
                ),
                "val_hierarchical_metastasis_recall": (
                    val_hierarchical_metastasis_metrics["recall"]
                    if val_hierarchical_metastasis_metrics is not None
                    else None
                ),
                "val_hierarchical_metastasis_fbeta": (
                    val_hierarchical_metastasis_metrics["fbeta"]
                    if val_hierarchical_metastasis_metrics is not None
                    else None
                ),
                "val_subtype_metastasis_precision": (
                    val_subtype_metastasis_metrics["precision"]
                    if val_subtype_metastasis_metrics is not None
                    else None
                ),
                "val_subtype_metastasis_recall": (
                    val_subtype_metastasis_metrics["recall"]
                    if val_subtype_metastasis_metrics is not None
                    else None
                ),
                "val_subtype_metastasis_fbeta": (
                    val_subtype_metastasis_metrics["fbeta"]
                    if val_subtype_metastasis_metrics is not None
                    else None
                ),
                "meets_accuracy_constraint": (
                    candidate.accuracy_constraint_met
                ),
                "meets_precision_constraint": (
                    candidate.precision_constraint_met
                ),
                "meets_selection_constraints": candidate.constraints_met,
                "min_val_accuracy": min_val_accuracy,
                "min_val_metastasis_precision": (
                    min_val_metastasis_precision
                ),
                "hierarchical_min_val_accuracy": (
                    min_val_accuracy
                    if capabilities["subtype"]
                    else None
                ),
                "hierarchical_min_val_metastasis_precision": (
                    min_val_metastasis_precision
                    if capabilities["subtype"]
                    else None
                ),
                "selection_metric": effective_selection_metric,
                "selection_candidate": {
                    **asdict(candidate),
                    "constraints_met": candidate.constraints_met,
                },
                "model_name": model_name,
                "model_capabilities": capabilities,
                "classification_alpha": classification_alpha,
                "sampling_stage": sampling_stage,
                "sampled_class_counts": epoch_sample_counts.tolist(),
                "class_aware_sampler": (
                    class_aware_sampler.metadata()
                    if resampling_active
                    else None
                ),
                "subtype_loss": (
                    "weighted_cross_entropy"
                    if use_subtype_loss
                    else None
                ),
                "subtype_class_weights": (
                    subtype_criterion.weight.detach().cpu().tolist()
                    if use_subtype_loss
                    else None
                ),
                "trainable_model_parts": configured_trainable_parts,
                "initialization": initialization_metadata,
                "train_config_path": str(config.__config_path__),
                "train_config": train_config_snapshot,
                "dataset_root": str(dataset_root),
                **get_classification_loss_metadata(
                    epoch_criterion,
                    class_counts,
                ),
            }, best_model_path)
            # 即使在 MIN_EPOCHS 内，也保存更好的模型
        else:
            if (
                best_candidate is not None
                and not best_candidate.constraints_met
            ):
                # 在尚未出现满足全部约束的候选前，不因 fallback 指标停训。
                patience_counter = 0
                continue
            # 只有当超过最小训练轮数后，才开始消耗耐心
            if epoch > MIN_EPOCHS:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(
                        f"\n[Early Stopping] Fold {current_fold} at epoch {epoch}. "
                        f"Best {best_candidate.primary_metric_name}: "
                        f"{best_candidate.primary_metric_value:.4f} "
                        f"(Ep {best_epoch})"
                    )
                    break
            else:
                # 保护期内，重置耐心，确保出了保护期是满血状态
                patience_counter = 0
    
    # 强制在结束时换行
    print(f"\n[Finished] Fold {current_fold} done. Model saved to: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a train_config.py file",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Experiment root containing DATA_ROOT/datasets, or datasets itself",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Training output root; weights are written to OUTPUT_ROOT/checkpoints",
    )
    parser.add_argument(
        "--init-checkpoint-root",
        default=None,
        help=(
            "Optional baseline experiment root (or its checkpoints directory). "
            "The matching sequence/fold checkpoint initializes shared weights."
        ),
    )
    # [修改] required=False 表示如果命令行不打 --seq 就是多通道
    parser.add_argument(
        "--seq",
        type=int,
        choices=range(1, len(ALL_SEQUENCES) + 1),
        required=False,
        default=None,
        help="Sequence ID (1-3). Leave empty for ALL channels.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--fold", type=int, required=True, choices=range(1, K_FOLDS + 1), help=f"Fold ID (1-{K_FOLDS})")
    args = parser.parse_args()
    
    main(args)
