import argparse
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from configs.train_config import *
from configs.global_config import *

from models.cnn3d import Simple3DCNN
from models.ResNet import ResNet10, ResNet18
from utils.train_and_test import set_seed, load_pt_dataset

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_pt(ckpt_path: Path, data_path: Path, model_name: str):
    """
    直接评估指定的 .pth 模型文件和 .pt 数据集文件
    """
    if not data_path.exists():
        print(f"[Error] Dataset not found at {data_path}")
        return
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at {ckpt_path}")
        return

    # ---------- 1. 加载数据并自动推断通道数 ----------
    print(f"Loading dataset from {data_path} ...")
    test_set = load_pt_dataset(data_path)
    
    # 获取第一个样本的输入维度 (C, D, H, W) 来自动确定 in_channels
    sample_x, _, _ = test_set[0]
    in_channels = sample_x.shape[0]

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"\n{'='*20} Evaluating {'='*20}")
    print(f"Model File  : {ckpt_path.name}")
    print(f"Data File   : {data_path.name}")
    print(f"Model Arch  : {model_name} (in_channels={in_channels})")

    # ---------- 2. 初始化并加载模型 ----------
    if model_name == "cnn3d":
        ModelClass = Simple3DCNN
    elif model_name == "ResNet":
        ModelClass = ResNet10
    elif model_name == "ResNet18":
        ModelClass = ResNet18
    else:
        raise ValueError(f"Unknown model: {model_name}")

    try:
        model = ModelClass(num_classes=NUM_CLASSES, in_channels=in_channels)
    except TypeError:
        print(f"[Warning] {model_name} does not accept 'in_channels'. Using default.")
        model = ModelClass(num_classes=NUM_CLASSES)
        
    model = model.to(DEVICE)
    
    # 兼容 DataParallel 保存的权重加载
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint  # 兼容纯 state_dict 保存方式

    model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation!")
        model = nn.DataParallel(model)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    # ---------- 3. 测试推理循环 ----------
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    # ---------- 4. 计算指标 ----------
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # ---------- 5. 打印结果 ----------
    print("\n===== Test Results =====")
    print(f"Test samples  : {len(test_set)}")
    print(f"Test loss     : {avg_loss:.4f}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    print("========================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a specific .pth model on a specific .pt dataset.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Absolute or relative path to the .pth model file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Absolute or relative path to the .pt dataset file (e.g., easy_inflammation.pt).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn3d", "ResNet", "ResNet18"],
        help="Which model architecture to use (must match the checkpoint).",
    )
    
    args = parser.parse_args()
    set_seed(SEED)

    evaluate_pt(
        ckpt_path=Path(args.ckpt_path),
        data_path=Path(args.data_path),
        model_name=args.model
    )