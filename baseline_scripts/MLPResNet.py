import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import sys
from pathlib import Path
import warnings

# 屏蔽 torch.load 的安全警告
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# 将项目根目录加入环境变量
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.train_and_test import load_pt_dataset
from configs.global_config import DATASET_ROOT, K_FOLDS, ALL_SEQUENCES, CLASS_NAMES
from configs.train_config import NUM_EPOCHS, MIN_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, PATIENCE

# === 1. 定义官方风格的 3D ResNet + MLP 分类头 ===

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet3D_MLP(nn.Module):
    def __init__(self, block, layers, num_classes=3, in_channels=1):
        super().__init__()
        self.in_planes = 64
        # Stem
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # MLP Classification Head
        self.mlp = nn.Sequential(
            nn.Linear(512 * block.expansion, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x

def resnet10_mlp(num_classes=3, in_channels=1):
    """采用更浅的层数以适配医疗图像并防止 OOM"""
    return ResNet3D_MLP(BasicBlock3D, [1, 1, 1, 1], num_classes=num_classes, in_channels=in_channels)

# === 2. 训练与测试逻辑 ===

def run_mlpresnet_fold(seq_id, fold):
    seq_name = ALL_SEQUENCES[seq_id - 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 权重保存目录
    checkpoint_dir = ROOT_DIR / "BLcheckpoints" / f"seq{seq_id}_{seq_name}" / "MLPResNet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / f"fold{fold}_model_best.pth"
    
    # 加载数据集
    dataset_dir = DATASET_ROOT / f"seq{seq_id}_{seq_name}" / f"fold{fold}"
    if not dataset_dir.exists():
        dataset_dir = DATASET_ROOT / f"seq{seq_id}" / f"fold{fold}"
        
    train_dataset = load_pt_dataset(dataset_dir / "train.pt")
    test_dataset = load_pt_dataset(dataset_dir / "test.pt")
    
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = resnet10_mlp(num_classes=len(CLASS_NAMES), in_channels=1).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    if best_model_path.exists():
        print(f"\n[Skip] Weight found at {best_model_path}. Skipping training and loading weights directly...")
    else:
        print(f"\n=== Training Fold {fold}/{K_FOLDS} | Seq: {seq_name} | Model: MLPResNet (1-Channel) ===")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        best_val_loss = float("inf")
        best_val_f1 = 0.0
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_dl, desc=f"Fold {fold} Ep {epoch}", leave=False)
            for x_batch, y_batch, _, _, _ in pbar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)
                
            train_loss /= len(train_dl)
            train_acc = train_correct / train_total

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for x_batch, y_batch, _, _, _ in test_dl:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_batch.cpu().numpy())

            val_loss /= len(test_dl)
            val_acc = val_correct / val_total
            val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            
            # --- 打印格式完全对齐 train_kfold.py ---
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
                  f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}   "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}")
            
            # --- 完整的 Early Stopping 与保护期逻辑 ---
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), best_model_path)
            else:
                if epoch > MIN_EPOCHS:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"\n[Early Stopping] Fold {fold} at epoch {epoch}. "
                              f"Best Val F1: {best_val_f1:.4f} (Ep {best_epoch})")
                        break
                else:
                    patience_counter = 0

        print(f"\n[Finished] Fold {fold} done. Model saved to: {best_model_path}")

    # === 测试环节 ===
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    final_preds, final_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch, _, _, _ in test_dl:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            final_preds.extend(preds)
            final_targets.extend(y_batch.numpy())

    y_test, y_pred = np.array(final_targets), np.array(final_preds)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== Test Results =====")
    print(f"Sequence      : {seq_name} (Fold {fold})")
    print(f"Test samples  : {len(y_test)}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))
    
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}

def main(args):
    seq_name = ALL_SEQUENCES[args.seq - 1]
    print(f"\n>>> Starting PyTorch MLP+ResNet K-Fold Evaluation for: {seq_name} <<<")
    
    folds_to_run = [args.fold] if args.fold is not None else list(range(1, K_FOLDS + 1))
    metrics_history = []
    
    for fold in folds_to_run:
        res = run_mlpresnet_fold(args.seq, fold)
        metrics_history.append(res)
        
    if len(metrics_history) > 1:
        print("\n" + "="*50)
        print(f"   K-FOLDS AVERAGE REPORT ({len(metrics_history)} folds)   ")
        print("="*50)

        avg_acc, std_acc = np.mean([r['acc'] for r in metrics_history]), np.std([r['acc'] for r in metrics_history])
        avg_f1, std_f1 = np.mean([r['f1'] for r in metrics_history]), np.std([r['f1'] for r in metrics_history])
        avg_prec, std_prec = np.mean([r['precision'] for r in metrics_history]), np.std([r['precision'] for r in metrics_history])
        avg_rec, std_rec = np.mean([r['recall'] for r in metrics_history]), np.std([r['recall'] for r in metrics_history])

        print(f"Sequence      : {seq_name}")
        print(f"Model         : MLP+ResNet3D")
        print("-" * 40)
        print(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
        print("-" * 40)
        print(f"{'Accuracy':<15} | {avg_acc:.4f}     | ±{std_acc:.4f}")
        print(f"{'Precision':<15} | {avg_prec:.4f}     | ±{std_prec:.4f}")
        print(f"{'Recall':<15} | {avg_rec:.4f}     | ±{std_rec:.4f}")
        print(f"{'F1-Score':<15} | {avg_f1:.4f}     | ±{std_f1:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=1, help="Sequence ID (1-3)")
    parser.add_argument("--fold", type=int, default=None, help="Fold index (1-5)")
    args = parser.parse_args()
    main(args)