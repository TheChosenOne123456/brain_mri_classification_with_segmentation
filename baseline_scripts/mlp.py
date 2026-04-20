import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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

# === 1. 定义 PyTorch MLP 模型 ===
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# === 2. 特征加载机制 (复用缓存) ===
def extract_features_and_labels(dataloader, desc="Extracting"):
    features = []
    labels = []
    print(f"[{desc}] Converting 3D volumes to flat vectors...")
    for x, y, mask, mask_flag, case_id in tqdm(dataloader, desc=desc):
        x_down = F.avg_pool3d(x, kernel_size=4, stride=4) 
        x_flat = x_down.view(x_down.size(0), -1).numpy()
        features.append(x_flat)
        labels.append(y.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def run_mlp_fold(seq_id, fold, epochs=50, lr=1e-4, batch_size=32):
    seq_name = ALL_SEQUENCES[seq_id - 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 加载特征 ---
    cache_dir = ROOT_DIR / "baseline_cache" / f"seq{seq_id}_{seq_name}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"fold{fold}_features.npz"
    
    if cache_file.exists():
        print(f"\n--> [Cache HIT] Loading pre-extracted features from {cache_file}...")
        data = np.load(cache_file)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
    else:
        print(f"\n--> [Cache MISS] Extracting features from dataset... This might take a while.")
        dataset_dir = DATASET_ROOT / f"seq{seq_id}_{seq_name}" / f"fold{fold}"
        if not dataset_dir.exists():
            dataset_dir = DATASET_ROOT / f"seq{seq_id}" / f"fold{fold}"
            
        train_dataset = load_pt_dataset(dataset_dir / "train.pt")
        test_dataset = load_pt_dataset(dataset_dir / "test.pt")
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
        
        X_train, y_train = extract_features_and_labels(train_loader, f"Train Fold {fold}")
        X_test, y_test = extract_features_and_labels(test_loader, f"Test Fold {fold}")
        
        np.savez_compressed(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # --- 准备 PyTorch DataLoader (针对 1D 特征) ---
    train_tensors = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_tensors = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_dl = DataLoader(train_tensors, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_tensors, batch_size=batch_size, shuffle=False)
    
    # --- 初始化模型与训练组件 ---
    input_dim = X_train.shape[1]
    model = SimpleMLP(input_dim, num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # --- 设置权重保存路径 ---
    checkpoint_dir = ROOT_DIR / "BLcheckpoints" / f"seq{seq_id}_MLP"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / f"fold{fold}_model_best.pth"
    
    # --- 训练或跳过判断 ---
    if best_model_path.exists():
        print(f"\n[Skip] Weight found at {best_model_path}. Skipping training and loading weights directly...")
    else:
        print(f"\nTraining PyTorch MLP on {device}... (Input Dim: {input_dim})")
        best_f1 = 0.0
        
        # --- 训练循环 ---
        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_dl, desc=f"Fold {fold} Ep {epoch}", leave=False)
            for bx, by in pbar:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == by).sum().item()
                train_total += by.size(0)
            
            train_loss /= len(train_dl)
            train_acc = train_correct / train_total

            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for bx, by in test_dl:
                    bx, by = bx.to(device), by.to(device)
                    logits = model(bx)
                    loss = criterion(logits, by)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == by).sum().item()
                    val_total += by.size(0)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(by.cpu().numpy())

            val_loss /= len(test_dl)
            val_acc = val_correct / val_total
            val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            
            # --- 打印格式对齐主项目 ---
            print(f"Epoch [{epoch}/{epochs}] "
                  f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}   "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}")
            
            # 保存最佳权重
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)

    # --- 加载最佳权重进行最终评估 ---
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    final_preds = []
    final_targets = []
    with torch.no_grad():
        for bx, by in test_dl:
            bx = bx.to(device)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            final_preds.extend(preds)
            final_targets.extend(by.numpy())

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
    print(f"\n>>> Starting PyTorch MLP K-Fold Evaluation for: {seq_name} <<<")
    
    folds_to_run = [args.fold] if args.fold is not None else list(range(1, K_FOLDS + 1))
    metrics_history = []
    
    for fold in folds_to_run:
        res = run_mlp_fold(args.seq, fold)
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
        print(f"Model         : Simple MLP (PyTorch)")
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