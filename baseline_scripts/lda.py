import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# 将项目根目录加入环境变量，以便导入 utils 和 configs
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.train_and_test import load_pt_dataset
from configs.global_config import DATASET_ROOT, K_FOLDS, ALL_SEQUENCES, CLASS_NAMES

def extract_features_and_labels(dataloader, desc="Extracting"):
    features = []
    labels = []
    
    print(f"[{desc}] Converting 3D volumes to flat vectors...")
    # 使用 tqdm 包装 dataloader 显示进度条
    for x, y, mask, mask_flag, case_id in tqdm(dataloader, desc=desc):
        # x shape: [B, C, D, H, W]
        # 3D 平均池化，大幅降低初始维度，防止普通内存溢出
        x_down = F.avg_pool3d(x, kernel_size=4, stride=4) 
        
        # 展平并转为 numpy
        x_flat = x_down.view(x_down.size(0), -1).numpy()
        
        features.append(x_flat)
        labels.append(y.numpy())
        
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def run_lda_fold(seq_id, fold, batch_size=8):
    seq_name = ALL_SEQUENCES[seq_id - 1]
    
    # 独立的缓存文件夹路径：放在项目根目录下的 baseline_cache
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
        # 增加一个小容错：如果找不到全名，尝试只用 seq{seq_id} 找
        if not dataset_dir.exists():
            dataset_dir = DATASET_ROOT / f"seq{seq_id}" / f"fold{fold}"
            
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

        train_dataset = load_pt_dataset(dataset_dir / "train.pt")
        test_dataset = load_pt_dataset(dataset_dir / "test.pt")
        
        # 加入 num_workers=4 加速多线程读取
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        X_train, y_train = extract_features_and_labels(train_loader, f"Train Fold {fold}")
        X_test, y_test = extract_features_and_labels(test_loader, f"Test Fold {fold}")
        
        # 保存到硬盘以便下次复用 (SVM等脚本也可以直接读这个)
        print(f"--> Saving extracted features to {cache_file}...")
        np.savez_compressed(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("Training LDA model...")
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    lda.fit(X_train_pca.astype(np.float64), y_train)

    y_pred = lda.predict(X_test_pca)
    
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
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main(args):
    seq_name = ALL_SEQUENCES[args.seq - 1]
    print(f"\n>>> Starting LDA K-Fold Evaluation for: {seq_name} <<<")
    
    if args.fold is not None:
        folds_to_run = [args.fold]
        print(f"Mode: Single Fold Evaluation (Fold {args.fold})")
    else:
        folds_to_run = list(range(1, K_FOLDS + 1))
        print(f"Mode: All {K_FOLDS} Folds Average")
    
    metrics_history = []
    
    for fold in folds_to_run:
        res = run_lda_fold(args.seq, fold)
        metrics_history.append(res)
        
    if len(metrics_history) > 1:
        print("\n" + "="*50)
        print(f"   K-FOLDS AVERAGE REPORT ({len(metrics_history)} folds)   ")
        print("="*50)

        avg_acc = np.mean([r['acc'] for r in metrics_history])
        std_acc = np.std([r['acc'] for r in metrics_history])
        
        avg_f1 = np.mean([r['f1'] for r in metrics_history])
        std_f1 = np.std([r['f1'] for r in metrics_history])
        
        avg_prec = np.mean([r['precision'] for r in metrics_history])
        std_prec = np.std([r['precision'] for r in metrics_history])
        
        avg_rec = np.mean([r['recall'] for r in metrics_history])
        std_rec = np.std([r['recall'] for r in metrics_history])

        print(f"Sequence      : {seq_name}")
        print(f"Model         : LDA")
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
    parser.add_argument("--fold", type=int, default=None, help="Fold index (1-5). Leave empty to run all folds.")
    args = parser.parse_args()
    
    main(args)