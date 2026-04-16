'''
K-Fold 训练脚本：指定 --fold 参数 (1~5) 进行训练
模型将保存为 fold{k}_model_best.pth
[适配新服务器：8x RTX 3080]
'''
import argparse
import sys
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
from pathlib import Path

from configs.train_config import *
from configs.global_config import *
from models.cnn3d import Simple3DCNN
from models.ResNet import ResNet10, ResNet18
from models.FoundationModel import FoundationModel
from utils.train_and_test import set_seed, load_pt_dataset

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

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
                # 如果是双头模型我们这里只测分类性能，取第一个返回值
                if hasattr(model, 'module') and isinstance(model.module, FoundationModel) or isinstance(model, FoundationModel):
                     outputs = model(x, return_seg=False)
                else:
                     outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

def main(args):
    set_seed(SEED)
    
    current_fold = args.fold
    model_name = args.model
    
    if args.seq is not None:
        # ---- 单通道模式 ----
        seq_id = args.seq
        seq_idx = seq_id - 1
        seq_name = ALL_SEQUENCES[seq_idx]
        in_channels = 1
        print(f"\n=== Training Fold {current_fold}/{K_FOLDS} | Seq: {seq_name} | Model: {model_name} (1-Channel) ===")

        dataset_dir = DATASET_DIRS[seq_idx] / f"fold{current_fold}"
        ckpt_dir = CKPT_DIRS[seq_idx] / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if not dataset_dir.exists():
            print(f"Error: Dataset for fold {current_fold} not found at {dataset_dir}")
            print("Please run 'python -m scripts.build_dataset_kfold' first.")
            sys.exit(1)

        # 加载数据
        train_set = load_pt_dataset(dataset_dir / "train.pt")
        val_set   = load_pt_dataset(dataset_dir / "val.pt")

    else:
        # ---- 多通道模式 ----
        in_channels = len(ALL_SEQUENCES)
        seq_name = f"Multi-Fusion ({in_channels} Channels)"
        print(f"\n=== Training Fold {current_fold}/{K_FOLDS} | Mode: {seq_name} | Model: {model_name} ===")

        train_sets_list = []
        val_sets_list = []
        
        for idx, s_name in enumerate(ALL_SEQUENCES):
            dataset_dir = DATASET_DIRS[idx] / f"fold{current_fold}"
            if not dataset_dir.exists():
                print(f"Error: Dataset missing for sequence {s_name} at {dataset_dir}")
                sys.exit(1)
            train_sets_list.append(load_pt_dataset(dataset_dir / "train.pt"))
            val_sets_list.append(load_pt_dataset(dataset_dir / "val.pt"))

        # 使用我们自定义的代理类包装
        train_set = MultiChannelDataset(train_sets_list)
        val_set   = MultiChannelDataset(val_sets_list)

        # 把多通道模型统一保存在独立路径中
        base_ckpt_dir = CKPT_DIRS[0].parent  # 例如 checkpoints/
        ckpt_dir = base_ckpt_dir / "multi_channel" / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)


    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 初始化模型
    if model_name == "cnn3d":
        ModelClass = Simple3DCNN
    elif model_name == "ResNet":
        ModelClass = ResNet10
    elif model_name == "ResNet18":
        ModelClass = ResNet18
    elif model_name == "FoundationModel":
        ModelClass = FoundationModel
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # [修改] 尝试动态传入 in_channels，保证兼容性
    try:
        model = ModelClass(num_classes=NUM_CLASSES, in_channels=in_channels)
    except TypeError:
        print(f"[Warning] {model_name} does not accept 'in_channels'. Using default.")
        model = ModelClass(num_classes=NUM_CLASSES)

    model = model.to(DEVICE)

    # 启用多卡 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    all_labels = train_set.labels.tolist() if isinstance(train_set.labels, torch.Tensor) else list(train_set.labels)
    class_counts = torch.bincount(torch.tensor(all_labels), minlength=NUM_CLASSES)
    
    # 权重计算
    total_samples = len(all_labels)
    raw_weights = total_samples / (NUM_CLASSES * class_counts.float() + 1e-6)
    class_weights = torch.pow(raw_weights, 0.5)
    
    # 将权重转到 device
    class_weights = class_weights.to(DEVICE)
    print(f"Class Weights: {class_weights.tolist()}") 

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 在设备上创建一个极其侧重类别 1 和 2 的权重
    # 假设背景权重很小(0.1)，炎症权重10.0，转移瘤权重10.0
    seg_class_weights = torch.tensor(SEG_CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)

    # 应用于分割交叉熵
    seg_criterion = nn.CrossEntropyLoss(weight=seg_class_weights, reduction='none')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 初始化混合精度 Scaler
    scaler = GradScaler(**scaler_args)

    # 训练循环 &早停
    best_val_loss = float("inf")
    best_val_f1 = 0.0  # 初始化为 0，越大越好
    patience_counter = 0
    best_epoch = 0

    # 模型保存文件名区分 fold
    best_model_path = ckpt_dir / f"fold{current_fold}_model_best.pth"

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0 
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Fold {current_fold} Ep {epoch}", leave=False)
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
                if model_name == "FoundationModel":
                    # [修复] 必须严格作为位置参数（不带 "return_seg="）传入，防止 DataParallel 解析乱套！
                    logits, seg_logits = model(x, True)
                    loss_cls = criterion(logits, y)
                    
                    # 取出没有经过reduction的各个像素的损失 (mask去掉第一维单通道的壳子)
                    gt_mask = mask.squeeze(1)
                    unreduced_seg_loss = seg_criterion(seg_logits, gt_mask)
                    
                    # 算出Batch里每张图平均的空间Loss：[B, D, H, W] -> [B]
                    per_sample_ce_loss = unreduced_seg_loss.mean(dim=[1, 2, 3])
                    
                    # [新增] 算出Batch里每张图的 Dice Loss: [B]
                    # 因为我们传入的 compute_dice_loss 会针对每张图返回平均损失，我们可以在里面稍微调整使其按样本返回，
                    # 简便起见，这里假设 compute_dice_loss 返回的是 [B] 形状的张量（你可以修改上方公式不进行 mean(0)）
                    
                    # --- 为配合上方的代码，将上方 compute_dice_loss 的返回值改为不对 Batch 取平均 ---
                    # 即 return (dice_loss / (num_classes - 1))  # 这个就是 [B] 大小的
                    per_sample_dice_loss = compute_dice_loss(seg_logits, gt_mask, num_classes=NUM_CLASSES)
                    
                    # 将两个 Loss 缝合 (通常按照 1:1 的比例即可)
                    per_sample_total_seg_loss = per_sample_ce_loss + per_sample_dice_loss
                    
                    # 使用 Flag 作为开关，把那些没有遮罩标注又不是正常脑炎的“伪标签图像”损失强行清零
                    masked_seg_loss = (per_sample_total_seg_loss * mask_flag).sum()
                    
                    valid_mask_count = mask_flag.sum()
                    if valid_mask_count > 0:
                        loss_seg = masked_seg_loss / valid_mask_count
                    else:
                        loss_seg = 0.0
                        
                    loss = loss_cls + SEG_ALPHA * loss_seg
                else:
                    # 单任务模型
                    logits = model(x)
                    loss = criterion(logits, y)
            
            # AMP 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total

                # --- Val ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_preds = []
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
                    if model_name == "FoundationModel":
                        # [修改] 开启验证集的分割预测 (传入 True)
                        logits, seg_logits = model(x, True)
                    else:
                        logits = model(x)
                        seg_logits = None
                    loss = criterion(logits, y)

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
                all_targets.extend(y.cpu().numpy())

                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        val_dice = np.mean(valid_seg_dices) if len(valid_seg_dices) > 0 else 0.0

        # --- 打印格式 ---
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}   "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | val_dice: {val_dice:.4f}")

        # --- Early Stopping check with MIN_EPOCHS ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            
            # 保存时剥离 DataParallel 包装，否则以后加载会报错
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            
            torch.save({
                "model_state": model_to_save.state_dict(),
                "fold": current_fold,
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1 
            }, best_model_path)
            # 即使在 MIN_EPOCHS 内，也保存更好的模型
        else:
            # 只有当超过最小训练轮数后，才开始消耗耐心
            if epoch > MIN_EPOCHS:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n[Early Stopping] Fold {current_fold} at epoch {epoch}. "
                          f"Best Val F1: {best_val_f1:.4f} (Ep {best_epoch})")
                    break
            else:
                # 保护期内，重置耐心，确保出了保护期是满血状态
                patience_counter = 0
    
    # 强制在结束时换行
    print(f"\n[Finished] Fold {current_fold} done. Model saved to: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [修改] required=False 表示如果命令行不打 --seq 就是多通道
    parser.add_argument("--seq", type=int, required=False, default=None, help="Sequence ID (1-3). Leave empty for ALL channels.")
    parser.add_argument("--model", type=str, required=True, choices=["cnn3d", "ResNet", "ResNet18", "FoundationModel"])
    parser.add_argument("--fold", type=int, required=True, choices=range(1, K_FOLDS + 1), help=f"Fold ID (1-{K_FOLDS})")
    args = parser.parse_args()
    
    main(args)