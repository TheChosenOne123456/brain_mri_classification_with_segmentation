'''
K-Fold 软投票评估脚本 (Late Fusion / Soft Voting)：
- 加载已经训练好的单通道模型 (T1, T2, FLAIR)。
- 对同一个样本，分别获取三个模型的预测概率，取平均值后进行最终决策。
- 如果指定 --fold N，则只评估第 N 折。
- 如果不指定 --fold，则自动评估所有 fold 并计算平均指标。
'''

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from configs.config_utils import (
    TRAIN_CONFIG_FIELDS,
    infer_data_dir,
    load_python_config,
    resolve_input_artifact_dir,
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
    model_capabilities,
)
from utils.train_and_test import set_seed, load_pt_dataset
from utils.fusion_diagnostics import (
    save_probability_diagnostics,
    write_probability_diagnostics_readme,
)

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


# ================== [数据集专区] ==================
class MultiSequenceDataset(Dataset):
    """
    为晚期融合设计：返回三个独立的 X (各自为单通道)，而不是拼接好的。
    """
    def __init__(self, datasets_list):
        self.datasets = datasets_list
        self.labels = datasets_list[0].labels

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        xs = []
        for ds in self.datasets:
            # 必须修改解包，加上 mask_tensor 和 has_mask_flag
            x, y, mask_tensor, has_mask_flag, case_id = ds[idx]  
            xs.append(x) # 保持独立的 [1, D, H, W]
        
        # 同样这里也要适配解包
        _, y, _, _, case_id = self.datasets[0][idx]
        return xs, y, case_id
# ================================================


def resolve_fusion_models(model_set, explicit_model_names):
    """解析三个序列实际使用的模型，并标记旧层级融合模式。"""
    if explicit_model_names is not None:
        return (
            tuple(explicit_model_names),
            "Custom Heterogeneous Soft Voting",
            False,
        )
    if model_set == "hierarchical":
        return (
            ("FoundationModelHierarchical",) * len(ALL_SEQUENCES),
            "Hierarchical Late Fusion",
            True,
        )
    return (
        (
            "FoundationModel_ori",
            "FoundationModel_ori",
            "FoundationModel",
        ),
        "Baseline Heterogeneous Soft Voting",
        False,
    )


def resolve_checkpoint_roots(shared_root, sequence_roots):
    """支持一个共享实验根，或按 seq1/seq2/seq3 分别指定实验根。"""
    if shared_root is not None and sequence_roots is not None:
        raise ValueError(
            "Specify either --checkpoint-root or --checkpoint-roots, not both"
        )
    if sequence_roots is not None:
        if len(sequence_roots) != len(ALL_SEQUENCES):
            raise ValueError(
                f"--checkpoint-roots requires {len(ALL_SEQUENCES)} paths"
            )
        return tuple(
            resolve_input_artifact_dir(root, "checkpoints")
            for root in sequence_roots
        )
    if shared_root is None:
        raise ValueError(
            "One of --checkpoint-root or --checkpoint-roots is required"
        )
    checkpoint_root = resolve_input_artifact_dir(shared_root, "checkpoints")
    return (checkpoint_root,) * len(ALL_SEQUENCES)


def combine_hierarchical_probabilities(
    classification_probabilities,
    subtype_probabilities,
):
    """把 normal/abnormal 主概率与异常条件概率组成三分类联合概率。"""
    abnormal_probability = classification_probabilities[:, 1:].sum(
        dim=1,
        keepdim=True,
    )
    return torch.cat(
        (
            classification_probabilities[:, :1],
            abnormal_probability * subtype_probabilities,
        ),
        dim=1,
    )


def evaluate_vote_single_fold(
    fold_idx,
    dataset_dirs,
    ckpt_dirs,
    processed_data_root,
    batch_size,
    num_workers,
    device,
    model_names,
    mode_name,
    use_legacy_hierarchical_fusion,
    report_root=None,
):
    """
    使用三个模型进行软投票的评估函数
    """
    print(f"\n{'='*20} Evaluating Fold {fold_idx} {'='*20}")
    print(f"Mode: {mode_name}")

    # ---------- 1. 加载数据 ----------
    test_sets_list = []
    print("  -> Loading test sets for all 3 sequences... ", end="", flush=True)
    t0 = time.time()
    for idx, s_name in enumerate(ALL_SEQUENCES):
        d_dir = dataset_dirs[idx] / f"fold{fold_idx}"
        if not d_dir.exists():
            print(f"\n[Warning] Dataset missing for {s_name} at {d_dir}. Skipping fold {fold_idx}.")
            return None
        test_sets_list.append(
            load_pt_dataset(
                d_dir / "test.pt",
                data_root=processed_data_root,
            )
        )
    print(f"Done in {time.time()-t0:.1f}s")
    
    test_set = MultiSequenceDataset(test_sets_list)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # ---------- 2. 初始化并加载 3 个模型 ----------
    models = []
    for seq_idx, s_name in enumerate(ALL_SEQUENCES):
        target_model_name = model_names[seq_idx]

        ckpt_dir = ckpt_dirs[seq_idx] / target_model_name
        ckpt_path = ckpt_dir / f"fold{fold_idx}_model_best.pth"
        
        if not ckpt_path.exists():
            print(f"\n[Warning] Model checkpoint missing for {s_name} at {ckpt_path}. Skipping fold {fold_idx}.")
            return None
        
        model = create_model(
            target_model_name,
            num_classes=NUM_CLASSES,
            in_channels=1,
            sequence_id=seq_idx + 1,
        )
        model = model.to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        model.eval()
        models.append(model)
        
    print(f"  -> Successfully loaded: {', '.join(model_names)}")

    # ---------- 3. 测试 (软投票机制) ----------
    all_preds = []
    all_main_preds = []
    all_labels = []
    all_case_ids = []
    all_base_probabilities = []
    all_fusion_probabilities = []
    misclassified_cases = []

    # 注意：投票模式下不计算 Loss，因为直接比较概率
    with torch.no_grad():
        for xs, y, case_ids in test_loader:
            y = y.to(device)
            
            # xs 包含 3 个 batch tensor (T1, T2, FLAIR)
            probs = []
            main_probs = []
            subtype_probs = []
            for i, x in enumerate(xs):
                x = x.to(device)
                capabilities = model_capabilities(models[i])
                outputs = forward_model(
                    models[i],
                    x,
                    return_subtype=capabilities["subtype"],
                )
                logits = outputs["classification"]
                
                # 将 logits 转换为概率分布
                main_prob = F.softmax(logits, dim=1)
                main_probs.append(main_prob)
                if capabilities["subtype"]:
                    subtype_prob = F.softmax(outputs["subtype"], dim=1)
                    subtype_probs.append(subtype_prob)
                    probs.append(
                        combine_hierarchical_probabilities(
                            main_prob,
                            subtype_prob,
                        )
                    )
                else:
                    probs.append(main_prob)
            
            # --- 核心：平均概率 (Soft Voting) ---
            # 也可以在这里改成加权平均，例如:
            # avg_prob = 0.5 * probs[0] + 0.25 * probs[1] + 0.25 * probs[2]
            avg_prob = (probs[0] + probs[1] + probs[2]) / 3.0
            
            main_preds = avg_prob.argmax(dim=1)
            if use_legacy_hierarchical_fusion:
                avg_main_prob = torch.stack(main_probs, dim=0).mean(dim=0)
                avg_subtype_prob = torch.stack(subtype_probs, dim=0).mean(dim=0)
                avg_prob = combine_hierarchical_probabilities(
                    avg_main_prob,
                    avg_subtype_prob,
                )
                main_preds = avg_main_prob.argmax(dim=1)
                preds = main_preds.clone()
                abnormal = preds != 0
                subtype_preds = avg_subtype_prob.argmax(dim=1) + 1
                preds[abnormal] = subtype_preds[abnormal]
            else:
                preds = main_preds

            preds_cpu = preds.cpu().numpy()
            main_preds_cpu = main_preds.cpu().numpy()
            labels_cpu = y.cpu().numpy()

            all_preds.extend(preds_cpu)
            all_main_preds.extend(main_preds_cpu)
            all_labels.extend(labels_cpu)
            all_case_ids.extend(str(case_id) for case_id in case_ids)
            if report_root is not None:
                all_base_probabilities.append(
                    torch.stack(probs, dim=1).cpu().numpy()
                )
                all_fusion_probabilities.append(avg_prob.cpu().numpy())

            # 收集误判 case
            for cid, p, gt in zip(case_ids, preds_cpu, labels_cpu):
                if p != gt:
                    misclassified_cases.append({
                        "case_id": cid,
                        "gt": int(gt),
                        "pred": int(p),
                    })

    # ---------- 4. 计算指标 ----------
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    if use_legacy_hierarchical_fusion:
        main_acc = accuracy_score(all_labels, all_main_preds)
        main_f1 = f1_score(
            all_labels,
            all_main_preds,
            average="macro",
            zero_division=0,
        )

    # ---------- 5. 打印结果 ----------
    print("\n===== Test Results =====")
    print(f"Sequence      : ALL ({mode_name}) (Fold {fold_idx})")
    print(f"Test samples  : {len(test_set)}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")
    if use_legacy_hierarchical_fusion:
        print(f"Main-head Acc : {main_acc:.4f}")
        print(f"Main-head F1  : {main_f1:.4f}")

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

    # ---------- 6. 打印误判 case ----------
    print("\n===== Misclassified Cases =====")
    print(f"Total misclassified: {len(misclassified_cases)}")

    if len(misclassified_cases) > 0:
        for item in misclassified_cases:
            print(
                f"CaseID: {item['case_id']} | "
                f"GT: {item['gt']} | Pred: {item['pred']}"
            )
    else:
        print("None")

    diagnostics_data = None
    if report_root is not None:
        diagnostics_data = {
            "case_ids": np.asarray(all_case_ids, dtype=str),
            "labels": np.asarray(all_labels, dtype=np.int64),
            "base_probabilities": np.concatenate(
                all_base_probabilities,
                axis=0,
            ),
            "fusion_probabilities": np.concatenate(
                all_fusion_probabilities,
                axis=0,
            ),
            "model_names": tuple(model_names),
        }
        saved = save_probability_diagnostics(
            report_root=report_root,
            fold_idx=fold_idx,
            **diagnostics_data,
        )
        print(f"\nProbability table : {saved['predictions_path']}")
        print(f"Calibration table : {saved['calibration_path']}")
        print(f"Diagnostic summary: {saved['summary_path']}")

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "main_acc": main_acc if use_legacy_hierarchical_fusion else acc,
        "main_f1": main_f1 if use_legacy_hierarchical_fusion else f1,
        "diagnostics_data": diagnostics_data,
    }


# ================== 主流程 ==================
def main(args):
    config = load_python_config(args.config, TRAIN_CONFIG_FIELDS)
    dataset_root = resolve_input_artifact_dir(args.data_root, "datasets")
    checkpoint_roots = resolve_checkpoint_roots(
        args.checkpoint_root,
        args.checkpoint_roots,
    )
    (
        model_names,
        mode_name,
        use_legacy_hierarchical_fusion,
    ) = resolve_fusion_models(args.model_set, args.model_names)
    processed_data_root = infer_data_dir(args.data_root)
    report_root = (
        Path(args.report_root).expanduser().resolve()
        if args.report_root is not None
        else None
    )
    if report_root is not None and use_legacy_hierarchical_fusion:
        raise ValueError(
            "Probability diagnostics do not support the legacy hard-gated "
            "hierarchical fusion mode; use explicit --model-names to evaluate "
            "joint three-class probabilities"
        )
    dataset_dirs = [
        dataset_root / f"seq{seq_id}_{seq_name}"
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
    ]
    ckpt_dirs = [
        checkpoint_roots[seq_id - 1] / f"seq{seq_id}_{seq_name}"
        for seq_id, seq_name in enumerate(ALL_SEQUENCES, start=1)
    ]

    set_seed(SEED)

    print(f"\n>>> Starting K-Fold Evaluation for: {mode_name} <<<")
    print(f"Evaluation config: {config.__config_path__}")
    print(f"Dataset input   : {dataset_root}")
    for seq_idx, (seq_name, model_name, checkpoint_root) in enumerate(
        zip(ALL_SEQUENCES, model_names, checkpoint_roots),
        start=1,
    ):
        print(
            f"Seq{seq_idx} {seq_name:<5}: {model_name} | "
            f"{checkpoint_root}"
        )
    if report_root is not None:
        readme_path = write_probability_diagnostics_readme(report_root)
        print(f"Report output   : {report_root}")
        print(f"Report guide    : {readme_path}")

    if args.fold is not None:
        folds_to_run = [args.fold]
        print(f"Mode: Single Fold Evaluation (Fold {args.fold})")
    else:
        folds_to_run = range(1, K_FOLDS + 1)
        print(f"Mode: All {K_FOLDS} Folds Average")

    metrics_history = []

    for k in folds_to_run:
        res = evaluate_vote_single_fold(
            k,
            dataset_dirs=dataset_dirs,
            ckpt_dirs=ckpt_dirs,
            processed_data_root=processed_data_root,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            device=config.DEVICE,
            model_names=model_names,
            mode_name=mode_name,
            use_legacy_hierarchical_fusion=(
                use_legacy_hierarchical_fusion
            ),
            report_root=report_root,
        )
        if res:
            metrics_history.append(res)
    
    # ---------- 打印综合平均报告 ----------
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

        if use_legacy_hierarchical_fusion:
            print("Method        : Hierarchical Late Fusion")
        else:
            print(f"Method        : {mode_name}")
        print(f"Models        : {', '.join(model_names)}")
        print("-" * 40)
        print(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
        print("-" * 40)
        print(f"{'Accuracy':<15} | {avg_acc:.4f}     | ±{std_acc:.4f}")
        print(f"{'Precision':<15} | {avg_prec:.4f}     | ±{std_prec:.4f}")
        print(f"{'Recall':<15} | {avg_rec:.4f}     | ±{std_rec:.4f}")
        print(f"{'F1-Score':<15} | {avg_f1:.4f}     | ±{std_f1:.4f}")
        if use_legacy_hierarchical_fusion:
            avg_main_acc = np.mean([r["main_acc"] for r in metrics_history])
            avg_main_f1 = np.mean([r["main_f1"] for r in metrics_history])
            print(f"{'Main-head Acc':<15} | {avg_main_acc:.4f}     |")
            print(f"{'Main-head F1':<15} | {avg_main_f1:.4f}     |")
        print("-" * 40)
    elif len(metrics_history) == 0:
        print("\n[Error] No folds were successfully evaluated.")

    if report_root is not None and metrics_history:
        diagnostic_results = [
            result["diagnostics_data"]
            for result in metrics_history
            if result["diagnostics_data"] is not None
        ]
        pooled = save_probability_diagnostics(
            report_root=report_root,
            case_ids=np.concatenate(
                [result["case_ids"] for result in diagnostic_results]
            ),
            labels=np.concatenate(
                [result["labels"] for result in diagnostic_results]
            ),
            base_probabilities=np.concatenate(
                [result["base_probabilities"] for result in diagnostic_results],
                axis=0,
            ),
            fusion_probabilities=np.concatenate(
                [result["fusion_probabilities"] for result in diagnostic_results],
                axis=0,
            ),
            model_names=diagnostic_results[0]["model_names"],
        )
        print(f"\nPooled probability table : {pooled['predictions_path']}")
        print(f"Pooled calibration table : {pooled['calibration_path']}")
        print(f"Pooled diagnostic summary: {pooled['summary_path']}")


# ================== CLI ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the train_config.py used for runtime evaluation settings",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Experiment root containing datasets, or the datasets directory itself",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Training output root containing checkpoints, or checkpoints itself",
    )
    parser.add_argument(
        "--checkpoint-roots",
        nargs=len(ALL_SEQUENCES),
        metavar=("SEQ1_ROOT", "SEQ2_ROOT", "SEQ3_ROOT"),
        default=None,
        help=(
            "Per-sequence training roots (or checkpoints directories). Mutually "
            "exclusive with --checkpoint-root."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, K_FOLDS + 1),
        help=f"Specific fold to evaluate (1~{K_FOLDS}). If not set, run all folds.",
    )
    parser.add_argument(
        "--model-set",
        choices=("baseline", "hierarchical"),
        default="baseline",
        help=(
            "baseline keeps the existing heterogeneous ensemble; hierarchical "
            "uses FoundationModelHierarchical for all three sequences"
        ),
    )
    parser.add_argument(
        "--model-names",
        nargs=len(ALL_SEQUENCES),
        choices=MODEL_CHOICES,
        metavar=("SEQ1_MODEL", "SEQ2_MODEL", "SEQ3_MODEL"),
        default=None,
        help=(
            "Explicit model class for each sequence. When supplied, this overrides "
            "the model names implied by --model-set and enables mixed model sets."
        ),
    )
    parser.add_argument(
        "--report-root",
        default=None,
        help=(
            "Optional directory for per-case probability, calibration, and soft-vote "
            "diagnostic tables."
        ),
    )
    args = parser.parse_args()

    main(args)
