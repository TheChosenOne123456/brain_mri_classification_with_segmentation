"""在各 fold validation 上诊断 FLAIRUNet3D 二值分割。

该脚本只读取 val.pt，不读取 test.pt。一次前向同时完成多阈值逐病例统计，
并在每折 validation 上独立选择“不增加正常病例平均 FP 体积”的 Dice 最优阈值。
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/brain_mri_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.config_utils import (
    infer_data_dir,
    load_python_config,
    resolve_input_artifact_dir,
)
from configs.global_config import CLASS_NAMES, K_FOLDS, NUM_CLASSES, SEED
from models.FLAIRUNet3D import load_flair_segmentation_state
from models.model_factory import create_model
from train_flair_unet import segmentation_supervision_indices
from utils.segmentation_inference import segmentation_logits_from_config
from utils.train_and_test import load_pt_dataset, set_seed


MODEL_NAME = "FLAIRUNet3D"
SEQUENCE_ID = 3
SEQUENCE_NAME = "FLAIR"
VOXEL_VOLUME_ML = 3.0 * 0.75 * 0.75 / 1000.0
DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.10, 0.901, 0.05), 2))


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        dataset_index = self.indices[item]
        return (*self.dataset[dataset_index], dataset_index)


def amp_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def resolve_report_root(checkpoint_root, report_root):
    if report_root is not None:
        return Path(report_root).expanduser().resolve()
    root = Path(checkpoint_root).expanduser().resolve()
    experiment_root = root.parent if root.name == "checkpoints" else root
    return experiment_root / "reports" / "validation_segmentation_analysis"


def checkpoint_path(checkpoint_dir, fold, model_name):
    return (
        checkpoint_dir
        / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}"
        / model_name
        / f"fold{fold}_model_best.pth"
    )


def load_model(checkpoint_dir, fold, device, model_name):
    path = checkpoint_path(checkpoint_dir, fold, model_name)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = create_model(
        model_name,
        num_classes=NUM_CLASSES,
        in_channels=1,
        sequence_id=SEQUENCE_ID,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    load_flair_segmentation_state(model, checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model, checkpoint, path


def load_validation_dataset(dataset_dir, data_root, fold):
    fold_root = (
        dataset_dir
        / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}"
        / f"fold{fold}"
    )
    dataset = load_pt_dataset(fold_root / "val.pt", data_root=data_root)
    positive_indices, normal_indices = segmentation_supervision_indices(dataset)
    return dataset, positive_indices, normal_indices


def safe_ratio(numerator, denominator, empty_value=0.0):
    return numerator / denominator if denominator > 0 else empty_value


def infer_fold(
    *,
    fold,
    dataset_dir,
    processed_data_root,
    checkpoint_dir,
    thresholds,
    batch_size,
    num_workers,
    device,
    config,
    model_name,
):
    dataset, positive_indices, normal_indices = load_validation_dataset(
        dataset_dir,
        processed_data_root,
        fold,
    )
    analysis_dataset = IndexedSubset(
        dataset,
        positive_indices + normal_indices,
    )
    loader = DataLoader(
        analysis_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model, checkpoint, path = load_model(
        checkpoint_dir,
        fold,
        device,
        model_name,
    )
    print(
        f"Fold {fold}: positive={len(positive_indices)}, "
        f"normal={len(normal_indices)}, checkpoint_epoch={checkpoint.get('epoch')}"
    )

    threshold_tensor = torch.as_tensor(
        thresholds,
        device=device,
        dtype=torch.float32,
    ).reshape(1, -1, 1, 1, 1)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validation Fold {fold}", leave=False):
            x, y, mask, mask_flag, case_ids, dataset_indices = batch
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            mask_flag = mask_flag.to(device, non_blocking=True)
            if not torch.all(mask_flag > 0.5):
                raise ValueError("Analysis subset unexpectedly contains invalid masks")

            with amp_context(device):
                probabilities = torch.sigmoid(
                    segmentation_logits_from_config(model, x, config).float()
                ).squeeze(1)

            target = (mask.squeeze(1) if mask.dim() == 5 else mask) > 0
            predictions = probabilities.unsqueeze(1) >= threshold_tensor
            expanded_target = target.unsqueeze(1)
            reduction_dims = (2, 3, 4)
            true_positive = (predictions & expanded_target).sum(reduction_dims)
            predicted_voxels = predictions.sum(reduction_dims)
            target_voxels = target.sum(dim=(1, 2, 3))
            false_positive = predicted_voxels - true_positive
            false_negative = target_voxels.unsqueeze(1) - true_positive
            total_voxels = int(np.prod(target.shape[1:]))

            arrays = {
                "tp": true_positive.cpu().numpy(),
                "fp": false_positive.cpu().numpy(),
                "fn": false_negative.cpu().numpy(),
                "pred": predicted_voxels.cpu().numpy(),
                "target": target_voxels.cpu().numpy(),
            }
            labels = y.numpy()
            indices = dataset_indices.numpy()
            for sample_index, (case_id, label, dataset_index) in enumerate(
                zip(case_ids, labels, indices)
            ):
                target_count = int(arrays["target"][sample_index])
                case_type = "positive" if int(label) > 0 else "normal"
                for threshold_index, threshold in enumerate(thresholds):
                    tp = int(arrays["tp"][sample_index, threshold_index])
                    fp = int(arrays["fp"][sample_index, threshold_index])
                    fn = int(arrays["fn"][sample_index, threshold_index])
                    pred_count = int(
                        arrays["pred"][sample_index, threshold_index]
                    )
                    dice_denominator = 2 * tp + fp + fn
                    rows.append(
                        {
                            "fold": fold,
                            "case_id": str(case_id),
                            "dataset_index": int(dataset_index),
                            "label": int(label),
                            "class_name": CLASS_NAMES[int(label)],
                            "case_type": case_type,
                            "threshold": float(threshold),
                            "dice": safe_ratio(
                                2 * tp,
                                dice_denominator,
                                empty_value=1.0,
                            ),
                            "precision": safe_ratio(tp, pred_count),
                            "recall": safe_ratio(tp, target_count),
                            "detected": int(tp > 0),
                            "true_positive_voxels": tp,
                            "false_positive_voxels": fp,
                            "false_negative_voxels": fn,
                            "predicted_voxels": pred_count,
                            "target_voxels": target_count,
                            "predicted_volume_ml": pred_count * VOXEL_VOLUME_ML,
                            "target_volume_ml": target_count * VOXEL_VOLUME_ML,
                            "false_positive_fraction": fp / total_voxels,
                            "false_positive_volume_ml": fp * VOXEL_VOLUME_ML,
                        }
                    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, {
        "fold": fold,
        "checkpoint_path": str(path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "positive_cases": len(positive_indices),
        "normal_cases": len(normal_indices),
    }


def percentile(series, value):
    return float(np.percentile(series.to_numpy(dtype=float), value))


def summarize_thresholds(case_metrics):
    rows = []
    scopes = [(f"fold{fold}", frame) for fold, frame in case_metrics.groupby("fold")]
    scopes.append(("pooled", case_metrics))
    for scope, frame in scopes:
        for threshold, threshold_frame in frame.groupby("threshold"):
            positive = threshold_frame[threshold_frame["case_type"] == "positive"]
            normal = threshold_frame[threshold_frame["case_type"] == "normal"]
            foreground_defined = threshold_frame[
                (threshold_frame["target_voxels"] + threshold_frame["predicted_voxels"])
                > 0
            ]
            pooled_dice_denominator = (
                2 * positive["true_positive_voxels"].sum()
                + positive["false_positive_voxels"].sum()
                + positive["false_negative_voxels"].sum()
            )
            rows.append(
                {
                    "scope": scope,
                    "threshold": float(threshold),
                    "positive_cases": len(positive),
                    "nnunet_foreground_mean_dice": foreground_defined[
                        "dice"
                    ].mean(),
                    "positive_mean_dice": positive["dice"].mean(),
                    "positive_median_dice": positive["dice"].median(),
                    "positive_dice_q25": percentile(positive["dice"], 25),
                    "positive_dice_q75": percentile(positive["dice"], 75),
                    "positive_pooled_voxel_dice": safe_ratio(
                        2 * positive["true_positive_voxels"].sum(),
                        pooled_dice_denominator,
                    ),
                    "positive_mean_precision": positive["precision"].mean(),
                    "positive_mean_recall": positive["recall"].mean(),
                    "positive_complete_miss_rate": 1.0 - positive["detected"].mean(),
                    "normal_cases": len(normal),
                    "normal_mean_fp_fraction": normal[
                        "false_positive_fraction"
                    ].mean(),
                    "normal_mean_fp_volume_ml": normal[
                        "false_positive_volume_ml"
                    ].mean(),
                    "normal_median_fp_volume_ml": normal[
                        "false_positive_volume_ml"
                    ].median(),
                    "normal_p95_fp_volume_ml": percentile(
                        normal["false_positive_volume_ml"],
                        95,
                    ),
                    "normal_zero_fp_rate": (
                        normal["false_positive_voxels"] == 0
                    ).mean(),
                }
            )
    return pd.DataFrame(rows)


def select_recommended_thresholds(threshold_summary):
    recommendations = []
    for scope, frame in threshold_summary.groupby("scope", sort=False):
        frame = frame.sort_values("threshold").reset_index(drop=True)
        baseline_index = (frame["threshold"] - 0.5).abs().idxmin()
        baseline = frame.loc[baseline_index]
        unconstrained = frame.sort_values(
            [
                "positive_mean_dice",
                "positive_mean_recall",
                "normal_mean_fp_volume_ml",
            ],
            ascending=[False, False, True],
        ).iloc[0]
        eligible = frame[
            frame["normal_mean_fp_volume_ml"]
            <= baseline["normal_mean_fp_volume_ml"] + 1e-9
        ]
        constrained = eligible.sort_values(
            [
                "positive_mean_dice",
                "positive_mean_recall",
                "normal_mean_fp_volume_ml",
            ],
            ascending=[False, False, True],
        ).iloc[0]
        recommendations.append(
            {
                "scope": scope,
                "baseline_threshold": float(baseline["threshold"]),
                "baseline_positive_mean_dice": baseline["positive_mean_dice"],
                "baseline_nnunet_foreground_mean_dice": baseline[
                    "nnunet_foreground_mean_dice"
                ],
                "baseline_positive_mean_precision": baseline[
                    "positive_mean_precision"
                ],
                "baseline_positive_mean_recall": baseline[
                    "positive_mean_recall"
                ],
                "baseline_complete_miss_rate": baseline[
                    "positive_complete_miss_rate"
                ],
                "baseline_normal_mean_fp_volume_ml": baseline[
                    "normal_mean_fp_volume_ml"
                ],
                "baseline_normal_p95_fp_volume_ml": baseline[
                    "normal_p95_fp_volume_ml"
                ],
                "unconstrained_threshold": float(unconstrained["threshold"]),
                "unconstrained_positive_mean_dice": unconstrained[
                    "positive_mean_dice"
                ],
                "unconstrained_normal_mean_fp_volume_ml": unconstrained[
                    "normal_mean_fp_volume_ml"
                ],
                "recommended_threshold": float(constrained["threshold"]),
                "recommended_positive_mean_dice": constrained[
                    "positive_mean_dice"
                ],
                "recommended_nnunet_foreground_mean_dice": constrained[
                    "nnunet_foreground_mean_dice"
                ],
                "recommended_positive_median_dice": constrained[
                    "positive_median_dice"
                ],
                "recommended_positive_mean_precision": constrained[
                    "positive_mean_precision"
                ],
                "recommended_positive_mean_recall": constrained[
                    "positive_mean_recall"
                ],
                "recommended_complete_miss_rate": constrained[
                    "positive_complete_miss_rate"
                ],
                "recommended_normal_mean_fp_volume_ml": constrained[
                    "normal_mean_fp_volume_ml"
                ],
                "recommended_normal_p95_fp_volume_ml": constrained[
                    "normal_p95_fp_volume_ml"
                ],
            }
        )
    return pd.DataFrame(recommendations)


def add_reference_comparison(recommendations, config):
    required = (
        "REFERENCE_POSITIVE_DICE",
        "REFERENCE_NNUNET_FOREGROUND_MEAN_DICE",
        "REFERENCE_POSITIVE_PRECISION",
        "REFERENCE_POSITIVE_RECALL",
        "REFERENCE_MAX_POSITIVE_MISS_RATE",
        "REFERENCE_MAX_NORMAL_MEAN_FP_VOLUME_ML",
        "REFERENCE_MAX_NORMAL_P95_FP_VOLUME_ML",
    )
    if not all(hasattr(config, name) for name in required):
        return recommendations
    recommendations = recommendations.copy()
    recommendations["baseline_meets_config_reference"] = (
        (recommendations["baseline_positive_mean_dice"] >= config.REFERENCE_POSITIVE_DICE)
        & (
            recommendations["baseline_nnunet_foreground_mean_dice"]
            >= config.REFERENCE_NNUNET_FOREGROUND_MEAN_DICE
        )
        & (
            recommendations["baseline_positive_mean_precision"]
            >= config.REFERENCE_POSITIVE_PRECISION
        )
        & (
            recommendations["baseline_positive_mean_recall"]
            >= config.REFERENCE_POSITIVE_RECALL
        )
        & (
            recommendations["baseline_complete_miss_rate"]
            <= config.REFERENCE_MAX_POSITIVE_MISS_RATE
        )
        & (
            recommendations["baseline_normal_mean_fp_volume_ml"]
            <= config.REFERENCE_MAX_NORMAL_MEAN_FP_VOLUME_ML
        )
        & (
            recommendations["baseline_normal_p95_fp_volume_ml"]
            <= config.REFERENCE_MAX_NORMAL_P95_FP_VOLUME_ML
        )
    )
    return recommendations


def selected_threshold_case_metrics(case_metrics, recommendations):
    selected = []
    for fold, frame in case_metrics.groupby("fold"):
        scope = f"fold{fold}"
        threshold = float(
            recommendations.loc[
                recommendations["scope"] == scope,
                "recommended_threshold",
            ].iloc[0]
        )
        selected.append(frame[np.isclose(frame["threshold"], threshold)].copy())
    return pd.concat(selected, ignore_index=True)


def summarize_size_strata(selected_metrics):
    positive = selected_metrics[
        selected_metrics["case_type"] == "positive"
    ].copy()
    q33, q67 = np.percentile(positive["target_volume_ml"], [33.333, 66.667])
    positive["size_group"] = pd.cut(
        positive["target_volume_ml"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["small", "medium", "large"],
        include_lowest=True,
    )
    rows = []
    scopes = [(f"fold{fold}", frame) for fold, frame in positive.groupby("fold")]
    scopes.append(("pooled", positive))
    for scope, frame in scopes:
        for size_group, group in frame.groupby("size_group", observed=True):
            rows.append(
                {
                    "scope": scope,
                    "size_group": str(size_group),
                    "global_small_upper_ml": q33,
                    "global_medium_upper_ml": q67,
                    "cases": len(group),
                    "target_volume_min_ml": group["target_volume_ml"].min(),
                    "target_volume_median_ml": group["target_volume_ml"].median(),
                    "target_volume_max_ml": group["target_volume_ml"].max(),
                    "mean_dice": group["dice"].mean(),
                    "median_dice": group["dice"].median(),
                    "dice_q25": percentile(group["dice"], 25),
                    "dice_q75": percentile(group["dice"], 75),
                    "mean_precision": group["precision"].mean(),
                    "mean_recall": group["recall"].mean(),
                    "complete_miss_rate": 1.0 - group["detected"].mean(),
                }
            )
    positive["size_group"] = positive["size_group"].astype(str)
    return positive, pd.DataFrame(rows)


def normalize_slice(image_slice):
    image_slice = np.asarray(image_slice, dtype=np.float32)
    finite = image_slice[np.isfinite(image_slice)]
    nonzero = finite[np.abs(finite) > 1e-6]
    values = nonzero if nonzero.size else finite
    if values.size == 0:
        return np.zeros_like(image_slice)
    lower, upper = np.percentile(values, [1, 99])
    if upper <= lower:
        return np.zeros_like(image_slice)
    return np.clip((image_slice - lower) / (upper - lower), 0.0, 1.0)


def add_contour(axis, mask, color, label):
    if np.any(mask):
        axis.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.0)
        axis.plot([], [], color=color, label=label)


def save_overlay(image, target, prediction, probability, row, output_path):
    target_areas = target.sum(axis=(1, 2))
    prediction_areas = prediction.sum(axis=(1, 2))
    gt_slice_index = int(np.argmax(target_areas))
    if prediction_areas.max() > 0:
        prediction_slice_index = int(np.argmax(prediction_areas))
    else:
        prediction_slice_index = int(np.argmax(probability.sum(axis=(1, 2))))
    slice_rows = (
        ("GT-max", gt_slice_index),
        ("Pred-max", prediction_slice_index),
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for plot_row, (slice_label, slice_index) in enumerate(slice_rows):
        image_slice = normalize_slice(image[slice_index])
        target_slice = target[slice_index]
        prediction_slice = prediction[slice_index]
        probability_slice = probability[slice_index]
        row_axes = axes[plot_row]
        for axis in row_axes:
            axis.imshow(image_slice, cmap="gray", origin="lower")
            axis.axis("off")
        row_axes[0].set_title(f"{slice_label} FLAIR z={slice_index}")
        add_contour(row_axes[1], target_slice, "red", "GT")
        row_axes[1].set_title("Ground truth")
        add_contour(row_axes[2], prediction_slice, "cyan", "Prediction")
        row_axes[2].set_title(f"Prediction t={row.threshold:.2f}")
        probability_artist = row_axes[3].imshow(
            probability_slice,
            cmap="magma",
            alpha=0.55,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        add_contour(row_axes[3], target_slice, "lime", "GT")
        add_contour(row_axes[3], prediction_slice, "cyan", "Prediction")
        handles, labels = row_axes[3].get_legend_handles_labels()
        if handles:
            row_axes[3].legend(handles, labels, loc="lower right", fontsize=7)
        row_axes[3].set_title("Probability + contours")
        fig.colorbar(
            probability_artist,
            ax=row_axes[3],
            fraction=0.046,
            pad=0.04,
        )
    fig.suptitle(
        f"Fold {row.fold} | case {row.case_id} | {row.class_name} | "
        f"Dice={row.dice:.3f}, P={row.precision:.3f}, R={row.recall:.3f} | "
        f"GT={row.target_volume_ml:.2f} mL, Pred={row.predicted_volume_ml:.2f} mL"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_overlays(
    *,
    selected_positive_metrics,
    top_k,
    dataset_dir,
    processed_data_root,
    checkpoint_dir,
    report_root,
    device,
    config,
    model_name,
):
    best = selected_positive_metrics.nlargest(top_k, "dice").copy()
    best["overlay_group"] = "best"
    worst = selected_positive_metrics.nsmallest(top_k, "dice").copy()
    worst["overlay_group"] = "worst"
    selected = pd.concat((best, worst), ignore_index=True)
    manifest = []

    for fold, fold_rows in selected.groupby("fold"):
        dataset, _, _ = load_validation_dataset(
            dataset_dir,
            processed_data_root,
            int(fold),
        )
        model, _, _ = load_model(
            checkpoint_dir,
            int(fold),
            device,
            model_name,
        )
        for row in fold_rows.itertuples(index=False):
            x, _, mask, _, _ = dataset[int(row.dataset_index)]
            with torch.no_grad(), amp_context(device):
                probability = torch.sigmoid(
                    segmentation_logits_from_config(
                        model,
                        x.unsqueeze(0).to(device),
                        config,
                    ).float()
                )[0, 0].cpu().numpy()
            image = x[0].numpy()
            target = mask[0].numpy() > 0
            prediction = probability >= float(row.threshold)
            filename = (
                f"fold{row.fold}_case{row.case_id}_"
                f"dice{row.dice:.3f}.png"
            )
            output_path = report_root / "overlays" / row.overlay_group / filename
            save_overlay(
                image,
                target,
                prediction,
                probability,
                row,
                output_path,
            )
            manifest.append(
                {
                    "overlay_group": row.overlay_group,
                    "fold": row.fold,
                    "case_id": row.case_id,
                    "dice": row.dice,
                    "precision": row.precision,
                    "recall": row.recall,
                    "target_volume_ml": row.target_volume_ml,
                    "predicted_volume_ml": row.predicted_volume_ml,
                    "path": str(output_path),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(manifest)


def save_threshold_plot(threshold_summary, recommendations, output_path):
    pooled = threshold_summary[threshold_summary["scope"] == "pooled"].sort_values(
        "threshold"
    )
    recommended = float(
        recommendations.loc[
            recommendations["scope"] == "pooled",
            "recommended_threshold",
        ].iloc[0]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for column, label in (
        ("positive_mean_dice", "Mean Dice"),
        ("positive_mean_precision", "Mean precision"),
        ("positive_mean_recall", "Mean recall"),
    ):
        axes[0].plot(
            pooled["threshold"],
            pooled[column],
            marker="o",
            label=label,
        )
    axes[0].axvline(0.5, color="gray", linestyle="--", label="Baseline 0.5")
    axes[0].axvline(
        recommended,
        color="black",
        linestyle=":",
        label=f"Recommended {recommended:.2f}",
    )
    axes[0].set_xlabel("Probability threshold")
    axes[0].set_ylabel("Positive-case metric")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for column, label in (
        ("normal_mean_fp_volume_ml", "Normal mean FP"),
        ("normal_p95_fp_volume_ml", "Normal P95 FP"),
    ):
        axes[1].plot(
            pooled["threshold"],
            pooled[column],
            marker="o",
            label=label,
        )
    axes[1].axvline(0.5, color="gray", linestyle="--")
    axes[1].axvline(recommended, color="black", linestyle=":")
    axes[1].set_xlabel("Probability threshold")
    axes[1].set_ylabel("False-positive volume (mL)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("FLAIRUNet3D validation threshold sweep (pooled fold-val pairs)")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_size_plot(selected_positive, output_path):
    group_order = ["small", "medium", "large"]
    dice_values = [
        selected_positive.loc[
            selected_positive["size_group"] == group,
            "dice",
        ].to_numpy()
        for group in group_order
    ]
    recall_values = [
        selected_positive.loc[
            selected_positive["size_group"] == group,
            "recall",
        ].to_numpy()
        for group in group_order
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].boxplot(dice_values, tick_labels=group_order, showfliers=False)
    axes[0].set_title("Dice by lesion-volume stratum")
    axes[0].set_ylabel("Dice")
    axes[1].boxplot(recall_values, tick_labels=group_order, showfliers=False)
    axes[1].set_title("Voxel recall by lesion-volume stratum")
    axes[1].set_ylabel("Recall")
    for axis in axes:
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def markdown_cell(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def dataframe_markdown(frame, columns=None):
    columns = list(frame.columns) if columns is None else list(columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        rows.append(
            "| " + " | ".join(markdown_cell(value) for value in values) + " |"
        )
    return "\n".join(rows)


def write_summary(
    report_root,
    recommendations,
    selected_metrics,
    size_summary,
    fold_metadata,
    thresholds,
):
    positive = selected_metrics[selected_metrics["case_type"] == "positive"]
    normal = selected_metrics[selected_metrics["case_type"] == "normal"]
    foreground_defined = selected_metrics[
        (selected_metrics["target_voxels"] + selected_metrics["predicted_voxels"])
        > 0
    ]
    pooled_size = size_summary[size_summary["scope"] == "pooled"]
    lines = [
        "# FLAIRUNet3D validation segmentation analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Only validation sets were used; no test data were read. Each fold threshold "
        "maximizes validation positive-case mean Dice among thresholds whose mean "
        "normal false-positive volume is no worse than threshold 0.5.",
        "",
        f"Threshold grid: {', '.join(f'{value:.2f}' for value in thresholds)}",
        "",
        "## Recommended thresholds",
        "",
        dataframe_markdown(
            recommendations,
            [
                "scope",
                "baseline_threshold",
                "baseline_positive_mean_dice",
                "baseline_nnunet_foreground_mean_dice",
                "baseline_meets_config_reference",
                "recommended_threshold",
                "recommended_positive_mean_dice",
                "recommended_nnunet_foreground_mean_dice",
                "recommended_positive_mean_precision",
                "recommended_positive_mean_recall",
                "recommended_normal_mean_fp_volume_ml",
            ],
        ),
        "",
        "## Per-case distribution at fold-specific thresholds",
        "",
        f"- Positive cases/model-case pairs: {len(positive)}",
        f"- Dice mean: {positive['dice'].mean():.4f}",
        "- nnU-Net foreground mean Dice: "
        f"{foreground_defined['dice'].mean():.4f}",
        f"- Dice median [IQR]: {positive['dice'].median():.4f} "
        f"[{percentile(positive['dice'], 25):.4f}, "
        f"{percentile(positive['dice'], 75):.4f}]",
        f"- Mean precision / recall: {positive['precision'].mean():.4f} / "
        f"{positive['recall'].mean():.4f}",
        f"- Complete miss rate: {(1.0 - positive['detected'].mean()):.4f}",
        f"- Normal mean FP volume: {normal['false_positive_volume_ml'].mean():.4f} mL",
        f"- Normal median / P95 FP volume: "
        f"{normal['false_positive_volume_ml'].median():.4f} / "
        f"{percentile(normal['false_positive_volume_ml'], 95):.4f} mL",
        "",
        "## Lesion-size strata",
        "",
        dataframe_markdown(
            pooled_size,
            [
                "size_group",
                "cases",
                "target_volume_min_ml",
                "target_volume_median_ml",
                "target_volume_max_ml",
                "mean_dice",
                "median_dice",
                "mean_precision",
                "mean_recall",
                "complete_miss_rate",
            ],
        ),
        "",
        "## Fold checkpoints",
        "",
        dataframe_markdown(pd.DataFrame(fold_metadata)),
        "",
        "## Files",
        "",
        "- `all_threshold_case_metrics.csv`: every case at every threshold",
        "- `threshold_summary.csv`: fold and pooled threshold curves",
        "- `recommended_thresholds.csv`: baseline, unconstrained, and FP-constrained choices",
        "- `selected_threshold_case_metrics.csv`: one validation-selected threshold per fold",
        "- `size_stratified_metrics.csv`: small/medium/large lesion summaries",
        "- `selected_positive_case_metrics.csv`: positive cases with assigned size strata",
        "- `threshold_sweep.png` and `lesion_size_strata.png`: diagnostic plots",
        "- `overlay_manifest.csv` and `overlays/`: best/worst qualitative examples",
        "",
    ]
    (report_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(args):
    set_seed(SEED)
    config = load_python_config(
        args.config,
        ("BATCH_SIZE", "DEVICE", "NUM_WORKERS"),
    )
    model_name = getattr(config, "MODEL_NAME", MODEL_NAME)
    device = torch.device(args.device or config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable")
    thresholds = tuple(sorted(set(float(value) for value in args.thresholds)))
    if not thresholds or any(value <= 0 or value >= 1 for value in thresholds):
        raise ValueError("All thresholds must lie strictly between 0 and 1")
    if not any(abs(value - 0.5) < 1e-8 for value in thresholds):
        raise ValueError("Threshold grid must include 0.5 as the baseline")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    dataset_dir = resolve_input_artifact_dir(args.data_root, "datasets")
    processed_data_root = infer_data_dir(args.data_root)
    checkpoint_dir = resolve_input_artifact_dir(
        args.checkpoint_root,
        "checkpoints",
    )
    report_root = resolve_report_root(args.checkpoint_root, args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    fold_metadata = []
    for fold in args.folds:
        rows, metadata = infer_fold(
            fold=fold,
            dataset_dir=dataset_dir,
            processed_data_root=processed_data_root,
            checkpoint_dir=checkpoint_dir,
            thresholds=thresholds,
            batch_size=args.batch_size or config.BATCH_SIZE,
            num_workers=(
                args.num_workers
                if args.num_workers is not None
                else config.NUM_WORKERS
            ),
            device=device,
            config=config,
            model_name=model_name,
        )
        all_rows.extend(rows)
        fold_metadata.append(metadata)

    case_metrics = pd.DataFrame(all_rows)
    threshold_summary = summarize_thresholds(case_metrics)
    recommendations = select_recommended_thresholds(threshold_summary)
    recommendations = add_reference_comparison(recommendations, config)
    selected_metrics = selected_threshold_case_metrics(
        case_metrics,
        recommendations,
    )
    selected_positive, size_summary = summarize_size_strata(selected_metrics)

    case_metrics.to_csv(report_root / "all_threshold_case_metrics.csv", index=False)
    threshold_summary.to_csv(report_root / "threshold_summary.csv", index=False)
    recommendations.to_csv(report_root / "recommended_thresholds.csv", index=False)
    selected_metrics.to_csv(
        report_root / "selected_threshold_case_metrics.csv",
        index=False,
    )
    selected_positive.to_csv(
        report_root / "selected_positive_case_metrics.csv",
        index=False,
    )
    size_summary.to_csv(report_root / "size_stratified_metrics.csv", index=False)

    save_threshold_plot(
        threshold_summary,
        recommendations,
        report_root / "threshold_sweep.png",
    )
    save_size_plot(
        selected_positive,
        report_root / "lesion_size_strata.png",
    )

    overlay_manifest = generate_overlays(
        selected_positive_metrics=selected_positive,
        top_k=args.top_k,
        dataset_dir=dataset_dir,
        processed_data_root=processed_data_root,
        checkpoint_dir=checkpoint_dir,
        report_root=report_root,
        device=device,
        config=config,
        model_name=model_name,
    )
    overlay_manifest.to_csv(report_root / "overlay_manifest.csv", index=False)
    write_summary(
        report_root,
        recommendations,
        selected_metrics,
        size_summary,
        fold_metadata,
        thresholds,
    )
    print(f"Analysis report: {report_root}")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--report-root", default=None)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(1, K_FOLDS + 1)),
        choices=range(1, K_FOLDS + 1),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    main(parser.parse_args())
