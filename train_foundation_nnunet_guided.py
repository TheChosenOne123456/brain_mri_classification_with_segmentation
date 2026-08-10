"""训练 Foundation + OOF nnU-Net soft mask 引导的独立 subtype expert。"""

import argparse
import hashlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from configs.config_utils import (
    infer_data_dir,
    load_python_config,
    resolve_input_artifact_dir,
    resolve_output_artifact_dir,
)
from configs.global_config import K_FOLDS, NUM_CLASSES, SEED
from models.FLAIRUNet3D import FLAIRUNet3DNNUNet
from models.FoundationModelNNUNetGuided import (
    GUIDANCE_FEATURE_DIM,
    FoundationModelNNUNetGuided,
)
from scripts.convert_nnunet_flair_checkpoint import convert_network_weights
from utils.segmentation_inference import segmentation_logits_from_config
from utils.dataset import load_nii_as_tensor
from utils.train_and_test import load_pt_dataset, set_seed


MODEL_NAME = "FoundationModelNNUNetGuided"
SEQUENCE_ID = 3
SEQUENCE_NAME = "FLAIR"
CACHE_SCHEMA_VERSION = 1
SELECTION_CONSTRAINED = "constrained_guidance"
SELECTION_DIAGNOSTIC = "diagnostic_unconstrained"
SELECTION_FALLBACK = "foundation_fallback"

CONFIG_FIELDS = (
    "NUM_EPOCHS",
    "MIN_EPOCHS",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "DEVICE",
    "NUM_WORKERS",
    "FEATURE_EXTRACTION_NUM_WORKERS",
    "PATIENCE",
    "SUBTYPE_POS_WEIGHT_POWER",
    "MIX_ALPHA_MIN",
    "MIX_ALPHA_MAX",
    "MIX_ALPHA_STEPS",
    "METASTASIS_F_BETA",
    "MIN_ACCURACY_DELTA",
    "MIN_MACRO_F1_DELTA",
    "MIN_METASTASIS_PRECISION_DELTA",
    "MIN_METASTASIS_RECALL_GAIN",
    "NNUNET_MODEL_RELATIVE_DIR",
    "NNUNET_CHECKPOINT_NAME",
    "SEGMENTATION_VALIDATION_INFERENCE",
    "SEG_VALIDATION_ROI_SIZE",
    "SEG_VALIDATION_OVERLAP",
    "SEG_VALIDATION_SW_BATCH_SIZE",
    "SEG_VALIDATION_MIRROR_AXES",
    "SEG_VALIDATION_CROP_NONZERO",
    "SEG_VALIDATION_RENORMALIZE_NONZERO",
)


class GuidedFeatureDataset(Dataset):
    def __init__(self, payload):
        self.features = payload["guidance_features"].float()
        self.base_logits = payload["base_logits"].float()
        self.labels = payload["labels"].long()
        self.case_ids = tuple(payload["case_ids"])
        if self.features.shape != (len(self.labels), GUIDANCE_FEATURE_DIM):
            raise ValueError(
                "Invalid cached guidance feature shape: "
                f"{tuple(self.features.shape)}"
            )
        if self.base_logits.shape != (len(self.labels), NUM_CLASSES):
            raise ValueError(
                f"Invalid cached base logits shape: {tuple(self.base_logits.shape)}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.features[index],
            self.base_logits[index],
            self.labels[index],
        )


class GuidanceInputDataset(Dataset):
    """缓存阶段只加载 FLAIR，不读取本阶段不使用的医生 mask。"""

    def __init__(self, cases):
        self.cases = tuple(cases)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        return (
            load_nii_as_tensor(Path(case["nii_path"])),
            torch.tensor(int(case["label"]), dtype=torch.long),
            normalize_case_id(case["case_id"]),
        )


def amp_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_case_id(case_id):
    value = str(case_id)
    return value[5:] if value.startswith("case_") else value


def dataset_dir(data_root, fold):
    datasets_root = resolve_input_artifact_dir(data_root, "datasets")
    result = datasets_root / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}" / f"fold{fold}"
    if not result.is_dir():
        raise FileNotFoundError(f"Dataset fold directory not found: {result}")
    return result


def load_split_dataset(data_root, fold, split):
    root = dataset_dir(data_root, fold)
    return load_pt_dataset(
        root / f"{split}.pt",
        data_root=infer_data_dir(data_root),
    )


def build_outer_test_fold_map(data_root):
    membership = {}
    for fold in range(1, int(K_FOLDS) + 1):
        dataset = load_split_dataset(data_root, fold, "test")
        for case in dataset.cases:
            case_id = normalize_case_id(case["case_id"])
            if case_id in membership:
                raise RuntimeError(
                    f"Case {case_id} occurs in multiple outer test folds"
                )
            membership[case_id] = fold
    return membership


def source_folds_for_split(dataset, split, target_fold, outer_test_folds):
    source_folds = []
    for case in dataset.cases:
        case_id = normalize_case_id(case["case_id"])
        if split == "train":
            if case_id not in outer_test_folds:
                raise KeyError(f"No outer test fold found for case {case_id}")
            source_fold = int(outer_test_folds[case_id])
            if source_fold == int(target_fold):
                raise RuntimeError(
                    f"Training case {case_id} unexpectedly belongs to target "
                    f"test fold {target_fold}"
                )
        else:
            source_fold = int(target_fold)
        source_folds.append(source_fold)
    return tuple(source_folds)


def foundation_checkpoint_path(checkpoint_root, fold):
    checkpoint_dir = resolve_input_artifact_dir(checkpoint_root, "checkpoints")
    path = (
        checkpoint_dir
        / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}"
        / "FoundationModel"
        / f"fold{fold}_model_best.pth"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Foundation checkpoint not found: {path}")
    return path


def native_nnunet_checkpoint_path(results_root, config, fold):
    path = (
        Path(results_root).expanduser().resolve()
        / str(config.NNUNET_MODEL_RELATIVE_DIR)
        / f"fold_{int(fold) - 1}"
        / str(config.NNUNET_CHECKPOINT_NAME)
    )
    if not path.is_file():
        raise FileNotFoundError(f"nnU-Net checkpoint not found: {path}")
    return path


def load_foundation_model(checkpoint_path, config, device):
    model = FoundationModelNNUNetGuided(
        num_classes=NUM_CLASSES,
        in_channels=1,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    source_state = checkpoint.get("model_state", checkpoint)
    copied = model.initialize_from_foundation_state(source_state)
    model.freeze_foundation()
    model.to(device)
    model.eval()
    print(f"Loaded Foundation checkpoint: {checkpoint_path} ({copied} tensors)")
    return model, checkpoint


def load_nnunet_model(checkpoint_path, device, use_data_parallel):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "network_weights" not in checkpoint:
        raise KeyError(f"network_weights not found in {checkpoint_path}")
    model = FLAIRUNet3DNNUNet(num_classes=NUM_CLASSES, in_channels=1)
    converted = convert_network_weights(model, checkpoint["network_weights"])
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    if use_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"nnU-Net inference uses DataParallel on {torch.cuda.device_count()} GPUs")
    print(f"Loaded OOF nnU-Net: {checkpoint_path} ({converted} tensors)")
    return model


def cache_signature(
    *,
    split,
    target_fold,
    foundation_path,
    foundation_sha256,
    nnunet_paths,
    nnunet_sha256,
    config,
):
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "split": str(split),
        "target_fold": int(target_fold),
        "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": str(foundation_sha256),
        "nnunet_checkpoints": {
            str(fold): str(path) for fold, path in nnunet_paths.items()
        },
        "nnunet_sha256": {
            str(fold): str(value) for fold, value in nnunet_sha256.items()
        },
        "inference": {
            "roi_size": tuple(int(x) for x in config.SEG_VALIDATION_ROI_SIZE),
            "overlap": float(config.SEG_VALIDATION_OVERLAP),
            "sw_batch_size": int(config.SEG_VALIDATION_SW_BATCH_SIZE),
            "mirror_axes": tuple(int(x) for x in config.SEG_VALIDATION_MIRROR_AXES),
            "crop_nonzero": bool(config.SEG_VALIDATION_CROP_NONZERO),
            "renormalize_nonzero": bool(
                config.SEG_VALIDATION_RENORMALIZE_NONZERO
            ),
        },
        "guidance_feature_dim": GUIDANCE_FEATURE_DIM,
    }


def validate_cached_payload(payload, signature, case_ids, source_folds):
    if payload.get("signature") != signature:
        raise RuntimeError("Cached feature signature does not match this run")
    if tuple(payload.get("case_ids", ())) != tuple(case_ids):
        raise RuntimeError("Cached feature case order does not match the dataset")
    cached_source_folds = tuple(
        int(value) for value in payload.get("source_folds", ())
    )
    if cached_source_folds != tuple(source_folds):
        raise RuntimeError("Cached OOF nnU-Net source folds do not match")
    if payload["guidance_features"].shape != (
        len(case_ids),
        GUIDANCE_FEATURE_DIM,
    ):
        raise RuntimeError("Cached guidance feature tensor has the wrong shape")
    if payload["base_logits"].shape != (len(case_ids), NUM_CLASSES):
        raise RuntimeError("Cached Foundation logits have the wrong shape")


def save_partial_cache(
    path,
    *,
    signature,
    source_fold,
    records,
):
    ordered = sorted(records, key=lambda record: record[0])
    torch.save(
        {
            "signature": signature,
            "source_fold": int(source_fold),
            "indices": torch.tensor(
                [record[0] for record in ordered],
                dtype=torch.long,
            ),
            "case_ids": [record[1] for record in ordered],
            "labels": torch.tensor(
                [record[2] for record in ordered],
                dtype=torch.long,
            ),
            "base_logits": torch.stack(
                [record[3] for record in ordered]
            ),
            "guidance_features": torch.stack(
                [record[4] for record in ordered]
            ),
        },
        path,
    )


def load_partial_records(path, signature, source_fold):
    if not path.is_file():
        return []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature:
        raise RuntimeError(
            f"Partial cache signature mismatch: {path}; use --rebuild-cache"
        )
    if int(payload.get("source_fold", -1)) != int(source_fold):
        raise RuntimeError(f"Partial cache source fold mismatch: {path}")
    return [
        (
            int(index),
            str(case_id),
            int(label),
            base_logits.float(),
            guidance_features.float(),
        )
        for index, case_id, label, base_logits, guidance_features in zip(
            payload["indices"].tolist(),
            payload["case_ids"],
            payload["labels"].tolist(),
            payload["base_logits"],
            payload["guidance_features"],
        )
    ]


def extract_feature_cache(
    *,
    split,
    dataset,
    source_folds,
    target_fold,
    model,
    foundation_path,
    foundation_sha256,
    nnunet_paths,
    nnunet_sha256,
    cache_dir,
    config,
    device,
    rebuild,
):
    case_ids = tuple(normalize_case_id(case["case_id"]) for case in dataset.cases)
    signature = cache_signature(
        split=split,
        target_fold=target_fold,
        foundation_path=foundation_path,
        foundation_sha256=foundation_sha256,
        nnunet_paths=nnunet_paths,
        nnunet_sha256=nnunet_sha256,
        config=config,
    )
    final_path = cache_dir / f"fold{target_fold}_{split}.pt"
    if final_path.is_file() and not rebuild:
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        validate_cached_payload(payload, signature, case_ids, source_folds)
        print(f"Reusing complete {split} cache: {final_path}")
        return final_path

    records_by_index = {}
    groups = {}
    for index, source_fold in enumerate(source_folds):
        groups.setdefault(int(source_fold), []).append(index)

    for source_fold, expected_indices in sorted(groups.items()):
        partial_path = (
            cache_dir
            / f"fold{target_fold}_{split}_source_fold{source_fold}.partial.pt"
        )
        records = [] if rebuild else load_partial_records(
            partial_path,
            signature,
            source_fold,
        )
        processed = {record[0] for record in records}
        expected_set = set(expected_indices)
        if not processed.issubset(expected_set):
            raise RuntimeError(f"Partial cache contains unexpected indices: {partial_path}")
        missing_indices = [index for index in expected_indices if index not in processed]
        if missing_indices:
            segmenter = load_nnunet_model(
                nnunet_paths[source_fold],
                device,
                bool(getattr(config, "NNUNET_USE_DATAPARALLEL", True)),
            )
            loader = DataLoader(
                Subset(GuidanceInputDataset(dataset.cases), missing_indices),
                batch_size=1,
                shuffle=False,
                num_workers=int(config.FEATURE_EXTRACTION_NUM_WORKERS),
                pin_memory=(device.type == "cuda"),
            )
            id_to_index = {case_ids[index]: index for index in missing_indices}
            progress = tqdm(
                loader,
                desc=f"Cache {split}: OOF nnU-Net fold {source_fold}",
            )
            save_every = int(getattr(config, "FEATURE_CACHE_SAVE_EVERY", 25))
            for step, (inputs, labels, batch_case_ids) in enumerate(
                progress,
                start=1,
            ):
                case_id = normalize_case_id(batch_case_ids[0])
                index = id_to_index[case_id]
                inputs = inputs.to(device, non_blocking=True)
                with torch.no_grad(), amp_context(device):
                    lesion_logits = segmentation_logits_from_config(
                        segmenter,
                        inputs,
                        config,
                    )
                    lesion_probability = torch.sigmoid(lesion_logits.float())
                    base_logits, multiscale = model.forward_multiscale_features(inputs)
                    guidance_features = model.guidance_feature_vector(
                        inputs,
                        multiscale,
                        lesion_probability,
                    )
                records.append(
                    (
                        index,
                        case_id,
                        int(labels.item()),
                        base_logits[0].detach().float().cpu(),
                        guidance_features[0].detach().float().cpu(),
                    )
                )
                if step % save_every == 0:
                    save_partial_cache(
                        partial_path,
                        signature=signature,
                        source_fold=source_fold,
                        records=records,
                    )
            save_partial_cache(
                partial_path,
                signature=signature,
                source_fold=source_fold,
                records=records,
            )
            del segmenter
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for record in records:
            if record[0] in records_by_index:
                raise RuntimeError(f"Duplicate cached case index: {record[0]}")
            records_by_index[record[0]] = record

    if set(records_by_index) != set(range(len(dataset))):
        missing = sorted(set(range(len(dataset))) - set(records_by_index))
        raise RuntimeError(f"Feature cache is incomplete; missing indices: {missing[:20]}")
    ordered = [records_by_index[index] for index in range(len(dataset))]
    payload = {
        "signature": signature,
        "case_ids": [record[1] for record in ordered],
        "source_folds": tuple(int(value) for value in source_folds),
        "labels": torch.tensor([record[2] for record in ordered], dtype=torch.long),
        "base_logits": torch.stack([record[3] for record in ordered]),
        "guidance_features": torch.stack([record[4] for record in ordered]),
    }
    validate_cached_payload(payload, signature, case_ids, source_folds)
    torch.save(payload, final_path)
    print(f"Completed {split} cache: {final_path}")
    return final_path


def metastasis_metrics(labels, predictions, beta):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    true_positive = int(np.sum((labels == 2) & (predictions == 2)))
    predicted_positive = int(np.sum(predictions == 2))
    actual_positive = int(np.sum(labels == 2))
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    beta_squared = float(beta) ** 2
    denominator = beta_squared * precision + recall
    fbeta = (
        (1.0 + beta_squared) * precision * recall / denominator
        if denominator > 0
        else 0.0
    )
    return float(precision), float(recall), float(fbeta)


def classification_metrics(labels, predictions, beta):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    precision, recall, fbeta = metastasis_metrics(labels, predictions, beta)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "metastasis_precision": precision,
        "metastasis_recall": recall,
        "metastasis_fbeta": fbeta,
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1, 2],
        ).tolist(),
    }


def mixture_metrics(labels, base_logits, expert_logits, alpha, beta):
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    base_logits = torch.as_tensor(base_logits, dtype=torch.float32)
    expert_logits = torch.as_tensor(expert_logits, dtype=torch.float32)
    base_predictions = base_logits.argmax(dim=1)
    metastasis_probability = (
        FoundationModelNNUNetGuided.mixed_subtype_probability(
            base_logits,
            expert_logits,
            float(alpha),
        )
    )
    predictions = FoundationModelNNUNetGuided.hierarchical_predictions(
        base_logits,
        metastasis_probability,
    )
    metrics = classification_metrics(
        labels_tensor.numpy(),
        predictions.numpy(),
        beta,
    )
    labels_array = labels_tensor.numpy()
    base_array = base_predictions.numpy()
    prediction_array = predictions.numpy()
    metrics.update(
        {
            "mix_alpha": float(alpha),
            "recovered_metastasis": int(
                np.sum(
                    (labels_array == 2)
                    & (base_array != 2)
                    & (prediction_array == 2)
                )
            ),
            "lost_metastasis": int(
                np.sum(
                    (labels_array == 2)
                    & (base_array == 2)
                    & (prediction_array != 2)
                )
            ),
            "new_inflammation_to_metastasis": int(
                np.sum(
                    (labels_array == 1)
                    & (base_array != 2)
                    & (prediction_array == 2)
                )
            ),
            "fixed_inflammation_from_metastasis": int(
                np.sum(
                    (labels_array == 1)
                    & (base_array == 2)
                    & (prediction_array == 1)
                )
            ),
        }
    )
    return metrics


def evaluate_expert(model, loader, device, criterion):
    model.eval()
    labels = []
    base_logits = []
    expert_logits = []
    total_loss = 0.0
    total_abnormal = 0
    with torch.no_grad():
        for features, batch_base_logits, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_expert_logits = model.guidance_expert_logit(features)
            abnormal = targets != 0
            if abnormal.any():
                subtype_target = (targets[abnormal] == 2).float()
                loss = criterion(batch_expert_logits[abnormal], subtype_target)
                count = int(abnormal.sum().item())
                total_loss += float(loss.item()) * count
                total_abnormal += count
            labels.append(targets.cpu())
            base_logits.append(batch_base_logits.float().cpu())
            expert_logits.append(batch_expert_logits.float().cpu())
    return {
        "labels": torch.cat(labels),
        "base_logits": torch.cat(base_logits),
        "expert_logits": torch.cat(expert_logits),
        "expert_loss": total_loss / max(total_abnormal, 1),
    }


def scan_mixture_workpoints(evaluation, config):
    alpha_min = float(config.MIX_ALPHA_MIN)
    alpha_max = float(config.MIX_ALPHA_MAX)
    alpha_steps = int(config.MIX_ALPHA_STEPS)
    if not 0.0 <= alpha_min <= alpha_max <= 1.0:
        raise ValueError("MIX_ALPHA_MIN/MAX must satisfy 0 <= min <= max <= 1")
    if alpha_steps < 2:
        raise ValueError("MIX_ALPHA_STEPS must be at least 2")
    baseline = mixture_metrics(
        evaluation["labels"],
        evaluation["base_logits"],
        evaluation["expert_logits"],
        0.0,
        config.METASTASIS_F_BETA,
    )
    candidates = [
        mixture_metrics(
            evaluation["labels"],
            evaluation["base_logits"],
            evaluation["expert_logits"],
            alpha,
            config.METASTASIS_F_BETA,
        )
        for alpha in np.linspace(alpha_min, alpha_max, alpha_steps)
    ]

    def rank(metrics):
        return (
            metrics["metastasis_fbeta"],
            metrics["metastasis_recall"],
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["metastasis_precision"],
            -metrics["mix_alpha"],
        )

    feasible = [
        metrics
        for metrics in candidates
        if constraints_met(metrics, baseline, config)
    ]
    constrained = max(feasible, key=rank) if feasible else None
    unconstrained = max(candidates, key=rank)
    return constrained, unconstrained, baseline


def constraints_met(metrics, baseline, config):
    return bool(
        metrics["accuracy"]
        >= baseline["accuracy"] + float(config.MIN_ACCURACY_DELTA)
        and metrics["macro_f1"]
        >= baseline["macro_f1"] + float(config.MIN_MACRO_F1_DELTA)
        and metrics["metastasis_precision"]
        >= baseline["metastasis_precision"]
        + float(config.MIN_METASTASIS_PRECISION_DELTA)
        and metrics["metastasis_recall"]
        >= baseline["metastasis_recall"]
        + float(config.MIN_METASTASIS_RECALL_GAIN)
    )


def checkpoint_payload(
    model,
    *,
    epoch,
    fold,
    metrics,
    baseline,
    constraints,
    expert_validation_loss,
    config,
    foundation_path,
    foundation_sha256,
    cache_paths,
):
    guidance_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("guidance_")
    }
    return {
        "model_name": MODEL_NAME,
        "stage": "oof-guidance-expert-mixture",
        "fold": int(fold),
        "epoch": int(epoch),
        "guidance_state": guidance_state,
        "foundation_checkpoint": str(foundation_path),
        "foundation_sha256": str(foundation_sha256),
        "cache_paths": {key: str(value) for key, value in cache_paths.items()},
        "validation": metrics,
        "expert_validation_loss": float(expert_validation_loss),
        "baseline_validation": baseline,
        "selection_constraints_met": bool(constraints),
        "selection_status": (
            SELECTION_CONSTRAINED if constraints else SELECTION_DIAGNOSTIC
        ),
        "train_config_path": str(config.__config_path__),
    }


def create_foundation_fallback_checkpoint(
    diagnostic_path,
    best_path,
    *,
    expected_fold,
):
    """把无安全增益的诊断 checkpoint 固化为 alpha=0 正式回退结果。"""
    diagnostic_path = Path(diagnostic_path).expanduser().resolve()
    best_path = Path(best_path).expanduser().resolve()
    if best_path.is_file():
        existing = torch.load(best_path, map_location="cpu", weights_only=False)
        if existing.get("selection_status") != SELECTION_FALLBACK:
            raise FileExistsError(
                f"Refusing to overwrite existing non-fallback checkpoint: {best_path}"
            )
        print(f"Foundation fallback checkpoint already exists: {best_path}")
        return best_path
    if not diagnostic_path.is_file():
        raise FileNotFoundError(
            f"Unconstrained diagnostic checkpoint not found: {diagnostic_path}"
        )

    diagnostic = torch.load(
        diagnostic_path,
        map_location="cpu",
        weights_only=False,
    )
    if diagnostic.get("model_name") != MODEL_NAME:
        raise RuntimeError(f"Unexpected model in diagnostic: {diagnostic_path}")
    if int(diagnostic.get("fold", -1)) != int(expected_fold):
        raise RuntimeError(f"Fold mismatch in diagnostic: {diagnostic_path}")
    baseline = diagnostic.get("baseline_validation")
    if not isinstance(baseline, dict) or float(baseline.get("mix_alpha", -1)) != 0.0:
        raise RuntimeError(
            f"Valid alpha=0 baseline missing from diagnostic: {diagnostic_path}"
        )
    guidance_state = diagnostic.get("guidance_state")
    if not isinstance(guidance_state, dict) or "guidance_mix_alpha" not in guidance_state:
        raise RuntimeError(f"Guidance state missing from diagnostic: {diagnostic_path}")
    fallback_guidance_state = {
        key: value.detach().cpu().clone()
        for key, value in guidance_state.items()
    }
    fallback_guidance_state["guidance_mix_alpha"].zero_()

    fallback = dict(diagnostic)
    fallback.update(
        {
            "epoch": 0,
            "source_expert_epoch": int(diagnostic.get("epoch", -1)),
            "guidance_state": fallback_guidance_state,
            "validation": dict(baseline),
            "selection_constraints_met": False,
            "selection_status": SELECTION_FALLBACK,
            "fallback_reason": (
                "No alpha > 0 met all validation Foundation-preservation "
                "constraints; alpha=0 exactly reproduces Foundation."
            ),
            "source_diagnostic_checkpoint": str(diagnostic_path),
        }
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fallback, best_path)
    print(
        "Saved formal Foundation fallback checkpoint: "
        f"{best_path} (alpha=0, source expert epoch "
        f"{fallback['source_expert_epoch']})"
    )
    return best_path


def train_guidance_expert(
    *,
    model,
    train_cache_path,
    val_cache_path,
    output_root,
    fold,
    config,
    device,
    foundation_path,
    foundation_sha256,
):
    train_payload = torch.load(
        train_cache_path,
        map_location="cpu",
        weights_only=False,
    )
    val_payload = torch.load(
        val_cache_path,
        map_location="cpu",
        weights_only=False,
    )
    train_dataset = GuidedFeatureDataset(train_payload)
    val_dataset = GuidedFeatureDataset(val_payload)
    abnormal_indices = torch.nonzero(
        train_dataset.labels != 0,
        as_tuple=False,
    ).flatten()
    abnormal_features = train_dataset.features[abnormal_indices]
    model.set_guidance_standardization(
        abnormal_features.mean(dim=0),
        abnormal_features.std(dim=0, unbiased=False),
    )
    model.freeze_foundation()
    model.to(device)

    train_loader = DataLoader(
        Subset(train_dataset, abnormal_indices.tolist()),
        batch_size=int(config.BATCH_SIZE),
        shuffle=True,
        num_workers=int(config.NUM_WORKERS),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.BATCH_SIZE),
        shuffle=False,
        num_workers=int(config.NUM_WORKERS),
        pin_memory=(device.type == "cuda"),
    )
    class_counts = torch.bincount(
        train_dataset.labels[abnormal_indices] - 1,
        minlength=2,
    )
    pos_weight = (
        class_counts[0].float() / class_counts[1].clamp_min(1).float()
    ).pow(float(config.SUBTYPE_POS_WEIGHT_POWER))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.guidance_expert_head.parameters(),
        lr=float(config.LEARNING_RATE),
        weight_decay=float(config.WEIGHT_DECAY),
    )

    checkpoint_dir = (
        resolve_output_artifact_dir(output_root, "checkpoints")
        / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}"
        / MODEL_NAME
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"fold{fold}_model_best.pth"
    unconstrained_path = (
        checkpoint_dir / f"fold{fold}_model_best_unconstrained.pth"
    )
    latest_path = checkpoint_dir / f"fold{fold}_model_latest.pth"

    initial_evaluation = evaluate_expert(
        model,
        val_loader,
        device,
        criterion,
    )
    _, _, baseline = scan_mixture_workpoints(initial_evaluation, config)
    print(
        "Foundation validation baseline: "
        f"acc={baseline['accuracy']:.4f}, macro_f1={baseline['macro_f1']:.4f}, "
        f"meta_precision={baseline['metastasis_precision']:.4f}, "
        f"meta_recall={baseline['metastasis_recall']:.4f}, "
        f"meta_f2={baseline['metastasis_fbeta']:.4f}"
    )
    zero_alpha_metrics = mixture_metrics(
        initial_evaluation["labels"],
        initial_evaluation["base_logits"],
        initial_evaluation["expert_logits"],
        0.0,
        config.METASTASIS_F_BETA,
    )
    if zero_alpha_metrics != baseline:
        raise RuntimeError("alpha=0 does not reproduce the Foundation baseline")

    best_rank = None
    best_unconstrained_rank = None
    best_expert_loss = float("inf")
    best_epoch = None
    stale_epochs = 0
    payload = None
    for epoch in range(1, int(config.NUM_EPOCHS) + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for features, _, targets in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            expert_logit = model.guidance_expert_logit(features)
            subtype_target = (targets == 2).float()
            loss = criterion(expert_logit, subtype_target)
            loss.backward()
            if float(getattr(config, "GRADIENT_CLIP_NORM", 0.0)) > 0:
                nn.utils.clip_grad_norm_(
                    model.guidance_expert_head.parameters(),
                    float(config.GRADIENT_CLIP_NORM),
                )
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1

        evaluation = evaluate_expert(
            model,
            val_loader,
            device,
            criterion,
        )
        constrained, unconstrained, current_baseline = scan_mixture_workpoints(
            evaluation,
            config,
        )
        if current_baseline != baseline:
            raise RuntimeError("Cached Foundation baseline changed during training")
        metrics = constrained if constrained is not None else unconstrained
        meets_constraints = constrained is not None

        def checkpoint_rank(workpoint):
            return (
                workpoint["metastasis_fbeta"],
                workpoint["metastasis_recall"],
                workpoint["accuracy"],
                workpoint["macro_f1"],
                workpoint["metastasis_precision"],
                -evaluation["expert_loss"],
            )

        improved = False
        unconstrained_rank = checkpoint_rank(unconstrained)
        if (
            best_unconstrained_rank is None
            or unconstrained_rank > best_unconstrained_rank
        ):
            best_unconstrained_rank = unconstrained_rank
            model.set_guidance_mix_alpha(unconstrained["mix_alpha"])
            unconstrained_payload = checkpoint_payload(
                model,
                epoch=epoch,
                fold=fold,
                metrics=unconstrained,
                baseline=baseline,
                constraints=False,
                expert_validation_loss=evaluation["expert_loss"],
                config=config,
                foundation_path=foundation_path,
                foundation_sha256=foundation_sha256,
                cache_paths={"train": train_cache_path, "val": val_cache_path},
            )
            torch.save(unconstrained_payload, unconstrained_path)
        if constrained is not None:
            constrained_rank = checkpoint_rank(constrained)
        else:
            constrained_rank = None
        if constrained_rank is not None and (
            best_rank is None or constrained_rank > best_rank
        ):
            best_rank = constrained_rank
            best_epoch = epoch
            improved = True
            model.set_guidance_mix_alpha(constrained["mix_alpha"])
            best_payload = checkpoint_payload(
                model,
                epoch=epoch,
                fold=fold,
                metrics=constrained,
                baseline=baseline,
                constraints=True,
                expert_validation_loss=evaluation["expert_loss"],
                config=config,
                foundation_path=foundation_path,
                foundation_sha256=foundation_sha256,
                cache_paths={"train": train_cache_path, "val": val_cache_path},
            )
            torch.save(best_payload, best_path)

        if evaluation["expert_loss"] < best_expert_loss - 1e-4:
            best_expert_loss = evaluation["expert_loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1

        model.set_guidance_mix_alpha(metrics["mix_alpha"])
        payload = checkpoint_payload(
            model,
            epoch=epoch,
            fold=fold,
            metrics=metrics,
            baseline=baseline,
            constraints=meets_constraints,
            expert_validation_loss=evaluation["expert_loss"],
            config=config,
            foundation_path=foundation_path,
            foundation_sha256=foundation_sha256,
            cache_paths={"train": train_cache_path, "val": val_cache_path},
        )
        if epoch % int(getattr(config, "SAVE_LATEST_EVERY", 5)) == 0:
            torch.save(payload, latest_path)
        print(
            f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
            f"train_loss={train_loss / max(train_batches, 1):.4f} | "
            f"expert_val_loss={evaluation['expert_loss']:.4f} | "
            f"alpha={metrics['mix_alpha']:.2f} | "
            f"val_acc={metrics['accuracy']:.4f} | "
            f"val_f1={metrics['macro_f1']:.4f} | "
            f"val_meta_precision={metrics['metastasis_precision']:.4f} | "
            f"val_meta_recall={metrics['metastasis_recall']:.4f} | "
            f"val_meta_f2={metrics['metastasis_fbeta']:.4f} | "
            f"recovered={metrics['recovered_metastasis']} | "
            f"lost={metrics['lost_metastasis']} | "
            f"new_infl_to_meta={metrics['new_inflammation_to_metastasis']} | "
            f"constraints={'yes' if meets_constraints else 'no'}"
        )
        if (
            epoch >= int(config.MIN_EPOCHS)
            and stale_epochs >= int(config.PATIENCE)
        ):
            print(
                f"Early stopping after epoch {epoch}; "
                f"best constrained epoch={best_epoch}"
            )
            break
        if improved:
            print(f"Saved constrained best checkpoint: {best_path}")

    if payload is not None:
        torch.save(payload, latest_path)
    if best_epoch is None:
        print(
            "No epoch met all Foundation-preservation constraints. "
            f"Diagnostic checkpoint: {unconstrained_path}"
        )
        create_foundation_fallback_checkpoint(
            unconstrained_path,
            best_path,
            expected_fold=fold,
        )
    else:
        print(f"Best constrained checkpoint: {best_path} (epoch {best_epoch})")


def main(args):
    config = load_python_config(args.config, CONFIG_FIELDS)
    set_seed(SEED)
    if args.fold < 1 or args.fold > int(K_FOLDS):
        raise ValueError(f"fold must lie in [1, {K_FOLDS}]")
    if args.finalize_existing_fallback:
        checkpoint_dir = (
            resolve_output_artifact_dir(args.output_root, "checkpoints")
            / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}"
            / MODEL_NAME
        )
        create_foundation_fallback_checkpoint(
            checkpoint_dir / f"fold{args.fold}_model_best_unconstrained.pth",
            checkpoint_dir / f"fold{args.fold}_model_best.pth",
            expected_fold=args.fold,
        )
        return

    device = torch.device(config.DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    foundation_path = foundation_checkpoint_path(
        args.foundation_checkpoint_root,
        args.fold,
    )
    foundation_sha256 = sha256_file(foundation_path)
    model, foundation_checkpoint = load_foundation_model(
        foundation_path,
        config,
        device,
    )
    train_dataset = load_split_dataset(args.data_root, args.fold, "train")
    val_dataset = load_split_dataset(args.data_root, args.fold, "val")
    outer_test_folds = build_outer_test_fold_map(args.data_root)
    train_source_folds = source_folds_for_split(
        train_dataset,
        "train",
        args.fold,
        outer_test_folds,
    )
    val_source_folds = source_folds_for_split(
        val_dataset,
        "val",
        args.fold,
        outer_test_folds,
    )
    required_source_folds = sorted(set(train_source_folds) | set(val_source_folds))
    nnunet_paths = {
        fold: native_nnunet_checkpoint_path(
            args.nnunet_results_root,
            config,
            fold,
        )
        for fold in required_source_folds
    }
    nnunet_sha256 = {
        fold: sha256_file(path) for fold, path in nnunet_paths.items()
    }

    cache_dir = Path(args.output_root).expanduser().resolve() / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        "OOF nnU-Net source distribution (train): "
        + ", ".join(
            f"fold{fold}={train_source_folds.count(fold)}"
            for fold in sorted(set(train_source_folds))
        )
    )
    print(f"OOF nnU-Net source (validation): fold{args.fold}")
    train_cache_path = extract_feature_cache(
        split="train",
        dataset=train_dataset,
        source_folds=train_source_folds,
        target_fold=args.fold,
        model=model,
        foundation_path=foundation_path,
        foundation_sha256=foundation_sha256,
        nnunet_paths=nnunet_paths,
        nnunet_sha256=nnunet_sha256,
        cache_dir=cache_dir,
        config=config,
        device=device,
        rebuild=args.rebuild_cache,
    )
    val_cache_path = extract_feature_cache(
        split="val",
        dataset=val_dataset,
        source_folds=val_source_folds,
        target_fold=args.fold,
        model=model,
        foundation_path=foundation_path,
        foundation_sha256=foundation_sha256,
        nnunet_paths=nnunet_paths,
        nnunet_sha256=nnunet_sha256,
        cache_dir=cache_dir,
        config=config,
        device=device,
        rebuild=args.rebuild_cache,
    )
    if args.cache_only:
        print("Feature caches completed; --cache-only requested, stopping before training")
        return
    print(
        f"Foundation source epoch: {foundation_checkpoint.get('epoch', 'unknown')}"
    )
    train_guidance_expert(
        model=model,
        train_cache_path=train_cache_path,
        val_cache_path=val_cache_path,
        output_root=args.output_root,
        fold=args.fold,
        config=config,
        device=device,
        foundation_path=foundation_path,
        foundation_sha256=foundation_sha256,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--foundation-checkpoint-root", required=True)
    parser.add_argument("--nnunet-results-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--finalize-existing-fallback",
        action="store_true",
        help=(
            "Create an alpha=0 formal checkpoint from an existing unconstrained "
            "diagnostic without rerunning feature extraction or training"
        ),
    )
    main(parser.parse_args())
