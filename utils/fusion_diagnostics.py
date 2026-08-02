"""晚期融合逐病例概率、校准和投票影响诊断。"""

import csv
import json
from pathlib import Path

import numpy as np

from configs.global_config import ALL_SEQUENCES, CLASS_NAMES, NUM_CLASSES


CALIBRATION_BIN_COUNT = 10
CONFIDENCE_THRESHOLDS = (0.90, 0.95, 0.99)
PROBABILITY_EPS = 1e-12


def _source_names():
    return [
        f"seq{seq_idx}_{seq_name.lower()}"
        for seq_idx, seq_name in enumerate(ALL_SEQUENCES, start=1)
    ]


def _validate_inputs(case_ids, labels, base_probabilities, fusion_probabilities):
    case_ids = np.asarray(case_ids, dtype=str)
    labels = np.asarray(labels, dtype=np.int64)
    base_probabilities = np.asarray(base_probabilities, dtype=np.float64)
    fusion_probabilities = np.asarray(fusion_probabilities, dtype=np.float64)

    expected_base_shape = (
        len(labels),
        len(ALL_SEQUENCES),
        NUM_CLASSES,
    )
    expected_fusion_shape = (len(labels), NUM_CLASSES)
    if len(case_ids) != len(labels):
        raise ValueError("case_ids and labels must contain the same number of cases")
    if base_probabilities.shape != expected_base_shape:
        raise ValueError(
            f"Expected base probabilities with shape {expected_base_shape}, "
            f"got {base_probabilities.shape}"
        )
    if fusion_probabilities.shape != expected_fusion_shape:
        raise ValueError(
            f"Expected fusion probabilities with shape {expected_fusion_shape}, "
            f"got {fusion_probabilities.shape}"
        )
    if np.any(labels < 0) or np.any(labels >= NUM_CLASSES):
        raise ValueError("labels contain an unknown class index")
    if not np.all(np.isfinite(base_probabilities)):
        raise ValueError("base probabilities contain non-finite values")
    if not np.all(np.isfinite(fusion_probabilities)):
        raise ValueError("fusion probabilities contain non-finite values")

    base_sums = base_probabilities.sum(axis=2)
    fusion_sums = fusion_probabilities.sum(axis=1)
    if not np.allclose(base_sums, 1.0, atol=1e-4):
        raise ValueError("base probabilities do not sum to one")
    if not np.allclose(fusion_sums, 1.0, atol=1e-4):
        raise ValueError("fusion probabilities do not sum to one")

    return case_ids, labels, base_probabilities, fusion_probabilities


def _safe_mean(values):
    values = np.asarray(values)
    return float(values.mean()) if len(values) else None


def _threshold_suffix(threshold):
    return f"{int(round(threshold * 100)):02d}"


def probability_metrics(labels, probabilities):
    """计算多分类概率的判别、校准和高置信错误指标。"""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(labels) == 0:
        return None

    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    true_probabilities = probabilities[np.arange(len(labels)), labels]
    one_hot = np.eye(NUM_CLASSES, dtype=np.float64)[labels]

    metrics = {
        "sample_count": int(len(labels)),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidences.mean()),
        "overconfidence_gap": float(confidences.mean() - correct.mean()),
        "mean_true_class_probability": float(true_probabilities.mean()),
        "negative_log_likelihood": float(
            -np.log(np.clip(true_probabilities, PROBABILITY_EPS, 1.0)).mean()
        ),
        "multiclass_brier": float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        ),
        "mean_confidence_correct": _safe_mean(confidences[correct]),
        "mean_confidence_wrong": _safe_mean(confidences[~correct]),
    }

    ece = 0.0
    maximum_calibration_gap = 0.0
    for calibration_bin in calibration_bins(labels, probabilities):
        weight = calibration_bin["sample_count"] / len(labels)
        gap = abs(calibration_bin["confidence_minus_accuracy"])
        ece += weight * gap
        maximum_calibration_gap = max(maximum_calibration_gap, gap)
    metrics["expected_calibration_error"] = float(ece)
    metrics["maximum_calibration_gap"] = float(maximum_calibration_gap)

    total_errors = int((~correct).sum())
    for threshold in CONFIDENCE_THRESHOLDS:
        selected = confidences >= threshold
        selected_errors = selected & ~correct
        suffix = _threshold_suffix(threshold)
        metrics[f"samples_conf_ge_{suffix}"] = int(selected.sum())
        metrics[f"errors_conf_ge_{suffix}"] = int(selected_errors.sum())
        metrics[f"error_rate_conf_ge_{suffix}"] = (
            float(selected_errors.sum() / selected.sum())
            if selected.any()
            else None
        )
        metrics[f"share_of_errors_conf_ge_{suffix}"] = (
            float(selected_errors.sum() / total_errors)
            if total_errors > 0
            else 0.0
        )
    return metrics


def calibration_bins(labels, probabilities, bin_count=CALIBRATION_BIN_COUNT):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    bin_indices = np.minimum(
        (confidences * bin_count).astype(np.int64),
        bin_count - 1,
    )

    rows = []
    for bin_idx in range(bin_count):
        selected = bin_indices == bin_idx
        if not selected.any():
            continue
        accuracy = float(correct[selected].mean())
        mean_confidence = float(confidences[selected].mean())
        rows.append(
            {
                "bin_index": bin_idx + 1,
                "confidence_lower": bin_idx / bin_count,
                "confidence_upper": (bin_idx + 1) / bin_count,
                "sample_count": int(selected.sum()),
                "correct_count": int(correct[selected].sum()),
                "error_count": int((~correct[selected]).sum()),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "confidence_minus_accuracy": mean_confidence - accuracy,
            }
        )
    return rows


def _case_diagnostics(labels, base_probabilities, fusion_probabilities):
    base_predictions = base_probabilities.argmax(axis=2)
    base_confidences = base_probabilities.max(axis=2)
    base_correct = base_predictions == labels[:, None]
    individual_correct_count = base_correct.sum(axis=1)

    fusion_predictions = fusion_probabilities.argmax(axis=1)
    fusion_confidences = fusion_probabilities.max(axis=1)
    fusion_correct = fusion_predictions == labels

    vote_counts = np.eye(NUM_CLASSES, dtype=np.int64)[base_predictions].sum(axis=1)
    maximum_vote_count = vote_counts.max(axis=1)
    hard_vote_tie = (vote_counts == maximum_vote_count[:, None]).sum(axis=1) > 1
    hard_vote_predictions = vote_counts.argmax(axis=1)
    hard_vote_correct = hard_vote_predictions == labels

    wrong_base = ~base_correct
    high_confidence_wrong_base = wrong_base & (base_confidences >= 0.90)
    max_wrong_confidence = np.where(wrong_base, base_confidences, -np.inf).max(axis=1)
    max_correct_confidence = np.where(base_correct, base_confidences, -np.inf).max(axis=1)
    has_correct_and_wrong_base = (
        (individual_correct_count > 0)
        & (individual_correct_count < len(ALL_SEQUENCES))
    )

    return {
        "base_predictions": base_predictions,
        "base_confidences": base_confidences,
        "base_correct": base_correct,
        "individual_correct_count": individual_correct_count,
        "fusion_predictions": fusion_predictions,
        "fusion_confidences": fusion_confidences,
        "fusion_correct": fusion_correct,
        "hard_vote_tie": hard_vote_tie,
        "hard_vote_predictions": hard_vote_predictions,
        "hard_vote_correct": hard_vote_correct,
        "fusion_corrected_any_individual_error": (
            fusion_correct & (individual_correct_count < len(ALL_SEQUENCES))
        ),
        "fusion_rescued_zero_or_one_correct": (
            fusion_correct & (individual_correct_count <= 1)
        ),
        "fusion_lost_available_correct": (
            ~fusion_correct & (individual_correct_count >= 1)
        ),
        "fusion_harmed_correct_majority": (
            ~fusion_correct & (individual_correct_count >= 2)
        ),
        "soft_correct_hard_wrong": (
            ~hard_vote_tie & fusion_correct & ~hard_vote_correct
        ),
        "soft_wrong_hard_correct": (
            ~hard_vote_tie & ~fusion_correct & hard_vote_correct
        ),
        "soft_resolved_hard_tie_correctly": hard_vote_tie & fusion_correct,
        "soft_resolved_hard_tie_incorrectly": hard_vote_tie & ~fusion_correct,
        "high_confidence_wrong_base_present": high_confidence_wrong_base.any(axis=1),
        "high_confidence_wrong_base_harmed": (
            ~fusion_correct
            & (individual_correct_count >= 1)
            & high_confidence_wrong_base.any(axis=1)
        ),
        "wrong_confidence_exceeds_correct": (
            has_correct_and_wrong_base
            & (max_wrong_confidence > max_correct_confidence)
        ),
        "max_wrong_base_confidence": max_wrong_confidence,
        "max_correct_base_confidence": max_correct_confidence,
    }


def fusion_effects(labels, base_probabilities, fusion_probabilities):
    diagnostics = _case_diagnostics(
        labels,
        base_probabilities,
        fusion_probabilities,
    )
    base_correct = diagnostics["base_correct"]
    fusion_correct = diagnostics["fusion_correct"]
    correct_count = diagnostics["individual_correct_count"]
    sample_count = len(labels)

    effects = {
        "sample_count": int(sample_count),
        "fusion_correct_count": int(fusion_correct.sum()),
        "fusion_wrong_count": int((~fusion_correct).sum()),
        "fusion_accuracy": float(fusion_correct.mean()),
        "oracle_any_model_correct_count": int((correct_count >= 1).sum()),
        "oracle_any_model_accuracy": float((correct_count >= 1).mean()),
        "all_models_correct_count": int((correct_count == len(ALL_SEQUENCES)).sum()),
        "all_models_wrong_count": int((correct_count == 0).sum()),
        "fusion_corrected_any_individual_error_count": int(
            diagnostics["fusion_corrected_any_individual_error"].sum()
        ),
        "fusion_rescued_zero_or_one_correct_count": int(
            diagnostics["fusion_rescued_zero_or_one_correct"].sum()
        ),
        "fusion_lost_available_correct_count": int(
            diagnostics["fusion_lost_available_correct"].sum()
        ),
        "fusion_harmed_correct_majority_count": int(
            diagnostics["fusion_harmed_correct_majority"].sum()
        ),
        "hard_vote_tie_count": int(diagnostics["hard_vote_tie"].sum()),
        "soft_correct_hard_wrong_count": int(
            diagnostics["soft_correct_hard_wrong"].sum()
        ),
        "soft_wrong_hard_correct_count": int(
            diagnostics["soft_wrong_hard_correct"].sum()
        ),
        "soft_resolved_hard_tie_correctly_count": int(
            diagnostics["soft_resolved_hard_tie_correctly"].sum()
        ),
        "soft_resolved_hard_tie_incorrectly_count": int(
            diagnostics["soft_resolved_hard_tie_incorrectly"].sum()
        ),
        "fusion_wrong_with_high_confidence_wrong_base_count": int(
            diagnostics["high_confidence_wrong_base_harmed"].sum()
        ),
        "mixed_correctness_wrong_confidence_exceeds_correct_count": int(
            diagnostics["wrong_confidence_exceeds_correct"].sum()
        ),
    }

    for correct_model_count in range(len(ALL_SEQUENCES) + 1):
        selected = correct_count == correct_model_count
        selected_count = int(selected.sum())
        selected_fusion_correct = int((selected & fusion_correct).sum())
        effects[f"cases_with_{correct_model_count}_correct_models_count"] = (
            selected_count
        )
        effects[
            f"fusion_correct_when_{correct_model_count}_models_correct_count"
        ] = selected_fusion_correct
        effects[
            f"fusion_accuracy_when_{correct_model_count}_models_correct"
        ] = (
            float(selected_fusion_correct / selected_count)
            if selected_count > 0
            else None
        )

    rescue_candidates = correct_count <= 1
    majority_correct = correct_count >= 2
    available_correct = correct_count >= 1
    effects["fusion_rescue_rate_when_zero_or_one_correct"] = (
        float(
            diagnostics["fusion_rescued_zero_or_one_correct"].sum()
            / rescue_candidates.sum()
        )
        if rescue_candidates.any()
        else None
    )
    effects["fusion_harm_rate_when_majority_correct"] = (
        float(
            diagnostics["fusion_harmed_correct_majority"].sum()
            / majority_correct.sum()
        )
        if majority_correct.any()
        else None
    )
    effects["fusion_loss_rate_when_any_model_correct"] = (
        float(
            diagnostics["fusion_lost_available_correct"].sum()
            / available_correct.sum()
        )
        if available_correct.any()
        else None
    )
    mixed_correctness = (
        (correct_count > 0)
        & (correct_count < len(ALL_SEQUENCES))
    )
    effects["fusion_wrong_high_confidence_wrong_base_rate"] = (
        float(
            diagnostics["high_confidence_wrong_base_harmed"].sum()
            / (~fusion_correct).sum()
        )
        if (~fusion_correct).any()
        else 0.0
    )
    effects[
        "fusion_wrong_high_confidence_wrong_base_rate_when_any_model_correct"
    ] = (
        float(
            diagnostics["high_confidence_wrong_base_harmed"].sum()
            / diagnostics["fusion_lost_available_correct"].sum()
        )
        if diagnostics["fusion_lost_available_correct"].any()
        else 0.0
    )
    effects["wrong_confidence_exceeds_correct_rate_on_mixed_cases"] = (
        float(
            diagnostics["wrong_confidence_exceeds_correct"].sum()
            / mixed_correctness.sum()
        )
        if mixed_correctness.any()
        else None
    )

    for seq_idx, source_name in enumerate(_source_names()):
        base_errors = ~base_correct[:, seq_idx]
        base_successes = base_correct[:, seq_idx]
        corrected = base_errors & fusion_correct
        harmed = base_successes & ~fusion_correct
        effects[f"{source_name}_accuracy"] = float(
            base_correct[:, seq_idx].mean()
        )
        effects[f"fusion_corrected_{source_name}_errors_count"] = int(
            corrected.sum()
        )
        effects[f"fusion_corrected_{source_name}_errors_rate"] = (
            float(corrected.sum() / base_errors.sum())
            if base_errors.any()
            else None
        )
        effects[f"fusion_harmed_{source_name}_correct_count"] = int(
            harmed.sum()
        )
        effects[f"fusion_harmed_{source_name}_correct_rate"] = (
            float(harmed.sum() / base_successes.sum())
            if base_successes.any()
            else None
        )
    return effects


def _prediction_rows(
    scope,
    case_ids,
    labels,
    base_probabilities,
    fusion_probabilities,
    model_names,
):
    diagnostics = _case_diagnostics(
        labels,
        base_probabilities,
        fusion_probabilities,
    )
    source_names = _source_names()
    rows = []
    for case_idx, case_id in enumerate(case_ids):
        label = int(labels[case_idx])
        fusion_pred = int(diagnostics["fusion_predictions"][case_idx])
        row = {
            "scope": scope,
            "case_id": case_id,
            "label": label,
            "label_name": CLASS_NAMES[label],
        }
        for seq_idx, source_name in enumerate(source_names):
            prediction = int(diagnostics["base_predictions"][case_idx, seq_idx])
            row[f"{source_name}_model"] = model_names[seq_idx]
            for class_idx, class_name in enumerate(CLASS_NAMES):
                row[f"{source_name}_prob_{class_name}"] = float(
                    base_probabilities[case_idx, seq_idx, class_idx]
                )
            row[f"{source_name}_pred"] = prediction
            row[f"{source_name}_pred_name"] = CLASS_NAMES[prediction]
            row[f"{source_name}_confidence"] = float(
                diagnostics["base_confidences"][case_idx, seq_idx]
            )
            row[f"{source_name}_true_probability"] = float(
                base_probabilities[case_idx, seq_idx, label]
            )
            row[f"{source_name}_correct"] = int(
                diagnostics["base_correct"][case_idx, seq_idx]
            )
            row[f"{source_name}_fusion_pred_vs_true_margin"] = float(
                base_probabilities[case_idx, seq_idx, fusion_pred]
                - base_probabilities[case_idx, seq_idx, label]
            )

        for class_idx, class_name in enumerate(CLASS_NAMES):
            row[f"soft_vote_prob_{class_name}"] = float(
                fusion_probabilities[case_idx, class_idx]
            )
        row.update(
            {
                "soft_vote_pred": fusion_pred,
                "soft_vote_pred_name": CLASS_NAMES[fusion_pred],
                "soft_vote_confidence": float(
                    diagnostics["fusion_confidences"][case_idx]
                ),
                "soft_vote_true_probability": float(
                    fusion_probabilities[case_idx, label]
                ),
                "soft_vote_correct": int(
                    diagnostics["fusion_correct"][case_idx]
                ),
                "individual_correct_count": int(
                    diagnostics["individual_correct_count"][case_idx]
                ),
                "fusion_corrected_any_individual_error": int(
                    diagnostics["fusion_corrected_any_individual_error"][case_idx]
                ),
                "fusion_rescued_zero_or_one_correct": int(
                    diagnostics["fusion_rescued_zero_or_one_correct"][case_idx]
                ),
                "fusion_lost_available_correct": int(
                    diagnostics["fusion_lost_available_correct"][case_idx]
                ),
                "fusion_harmed_correct_majority": int(
                    diagnostics["fusion_harmed_correct_majority"][case_idx]
                ),
                "hard_vote_tie": int(
                    diagnostics["hard_vote_tie"][case_idx]
                ),
                "hard_vote_pred": (
                    ""
                    if diagnostics["hard_vote_tie"][case_idx]
                    else int(diagnostics["hard_vote_predictions"][case_idx])
                ),
                "hard_vote_correct": (
                    ""
                    if diagnostics["hard_vote_tie"][case_idx]
                    else int(diagnostics["hard_vote_correct"][case_idx])
                ),
                "soft_correct_hard_wrong": int(
                    diagnostics["soft_correct_hard_wrong"][case_idx]
                ),
                "soft_wrong_hard_correct": int(
                    diagnostics["soft_wrong_hard_correct"][case_idx]
                ),
                "high_confidence_wrong_base_present": int(
                    diagnostics["high_confidence_wrong_base_present"][case_idx]
                ),
                "high_confidence_wrong_base_harmed": int(
                    diagnostics["high_confidence_wrong_base_harmed"][case_idx]
                ),
                "wrong_confidence_exceeds_correct": int(
                    diagnostics["wrong_confidence_exceeds_correct"][case_idx]
                ),
                "max_wrong_base_confidence": (
                    ""
                    if not np.isfinite(
                        diagnostics["max_wrong_base_confidence"][case_idx]
                    )
                    else float(
                        diagnostics["max_wrong_base_confidence"][case_idx]
                    )
                ),
                "max_correct_base_confidence": (
                    ""
                    if not np.isfinite(
                        diagnostics["max_correct_base_confidence"][case_idx]
                    )
                    else float(
                        diagnostics["max_correct_base_confidence"][case_idx]
                    )
                ),
            }
        )
        rows.append(row)
    return rows


def _metrics_rows(scope, labels, base_probabilities, fusion_probabilities, model_names):
    source_names = _source_names() + ["soft_vote"]
    source_models = list(model_names) + ["equal_probability_average"]
    probabilities = [
        base_probabilities[:, seq_idx, :]
        for seq_idx in range(len(ALL_SEQUENCES))
    ] + [fusion_probabilities]

    rows = []
    for source_name, model_name, source_probabilities in zip(
        source_names,
        source_models,
        probabilities,
    ):
        populations = [("all", np.ones(len(labels), dtype=bool))]
        populations.extend(
            (
                f"true_{class_name}",
                labels == class_idx,
            )
            for class_idx, class_name in enumerate(CLASS_NAMES)
        )
        for population_name, selected in populations:
            metrics = probability_metrics(
                labels[selected],
                source_probabilities[selected],
            )
            if metrics is not None:
                rows.append(
                    {
                        "scope": scope,
                        "population": population_name,
                        "source": source_name,
                        "model": model_name,
                        **metrics,
                    }
                )
    return rows


def _calibration_rows(scope, labels, base_probabilities, fusion_probabilities):
    source_names = _source_names() + ["soft_vote"]
    probabilities = [
        base_probabilities[:, seq_idx, :]
        for seq_idx in range(len(ALL_SEQUENCES))
    ] + [fusion_probabilities]
    rows = []
    for source_name, source_probabilities in zip(source_names, probabilities):
        for row in calibration_bins(labels, source_probabilities):
            rows.append({"scope": scope, "source": source_name, **row})
    return rows


def _write_csv(output_path, rows):
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(output_path, value):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def save_probability_diagnostics(
    report_root,
    case_ids,
    labels,
    base_probabilities,
    fusion_probabilities,
    model_names,
    fold_idx=None,
):
    """保存一个 fold 或 pooled 五折的逐病例与校准诊断。"""
    (
        case_ids,
        labels,
        base_probabilities,
        fusion_probabilities,
    ) = _validate_inputs(
        case_ids,
        labels,
        base_probabilities,
        fusion_probabilities,
    )
    if len(model_names) != len(ALL_SEQUENCES):
        raise ValueError("model_names must contain one model name per sequence")

    report_root = Path(report_root).expanduser().resolve()
    scope = f"fold{fold_idx}" if fold_idx is not None else "pooled"
    predictions_path = report_root / f"{scope}_predictions.csv"
    metrics_path = report_root / f"{scope}_model_metrics.csv"
    calibration_path = report_root / f"{scope}_calibration.csv"
    summary_path = report_root / f"{scope}_summary.json"

    prediction_rows = _prediction_rows(
        scope,
        case_ids,
        labels,
        base_probabilities,
        fusion_probabilities,
        model_names,
    )
    metrics_rows = _metrics_rows(
        scope,
        labels,
        base_probabilities,
        fusion_probabilities,
        model_names,
    )
    calibration_rows_for_scope = _calibration_rows(
        scope,
        labels,
        base_probabilities,
        fusion_probabilities,
    )
    effects = fusion_effects(
        labels,
        base_probabilities,
        fusion_probabilities,
    )
    summary = {
        "scope": scope,
        "model_names": list(model_names),
        "fusion": "equal_probability_average",
        "fusion_effects": effects,
        "all_case_probability_metrics": {
            row["source"]: {
                key: value
                for key, value in row.items()
                if key not in {"scope", "population", "source", "model"}
            }
            for row in metrics_rows
            if row["population"] == "all"
        },
    }

    _write_csv(predictions_path, prediction_rows)
    _write_csv(metrics_path, metrics_rows)
    _write_csv(calibration_path, calibration_rows_for_scope)
    _write_json(summary_path, summary)
    return {
        "scope": scope,
        "predictions_path": predictions_path,
        "metrics_path": metrics_path,
        "calibration_path": calibration_path,
        "summary_path": summary_path,
        "summary": summary,
    }


def write_probability_diagnostics_readme(report_root):
    report_root = Path(report_root).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    readme_path = report_root / "README.md"
    readme_path.write_text(
        """# Fusion probability diagnostics

- `foldN_predictions.csv`: 每个病例的三个单序列概率、等权融合概率、预测和纠错标记。
- `foldN_model_metrics.csv`: 单模型和融合的 accuracy、置信度、NLL、Brier、ECE 与高置信错误。
- `foldN_calibration.csv`: 10 个置信度分箱的准确率与平均置信度。
- `foldN_summary.json`: 当前 fold 的融合效果与校准摘要。
- `pooled_*`: 五个互斥 test fold 合并后的同类诊断。

若某个序列模型带 subtype head，表中的三分类概率是用于融合的联合概率：
`P(normal)` 保持主头输出，两个异常类分别为
`P(abnormal) * P(subtype | abnormal)`；不是原三分类主头的原始概率。

关键标记：

- `fusion_rescued_zero_or_one_correct`: 三个单模型最多只有一个预测正确，但软投票最终正确。
- `fusion_harmed_correct_majority`: 至少两个单模型预测正确，但软投票最终错误。
- `soft_wrong_hard_correct`: 硬多数投票正确、软投票错误，说明概率幅度在该病例上产生负面作用。
- `soft_correct_hard_wrong`: 硬多数投票错误、软投票正确，说明概率幅度在该病例上产生正面作用。
- `high_confidence_wrong_base_harmed`: 至少存在一个正确单模型，但包含置信度不低于 0.90 的错误单模型，且软投票错误。

ECE 和 `overconfidence_gap` 只用于诊断校准，不应在 test 标签上拟合温度或阈值。
""",
        encoding="utf-8",
    )
    return readme_path
