"""验证集温度拟合与多分类概率缩放。"""

import math

import numpy as np
import torch


PROBABILITY_EPS = 1e-12
DEFAULT_TEMPERATURE_BOUNDS = (0.05, 20.0)


def _validate_temperature(temperature):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    return temperature


def temperature_scale_probabilities(probabilities, temperature):
    """
    对概率做标量温度缩放，同时保持每个样本的类别排序不变。

    对普通 softmax 输出，这与 logits / T 完全等价；对层级模型的联合
    三分类概率，则把 log(probability) 作为其规范化 logits。
    """
    temperature = _validate_temperature(temperature)
    if torch.is_tensor(probabilities):
        if probabilities.ndim != 2:
            raise ValueError("probabilities must have shape [N, C]")
        log_probabilities = probabilities.clamp_min(PROBABILITY_EPS).log()
        return torch.softmax(log_probabilities / temperature, dim=1)

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, C]")
    log_probabilities = np.log(
        np.clip(probabilities, PROBABILITY_EPS, None)
    )
    scaled_logits = log_probabilities / temperature
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    scaled_probabilities = np.exp(scaled_logits)
    return scaled_probabilities / scaled_probabilities.sum(
        axis=1,
        keepdims=True,
    )


def multiclass_nll(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("labels and probabilities have incompatible shapes")
    if len(labels) == 0:
        raise ValueError("temperature fitting requires at least one sample")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels contain an unknown class index")
    true_probabilities = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(true_probabilities, PROBABILITY_EPS, 1.0)).mean())


def fit_temperature(
    probabilities,
    labels,
    bounds=DEFAULT_TEMPERATURE_BOUNDS,
    iterations=100,
):
    """用一维有界搜索最小化验证集多分类 NLL。"""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("labels and probabilities have incompatible shapes")
    if len(labels) == 0:
        raise ValueError("temperature fitting requires at least one sample")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain non-finite values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("probabilities do not sum to one")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("labels contain an unknown class index")

    lower, upper = (float(value) for value in bounds)
    if lower <= 0 or upper <= lower:
        raise ValueError("temperature bounds must satisfy 0 < lower < upper")

    log_probabilities = np.log(
        np.clip(probabilities, PROBABILITY_EPS, None)
    )

    def objective(log_temperature):
        temperature = math.exp(log_temperature)
        scaled_logits = log_probabilities / temperature
        max_logits = scaled_logits.max(axis=1, keepdims=True)
        log_normalizer = (
            max_logits[:, 0]
            + np.log(np.exp(scaled_logits - max_logits).sum(axis=1))
        )
        true_logits = scaled_logits[np.arange(len(labels)), labels]
        return float((log_normalizer - true_logits).mean())

    left = math.log(lower)
    right = math.log(upper)
    inverse_golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    point_left = right - inverse_golden_ratio * (right - left)
    point_right = left + inverse_golden_ratio * (right - left)
    value_left = objective(point_left)
    value_right = objective(point_right)

    for _ in range(iterations):
        if value_left <= value_right:
            right = point_right
            point_right = point_left
            value_right = value_left
            point_left = right - inverse_golden_ratio * (right - left)
            value_left = objective(point_left)
        else:
            left = point_left
            point_left = point_right
            value_left = value_right
            point_right = left + inverse_golden_ratio * (right - left)
            value_right = objective(point_right)

    temperature = math.exp((left + right) / 2.0)
    scaled_probabilities = temperature_scale_probabilities(
        probabilities,
        temperature,
    )
    predictions_before = probabilities.argmax(axis=1)
    predictions_after = scaled_probabilities.argmax(axis=1)
    if not np.array_equal(predictions_before, predictions_after):
        raise RuntimeError("scalar temperature scaling changed class ordering")

    return {
        "temperature": float(temperature),
        "sample_count": int(len(labels)),
        "nll_before": multiclass_nll(labels, probabilities),
        "nll_after": multiclass_nll(labels, scaled_probabilities),
        "accuracy_before": float((predictions_before == labels).mean()),
        "accuracy_after": float((predictions_after == labels).mean()),
        "mean_confidence_before": float(probabilities.max(axis=1).mean()),
        "mean_confidence_after": float(
            scaled_probabilities.max(axis=1).mean()
        ),
        "lower_bound": lower,
        "upper_bound": upper,
    }


def fit_fusion_temperatures(
    base_probabilities,
    labels,
    initial_temperatures=None,
    bounds=DEFAULT_TEMPERATURE_BOUNDS,
    max_iterations=250,
):
    """联合拟合各序列温度，直接最小化等权融合概率的验证集 NLL。"""
    base_probabilities = np.asarray(base_probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if base_probabilities.ndim != 3:
        raise ValueError("base_probabilities must have shape [N, S, C]")
    if len(base_probabilities) != len(labels) or len(labels) == 0:
        raise ValueError("labels and base_probabilities have incompatible shapes")
    if not np.all(np.isfinite(base_probabilities)):
        raise ValueError("base_probabilities contain non-finite values")
    if not np.allclose(base_probabilities.sum(axis=2), 1.0, atol=1e-4):
        raise ValueError("base probabilities do not sum to one")
    if np.any(labels < 0) or np.any(labels >= base_probabilities.shape[2]):
        raise ValueError("labels contain an unknown class index")

    lower, upper = (float(value) for value in bounds)
    if lower <= 0 or upper <= lower:
        raise ValueError("temperature bounds must satisfy 0 < lower < upper")

    sequence_count = base_probabilities.shape[1]
    if initial_temperatures is None:
        initial_temperatures = np.ones(sequence_count, dtype=np.float64)
    initial_temperatures = np.asarray(
        initial_temperatures,
        dtype=np.float64,
    )
    if initial_temperatures.shape != (sequence_count,):
        raise ValueError(
            f"initial_temperatures must have shape ({sequence_count},)"
        )
    if not np.all(np.isfinite(initial_temperatures)) or np.any(
        initial_temperatures <= 0
    ):
        raise ValueError("initial_temperatures must be finite and positive")
    initial_temperatures = np.clip(initial_temperatures, lower, upper)

    probability_tensor = torch.as_tensor(
        base_probabilities,
        dtype=torch.float64,
    )
    log_probability_tensor = probability_tensor.clamp_min(
        PROBABILITY_EPS
    ).log()
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    row_indices = torch.arange(len(labels))

    def fusion_nll_tensor(temperatures):
        scaled_probabilities = torch.softmax(
            log_probability_tensor / temperatures.view(1, -1, 1),
            dim=2,
        )
        fusion_probabilities = scaled_probabilities.mean(dim=1)
        true_probabilities = fusion_probabilities[row_indices, label_tensor]
        return -true_probabilities.clamp_min(PROBABILITY_EPS).log().mean()

    def temperatures_to_raw(temperatures):
        fraction = (temperatures - lower) / (upper - lower)
        fraction = np.clip(fraction, 1e-8, 1.0 - 1e-8)
        return np.log(fraction / (1.0 - fraction))

    def raw_to_temperatures(raw_temperatures):
        return lower + (upper - lower) * torch.sigmoid(raw_temperatures)

    start_points = [
        np.ones(sequence_count, dtype=np.float64),
        initial_temperatures,
        np.sqrt(initial_temperatures),
    ]
    unique_start_points = []
    for start_point in start_points:
        start_point = np.clip(start_point, lower, upper)
        if not any(
            np.allclose(start_point, existing, atol=1e-10)
            for existing in unique_start_points
        ):
            unique_start_points.append(start_point)

    candidates = []
    optimization_runs = []
    for start_point in unique_start_points:
        start_tensor = torch.as_tensor(start_point, dtype=torch.float64)
        start_nll = float(fusion_nll_tensor(start_tensor).item())
        candidates.append((start_nll, start_point.copy()))

        raw_temperatures = torch.tensor(
            temperatures_to_raw(start_point),
            dtype=torch.float64,
            requires_grad=True,
        )
        optimizer = torch.optim.LBFGS(
            [raw_temperatures],
            lr=0.5,
            max_iter=max_iterations,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            temperatures = raw_to_temperatures(raw_temperatures)
            loss = fusion_nll_tensor(temperatures)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            fitted_temperatures = raw_to_temperatures(
                raw_temperatures
            ).cpu().numpy()
            fitted_nll = float(
                fusion_nll_tensor(
                    torch.as_tensor(
                        fitted_temperatures,
                        dtype=torch.float64,
                    )
                ).item()
            )
        candidates.append((fitted_nll, fitted_temperatures.copy()))
        optimizer_state = optimizer.state[raw_temperatures]
        optimization_runs.append(
            {
                "start_temperatures": start_point.tolist(),
                "start_nll": start_nll,
                "fitted_temperatures": fitted_temperatures.tolist(),
                "fitted_nll": fitted_nll,
                "iterations": int(optimizer_state.get("n_iter", 0)),
                "function_evaluations": int(
                    optimizer_state.get("func_evals", 0)
                ),
            }
        )

    fitted_nll, fitted_temperatures = min(
        candidates,
        key=lambda candidate: candidate[0],
    )
    calibrated_base_probabilities = np.stack(
        [
            temperature_scale_probabilities(
                base_probabilities[:, seq_idx, :],
                temperature,
            )
            for seq_idx, temperature in enumerate(fitted_temperatures)
        ],
        axis=1,
    )
    raw_fusion_probabilities = base_probabilities.mean(axis=1)
    calibrated_fusion_probabilities = calibrated_base_probabilities.mean(
        axis=1
    )
    predictions_before = raw_fusion_probabilities.argmax(axis=1)
    predictions_after = calibrated_fusion_probabilities.argmax(axis=1)

    return {
        "temperatures": fitted_temperatures.tolist(),
        "sample_count": int(len(labels)),
        "nll_before": multiclass_nll(labels, raw_fusion_probabilities),
        "nll_after": multiclass_nll(
            labels,
            calibrated_fusion_probabilities,
        ),
        "accuracy_before": float((predictions_before == labels).mean()),
        "accuracy_after": float((predictions_after == labels).mean()),
        "changed_prediction_count": int(
            (predictions_before != predictions_after).sum()
        ),
        "initial_temperatures": initial_temperatures.tolist(),
        "lower_bound": lower,
        "upper_bound": upper,
        "optimization_runs": optimization_runs,
        "best_objective_value": float(fitted_nll),
    }
