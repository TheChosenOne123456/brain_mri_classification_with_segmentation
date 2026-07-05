import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_balanced_weights(samples_per_class, beta):
    """按 effective number 计算类别权重，并归一化到权重和等于类别数。"""
    counts = torch.as_tensor(samples_per_class, dtype=torch.float64)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("samples_per_class must be a non-empty 1D sequence")
    if torch.any(counts <= 0):
        raise ValueError(f"Every class needs training samples, got {counts.tolist()}")
    if not 0 <= beta < 1:
        raise ValueError(f"beta must be in [0, 1), got {beta}")

    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * counts.numel()
    return weights.float()


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        weights = compute_class_balanced_weights(samples_per_class, beta)
        self.register_buffer("class_weights", weights)
        self.gamma = gamma

    def forward(self, logits, targets):
        # 在 float32 下计算 log-softmax，避免 AMP 下极小概率造成数值不稳定。
        log_probs = F.log_softmax(logits.float(), dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - log_pt.exp()).pow(self.gamma)
        alpha = self.class_weights[targets]
        return (-alpha * focal_factor * log_pt).mean()
