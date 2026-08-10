"""Reproducible class-aware epoch sampling for long-tailed classification."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DeferredResamplingConfig:
    start_epoch: int
    target_class_probabilities: tuple
    post_switch_loss: str

    def as_dict(self):
        return {
            "start_epoch": self.start_epoch,
            "target_class_probabilities": list(
                self.target_class_probabilities
            ),
            "post_switch_loss": self.post_switch_loss,
        }


def parse_deferred_resampling_config(value, num_classes):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            "DEFERRED_CLASS_AWARE_RESAMPLING must be a mapping"
        )

    start_epoch = int(value.get("start_epoch", 0))
    if start_epoch < 2:
        raise ValueError(
            "Deferred resampling start_epoch must be at least 2"
        )

    probabilities = tuple(
        float(item)
        for item in value.get("target_class_probabilities", ())
    )
    if len(probabilities) != num_classes:
        raise ValueError(
            "target_class_probabilities must contain exactly "
            f"{num_classes} values"
        )
    if any(probability <= 0.0 for probability in probabilities):
        raise ValueError(
            "Every target class probability must be positive"
        )
    probability_sum = sum(probabilities)
    if abs(probability_sum - 1.0) > 1e-6:
        raise ValueError(
            "target_class_probabilities must sum to 1.0, got "
            f"{probability_sum:.8f}"
        )

    post_switch_loss = str(
        value.get("post_switch_loss", "cross_entropy")
    )
    if post_switch_loss != "cross_entropy":
        raise ValueError(
            "Deferred resampling currently requires unweighted "
            "post_switch_loss='cross_entropy' to prevent double correction"
        )

    return DeferredResamplingConfig(
        start_epoch=start_epoch,
        target_class_probabilities=probabilities,
        post_switch_loss=post_switch_loss,
    )


class ClassAwareEpochSampler(Sampler):
    """Generate an exact class mix while cycling through each class pool."""

    def __init__(
        self,
        labels: Sequence[int],
        target_class_probabilities: Sequence[float],
        *,
        num_samples=None,
        seed=0,
    ):
        labels = torch.as_tensor(labels, dtype=torch.long)
        probabilities = torch.as_tensor(
            target_class_probabilities,
            dtype=torch.float64,
        )
        if labels.ndim != 1 or labels.numel() == 0:
            raise ValueError("labels must be a non-empty one-dimensional sequence")
        if probabilities.ndim != 1 or probabilities.numel() == 0:
            raise ValueError(
                "target_class_probabilities must be one-dimensional"
            )
        if not torch.isfinite(probabilities).all() or torch.any(
            probabilities <= 0
        ):
            raise ValueError(
                "target_class_probabilities must be finite and positive"
            )
        if labels.min().item() < 0 or labels.max().item() >= len(probabilities):
            raise ValueError("labels contain a class outside the configured range")

        self.labels = labels
        self.num_classes = len(probabilities)
        self.num_samples = int(
            labels.numel() if num_samples is None else num_samples
        )
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.seed = int(seed)
        self.epoch = 0
        self.class_indices = tuple(
            torch.nonzero(labels == class_index, as_tuple=False).flatten()
            for class_index in range(self.num_classes)
        )
        self.class_counts = tuple(
            int(indices.numel()) for indices in self.class_indices
        )
        if any(count == 0 for count in self.class_counts):
            raise ValueError(
                "Every configured class must contain at least one sample, got "
                f"{self.class_counts}"
            )

        normalized = probabilities / probabilities.sum()
        self.target_class_probabilities = tuple(normalized.tolist())
        self.target_class_counts = self._allocate_target_counts(normalized)

    def _allocate_target_counts(self, probabilities):
        exact_counts = probabilities * self.num_samples
        target_counts = torch.floor(exact_counts).to(torch.long)
        remainder = self.num_samples - int(target_counts.sum().item())
        if remainder:
            fractional = exact_counts - target_counts
            # When fractional remainders tie, favor the currently rarer class.
            priority = sorted(
                range(self.num_classes),
                key=lambda class_index: (
                    -float(fractional[class_index]),
                    self.class_counts[class_index],
                    class_index,
                ),
            )
            for class_index in priority[:remainder]:
                target_counts[class_index] += 1
        return tuple(int(item) for item in target_counts.tolist())

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _class_cycle(self, class_index, cycle_index):
        generator = torch.Generator()
        generator.manual_seed(
            self.seed
            + class_index * 1_000_003
            + cycle_index * 10_007
        )
        permutation = torch.randperm(
            self.class_counts[class_index],
            generator=generator,
        )
        return self.class_indices[class_index][permutation]

    def _sample_class(self, class_index, target_count):
        pool_size = self.class_counts[class_index]
        absolute_start = self.epoch * target_count
        selected = []
        remaining = target_count
        position = absolute_start
        while remaining:
            cycle_index, offset = divmod(position, pool_size)
            cycle = self._class_cycle(class_index, cycle_index)
            take = min(remaining, pool_size - offset)
            selected.append(cycle[offset : offset + take])
            remaining -= take
            position += take
        return torch.cat(selected)

    def __iter__(self):
        selected = torch.cat([
            self._sample_class(class_index, target_count)
            for class_index, target_count in enumerate(
                self.target_class_counts
            )
        ])
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 104_729 + 17)
        order = torch.randperm(selected.numel(), generator=generator)
        return iter(selected[order].tolist())

    def __len__(self):
        return self.num_samples

    def metadata(self):
        return {
            "epoch": self.epoch,
            "source_class_counts": list(self.class_counts),
            "target_class_probabilities": list(
                self.target_class_probabilities
            ),
            "target_class_counts": list(self.target_class_counts),
            "num_samples": self.num_samples,
            "seed": self.seed,
        }
