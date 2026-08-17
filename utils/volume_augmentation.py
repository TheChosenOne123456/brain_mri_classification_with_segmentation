"""训练集专用的轻量三维影像/分割 mask 同步增强。"""

from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
import torch
from scipy.ndimage import affine_transform, zoom
from torch.utils.data import Dataset


def _triple(value, name, *, allow_zero=True):
    if isinstance(value, Real):
        values = (float(value),) * 3
    else:
        values = tuple(float(item) for item in value)
        if len(values) != 3:
            raise ValueError(f"{name} must contain three D/H/W values")
    minimum = 0.0 if allow_zero else np.finfo(np.float32).eps
    if any(item < minimum for item in values):
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"Every {name} value must be {comparator}: {values}")
    return values


def _probability(value, name):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return value


@dataclass(frozen=True)
class VolumeAugmentationConfig:
    """可序列化的轻量三维增强参数。"""

    enabled: bool = False
    spatial_probability: float = 0.0
    rotation_degrees: tuple = (0.0, 0.0, 0.0)
    translation_voxels: tuple = (0.0, 0.0, 0.0)
    scale_range: tuple = (1.0, 1.0)
    voxel_spacing: tuple = (1.0, 1.0, 1.0)
    left_right_flip_probability: float = 0.0
    bias_field_probability: float = 0.0
    bias_field_log_amplitude: float = 0.0
    bias_field_control_points: tuple = (4, 4, 4)

    def as_dict(self):
        return asdict(self)


def parse_volume_augmentation_config(raw_config):
    """解析配置字典；未配置时保持旧训练路径不变。"""
    if raw_config is None:
        return VolumeAugmentationConfig()
    if not isinstance(raw_config, dict):
        raise TypeError("TRAIN_AUGMENTATION must be a dictionary")

    allowed = {
        "enabled",
        "spatial_probability",
        "rotation_degrees",
        "translation_voxels",
        "scale_range",
        "voxel_spacing",
        "left_right_flip_probability",
        "bias_field_probability",
        "bias_field_log_amplitude",
        "bias_field_control_points",
    }
    unknown = sorted(set(raw_config) - allowed)
    if unknown:
        raise ValueError(
            "Unknown TRAIN_AUGMENTATION fields: " + ", ".join(unknown)
        )

    scale_range = tuple(
        float(item) for item in raw_config.get("scale_range", (1.0, 1.0))
    )
    if len(scale_range) != 2 or not 0 < scale_range[0] <= scale_range[1]:
        raise ValueError(
            "TRAIN_AUGMENTATION scale_range must satisfy 0 < min <= max"
        )

    control_points = tuple(
        int(item)
        for item in raw_config.get("bias_field_control_points", (4, 4, 4))
    )
    if len(control_points) != 3 or any(item < 2 for item in control_points):
        raise ValueError(
            "bias_field_control_points must contain three integers >= 2"
        )

    config = VolumeAugmentationConfig(
        enabled=bool(raw_config.get("enabled", True)),
        spatial_probability=_probability(
            raw_config.get("spatial_probability", 0.0),
            "spatial_probability",
        ),
        rotation_degrees=_triple(
            raw_config.get("rotation_degrees", 0.0),
            "rotation_degrees",
        ),
        translation_voxels=_triple(
            raw_config.get("translation_voxels", 0.0),
            "translation_voxels",
        ),
        scale_range=scale_range,
        voxel_spacing=_triple(
            raw_config.get("voxel_spacing", (1.0, 1.0, 1.0)),
            "voxel_spacing",
            allow_zero=False,
        ),
        left_right_flip_probability=_probability(
            raw_config.get("left_right_flip_probability", 0.0),
            "left_right_flip_probability",
        ),
        bias_field_probability=_probability(
            raw_config.get("bias_field_probability", 0.0),
            "bias_field_probability",
        ),
        bias_field_log_amplitude=float(
            raw_config.get("bias_field_log_amplitude", 0.0)
        ),
        bias_field_control_points=control_points,
    )
    if config.bias_field_log_amplitude < 0:
        raise ValueError("bias_field_log_amplitude must be >= 0")
    return config


def _random_uniform(low, high):
    if low == high:
        return float(low)
    return float(torch.empty(()).uniform_(float(low), float(high)).item())


def _rotation_matrix_dhw(rotation_degrees):
    angles = np.deg2rad(
        [
            _random_uniform(-maximum, maximum)
            for maximum in rotation_degrees
        ]
    )
    d_angle, h_angle, w_angle = angles
    cd, sd = np.cos(d_angle), np.sin(d_angle)
    ch, sh = np.cos(h_angle), np.sin(h_angle)
    cw, sw = np.cos(w_angle), np.sin(w_angle)

    around_d = np.asarray(
        [[1, 0, 0], [0, cd, -sd], [0, sd, cd]], dtype=np.float64
    )
    around_h = np.asarray(
        [[ch, 0, sh], [0, 1, 0], [-sh, 0, ch]], dtype=np.float64
    )
    around_w = np.asarray(
        [[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]], dtype=np.float64
    )
    return around_w @ around_h @ around_d


def _sample_affine(shape, config):
    scale = _random_uniform(*config.scale_range)
    # 在物理坐标中旋转，再映射回体素坐标，避免厚层 Z 轴被当作各向同性。
    spacing = np.diag(np.asarray(config.voxel_spacing, dtype=np.float64))
    matrix_physical = scale * _rotation_matrix_dhw(config.rotation_degrees)
    matrix = np.linalg.inv(spacing) @ matrix_physical @ spacing
    translation = np.asarray(
        [
            _random_uniform(-maximum, maximum)
            for maximum in config.translation_voxels
        ],
        dtype=np.float64,
    )
    center = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    # scipy 接收 output-index -> input-index；减去平移量可令内容沿采样值正向移动。
    offset = center - matrix @ center - translation
    return matrix, offset


def _apply_affine(image, mask, matrix, offset):
    image_output = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[0]):
        affine_transform(
            image[channel],
            matrix=matrix,
            offset=offset,
            output_shape=image.shape[1:],
            output=image_output[channel],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    mask_output = np.empty(mask.shape, dtype=np.int16)
    for channel in range(mask.shape[0]):
        affine_transform(
            mask[channel],
            matrix=matrix,
            offset=offset,
            output_shape=mask.shape[1:],
            output=mask_output[channel],
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
    return image_output, mask_output


def _smooth_bias_field(shape, config):
    coarse = np.empty(config.bias_field_control_points, dtype=np.float32)
    coarse_values = torch.empty(config.bias_field_control_points).uniform_(
        -config.bias_field_log_amplitude,
        config.bias_field_log_amplitude,
    )
    coarse[...] = coarse_values.numpy()
    factors = tuple(size / points for size, points in zip(shape, coarse.shape))
    # 四点控制网格的三线性插值已足够平滑，并显著低于三次样条的 CPU 开销。
    field = zoom(coarse, factors, order=1, mode="reflect", prefilter=False)
    field = field[tuple(slice(0, size) for size in shape)]
    if field.shape != tuple(shape):
        raise RuntimeError(
            f"Bias field shape mismatch: expected {shape}, got {field.shape}"
        )
    limit = 2.0 * config.bias_field_log_amplitude
    return np.clip(field, -limit, limit).astype(np.float32, copy=False)


def _apply_bias_field(image, config):
    foreground = np.any(np.abs(image) > 1e-6, axis=0)
    if not foreground.any():
        return image
    log_field = _smooth_bias_field(image.shape[1:], config)
    log_field -= float(log_field[foreground].mean())
    gain = np.exp(log_field).astype(np.float32, copy=False)
    return image * gain[None]


class SynchronizedVolumeAugmentationDataset(Dataset):
    """只包装 train dataset，并让空间变换严格同步到离散 mask。"""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.labels = dataset.labels
        if hasattr(dataset, "cases"):
            self.cases = dataset.cases

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label, mask, has_mask, case_id = self.dataset[index]
        if image.ndim != 4 or mask.ndim != 4:
            raise ValueError(
                "Volume augmentation expects image/mask shaped [C,D,H,W], "
                f"got {tuple(image.shape)} and {tuple(mask.shape)}"
            )
        if tuple(image.shape[1:]) != tuple(mask.shape[1:]):
            raise ValueError(
                f"Image/mask spatial shapes differ: {image.shape} vs {mask.shape}"
            )

        image_np = image.detach().cpu().numpy().astype(np.float32, copy=True)
        mask_np = mask.detach().cpu().numpy().astype(np.int16, copy=True)

        if torch.rand(()).item() < self.config.spatial_probability:
            matrix, offset = _sample_affine(image_np.shape[1:], self.config)
            image_np, mask_np = _apply_affine(
                image_np,
                mask_np,
                matrix,
                offset,
            )

        # 当前数据加载顺序是 [C,Z,Y,X]；SynthStrip 数据的 X 轴为 L/R。
        if torch.rand(()).item() < self.config.left_right_flip_probability:
            image_np = np.flip(image_np, axis=-1).copy()
            mask_np = np.flip(mask_np, axis=-1).copy()

        if (
            self.config.bias_field_log_amplitude > 0
            and torch.rand(()).item() < self.config.bias_field_probability
        ):
            image_np = _apply_bias_field(image_np, self.config)

        return (
            torch.from_numpy(np.ascontiguousarray(image_np)),
            label,
            torch.from_numpy(np.ascontiguousarray(mask_np)).long(),
            has_mask,
            case_id,
        )


def wrap_training_dataset(dataset, config):
    """仅在显式启用时包装数据集，旧配置返回原对象。"""
    if not config.enabled:
        return dataset
    return SynchronizedVolumeAugmentationDataset(dataset, config)
