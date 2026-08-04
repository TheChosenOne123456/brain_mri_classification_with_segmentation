"""nnU-Net 风格的 FLAIR 分割 patch 采样与增强。"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


class FixedIterationBatchSampler(Sampler):
    """每个 epoch 固定 batch 数，并在 batch 尾部强制前景 patch。"""

    def __init__(
        self,
        dataset_size,
        batch_size,
        num_batches,
        foreground_oversample=0.33,
        seed=0,
    ):
        if int(dataset_size) <= 0:
            raise ValueError("dataset_size must be positive")
        if int(batch_size) <= 0 or int(num_batches) <= 0:
            raise ValueError("batch_size and num_batches must be positive")
        if not 0 <= float(foreground_oversample) <= 1:
            raise ValueError("foreground_oversample must lie in [0, 1]")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.foreground_oversample = float(foreground_oversample)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        random_count = round(
            self.batch_size * (1.0 - self.foreground_oversample)
        )
        for _ in range(self.num_batches):
            indices = generator.integers(
                0,
                self.dataset_size,
                size=self.batch_size,
            )
            yield [
                (int(index), position >= random_count)
                for position, index in enumerate(indices)
            ]


class NNUNetPatchDataset(Dataset):
    """从项目全体积按 nnU-Net 规则生成各向异性训练 patch。"""

    def __init__(
        self,
        dataset,
        patch_size,
        *,
        augment=True,
        renormalize_nonzero=True,
    ):
        self.dataset = dataset
        self.patch_size = tuple(int(value) for value in patch_size)
        if len(self.patch_size) != 3 or any(value <= 0 for value in self.patch_size):
            raise ValueError("patch_size must contain three positive integers")
        self.augment = bool(augment)
        self.renormalize_nonzero = bool(renormalize_nonzero)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _nonzero_crop(image, mask):
        nonzero = image.abs().sum(dim=0) > 0
        coordinates = nonzero.nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            return image, mask
        lower = coordinates.min(dim=0).values
        upper = coordinates.max(dim=0).values + 1
        slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(lower, upper)
        )
        return image[(slice(None), *slices)], mask[(slice(None), *slices)]

    @staticmethod
    def _pad_to_patch(image, mask, patch_size):
        spatial_shape = image.shape[1:]
        padding_per_axis = []
        for size, target in zip(spatial_shape, patch_size):
            required = max(target - int(size), 0)
            lower = required // 2
            padding_per_axis.append((lower, required - lower))
        d_pad, h_pad, w_pad = padding_per_axis
        padding = (
            w_pad[0],
            w_pad[1],
            h_pad[0],
            h_pad[1],
            d_pad[0],
            d_pad[1],
        )
        if any(padding):
            image = F.pad(image, padding, value=0.0)
            mask = F.pad(mask, padding, value=0)
        return image, mask

    @staticmethod
    def _sample_patch(image, mask, patch_size, force_foreground):
        image, mask = NNUNetPatchDataset._pad_to_patch(
            image,
            mask,
            patch_size,
        )
        spatial_shape = image.shape[1:]
        foreground = (mask > 0).squeeze(0).nonzero(as_tuple=False)
        starts = []
        selected_voxel = None
        if force_foreground and foreground.numel() > 0:
            selected_voxel = foreground[
                random.randrange(int(foreground.shape[0]))
            ]
        for axis, (size, target) in enumerate(zip(spatial_shape, patch_size)):
            maximum = int(size) - target
            if selected_voxel is None:
                start = random.randint(0, maximum) if maximum > 0 else 0
            else:
                start = int(selected_voxel[axis]) - target // 2
                start = min(max(start, 0), maximum)
            starts.append(start)
        slices = tuple(
            slice(start, start + target)
            for start, target in zip(starts, patch_size)
        )
        return image[(slice(None), *slices)], mask[(slice(None), *slices)]

    @staticmethod
    def _renormalize(image):
        brain = image.abs().sum(dim=0, keepdim=True) > 0
        values = image[brain]
        if values.numel() > 1:
            std = values.std(unbiased=False)
            if std > 0:
                image = torch.where(brain, (image - values.mean()) / std, 0.0)
        return image, brain

    @staticmethod
    def _spatial_transform(image, mask, brain):
        rotate = random.random() < 0.2
        rescale = random.random() < 0.2
        if not rotate and not rescale:
            return image, mask, brain

        angle = random.uniform(-math.pi, math.pi) if rotate else 0.0
        scale = random.uniform(0.7, 1.4) if rescale else 1.0
        cosine = math.cos(angle) * scale
        sine = math.sin(angle) * scale
        theta = image.new_tensor(
            [
                [cosine, -sine, 0.0, 0.0],
                [sine, cosine, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ).unsqueeze(0)
        output_size = (1, 1, *image.shape[1:])
        grid = F.affine_grid(theta, output_size, align_corners=False)
        image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
        mask = F.grid_sample(
            mask.float().unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0).to(mask.dtype)
        brain = F.grid_sample(
            brain.float().unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0) > 0.5
        return image, mask, brain

    @staticmethod
    def _gamma(image, brain, invert=False):
        values = image[brain]
        if values.numel() < 2:
            return image
        original_mean = values.mean()
        original_std = values.std(unbiased=False)
        working = -image if invert else image
        active = working[brain]
        minimum = active.min()
        maximum = active.max()
        if maximum <= minimum:
            return image
        gamma = random.uniform(0.7, 1.5)
        transformed = ((working - minimum) / (maximum - minimum)).clamp(0, 1)
        transformed = transformed.pow(gamma) * (maximum - minimum) + minimum
        transformed = -transformed if invert else transformed
        new_values = transformed[brain]
        new_std = new_values.std(unbiased=False)
        if new_std > 0 and original_std > 0:
            transformed = (
                (transformed - new_values.mean())
                * (original_std / new_std)
                + original_mean
            )
        return torch.where(brain, transformed, 0.0)

    @staticmethod
    def _gaussian_blur_in_plane(image):
        output = image.unsqueeze(0)
        for spatial_axis in (3, 4):
            sigma = random.uniform(0.5, 1.0)
            raw_kernel_size = sigma * 6 + 0.5
            kernel_size = int(round(raw_kernel_size))
            if kernel_size % 2 == 0:
                kernel_size += 1 if raw_kernel_size >= kernel_size else -1
            kernel_size = max(3, kernel_size)
            radius = kernel_size // 2
            coordinates = torch.arange(
                -radius,
                radius + 1,
                device=image.device,
                dtype=image.dtype,
            )
            kernel = torch.exp(-0.5 * (coordinates / sigma).square())
            kernel = kernel / kernel.sum()
            if spatial_axis == 3:
                weight = kernel.reshape(1, 1, 1, kernel_size, 1)
                padding = (0, 0, radius, radius, 0, 0)
            else:
                weight = kernel.reshape(1, 1, 1, 1, kernel_size)
                padding = (radius, radius, 0, 0, 0, 0)
            output = F.conv3d(
                F.pad(output, padding, mode="reflect"),
                weight,
            )
        return output.squeeze(0)

    @staticmethod
    def _augment(image, mask, brain):
        image, mask, brain = NNUNetPatchDataset._spatial_transform(
            image,
            mask,
            brain,
        )

        if random.random() < 0.1:
            std = random.uniform(0.0, 0.1)
            image = image + torch.randn_like(image) * std
        if random.random() < 0.2:
            image = NNUNetPatchDataset._gaussian_blur_in_plane(image)
        if random.random() < 0.15:
            image = image * random.uniform(0.75, 1.25)
        if random.random() < 0.15:
            values = image[brain]
            if values.numel() > 0:
                minimum, maximum = values.min(), values.max()
                mean = values.mean()
                image = ((image - mean) * random.uniform(0.75, 1.25) + mean).clamp(
                    minimum,
                    maximum,
                )
        if random.random() < 0.25:
            scale = random.uniform(0.5, 1.0)
            depth, height, width = image.shape[1:]
            low_shape = (
                depth,
                max(1, int(round(height * scale))),
                max(1, int(round(width * scale))),
            )
            low_resolution = F.interpolate(
                image.unsqueeze(0),
                size=low_shape,
                mode="trilinear",
                align_corners=False,
            )
            image = F.interpolate(
                low_resolution,
                size=(depth, height, width),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        if random.random() < 0.1:
            image = NNUNetPatchDataset._gamma(image, brain, invert=True)
        if random.random() < 0.3:
            image = NNUNetPatchDataset._gamma(image, brain, invert=False)

        for spatial_axis in range(3):
            if random.random() < 0.5:
                tensor_axis = spatial_axis + 1
                image = torch.flip(image, dims=(tensor_axis,))
                mask = torch.flip(mask, dims=(tensor_axis,))
                brain = torch.flip(brain, dims=(tensor_axis,))
        image = torch.where(brain, image, 0.0)
        return image.contiguous(), mask.contiguous()

    def __getitem__(self, item):
        if isinstance(item, tuple):
            index, force_foreground = item
        else:
            index, force_foreground = item, False
        image, label, mask, mask_flag, case_id = self.dataset[int(index)]
        image = image.float()
        mask = mask.long()
        image, mask = self._nonzero_crop(image, mask)
        if self.renormalize_nonzero:
            # nnU-Net 在整例非零裁剪区域归一化，不能对每个随机 patch
            # 分别计算均值和方差，否则训练时的灰度标尺会随采样位置漂移。
            image, _ = self._renormalize(image)
        image, mask = self._sample_patch(
            image,
            mask,
            self.patch_size,
            bool(force_foreground),
        )
        brain = image.abs().sum(dim=0, keepdim=True) > 0
        if self.augment:
            image, mask = self._augment(image, mask, brain)
        return image, label, mask, mask_flag, case_id
