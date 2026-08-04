"""把 Dataset501 nnU-Net 权重转换为项目 FLAIRUNet3DNNUNet checkpoint。"""

import argparse
from pathlib import Path

import torch

from configs.config_utils import resolve_output_artifact_dir
from models.model_factory import create_model


MODEL_NAME = "FLAIRUNet3DNNUNet"
SEQUENCE_ID = 3
SEQUENCE_NAME = "FLAIR"


def _copy_tensor(target, target_key, source, source_key):
    if target_key not in target:
        raise KeyError(f"Project model key not found: {target_key}")
    if source_key not in source:
        raise KeyError(f"nnU-Net key not found: {source_key}")
    if target[target_key].shape != source[source_key].shape:
        raise ValueError(
            f"Shape mismatch for {target_key}: {tuple(target[target_key].shape)} "
            f"vs {source_key}: {tuple(source[source_key].shape)}"
        )
    target[target_key] = source[source_key].detach().clone()


def convert_network_weights(model, network_weights):
    """逐层映射同构网络；项目分类分支保留自己的随机初始化。"""
    source = {
        key.removeprefix("_orig_mod."): value
        for key, value in network_weights.items()
    }
    target = model.state_dict()
    converted_keys = set()

    for stage in range(6):
        for convolution, (conv_index, norm_index) in enumerate(((0, 1), (3, 4))):
            source_prefix = f"encoder.stages.{stage}.0.convs.{convolution}"
            target_prefix = f"encoder.stages.{stage}.block"
            pairs = (
                (f"{target_prefix}.{conv_index}.weight", f"{source_prefix}.conv.weight"),
                (f"{target_prefix}.{conv_index}.bias", f"{source_prefix}.conv.bias"),
                (f"{target_prefix}.{norm_index}.weight", f"{source_prefix}.norm.weight"),
                (f"{target_prefix}.{norm_index}.bias", f"{source_prefix}.norm.bias"),
            )
            for target_key, source_key in pairs:
                _copy_tensor(target, target_key, source, source_key)
                converted_keys.add(target_key)

    for level in range(5):
        for suffix in ("weight", "bias"):
            target_key = f"decoder.upconvs.{level}.{suffix}"
            source_key = f"decoder.transpconvs.{level}.{suffix}"
            _copy_tensor(target, target_key, source, source_key)
            converted_keys.add(target_key)

        for convolution, (conv_index, norm_index) in enumerate(((0, 1), (3, 4))):
            source_prefix = f"decoder.stages.{level}.convs.{convolution}"
            target_prefix = f"decoder.blocks.{level}.block"
            pairs = (
                (f"{target_prefix}.{conv_index}.weight", f"{source_prefix}.conv.weight"),
                (f"{target_prefix}.{conv_index}.bias", f"{source_prefix}.conv.bias"),
                (f"{target_prefix}.{norm_index}.weight", f"{source_prefix}.norm.weight"),
                (f"{target_prefix}.{norm_index}.bias", f"{source_prefix}.norm.bias"),
            )
            for target_key, source_key in pairs:
                _copy_tensor(target, target_key, source, source_key)
                converted_keys.add(target_key)

        target_head = (
            f"decoder.deep_supervision_heads.{level}"
            if level < 4
            else 'aux_heads.seg_head'
        )
        for suffix in ("weight", "bias"):
            target_key = f"{target_head}.{suffix}"
            source_key = f"decoder.seg_layers.{level}.{suffix}"
            _copy_tensor(target, target_key, source, source_key)
            converted_keys.add(target_key)

    expected_segmentation_keys = {
        key
        for key in target
        if key.startswith(("encoder.", "decoder.", "aux_heads.seg_head."))
    }
    if converted_keys != expected_segmentation_keys:
        missing = sorted(expected_segmentation_keys - converted_keys)
        unexpected = sorted(converted_keys - expected_segmentation_keys)
        raise RuntimeError(
            f"Incomplete conversion; missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(target, strict=True)
    return len(converted_keys)


def main(args):
    source_path = Path(args.nnunet_checkpoint).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"nnU-Net checkpoint not found: {source_path}")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if "network_weights" not in checkpoint:
        raise KeyError(f"network_weights not found in {source_path}")

    model = create_model(
        MODEL_NAME,
        num_classes=3,
        in_channels=1,
        sequence_id=SEQUENCE_ID,
    )
    converted_count = convert_network_weights(model, checkpoint["network_weights"])

    checkpoint_root = resolve_output_artifact_dir(args.output_root, "checkpoints")
    destination_dir = (
        checkpoint_root / f"seq{SEQUENCE_ID}_{SEQUENCE_NAME}" / MODEL_NAME
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"fold{args.fold}_model_best.pth"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": MODEL_NAME,
            "model_capabilities": {
                "classification": True,
                "subtype": False,
                "segmentation": True,
            },
            "segmentation_target_mode": "binary_lesion",
            "stage": "segmentation",
            "fold": int(args.fold),
            "epoch": int(checkpoint.get("current_epoch", 1000)),
            "validation": None,
            "initialization_path": str(source_path),
            "native_trainer_name": checkpoint.get("trainer_name"),
            "native_mirror_axes": checkpoint.get(
                "inference_allowed_mirroring_axes"
            ),
        },
        destination,
    )
    print(f"Converted tensors : {converted_count}")
    print(f"Source epoch      : {checkpoint.get('current_epoch')}")
    print(f"Project checkpoint: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    main(parser.parse_args())
