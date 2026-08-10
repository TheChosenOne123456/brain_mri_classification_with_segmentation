from models.FoundationModel import FoundationModel
from models.FoundationModelHierarchical import FoundationModelHierarchical
from models.FoundationModelLesionAwareHierarchical import (
    FoundationModelLesionAwareHierarchical,
)
from models.FLAIRUNet3D import FLAIRUNet3D, FLAIRUNet3DNNUNet
from models.FoundationModel_ori import FoundationModel as FoundationModelOri
from models.MedicalNetResNet18 import MedicalNetResNet18
from models.ResNet import ResNet10, ResNet18
from models.cnn3d import Simple3DCNN


MODEL_CHOICES = (
    "cnn3d",
    "ResNet",
    "ResNet18",
    "FoundationModel",
    "FoundationModel_ori",
    "MedicalNetResNet18",
    "FoundationModelHierarchical",
    "FoundationModelLesionAwareHierarchical",
    "FLAIRUNet3D",
    "FLAIRUNet3DNNUNet",
)

HIERARCHICAL_TRAIN_CONFIG_FIELDS = (
    "SUBTYPE_ALPHA",
    "SUBTYPE_CLASS_WEIGHT_POWER",
    "HIERARCHICAL_MIN_VAL_ACCURACY",
)

MODEL_REQUIRED_TRAIN_CONFIG_FIELDS = {
    "FoundationModelHierarchical": HIERARCHICAL_TRAIN_CONFIG_FIELDS,
    "FoundationModelLesionAwareHierarchical": (
        HIERARCHICAL_TRAIN_CONFIG_FIELDS
    ),
}


def required_train_config_fields(model_name):
    if model_name not in MODEL_CHOICES:
        raise ValueError(
            f"Unknown model: {model_name}. Expected one of {', '.join(MODEL_CHOICES)}"
        )
    return MODEL_REQUIRED_TRAIN_CONFIG_FIELDS.get(model_name, ())


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def model_capabilities(model):
    base_model = unwrap_model(model)
    required = (
        "has_classification_head",
        "has_subtype_head",
        "has_segmentation_head",
    )
    missing = [name for name in required if not hasattr(base_model, name)]
    if missing:
        raise AttributeError(
            f"{base_model.__class__.__name__} must declare model capabilities: "
            f"{', '.join(missing)}"
        )
    return {
        "classification": bool(base_model.has_classification_head),
        "subtype": bool(base_model.has_subtype_head),
        "segmentation": bool(base_model.has_segmentation_head),
    }


def forward_model(
    model,
    x,
    return_subtype=False,
    return_seg=False,
    segmentation_only=False,
):
    capabilities = model_capabilities(model)
    if not capabilities["classification"]:
        raise ValueError("The selected model does not have a classification head")
    if return_subtype and not capabilities["subtype"]:
        raise ValueError("Subtype output requested from a model without a subtype head")
    if return_seg and not capabilities["segmentation"]:
        raise ValueError(
            "Segmentation output requested from a model without a segmentation head"
        )

    base_model = unwrap_model(model)
    if getattr(base_model, "uses_capability_interface", False):
        # DataParallel 会独立 scatter inputs/kwargs。最后一个 batch 小于 GPU
        # 数量时，非 Tensor kwargs 仍可能复制到所有卡，产生“有 kwargs 但无 x”
        # 的空 replica。将控制标志全部作为位置参数传递，tuple scatter 会按 x
        # 的实际分片数量截断，从而不会调用无输入的 replica。
        if segmentation_only:
            if not getattr(
                base_model,
                "supports_segmentation_only_forward",
                False,
            ):
                raise ValueError(
                    f"{base_model.__class__.__name__} does not support "
                    "segmentation-only forward"
                )
            return model(x, return_seg, return_subtype, True, True)
        return model(x, return_seg, return_subtype, True)

    if return_subtype or return_seg:
        raise ValueError(
            f"{base_model.__class__.__name__} cannot return the requested auxiliary output"
        )
    return {"classification": model(x)}


def hierarchical_predictions(classification_logits, subtype_logits):
    """
    主头负责 normal 门控；主头判为异常时，子类头决定 inflammation/metastasis。
    """
    predictions = classification_logits.argmax(dim=1)
    abnormal = predictions != 0
    if abnormal.any():
        subtype_predictions = subtype_logits.argmax(dim=1) + 1
        predictions = predictions.clone()
        predictions[abnormal] = subtype_predictions[abnormal]
    return predictions


def create_model(model_name, num_classes, in_channels=1, sequence_id=None):
    if model_name not in MODEL_CHOICES:
        raise ValueError(
            f"Unknown model: {model_name}. Expected one of {', '.join(MODEL_CHOICES)}"
        )

    if model_name == "FoundationModelHierarchical":
        if sequence_id is None:
            raise ValueError(
                "FoundationModelHierarchical is designed for separate single-sequence "
                "training; specify --seq 1, --seq 2, or --seq 3"
            )
        return FoundationModelHierarchical(
            num_classes=num_classes,
            in_channels=in_channels,
            enable_segmentation=(sequence_id == 3),
        )

    if model_name == "FoundationModelLesionAwareHierarchical":
        if sequence_id != 3:
            raise ValueError(
                "FoundationModelLesionAwareHierarchical requires "
                "--seq 3 (FLAIR)"
            )
        return FoundationModelLesionAwareHierarchical(
            num_classes=num_classes,
            in_channels=in_channels,
        )

    if model_name in ("FLAIRUNet3D", "FLAIRUNet3DNNUNet"):
        if sequence_id != 3:
            raise ValueError(f"{model_name} requires --seq 3 (FLAIR)")
        constructor = (
            FLAIRUNet3D
            if model_name == "FLAIRUNet3D"
            else FLAIRUNet3DNNUNet
        )
        return constructor(
            num_classes=num_classes,
            in_channels=in_channels,
        )

    constructors = {
        "cnn3d": Simple3DCNN,
        "ResNet": ResNet10,
        "ResNet18": ResNet18,
        "FoundationModel": FoundationModel,
        "FoundationModel_ori": FoundationModelOri,
        "MedicalNetResNet18": MedicalNetResNet18,
    }
    constructor = constructors[model_name]
    try:
        return constructor(num_classes=num_classes, in_channels=in_channels)
    except TypeError:
        return constructor(num_classes=num_classes)
