"""Deep learning and classical baseline model architectures for CASSAVISION."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models


class CassavaScratchCNN(nn.Module):
    """
    Lightweight 4-stage Residual/BatchNorm CNN trained from scratch as the deep learning baseline.
    Input: (B, 3, 224, 224) -> Output: (B, num_classes).
    Exposes `self.features` for Grad-CAM compatibility on the final convolutional block (`features[-1]`).
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: 28 -> 14 (Target layer for Grad-CAM)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat)
        return self.classifier(pooled)


def build_classifier_model(
    arch_name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.25,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Instantiate a candidate computer vision architecture with a 5-class head and metadata.
    Supported architectures:
      - 'custom_cnn': CassavaScratchCNN trained from scratch (no pretrained weights)
      - 'mobilenet_v2': MobileNetV2 with ImageNet-1K V1 weights (inverted residuals + linear bottlenecks)
      - 'mobilenet_v3_small': MobileNetV3-Small with ImageNet-1K V1 weights (Squeeze-and-Excitation + HardSwish)
      - 'efficientnet_b0': EfficientNet-B0 with ImageNet-1K V1 weights (compound scaling MBConv)
      - 'resnet18': ResNet-18 with ImageNet-1K V1 weights (residual skip connections)
    """
    arch = arch_name.lower().strip()
    weights_origin = "Random Initialization (Kaiming Normal)"

    if arch == "custom_cnn":
        model = CassavaScratchCNN(num_classes=num_classes, dropout=dropout)
        target_layer_name = "features.14"

    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        if pretrained:
            weights_origin = "TorchVision MobileNet_V2_Weights.IMAGENET1K_V1 (BSD-3-Clause)"
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        target_layer_name = "features.18"

    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        if pretrained:
            weights_origin = "TorchVision MobileNet_V3_Small_Weights.IMAGENET1K_V1 (BSD-3-Clause)"
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        target_layer_name = "features.12"

    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        if pretrained:
            weights_origin = "TorchVision EfficientNet_B0_Weights.IMAGENET1K_V1 (BSD-3-Clause)"
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        target_layer_name = "features.8"

    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        if pretrained:
            weights_origin = "TorchVision ResNet18_Weights.IMAGENET1K_V1 (BSD-3-Clause)"
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        target_layer_name = "layer4.1"

    else:
        raise ValueError(
            f"Unsupported architecture '{arch_name}'. Choose from: custom_cnn, mobilenet_v2, mobilenet_v3_small, efficientnet_b0, resnet18."
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024.0 * 1024.0)

    meta = {
        "architecture": arch,
        "num_classes": num_classes,
        "pretrained": pretrained and (arch != "custom_cnn"),
        "weights_origin": weights_origin,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "parameter_size_mb": round(float(size_mb), 3),
        "gradcam_target_layer": target_layer_name,
    }
    return model, meta


def get_gradcam_target_layer(model: nn.Module, arch_name: str) -> nn.Module:
    """Return the final spatial convolutional module for Grad-CAM activation & gradient hooking."""
    arch = arch_name.lower().strip()
    if arch in ("custom_cnn", "mobilenet_v2", "mobilenet_v3_small", "efficientnet_b0"):
        return model.features[-1]
    elif arch == "resnet18":
        return model.layer4[-1]
    raise ValueError(f"Cannot resolve Grad-CAM target layer for architecture '{arch_name}'.")


def set_backbone_trainable(model: nn.Module, arch_name: str, trainable: bool) -> None:
    """Freeze or unfreeze backbone feature extractor layers while keeping the classification head trainable."""
    arch = arch_name.lower().strip()
    if arch == "custom_cnn":
        for p in model.parameters():
            p.requires_grad = True
        return

    if hasattr(model, "features"):
        for p in model.features.parameters():
            p.requires_grad = trainable
    elif arch == "resnet18":
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = trainable

    # Ensure classifier head is always trainable
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
