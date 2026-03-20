from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torchvision.models as tv_models


@dataclass
class ModelLoadResult:
    model: nn.Module
    message: str


class MicroCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(48 * 4 * 4, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        feature1 = self.block1(inputs)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)
        representation = torch.flatten(feature3, 1)
        return {
            "block1": feature1,
            "block2": feature2,
            "block3": feature3,
            "representation": representation,
        }

    def get_representation(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_features(inputs)["representation"]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        return self.classifier(features["representation"])


class SmallMNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(12 * 8 * 8, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        feature1 = self.block1(inputs)
        feature2 = self.block2(feature1)
        representation = torch.flatten(feature2, 1)
        return {
            "block1": feature1,
            "block2": feature2,
            "representation": representation,
        }

    def get_representation(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_features(inputs)["representation"]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        return self.classifier(features["representation"])


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        feature1 = self.block1(inputs)
        feature2 = self.block2(feature1)
        pooled1 = self.pool1(feature2)
        feature3 = self.block3(pooled1)
        feature4 = self.block4(feature3)
        pooled2 = self.pool2(feature4)
        representation = torch.flatten(self.avgpool(pooled2), 1)
        return {
            "block1": feature1,
            "block2": feature2,
            "pool1": pooled1,
            "block3": feature3,
            "block4": feature4,
            "pool2": pooled2,
            "representation": representation,
        }

    def get_representation(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_features(inputs)["representation"]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        return self.classifier(features["representation"])


def _load_with_fallback(builder: Callable, weights, weight_message: str) -> ModelLoadResult:
    try:
        model = builder(weights=weights)
        return ModelLoadResult(model=model, message=weight_message)
    except Exception as error:  # pragma: no cover - depends on runtime connectivity
        model = builder(weights=None)
        return ModelLoadResult(
            model=model,
            message=f"Fell back to randomly initialized weights because pretrained weights were unavailable: {error}",
        )


def build_resnet50_teacher(num_classes: int = 3) -> ModelLoadResult:
    weights = tv_models.ResNet50_Weights.DEFAULT
    load_result = _load_with_fallback(
        builder=tv_models.resnet50,
        weights=weights,
        weight_message="Loaded ResNet-50 with ImageNet pretrained weights.",
    )
    model = load_result.model
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return ModelLoadResult(model=model, message=load_result.message)


def build_resnet18_student(num_classes: int = 3) -> ModelLoadResult:
    weights = tv_models.ResNet18_Weights.DEFAULT
    load_result = _load_with_fallback(
        builder=tv_models.resnet18,
        weights=weights,
        weight_message="Loaded ResNet-18 with ImageNet pretrained weights.",
    )
    model = load_result.model
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return ModelLoadResult(model=model, message=load_result.message)


def build_vgg16_teacher(num_classes: int = 4) -> ModelLoadResult:
    weights = tv_models.VGG16_Weights.DEFAULT
    load_result = _load_with_fallback(
        builder=tv_models.vgg16,
        weights=weights,
        weight_message="Loaded VGG-16 with ImageNet pretrained weights.",
    )
    model = load_result.model
    classifier = list(model.classifier.children())
    classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)
    model.classifier = nn.Sequential(*classifier)
    return ModelLoadResult(model=model, message=load_result.message)
