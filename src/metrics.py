from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .utils import adapt_inputs_to_model


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    n: int = 100,
    warmup: int = 10,
) -> float:
    model.eval()
    sample = torch.randn(*input_shape, device=device)
    sample = adapt_inputs_to_model(model, sample)
    for _ in range(warmup):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / n) * 1000.0


def cosine_similarity_matrix(prototypes: torch.Tensor) -> np.ndarray:
    normalized = F.normalize(prototypes, dim=1)
    return (normalized @ normalized.T).cpu().numpy()


def class_prototypes(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    prototypes = []
    for class_index in range(num_classes):
        class_features = features[labels == class_index]
        prototypes.append(class_features.mean(dim=0))
    return torch.stack(prototypes)
