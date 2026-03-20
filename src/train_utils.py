from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .kd_losses import kd_loss
from .utils import adapt_inputs_to_model


@dataclass
class HistoryEntry:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float


class FeatureHook:
    def __init__(self, module: nn.Module):
        self.output = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.output = output

    def close(self) -> None:
        self.handle.remove()


def _batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        adapted_inputs = adapt_inputs_to_model(model, inputs)
        logits = model(adapted_inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_acc += _batch_accuracy(logits, labels)
        total_batches += 1
    return {
        "loss": running_loss / max(total_batches, 1),
        "acc": running_acc / max(total_batches, 1),
    }


def train_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    T: float,
    alpha: float,
    extra_loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        student_inputs = adapt_inputs_to_model(student, inputs)
        student_logits = student(student_inputs)

        with torch.no_grad():
            teacher_inputs = adapt_inputs_to_model(teacher, inputs)
            teacher_logits = teacher(teacher_inputs)

        loss = kd_loss(student_logits, teacher_logits, labels, T=T, alpha=alpha)
        if extra_loss_fn is not None:
            loss = extra_loss_fn(loss, student_logits, teacher_logits)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += _batch_accuracy(student_logits, labels)
        total_batches += 1

    return {
        "loss": running_loss / max(total_batches, 1),
        "acc": running_acc / max(total_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        adapted_inputs = adapt_inputs_to_model(model, inputs)
        logits = model(adapted_inputs)
        running_loss += F.cross_entropy(logits, labels).item()
        running_acc += _batch_accuracy(logits, labels)
        total_batches += 1
    return {
        "loss": running_loss / max(total_batches, 1),
        "acc": running_acc / max(total_batches, 1),
    }


def fit_model(
    student: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    teacher: nn.Module | None = None,
    T: float = 1.0,
    alpha: float = 1.0,
    extra_loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> pd.DataFrame:
    history: list[HistoryEntry] = []
    for epoch in range(1, epochs + 1):
        if teacher is None:
            train_metrics = train_supervised_epoch(student, train_loader, optimizer, device)
        else:
            train_metrics = train_epoch(
                student=student,
                teacher=teacher,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                T=T,
                alpha=alpha,
                extra_loss_fn=extra_loss_fn,
            )
        test_metrics = evaluate(student, test_loader, device)
        history.append(
            HistoryEntry(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_acc=train_metrics["acc"],
                test_loss=test_metrics["loss"],
                test_acc=test_metrics["acc"],
            )
        )
    return pd.DataFrame(history)


@torch.no_grad()
def collect_representations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    representation_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    max_items: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features = []
    labels = []
    collected = 0
    for inputs, batch_labels in loader:
        inputs = inputs.to(device)
        batch_labels = batch_labels.to(device)
        adapted_inputs = adapt_inputs_to_model(model, inputs)
        representations = representation_fn(model, adapted_inputs)
        features.append(representations.detach().cpu())
        labels.append(batch_labels.detach().cpu())
        collected += batch_labels.size(0)
        if max_items is not None and collected >= max_items:
            break
    feature_tensor = torch.cat(features, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    if max_items is not None:
        feature_tensor = feature_tensor[:max_items]
        label_tensor = label_tensor[:max_items]
    return feature_tensor, label_tensor
