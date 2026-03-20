from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_project_dirs() -> None:
    for folder in (FIGURES_DIR, RESULTS_DIR, REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model: torch.nn.Module, trainable_only: bool = False, verbose: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in parameters)
    if verbose:
        print(f"{model.__class__.__name__}: {total / 1_000:.2f} K parameters")
    return total


def model_size_kb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.nelement() * tensor.element_size()
    return total_bytes / 1024.0


def save_dataframe(frame: pd.DataFrame, filename: str) -> Path:
    ensure_project_dirs()
    target = RESULTS_DIR / filename
    frame.to_csv(target, index=False)
    return target


def format_size_kb(size_kb: float) -> str:
    if size_kb >= 1024:
        return f"{size_kb / 1024:.2f} MB"
    return f"{size_kb:.2f} KB"


def extract_targets(dataset) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("Dataset does not expose a `targets` attribute.")
    if isinstance(targets, torch.Tensor):
        return targets.tolist()
    return [int(target) for target in targets]


class LabelMappedSubset(Dataset):
    def __init__(self, dataset, keep_labels: Sequence[int], label_map: Dict[int, int]):
        self.dataset = dataset
        self.keep_labels = set(int(label) for label in keep_labels)
        self.label_map = {int(source): int(target) for source, target in label_map.items()}
        dataset_targets = extract_targets(dataset)
        self.indices = [index for index, label in enumerate(dataset_targets) if int(label) in self.keep_labels]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        item, label = self.dataset[self.indices[index]]
        return item, self.label_map[int(label)]


def build_label_filtered_dataset(dataset, keep_labels: Sequence[int], label_map: Dict[int, int]) -> LabelMappedSubset:
    return LabelMappedSubset(dataset=dataset, keep_labels=keep_labels, label_map=label_map)


def select_indices_by_label(dataset, per_label: int, num_classes: int) -> list[int]:
    counts = {label: 0 for label in range(num_classes)}
    selected: list[int] = []
    for index, (_, label) in enumerate(dataset):
        label = int(label)
        if counts[label] >= per_label:
            continue
        counts[label] += 1
        selected.append(index)
        if all(count >= per_label for count in counts.values()):
            break
    return selected


def repeat_channels_if_needed(inputs: torch.Tensor, target_channels: int) -> torch.Tensor:
    if inputs.ndim != 4:
        raise ValueError("Expected a 4D tensor shaped as (batch, channels, height, width).")
    if inputs.size(1) == target_channels:
        return inputs
    if inputs.size(1) == 1 and target_channels == 3:
        return inputs.repeat(1, 3, 1, 1)
    raise ValueError(f"Cannot adapt {inputs.size(1)} input channels to {target_channels}.")


def first_conv_in_channels(model: torch.nn.Module) -> int:
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.in_channels)
    raise ValueError("Model does not contain a Conv2d layer.")


def adapt_inputs_to_model(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    target_channels = first_conv_in_channels(model)
    return repeat_channels_if_needed(inputs, target_channels)


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: str | Path, map_location="cpu") -> bool:
    path = Path(checkpoint_path)
    if not path.exists():
        return False
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return True


def save_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def summarize_model_table(rows: Iterable[dict], filename: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if filename is not None and not frame.empty:
        save_dataframe(frame, filename)
    return frame
