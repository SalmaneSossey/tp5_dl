from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import FIGURES_DIR, ensure_project_dirs

sns.set_theme(style="whitegrid")


def _finalize_figure(fig: plt.Figure, filename: str | None = None) -> Path | None:
    ensure_project_dirs()
    if filename is None:
        return None
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=180)
    return path


def plot_soft_label_bars(
    images: np.ndarray,
    probs_t1: np.ndarray,
    probs_t4: np.ndarray,
    class_names: Sequence[str],
    title: str,
    filename: str | None = None,
):
    rows = len(images)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for row in range(rows):
        axes[row, 0].imshow(images[row], cmap="gray")
        axes[row, 0].set_title("Digit image")
        axes[row, 0].axis("off")
        axes[row, 1].bar(class_names, probs_t1[row], color="#457b9d")
        axes[row, 1].set_ylim(0, 1)
        axes[row, 1].set_title("Teacher probs, T=1")
        axes[row, 2].bar(class_names, probs_t4[row], color="#e76f51")
        axes[row, 2].set_ylim(0, 1)
        axes[row, 2].set_title("Teacher probs, T=4")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _finalize_figure(fig, filename)
    return fig


def plot_history_curves(history_frames: dict[str, object], filename: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, frame in history_frames.items():
        ax.plot(frame["epoch"], frame["test_acc"], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Validation/Test Accuracy Curves")
    ax.legend()
    fig.tight_layout()
    _finalize_figure(fig, filename)
    return fig


def plot_temperature_curve(frame, filename: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["temperature"], frame["test_acc"], marker="o", linewidth=2, color="#1d3557")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Effect of temperature on KD accuracy")
    fig.tight_layout()
    _finalize_figure(fig, filename)
    return fig


def plot_heatmap_grid(
    heatmaps: Sequence[np.ndarray],
    titles: Sequence[str],
    cmap: str = "magma",
    filename: str | None = None,
    annot: bool = False,
):
    columns = len(heatmaps)
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4))
    if columns == 1:
        axes = [axes]
    for axis, heatmap, title in zip(axes, heatmaps, titles):
        sns.heatmap(heatmap, ax=axis, cmap=cmap, annot=annot, cbar=True, square=True)
        axis.set_title(title)
    fig.tight_layout()
    _finalize_figure(fig, filename)
    return fig


def plot_attention_maps(
    image: np.ndarray,
    teacher_map: np.ndarray,
    student_before: np.ndarray,
    student_after: np.ndarray,
    filename: str | None = None,
):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")
    for axis, heatmap, title in zip(
        axes[1:],
        [teacher_map, student_before, student_after],
        ["Teacher", "Student before", "Student after"],
    ):
        axis.imshow(heatmap, cmap="inferno")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    _finalize_figure(fig, filename)
    return fig


def plot_tsne_triptych(embeddings: dict[str, np.ndarray], labels: np.ndarray, class_names: Sequence[str], filename: str | None = None):
    fig, axes = plt.subplots(1, len(embeddings), figsize=(5 * len(embeddings), 4))
    if len(embeddings) == 1:
        axes = [axes]
    palette = sns.color_palette("deep", n_colors=len(class_names))
    for axis, (title, points) in zip(axes, embeddings.items()):
        for class_index, class_name in enumerate(class_names):
            mask = labels == class_index
            axis.scatter(points[mask, 0], points[mask, 1], s=28, alpha=0.8, color=palette[class_index], label=class_name)
        axis.set_title(title)
    axes[0].legend()
    fig.tight_layout()
    _finalize_figure(fig, filename)
    return fig
