from __future__ import annotations

import itertools
import random

import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    true_labels: torch.Tensor,
    T: float,
    alpha: float,
) -> torch.Tensor:
    hard_loss = F.cross_entropy(student_logits, true_labels)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    # A class-wise mean keeps the loss scale stable for small classification heads.
    soft_loss = -(teacher_probs * student_log_probs).mean(dim=1).mean()
    return alpha * hard_loss + (1.0 - alpha) * soft_loss


def attention_map(feature_map: torch.Tensor) -> torch.Tensor:
    attention = feature_map.pow(2).mean(dim=1)
    flattened = attention.flatten(1)
    normalized = F.normalize(flattened, p=2, dim=1)
    return normalized.view_as(attention)


def at_loss(student_features: torch.Tensor, teacher_features: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    student_attention = attention_map(student_features)
    teacher_attention = attention_map(teacher_features.detach())
    if student_attention.shape[-2:] != teacher_attention.shape[-2:]:
        teacher_attention = F.interpolate(
            teacher_attention.unsqueeze(1),
            size=student_attention.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return beta * F.mse_loss(student_attention, teacher_attention)


def fitnets_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    adapter: torch.nn.Module,
    gamma: float,
    kd_loss_val: torch.Tensor,
) -> torch.Tensor:
    adapted_student = adapter(student_feat)
    teacher_target = teacher_feat.detach()
    if adapted_student.shape[-2:] != teacher_target.shape[-2:]:
        adapted_student = F.interpolate(
            adapted_student,
            size=teacher_target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    hint_loss = F.mse_loss(adapted_student, teacher_target)
    return kd_loss_val + gamma * hint_loss


def _pairwise_euclidean(features: torch.Tensor) -> torch.Tensor:
    return torch.cdist(features, features, p=2)


def rkd_distance_loss(feat_T: torch.Tensor, feat_S: torch.Tensor) -> torch.Tensor:
    teacher = feat_T.detach().flatten(1)
    student = feat_S.flatten(1)
    with torch.no_grad():
        teacher_dist = _pairwise_euclidean(teacher)
        teacher_dist = teacher_dist / teacher_dist[teacher_dist > 0].mean().clamp_min(1e-8)
    student_dist = _pairwise_euclidean(student)
    student_dist = student_dist / student_dist[student_dist > 0].mean().clamp_min(1e-8)
    return F.smooth_l1_loss(student_dist, teacher_dist)


def rkd_angle_loss(feat_T: torch.Tensor, feat_S: torch.Tensor, triplets: int = 64) -> torch.Tensor:
    teacher = feat_T.detach().flatten(1)
    student = feat_S.flatten(1)
    batch_size = teacher.size(0)
    if batch_size < 3:
        return torch.zeros((), device=teacher.device)

    all_triplets = list(itertools.combinations(range(batch_size), 3))
    sampled_triplets = random.sample(all_triplets, k=min(triplets, len(all_triplets)))
    teacher_angles = []
    student_angles = []
    for i, j, k in sampled_triplets:
        teacher_ij = F.normalize(teacher[i] - teacher[j], dim=0)
        teacher_ik = F.normalize(teacher[i] - teacher[k], dim=0)
        student_ij = F.normalize(student[i] - student[j], dim=0)
        student_ik = F.normalize(student[i] - student[k], dim=0)
        teacher_angles.append(torch.sum(teacher_ij * teacher_ik))
        student_angles.append(torch.sum(student_ij * student_ik))
    teacher_tensor = torch.stack(teacher_angles)
    student_tensor = torch.stack(student_angles)
    return F.smooth_l1_loss(student_tensor, teacher_tensor)
