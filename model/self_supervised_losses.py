"""MoCo, DINO and iBOT++ losses used by font-style pretraining."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class MoCoQueueLoss(nn.Module):
    """InfoNCE against a momentum-feature queue.

    When style ids are available, queued samples from the same font can be
    excluded from negatives. This avoids teaching the encoder that two different
    texts rendered by one font are different styles.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        queue_size: int = 65_536,
        temperature: float = 0.2,
        exclude_same_style_negatives: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or queue_size <= 0 or temperature <= 0:
            raise ValueError("feature_dim, queue_size and temperature must be positive")
        queue = F.normalize(torch.randn(queue_size, feature_dim), dim=1)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_style_ids", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros((), dtype=torch.long))
        self.temperature = float(temperature)
        self.exclude_same_style_negatives = bool(exclude_same_style_negatives)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        style_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.autocast(device_type=query.device.type, enabled=False):
            queue_size = self.queue.shape[0]
            if self.queue_filled < queue_size:
                queue_positions = torch.arange(queue_size, device=query.device)
                valid_queue_mask = queue_positions < self.queue_filled
                combined_mask = (~valid_queue_mask).unsqueeze(0).expand(query.shape[0], -1)
            else:
                combined_mask = torch.zeros((query.shape[0], queue_size), dtype=torch.bool, device=query.device)

            query = F.normalize(query.float(), dim=-1)
            key = F.normalize(key.detach().float(), dim=-1)
            positive_logits = torch.sum(query * key, dim=-1, keepdim=True)
            negative_logits = query @ self.queue.detach().float().t()

            if self.exclude_same_style_negatives and style_ids is not None:
                same_style_mask = style_ids.view(-1, 1).eq(
                    self.queue_style_ids.view(1, -1)
                )
                combined_mask = combined_mask | same_style_mask
                negative_logits = negative_logits.masked_fill(combined_mask, -torch.inf)

            logits = torch.cat((positive_logits, negative_logits), dim=1) / self.temperature
            labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
            loss = F.cross_entropy(logits, labels)
            accuracy = logits.argmax(dim=1).eq(0).float().mean()
            finite_negatives = negative_logits[torch.isfinite(negative_logits)]
            negative_similarity = (
                finite_negatives.mean()
                if finite_negatives.numel()
                else negative_logits.new_zeros(())
            )

        return loss, {
            "positive_similarity": positive_logits.mean().detach(),
            "negative_similarity": negative_similarity.detach(),
            "top1_accuracy": accuracy.detach(),
        }

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor, style_ids: torch.Tensor | None = None) -> None:
        keys = F.normalize(keys.detach(), dim=-1)
        if style_ids is None:
            style_ids = torch.full((keys.shape[0],), -1, dtype=torch.long, device=keys.device)
        else:
            style_ids = style_ids.detach().long()
        queue_size = self.queue.shape[0]
        if keys.shape[0] >= queue_size:
            self.queue.copy_(keys[-queue_size:])
            self.queue_style_ids.copy_(style_ids[-queue_size:])
            self.queue_pointer.zero_()
            self.queue_filled.fill_(queue_size)
            return

        pointer = int(self.queue_pointer)
        first_count = min(keys.shape[0], queue_size - pointer)
        self.queue[pointer:pointer + first_count] = keys[:first_count]
        self.queue_style_ids[pointer:pointer + first_count] = style_ids[:first_count]
        remaining = keys.shape[0] - first_count
        if remaining:
            self.queue[:remaining] = keys[first_count:]
            self.queue_style_ids[:remaining] = style_ids[first_count:]
        self.queue_pointer.fill_((pointer + keys.shape[0]) % queue_size)
        self.queue_filled.add_(keys.shape[0]).clamp_(max=queue_size)


class DINOLoss(nn.Module):
    """Cross-view self-distillation with teacher centering and sharpening."""

    def __init__(
        self,
        output_dim: int,
        student_temperature: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        if output_dim <= 0 or student_temperature <= 0:
            raise ValueError("output_dim and student_temperature must be positive")
        self.student_temperature = float(student_temperature)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, output_dim))

    def forward(
        self,
        student_logits: Sequence[torch.Tensor],
        teacher_logits: Sequence[torch.Tensor],
        teacher_temperature: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if teacher_temperature <= 0:
            raise ValueError("teacher_temperature must be positive")
        student_log_probabilities = [
            F.log_softmax(logits.float() / self.student_temperature, dim=-1)
            for logits in student_logits
        ]
        teacher_probabilities = [
            F.softmax((logits.detach().float() - self.center) / teacher_temperature, dim=-1)
            for logits in teacher_logits
        ]

        losses = []
        for teacher_index, probabilities in enumerate(teacher_probabilities):
            for student_index, log_probabilities in enumerate(student_log_probabilities):
                if teacher_index == student_index:
                    continue
                losses.append(-(probabilities * log_probabilities).sum(dim=-1).mean())
        if not losses:
            raise ValueError("DINO needs at least two student views")
        loss = torch.stack(losses).mean()

        with torch.no_grad():
            batch_center = torch.cat([logits.detach().float() for logits in teacher_logits]).mean(
                dim=0, keepdim=True
            )
            self.center.mul_(self.center_momentum).add_(
                batch_center, alpha=1.0 - self.center_momentum
            )
            teacher_entropy = torch.stack(
                [
                    -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    for probabilities in teacher_probabilities
                ]
            ).mean()
        return loss, {
            "teacher_entropy": teacher_entropy,
        }


class IBOTPlusPlusPatchLoss(nn.Module):
    """iBOT++ patch distillation over every valid global-view patch.

    The student still receives a block-masked image while the teacher receives
    the unmasked image. Unlike iBOT, visible student patches are supervised as
    well. ``visible_loss_weight=0`` recovers the masked-only iBOT objective;
    the default value of 1 gives every valid patch equal weight.
    """

    def __init__(
        self,
        output_dim: int,
        student_temperature: float = 0.1,
        center_momentum: float = 0.9,
        visible_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if output_dim <= 0 or student_temperature <= 0:
            raise ValueError("output_dim and student_temperature must be positive")
        if not 0.0 <= center_momentum <= 1.0:
            raise ValueError("center_momentum must be in [0, 1]")
        if visible_loss_weight < 0:
            raise ValueError("visible_loss_weight must be non-negative")
        self.student_temperature = float(student_temperature)
        self.center_momentum = float(center_momentum)
        self.visible_loss_weight = float(visible_loss_weight)
        self.register_buffer("center", torch.zeros(1, 1, output_dim))

    def forward(
        self,
        student_logits: Sequence[torch.Tensor],
        teacher_logits: Sequence[torch.Tensor],
        ibot_masks: Sequence[torch.Tensor],
        padding_masks: Sequence[torch.Tensor],
        teacher_temperature: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if teacher_temperature <= 0:
            raise ValueError("teacher_temperature must be positive")
        if not (
            len(student_logits)
            == len(teacher_logits)
            == len(ibot_masks)
            == len(padding_masks)
        ):
            raise ValueError("iBOT++ inputs must contain the same number of global views")
        if not student_logits:
            raise ValueError("iBOT++ needs at least one global view")
        losses = []
        masked_count = 0
        visible_count = 0
        masked_loss_sum = self.center.new_zeros((), dtype=torch.float32)
        visible_loss_sum = self.center.new_zeros((), dtype=torch.float32)
        center_sum = self.center.new_zeros((self.center.shape[-1],), dtype=torch.float32)
        center_count = 0

        for student, teacher, ibot_mask, padding_mask in zip(
            student_logits, teacher_logits, ibot_masks, padding_masks
        ):
            if student.shape != teacher.shape:
                raise ValueError("student and teacher iBOT++ logits must have identical shapes")
            if ibot_mask.shape != student.shape[:2] or padding_mask.shape != student.shape[:2]:
                raise ValueError("iBOT++ masks must match the logits' batch and token dimensions")
            valid_mask = ~padding_mask
            masked_mask = ibot_mask & valid_mask
            visible_mask = (~ibot_mask) & valid_mask
            student_log_probabilities = F.log_softmax(
                student.float() / self.student_temperature, dim=-1
            )
            teacher_probabilities = F.softmax(
                (teacher.detach().float() - self.center) / teacher_temperature, dim=-1
            )
            token_losses = -(teacher_probabilities * student_log_probabilities).sum(dim=-1)

            # Normalize per image so arbitrary resolutions do not make images with
            # more patches dominate the batch. At the default visible weight of 1,
            # this is exactly the mean distillation loss over all valid patches.
            token_weights = masked_mask.to(token_losses.dtype)
            token_weights = token_weights + visible_mask.to(token_losses.dtype) * self.visible_loss_weight
            loss_per_image = (token_losses * token_weights).sum(dim=1) / token_weights.sum(dim=1).clamp_min(1e-8)
            losses.append(loss_per_image.mean())

            current_masked_count = int(masked_mask.sum())
            current_visible_count = int(visible_mask.sum())
            if current_masked_count:
                masked_loss_sum += token_losses[masked_mask].detach().sum()
                masked_count += current_masked_count
            if current_visible_count:
                visible_loss_sum += token_losses[visible_mask].detach().sum()
                visible_count += current_visible_count
            valid_teacher = teacher.detach().float()[valid_mask]
            center_sum += valid_teacher.sum(dim=0)
            center_count += valid_teacher.shape[0]

        if losses:
            loss = torch.stack(losses).mean()
        else:
            # Keep a differentiable zero for a fully padded batch, or for legacy
            # masked-only mode when a batch happens to contain no masked patches.
            loss = sum(logits.sum() for logits in student_logits) * 0.0
        with torch.no_grad():
            if center_count:
                batch_center = (center_sum / center_count).view(1, 1, -1)
                self.center.mul_(self.center_momentum).add_(
                    batch_center, alpha=1.0 - self.center_momentum
                )
        return loss, {
            "masked_patch_count": loss.new_tensor(masked_count),
            "visible_patch_count": loss.new_tensor(visible_count),
            "masked_token_loss": masked_loss_sum / max(1, masked_count),
            "visible_token_loss": visible_loss_sum / max(1, visible_count),
        }
