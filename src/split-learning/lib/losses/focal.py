"""Multi-class Focal Loss (Lin et al, 2017)."""

from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._registry import register


AlphaT = Optional[Union[float, Iterable[float], torch.Tensor]]


@register("focal")
class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: AlphaT = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean'|'sum'|'none', got {reduction!r}")
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction

        if alpha is None:
            self.alpha_scalar: Optional[float] = None
            self.register_buffer("alpha_vec", torch.empty(0), persistent=False)
        elif isinstance(alpha, (int, float)):
            self.alpha_scalar = float(alpha)
            self.register_buffer("alpha_vec", torch.empty(0), persistent=False)
        else:
            vec = torch.as_tensor(list(alpha), dtype=torch.float32)
            if vec.ndim != 1:
                raise ValueError("alpha must be 1-D when given as a sequence")
            self.alpha_scalar = None
            self.register_buffer("alpha_vec", vec, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim < 2:
            raise ValueError(f"logits must be (N, C[, ...]); got shape {tuple(logits.shape)}")

        log_probs = F.log_softmax(logits, dim=1)

        valid = target != self.ignore_index
        target_safe = target.clone()
        target_safe[~valid] = 0

        log_pt = log_probs.gather(1, target_safe.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        ce = -log_pt
        if self.label_smoothing > 0.0:
            smooth = -log_probs.mean(dim=1)
            ce = (1.0 - self.label_smoothing) * ce + self.label_smoothing * smooth

        focal_weight = (1.0 - pt).clamp(min=0.0).pow(self.gamma)
        loss = focal_weight * ce

        if self.alpha_vec.numel() > 0:
            alpha_t = self.alpha_vec.to(logits.device, logits.dtype)[target_safe]
            loss = alpha_t * loss
        elif self.alpha_scalar is not None:
            loss = self.alpha_scalar * loss

        loss = loss * valid.to(loss.dtype)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            denom = valid.sum().clamp(min=1).to(loss.dtype)
            return loss.sum() / denom
        return loss
