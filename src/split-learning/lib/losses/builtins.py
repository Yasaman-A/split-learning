"""Registrations for standard torch loss modules."""

from typing import Any

import torch
import torch.nn as nn

from ._registry import register


def _as_float_tensor(x):
    if x is None or isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.float32)


@register("ce")
def _ce(**kw: Any) -> nn.Module:
    if "weight" in kw:
        kw["weight"] = _as_float_tensor(kw["weight"])
    return nn.CrossEntropyLoss(**kw)
