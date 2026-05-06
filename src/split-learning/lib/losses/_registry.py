"""Loss registry + factory."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Union

import torch.nn as nn


_Builder = Callable[..., nn.Module]
_REGISTRY: Dict[str, _Builder] = {}


def register(name: str, builder: _Builder | None = None):
    """Register a loss under name. Use as a decorator or direct call."""
    key = name.strip().lower()

    def _do(b: _Builder) -> _Builder:
        if key in _REGISTRY:
            raise ValueError(f"Loss {name!r} already registered")
        _REGISTRY[key] = b
        return b

    if builder is not None:
        return _do(builder)
    return _do


def list_losses() -> list[str]:
    return sorted(_REGISTRY.keys())


LossSpec = Union[str, Mapping[str, Any], None]


def get_loss(spec: LossSpec = "CE") -> nn.Module:
    """Build a loss module from a name string or mapping spec."""
    if spec is None:
        spec = "CE"

    if isinstance(spec, str):
        name, kwargs = spec, {}
    elif isinstance(spec, Mapping):
        if "name" not in spec:
            raise ValueError(f"loss_function dict needs 'name' key, got {dict(spec)!r}")
        kwargs = {k: v for k, v in spec.items() if k != "name"}
        name = spec["name"]
    else:
        raise ValueError(f"loss_function must be str or mapping, got {type(spec).__name__}")

    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown loss_function {name!r}. Registered: {', '.join(list_losses())}")
    return _REGISTRY[key](**kwargs)
