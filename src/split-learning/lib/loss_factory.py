"""Back-compat shim. Prefer importing from `..lib.losses` in new code."""

from .losses import get_loss, list_losses, register

__all__ = ["get_loss", "list_losses", "register"]
