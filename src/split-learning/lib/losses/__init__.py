"""Loss registry. Importing this package registers all bundled losses."""

from ._registry import get_loss, list_losses, register

from . import builtins  # noqa: F401
from . import focal     # noqa: F401

__all__ = ["get_loss", "list_losses", "register"]
