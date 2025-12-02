import pkgutil
import importlib
from pathlib import Path

# imports all models in this directory
models_dir = Path(__file__).parent / "models"
for module in pkgutil.iter_modules([str(models_dir)]):
    importlib.import_module(f"{__package__}.models.{module.name}")

from .model_registry import get


def get_architecture_bundle(name):
    bundle = get(name)
    return bundle
