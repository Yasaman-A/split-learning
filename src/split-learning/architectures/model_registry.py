import sys

REGISTRY = {}


def register(name):
    def decorator(fn):
        REGISTRY[name.lower()] = fn
        return fn

    return decorator


def get(name):
    key = name.lower()
    if key not in REGISTRY:
        print(
            f"Error: Model Architecture {key} not included in registry. Did you define the model correctly?"
        )
        print(f"Registry contains: {REGISTRY}")
        sys.exit("Exiting due to unknown architecture")
    return REGISTRY[key]()
