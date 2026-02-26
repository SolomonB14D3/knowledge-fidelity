"""Behavior plugin registry for rho-eval.

Each behavior is a class that inherits from ABCBehavior and is registered
via the @register decorator. External users can add custom behaviors by
subclassing ABCBehavior and calling register().

Usage:
    from rho_eval.behaviors import get_behavior, list_behaviors

    behavior = get_behavior("factual")
    probes = behavior.load_probes()
    result = behavior.evaluate(model, tokenizer, probes, device="mps")

Adding a custom behavior:
    from rho_eval.behaviors import ABCBehavior, register

    @register
    class MyBehavior(ABCBehavior):
        name = "my_behavior"
        description = "My custom behavioral probe"
        probe_type = "confidence"
        default_n = 50

        def load_probes(self, n=None, seed=42, **kwargs):
            ...
        def evaluate(self, model, tokenizer, probes, device="cpu", **kwargs):
            ...
"""

from .base import ABCBehavior, BehaviorResult

_REGISTRY: dict[str, type[ABCBehavior]] = {}


def register(cls: type[ABCBehavior]) -> type[ABCBehavior]:
    """Register a behavior class. Use as @register decorator or call directly."""
    _REGISTRY[cls.name] = cls
    return cls


def get_behavior(name: str) -> ABCBehavior:
    """Instantiate a registered behavior by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown behavior '{name}'. Available: {available}")
    return _REGISTRY[name]()


def list_behaviors() -> list[str]:
    """Return names of all registered behaviors."""
    return sorted(_REGISTRY.keys())


def get_all_behaviors() -> dict[str, ABCBehavior]:
    """Return dict of all registered behavior instances."""
    return {name: cls() for name, cls in sorted(_REGISTRY.items())}


# Auto-register built-in behaviors on import
from . import factual     # noqa: F401, E402
from . import toxicity    # noqa: F401, E402
from . import bias        # noqa: F401, E402
from . import sycophancy  # noqa: F401, E402
from . import reasoning   # noqa: F401, E402
from . import refusal     # noqa: F401, E402
from . import deception   # noqa: F401, E402
from . import overrefusal  # noqa: F401, E402

__all__ = [
    "ABCBehavior",
    "BehaviorResult",
    "register",
    "get_behavior",
    "list_behaviors",
    "get_all_behaviors",
]
