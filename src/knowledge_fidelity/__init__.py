"""
Knowledge Fidelity — Backward-compatible shim for rho-eval.

This package has been renamed to rho-eval. All imports are re-exported
from the rho_eval package. Please update your imports:

    # Old:
    from knowledge_fidelity import audit
    # New:
    from rho_eval import audit

All v1 and v2 APIs are available through either import path.
"""

# Re-export everything from rho_eval
from rho_eval import *  # noqa: F401, F403
from rho_eval import __version__, __all__  # noqa: F401
