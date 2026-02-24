"""Backward compatibility shim for knowledge_fidelity imports.

This module is imported by the knowledge_fidelity package (if installed
alongside rho_eval) to re-export the new API under the old name.

Users who have `from knowledge_fidelity import X` in existing code will
continue to work after the rename.
"""

# This file exists as documentation. The actual backward compat is provided
# by keeping the knowledge_fidelity/ package in the source tree, which
# imports directly from its own modules (unchanged from v1.x).
#
# For PyPI, we'll ship a thin `knowledge-fidelity` v1.4.0 package that
# depends on `rho-eval>=2.0` and re-exports everything.
