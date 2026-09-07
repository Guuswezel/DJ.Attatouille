"""Offline transition-learning tools for DJ.Attatouille.

Nothing in this package is imported by the live audio worker.  Training
exports a compact, versioned JSON policy which the worker evaluates with
NumPy during transition planning.
"""

from .config import FEATURE_VERSION

__all__ = ["FEATURE_VERSION"]
