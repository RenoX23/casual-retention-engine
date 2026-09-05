"""Causal-Retain: Uplift Modeling & Customer Revenue Recovery Engine."""

import contextlib

# Ensure LightGBM and OpenMP runtime DLLs initialize first on Windows
with contextlib.suppress(ImportError):
    import lightgbm  # noqa: F401

__version__ = "0.1.0"
