"""PyTest global configuration and early C-extension runtime initialization.

Pre-imports LightGBM to establish OpenMP runtime linkage before numpy/scipy
initialization on Windows platforms, preventing DLL runtime collision.
"""

from __future__ import annotations

import contextlib
import os

# Set joblib loky cpu count to avoid subprocess spawn on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

with contextlib.suppress(ImportError):
    import lightgbm  # noqa: F401
