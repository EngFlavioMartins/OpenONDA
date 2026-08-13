"""
Taichi backend initialization and management (compatibility shim).
==================================================================
The runtime backend lifecycle now lives in :mod:`source.solvers.VPM.runtime.backend`
so that ``config`` remains a leaf package (see ARCHITECTURE.md).  This module
re-exports the public API for backwards compatibility only; new code should
import from ``source.solvers.VPM.runtime.backend``.

Author: Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from ..runtime.backend import (  # noqa: F401
    _build_backend_chain,
    _clear_stale_taichi_cache,
    _cpu_candidates,
    _has_nvidia_gpu,
    _is_apple_silicon,
    _is_likely_integrated_gpu,
    _pool_bytes_from_kwargs,
    _probe_taichi_backend,
    _query_vulkan_budget,
    _resolve_gpu_backend,
    _safe_device_memory_for_init,
    initialize_taichi_backend,
    reset_taichi_backend,
)
