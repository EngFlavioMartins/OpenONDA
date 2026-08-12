"""
Taichi backend initialization and management.
=============================================
This module provides a centralized way to initialize Taichi with consistent
settings across different solvers (VPM, VLM, Couplers).

Author: Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import contextlib
import gc
import glob
import logging
import os
import platform
import re
import shutil
import subprocess
import sys

import taichi as ti

import source.solvers.VPM.config.constants as constants_module

_logger = logging.getLogger(__name__)


def _clear_stale_taichi_cache() -> None:
    """Remove all Taichi offline caches to prevent stale kernels.

    Taichi 1.7.x stores JIT-compiled kernels in ``~/.cache/taichi/ticache/``
    (or the path given by ``TI_OFFLINE_CACHE_FILE_PATH``).  Stale files from
    previous runs can cause ``Failed to allocate ext arr buffer`` errors even
    when ``offline_cache=False`` is passed to ``ti.init()``.

    This function clears **both** the default location and any custom path
    from the ``TI_OFFLINE_CACHE_FILE_PATH`` environment variable.
    """
    dirs_to_clear = [
        os.path.join(os.path.expanduser("~"), ".cache", "taichi", "ticache"),
    ]
    custom = os.environ.get("TI_OFFLINE_CACHE_FILE_PATH")
    if custom and os.path.isabs(custom):
        dirs_to_clear.append(custom)

    for cache_dir in dirs_to_clear:
        if os.path.isdir(cache_dir) and os.listdir(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                _logger.debug("Cleared stale Taichi cache at %s", cache_dir)
            except OSError as exc:
                _logger.debug("Could not clear Taichi cache at %s: %s", cache_dir, exc)

    # Keep an explicitly selected cache path active.  It has just been
    # cleared above, and callers use this override when the default user cache
    # is read-only (for example in a sandboxed batch run).


# -- Integrated-GPU detection -------------------------------------------
# On integrated GPUs the Vulkan heap "size" is total system RAM, but the
# actual usable "budget" is much smaller.  Taichi 1.7.x computes its
# memory pool as  heap_size × device_memory_fraction,  which can vastly
# exceed the budget, leaving nothing for ext-arr staging buffers.
#
# The helpers below detect this situation so that device_memory_fraction
# can be scaled down automatically.


def _query_vulkan_budget() -> tuple[int, int] | None:
    """Return (heap_size, heap_budget) in bytes for the first device-local heap.

    Uses ``vulkaninfo`` (if available) to parse VK_EXT_memory_budget data.
    Returns *None* when the information cannot be determined.
    """
    try:
        proc = subprocess.run(
            ["vulkaninfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        # Match: size = <bytes> ... budget = <bytes> (first occurrence)
        m = re.search(
            r"memoryHeaps\[\d+\]:\s*\n"
            r"\s*size\s*=\s*(\d+).*?\n"
            r"\s*budget\s*=\s*(\d+)",
            proc.stdout,
        )
        if m:
            return int(m.group(1)), int(m.group(2))
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _is_apple_silicon() -> bool:
    """True when running on Apple Silicon (arm64 Mac with unified memory)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _is_likely_integrated_gpu() -> bool:
    """Heuristic: ``True`` when the primary GPU uses shared system memory."""
    # Apple Silicon always uses unified memory (GPU shares system RAM).
    if _is_apple_silicon():
        return True
    # Intel i915 / Xe drivers on Linux → always integrated
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        driver = os.path.join(card, "device", "driver")
        if os.path.islink(driver):
            name = os.path.basename(os.readlink(driver))
            if name in ("i915", "xe"):
                return True
    return False


# Target pool size (bytes) on integrated GPUs.  A production treecode run owns
# considerably more than the particle container itself: the LBVH nodes, two
# traversal stacks, RK scratch fields, target fields, and a fixed GBD/DVH grid
# coexist.  The former 768 MiB pool could therefore be exhausted silently by a
# nominal 500k-particle coupled run even when the Vulkan heap still had several
# GiB available.  Keep a fixed (rather than heap-fraction) policy for unified
# memory, but reserve enough for the complete solver.  The runtime still caps
# this at 50% of the driver's current budget below.
_INTEGRATED_GPU_POOL_BYTES: int = 1536 * (1 << 20)  # 1.5 GiB


def _safe_device_memory_for_init(
    desired_fraction: float,
    backend: str = "VULKAN",
) -> dict[str, float]:
    """Return ``ti.init()`` memory kwargs appropriate for the current GPU.

    On **discrete GPUs** (or when detection fails), the user-supplied
    *desired_fraction* is returned as ``device_memory_fraction``.

    On **integrated GPUs / unified-memory architectures** the heap ``size``
    equals total system RAM, so a fraction-based approach is misleading.
    Instead we use Taichi's ``device_memory_GB`` parameter to set a small,
    fixed-size pool.

    **macOS / Metal**: Vulkan is not available, so Vulkan budget queries are
    skipped.  Apple Silicon uses unified memory (GPU shares system RAM), so
    a fixed 2 GiB pool is used.  Intel Macs with a dedicated GPU fall back
    to the fraction-based approach.

    Returns a dict with either ``{"device_memory_fraction": ...}`` or
    ``{"device_memory_GB": ...}``.
    """
    # -- Metal (macOS) path --------------------------------------------------
    if platform.system() == "Darwin" or backend == "METAL":
        if _is_apple_silicon():
            # Apple Silicon: GPU shares system RAM (unified memory).
            # Metal backend does NOT accept device_memory_GB / device_memory_fraction.
            print(
                "[OpenONDA] Apple Silicon (Metal) detected — "
                "Taichi memory pool managed automatically.",
                file=sys.stderr,
            )
            return {}
        # Intel Mac with discrete GPU: Metal still manages memory itself.
        return {}

    # -- Linux / Vulkan / CUDA path --------------------------------------
    if backend in {"CUDA"}:
        return {"device_memory_fraction": desired_fraction}

    is_integrated = False
    budget_info = _query_vulkan_budget()

    if budget_info is not None:
        heap_size, heap_budget = budget_info
        # NVIDIA Optimus / hybrid laptops report an inflated device-local
        # heap_size that combines VRAM + BAR regions (e.g. 11 GiB on a 6 GiB
        # card).
        if 0 < heap_budget < heap_size * 0.8 and heap_budget < 4 * (1 << 30):
            is_integrated = True

    if not is_integrated:
        is_integrated = _is_likely_integrated_gpu()

    if is_integrated:
        pool = _INTEGRATED_GPU_POOL_BYTES
        # Never use more than 50 % of the current Vulkan budget.
        if budget_info is not None:
            _, heap_budget = budget_info
            pool = min(pool, int(heap_budget * 0.5))
        pool_mb = pool / (1 << 20)
        pool_gb = pool / (1 << 30)
        budget_mb = budget_info[1] / (1 << 20) if budget_info else -1

        print(
            f"[OpenONDA] Integrated GPU detected — "
            f"Taichi pool: {pool_mb:.0f} MB  "
            f"(Vulkan budget: {budget_mb:.0f} MB)",
            file=sys.stderr,
        )
        return {"device_memory_GB": pool_gb}

    return {"device_memory_fraction": desired_fraction}


def _pool_bytes_from_kwargs(memory_kwargs: dict, backend: str) -> int | None:
    """Taichi pool size in bytes implied by the chosen init kwargs, or None."""
    if "device_memory_GB" in memory_kwargs:
        return int(float(memory_kwargs["device_memory_GB"]) * (1 << 30))
    if backend in {"CUDA", "VULKAN"} and "device_memory_fraction" in memory_kwargs:
        budget = _query_vulkan_budget()
        if budget is not None:
            return int(budget[1] * float(memory_kwargs["device_memory_fraction"]))
    return None


_PRECISION_MAP: dict[str, tuple] = {
    "f32": (ti.f32, ti.i32),
    # Floating-point precision does not require 64-bit field indices.  Taichi
    # fields and particle counts use i32; making every integer literal i64 in
    # an f64 run produces thousands of implicit-index and atomic-cast warnings
    # without extending any practical VPM allocation limit.
    "f64": (ti.f64, ti.i32),
}


def _has_nvidia_gpu() -> bool:
    """True when ``nvidia-smi`` reports at least one working CUDA device."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.returncode == 0 and bool(proc.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _resolve_gpu_backend() -> tuple:
    """Return the best GPU ``(ti_arch, name)`` for the current platform.

    Resolution order:
    * **macOS**             → Metal  (only GPU API available on Apple hardware)
    * **Linux / Windows**   → CUDA   (when ``nvidia-smi`` reports a working device)
    * **Linux / Windows**   → Vulkan (universal fallback; works on AMD, Intel, and
                                      NVIDIA cards even without the CUDA toolkit)

    This function never raises; it always returns a valid ``(arch, name)`` pair.
    """
    if platform.system() == "Darwin":
        return (ti.metal, "METAL")
    if _has_nvidia_gpu():
        return (ti.cuda, "CUDA")
    # Default GPU on Linux / Windows: Vulkan (driver-agnostic).
    return (ti.vulkan, "VULKAN")


def _cpu_candidates() -> list[tuple]:
    """Ordered list of CPU ``(arch, name)`` pairs to try as the final fallback."""
    cands: list[tuple] = []
    if hasattr(ti, "cpu"):
        cands.append((ti.cpu, "CPU"))
    if hasattr(ti, "x64") and ti.x64 != getattr(ti, "cpu", None):
        cands.append((ti.x64, "CPU"))
    return cands or [(ti.cpu, "CPU")]


def _build_backend_chain(preferred_backend: str, precision: str = "f32") -> list[tuple]:
    """Return compatible backends without silently replacing a GPU by the CPU."""
    cpu_chain = _cpu_candidates()
    system = platform.system()

    if preferred_backend == "CPU":
        return cpu_chain

    if preferred_backend == "METAL":
        if system != "Darwin":
            raise ValueError("processing_unit='METAL' is supported only on macOS")
        if precision == "f64":
            raise ValueError(
                "precision='f64' is not supported by the Metal backend; use f32 or CPU"
            )
        return [(ti.metal, "METAL")]

    if preferred_backend == "CUDA":
        return [(ti.cuda, "CUDA")]

    if preferred_backend == "VULKAN":
        if system == "Darwin":
            raise ValueError("processing_unit='VULKAN' is unavailable on macOS; use AUTO or METAL")
        return [(ti.vulkan, "VULKAN")]

    if system == "Darwin":
        if precision == "f64":
            raise ValueError(
                "precision='f64' is not supported by the macOS GPU backend; "
                "request CPU explicitly or use f32"
            )
        return [(ti.metal, "METAL")]

    vulkan = (ti.vulkan, "VULKAN")
    cuda = (ti.cuda, "CUDA")
    best = _resolve_gpu_backend()
    gpu_order = [best, cuda if best[1] != "CUDA" else vulkan]
    chain: list[tuple] = []
    for cand in gpu_order:
        if cand not in chain:
            chain.append(cand)
    return chain


def reset_taichi_backend() -> None:
    """Fully reset the Taichi runtime, releasing all GPU memory.

    Call this **before** creating a new :class:`Solver` when running multiple
    VPM simulations sequentially in the same Python process.  After this call
    every Taichi field, kernel, and ndarray from the previous run is invalid;
    the next :class:`Solver` constructor will re-initialise Taichi from scratch.

    Typical usage in a script that runs several cases back-to-back::

        from source.solvers.VPM import Solver, VPMSetup

        for case in cases:
            Solver.reset_gpu()          # free all GPU memory from previous run
            solver = Solver(setup=case)
            for _ in range(num_steps):
                solver.update_state()

    This prevents the ``Failed to allocate ext arr buffer`` Taichi error that
    occurs when accumulated GPU allocations leave no room for staging buffers.
    """
    # Flush all pending GPU work before tearing down the runtime so the
    # Vulkan driver can release resources immediately.
    with contextlib.suppress(Exception):
        ti.sync()
    with contextlib.suppress(Exception):
        ti.reset()
    # Force CPython to destroy C++ Taichi objects now (leaked fields,
    # reference cycles, etc.) so the Vulkan device is fully cleaned up
    # before the process exits or the next ti.init() runs.
    gc.collect()
    # Clear the cached backend flag so the next init runs unconditionally.
    constants_module.TAICHI_BACKEND = "UNKNOWN"


def _probe_taichi_backend() -> None:
    """Verify that the initialized backend can allocate and access a field."""
    probe = ti.field(dtype=ti.i32, shape=())
    probe[None] = 1
    if probe[None] != 1:
        raise RuntimeError("Taichi backend field probe failed")


def initialize_taichi_backend(
    preferred_backend: str = "AUTO",
    debug_mode: bool = False,
    precision: str = "f32",
    device_memory_fraction: float = 0.5,
    random_seed: int = 42,
) -> str:
    """
    Initialize Taichi with user-specified backend and precision settings.

    ``AUTO`` selects a compatible GPU for f32 runs. CPU execution must be
    requested explicitly, so a missing GPU cannot turn a production run into
    an unexpectedly slow CPU run.

    Args:
          preferred_backend: ``'AUTO'``, ``'METAL'``, ``'VULKAN'``,
              ``'CUDA'``, or ``'CPU'``.
          debug_mode: Enable Taichi debug features (default ``False``).
          precision: Floating-point precision — ``'f32'`` (default) or
              ``'f64'`` (CPU only; Vulkan/Metal have limited f64 support).
          device_memory_fraction: Fraction of GPU VRAM reserved for
              Taichi's internal memory pool (default 0.5).  Lower this
              value (e.g. 0.3) if you see ``Failed to allocate ext arr
              buffer`` errors.  Clamped to [0.1, 0.7].
          random_seed: Seed for Taichi's RNG (default 42).  The Random Walk
              Method draws its Brownian displacements from it, so a fixed seed
              makes a run reproducible but makes every run of an ensemble
              identical; vary it across members to average the stochastic
              diffusion.  Ignored on Metal, which does not accept the kwarg.

    Returns:
          str: Name of the successfully initialised backend
              (``'METAL'``, ``'VULKAN'``, ``'CUDA'``, or ``'CPU'``).
    """
    # Check if Taichi is already initialized
    if ti.lang.impl.get_runtime().prog is not None:
        return getattr(constants_module, "TAICHI_BACKEND", "INITIALIZED")

    if precision not in _PRECISION_MAP:
        raise ValueError(f"precision must be 'f32' or 'f64', got '{precision}'")

    _VALID_ENV_KEYS = {"AUTO", "CPU", "METAL", "VULKAN", "CUDA"}
    env_unit = os.environ.get("OPENONDA_PROCESSING_UNIT", "").strip().upper()
    if preferred_backend == "AUTO" and env_unit in _VALID_ENV_KEYS:
        preferred_backend = env_unit

    # Build the ordered list of compatible backends.
    chain = _build_backend_chain(preferred_backend, precision)
    strict_gpu = preferred_backend in {"AUTO", "METAL", "VULKAN", "CUDA"} and precision == "f32"

    # Clamp to a safe range.
    clamped_fraction = max(0.1, min(device_memory_fraction, 0.7))
    if clamped_fraction != device_memory_fraction:
        print(
            f"[OpenONDA] device_memory_fraction={device_memory_fraction:.3g} is outside the "
            f"safe GPU range [0.1, 0.7]; using {clamped_fraction:.3g} to avoid allocation "
            f"failures.",
            file=sys.stderr,
        )
    device_memory_fraction = clamped_fraction

    default_fp, default_ip = _PRECISION_MAP[precision]

    last_exc: Exception | None = None
    for arch, name in chain:
        # Per-backend memory kwargs: Metal manages its own pool; CUDA/Vulkan
        # use integrated-GPU detection; CPU takes a host memory fraction.
        if name == "CPU":
            memory_kwargs: dict = {"device_memory_fraction": device_memory_fraction}
        elif name == "METAL":
            memory_kwargs = {}  # Metal does not accept device_memory_* kwargs
        else:
            memory_kwargs = _safe_device_memory_for_init(device_memory_fraction, name)

        # Metal does not accept advanced_optimization / random_seed — keep its
        # init kwargs minimal.
        init_kwargs = {
            "arch": arch,
            "default_fp": default_fp,
            "default_ip": default_ip,
            "debug": debug_mode,
            "kernel_profiler": False,
            "offline_cache": False,
        }
        if name != "METAL":
            init_kwargs["random_seed"] = random_seed
            init_kwargs["advanced_optimization"] = True
        if name == "CPU":
            cpu_threads = os.environ.get("OPENONDA_CPU_THREADS", "").strip()
            if cpu_threads:
                init_kwargs["cpu_max_num_threads"] = max(1, int(cpu_threads))
        if memory_kwargs:
            init_kwargs.update(memory_kwargs)
        constants_module.TAICHI_POOL_BYTES = _pool_bytes_from_kwargs(memory_kwargs, name)

        try:
            ti.init(**init_kwargs)
            active_arch = ti.lang.impl.current_cfg().arch
            if active_arch != arch:
                raise RuntimeError(
                    f"Taichi requested {name} but initialized {active_arch}; "
                    "refusing an implicit backend fallback"
                )
            _probe_taichi_backend()
            constants_module.TAICHI_BACKEND = name
            _logger.info("Taichi initialized: backend=%s, precision=%s", name, precision)
            return name
        except Exception as exc:
            last_exc = exc
            _logger.debug("Backend '%s' init failed: %s", name, exc)
            # Reset any partial runtime state before trying the next candidate.
            with contextlib.suppress(Exception):
                ti.reset()

    if strict_gpu:
        raise RuntimeError(
            f"Requested Taichi backend {preferred_backend} failed to initialise"
        ) from last_exc
    _logger.error("All Taichi backends failed to initialise (last error: %s)", last_exc)
    return getattr(constants_module, "TAICHI_BACKEND", "UNKNOWN")
