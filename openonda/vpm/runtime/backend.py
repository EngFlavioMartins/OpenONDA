"""
Taichi runtime start-up for the package.
========================================
:func:`start_taichi` is the only caller of ``ti.init`` in OpenONDA. It starts
the process-global Taichi runtime on the requested device, or joins a running
runtime that is on that device, and raises otherwise.
"""

import contextlib
import gc
import glob
from io import StringIO
import os
import platform
import re
import subprocess
import threading
import weakref

import taichi as ti

from openonda.runtime import worker_thread_count
from openonda.vpm.io.logging import Logging

# Taichi owns one process-global runtime.  VPM solvers may share that runtime
# only when they use the same device and precision.  A weak solver set
# avoids keeping abandoned interactive solver objects alive, while ensuring
# that closing one solver never silently tears down another live solver.
# ``_BACKEND_CONFIGURATION`` is the (device, precision) of a runtime started
# here; it is ``None`` when no runtime was started by this module.
_BACKEND_SOLVERS: weakref.WeakSet[object] = weakref.WeakSet()
_BACKEND_CONFIGURATION: tuple[str, str] | None = None
_BACKEND_SOLVER_LOCK = threading.RLock()


# -- Integrated-GPU detection -------------------------------------------
# On integrated GPUs the Vulkan heap "size" is total system RAM, but the
# actual usable "budget" is much smaller.  Taichi 1.7.x computes its
# memory pool as  heap_size × device_memory_fraction,  which can vastly
# exceed the budget, leaving nothing for ext-arr staging buffers.
#
# Size the Vulkan pool from its reported usable budget.


def _query_vulkan_budget() -> tuple[int, int] | None:
    """Return (heap_size, heap_budget) in bytes for the first device-local heap.

    Uses ``vulkaninfo`` (if available) to parse VK_EXT_memory_budget data.
    Returns *None* when the information cannot be determined.
    """
    try:
        # Memory probing needs no window surface. A stale DISPLAY (common in
        # SSH, batch jobs and desktop sandboxes) makes vulkaninfo abort before
        # reporting otherwise usable compute devices.
        probe_environment = os.environ.copy()
        probe_environment.pop("DISPLAY", None)
        proc = subprocess.run(
            ["vulkaninfo"],
            capture_output=True,
            text=True,
            timeout=10,
            env=probe_environment,
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


def _is_likely_integrated_gpu() -> bool:
    """Heuristic: ``True`` when the primary GPU uses shared system memory."""
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
# GiB available. Keep a fixed (rather than heap-fraction) allocation for unified
# memory, but reserve enough for the complete solver.  The runtime still caps
# this at 50% of the driver's current budget below.
_INTEGRATED_GPU_POOL_BYTES: int = 1536 * (1 << 20)  # 1.5 GiB


def _safe_device_memory_for_init(
    desired_fraction: float,
    backend: str = "vulkan",
    minimum_pool_bytes: int = 0,
) -> dict[str, float]:
    """Return ``ti.init()`` memory kwargs for a CUDA or Vulkan device.

    On **discrete GPUs** (or when detection fails), the internal
    *desired_fraction* is returned as ``device_memory_fraction``.

    On **integrated GPUs / unified-memory architectures** the heap ``size``
    equals total system RAM, so a fraction-based approach is misleading.
    Instead we use Taichi's ``device_memory_GB`` parameter to set a small,
    fixed-size pool.

    ``minimum_pool_bytes`` sizes an integrated-GPU pool for a known fixed
    workspace, subject to the driver's current memory budget.
    """
    if backend == "cuda":
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
        pool = max(_INTEGRATED_GPU_POOL_BYTES, minimum_pool_bytes)
        # Large fixed workspaces need more than the default pool, while leaving
        # room in the device budget for external-array transfers.
        if budget_info is not None:
            _, heap_budget = budget_info
            budget_share = 0.7 if minimum_pool_bytes > _INTEGRATED_GPU_POOL_BYTES else 0.5
            pool = min(pool, int(heap_budget * budget_share))
        pool_mb = pool / (1 << 20)
        pool_gb = pool / (1 << 30)
        budget_mb = budget_info[1] / (1 << 20) if budget_info else -1

        Logging.record(
            f"backend {backend}",
            ("memory pool", f"{pool_mb:.0f}", "MiB"),
            ("device budget", f"{budget_mb:.0f}", "MiB"),
            ("gpu type", "integrated"),
        )
        return {"device_memory_GB": pool_gb}

    return {"device_memory_fraction": desired_fraction}


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
    """Return the GPU ``(ti_arch, name)`` that ``auto`` selects on this platform.

    Resolution order:
    * **macOS**             → Metal  (only GPU API available on Apple hardware)
    * **Linux / Windows**   → CUDA   (when ``nvidia-smi`` reports a working device)
    * **Linux / Windows**   → Vulkan (AMD, Intel, and NVIDIA cards without the
                                      CUDA toolkit)
    """
    if platform.system() == "Darwin":
        return (ti.metal, "metal")
    if _has_nvidia_gpu():
        return (ti.cuda, "cuda")
    return (ti.vulkan, "vulkan")


def _select_backend(
    device: str,
    precision: str,
    supported_devices: set[str] | frozenset[str] | None,
) -> tuple:
    """Return the one ``(ti_arch, name)`` that ``device`` starts, or raise."""
    system = platform.system()
    if device == "cpu":
        selected = (ti.cpu, "cpu")
    elif device == "metal":
        if system != "Darwin":
            raise ValueError("compute_device='metal' is supported only on macOS")
        if precision == "f64":
            raise ValueError(
                "precision='f64' is not supported by the Metal backend; use f32 or CPU"
            )
        selected = (ti.metal, "metal")
    elif device == "cuda":
        selected = (ti.cuda, "cuda")
    elif device == "vulkan":
        if system == "Darwin":
            raise ValueError("compute_device='vulkan' is unavailable on macOS; use AUTO or METAL")
        selected = (ti.vulkan, "vulkan")
    elif device == "auto":
        if system == "Darwin" and precision == "f64":
            raise ValueError(
                "precision='f64' is not supported by the macOS GPU backend; "
                "request CPU explicitly or use f32"
            )
        selected = _resolve_gpu_backend()
    else:
        raise ValueError(f"Unknown compute_device {device!r}")
    if supported_devices is not None and selected[1] not in supported_devices:
        raise ValueError(
            f"compute_device={device!r} selects {selected[1]!r}, which this numerical "
            f"method does not support; supported devices: {sorted(supported_devices)}"
        )
    return selected


def reset_taichi_backend(*, solver: object | None = None) -> None:
    """Fully reset the Taichi runtime, releasing all GPU memory.

    Call this **after** the active :class:`VPMSolver` has been closed, or pass
    that solver as ``solver`` from its cleanup path.  Resetting a runtime owned
    by another live solver is rejected because every Taichi field, kernel, and
    ndarray from that solver becomes invalid after ``ti.reset()``.

    Typical usage in a script that runs several cases back-to-back::

        from openonda.vpm import VPMSolver
        from openonda.vpm.runtime.backend import reset_taichi_backend

        for case in cases:
            reset_taichi_backend()         # free all GPU memory from previous run
            solver = VPMSolver(case)
            for _ in range(n_steps):
                solver.advance()

    This prevents the ``Failed to allocate ext arr buffer`` Taichi error that
    occurs when accumulated GPU allocations leave no room for staging buffers.
    """
    global _BACKEND_CONFIGURATION
    with _BACKEND_SOLVER_LOCK:
        # Remove weak solver references held only by abandoned reference cycles before
        # enforcing the live-solver safety check.  An explicit class-level
        # reset should be useful for ordinary sequential scripts even when a
        # previous interactive solver forgot to call ``close()``.
        gc.collect()
        if solver is None and _BACKEND_SOLVERS:
            raise RuntimeError(
                "Cannot reset the Taichi backend while a VPM solver is live; "
                "close all live solvers first"
            )
        if solver is not None:
            if solver not in _BACKEND_SOLVERS:
                return
            _BACKEND_SOLVERS.discard(solver)
            if _BACKEND_SOLVERS:
                # Other compatible solvers still use this process-global
                # runtime.  Releasing this scope must not invalidate them.
                return
        try:
            _reset_runtime()
            # Force CPython to destroy C++ Taichi objects now (leaked fields,
            # reference cycles, etc.) so the Vulkan device is fully cleaned up
            # before the process exits or the next ti.init() runs.
            gc.collect()
        finally:
            _BACKEND_CONFIGURATION = None


def _reset_runtime() -> None:
    """Finish the queued device work and destroy the Taichi runtime."""
    if ti.lang.impl.get_runtime().prog is not None:
        # Flush all pending GPU work before tearing down the runtime so the
        # Vulkan driver can release resources immediately.
        ti.sync()
    ti.reset()


def _probe_taichi_backend() -> None:
    """Verify that the initialized backend can allocate and access a field."""
    probe = ti.field(dtype=ti.i32, shape=())
    probe[None] = 1
    if probe[None] != 1:
        raise RuntimeError("Taichi backend field probe failed")


def _active_backend_name() -> str:
    """Return the device name of the running Taichi runtime."""
    active_arch = ti.lang.impl.current_cfg().arch
    for arch, name in (
        (ti.cpu, "cpu"),
        (ti.vulkan, "vulkan"),
        (ti.cuda, "cuda"),
        (ti.metal, "metal"),
    ):
        if active_arch == arch:
            return name
    raise RuntimeError(f"The running Taichi runtime uses the unsupported arch {active_arch}")


def _active_precision() -> str:
    """Return the default floating-point precision of the running Taichi runtime."""
    return "f64" if ti.lang.impl.current_cfg().default_fp == ti.f64 else "f32"


def start_taichi(
    solver: object | None,
    *,
    device: str,
    precision: str,
    debug: bool = False,
    random_seed: int = 42,
    supported_devices: set[str] | frozenset[str] | None = None,
    minimum_pool_bytes: int = 0,
) -> str:
    """Start the Taichi runtime on ``device``, or join the running one.

    A running runtime is joined when it was started here on the requested
    device (``auto`` joins it on any device) at the requested precision, or
    when the caller asks for the CPU and a CPU runtime at that precision was
    started elsewhere.  Every other request on a running runtime raises; the
    device is never substituted.  Without a running runtime, ``auto`` resolves
    to one GPU (Metal, CUDA or Vulkan), which is logged, and a failed start
    raises.

    Parameters
    ----------
    solver : object or None
        Solver that holds the runtime until :func:`reset_taichi_backend` is
        called with it; ``None`` for callers that do not hold it.
    device : {"auto", "cpu", "vulkan", "cuda", "metal"}
        Requested compute device.
    precision : {"f32", "f64"}
        Default floating-point precision of the runtime.
    debug : bool, default=False
        Enable the Taichi debug mode.
    random_seed : int, default=42
        Seed of Taichi's RNG; ignored on Metal.
    supported_devices : set of str, optional
        Devices qualified for the numerical method.
    minimum_pool_bytes : int, default=0
        Fixed workspace that an integrated-GPU memory pool must hold.

    Returns
    -------
    str
        Device of the runtime: ``"cpu"``, ``"vulkan"``, ``"cuda"`` or ``"metal"``.

    Raises
    ------
    RuntimeError
        If the running runtime is not on the requested device or precision,
        or if Taichi cannot start the requested device.
    ValueError
        If the request is invalid for this platform or numerical method.
    """
    global _BACKEND_CONFIGURATION
    device = str(device).lower()
    precision = str(precision).lower()
    if precision not in _PRECISION_MAP:
        raise ValueError(f"precision must be 'f32' or 'f64', got '{precision}'")
    with _BACKEND_SOLVER_LOCK:
        if ti.lang.impl.get_runtime().prog is None:
            _BACKEND_CONFIGURATION = None
        if _BACKEND_CONFIGURATION is not None and (
            _BACKEND_CONFIGURATION[1] != precision
            or device not in {"auto", _BACKEND_CONFIGURATION[0]}
        ):
            # A solver abandoned without ``close()`` can be held in a
            # reference cycle; collect it before deciding whether the runtime
            # it started is still in use.
            gc.collect()
            if not _BACKEND_SOLVERS:
                _reset_runtime()
                _BACKEND_CONFIGURATION = None

        if ti.lang.impl.get_runtime().prog is not None:
            resolved = _join_runtime(device, precision, supported_devices)
        else:
            resolved = _initialize_runtime(
                device, precision, debug, random_seed, supported_devices, minimum_pool_bytes
            )
        _BACKEND_CONFIGURATION = (resolved, precision)
        if solver is not None:
            _BACKEND_SOLVERS.add(solver)
        return resolved


def _join_runtime(
    device: str,
    precision: str,
    supported_devices: set[str] | frozenset[str] | None,
) -> str:
    """Return the device of the running runtime if ``device`` may use it, or raise."""
    active = _active_backend_name()
    active_precision = _active_precision()
    started_here = _BACKEND_CONFIGURATION is not None and _BACKEND_CONFIGURATION[0] == active
    if started_here:
        if device not in {"auto", active}:
            raise RuntimeError(
                f"The Taichi runtime runs on {active!r} for a live solver; "
                f"compute_device={device!r} was requested. Close the live solvers first."
            )
    elif device != "cpu" or active != "cpu":
        raise RuntimeError(
            f"A Taichi runtime on {active!r} was started outside OpenONDA; "
            f"compute_device={device!r} cannot use it. Call ti.reset() before building "
            "the solver, or request compute_device='cpu' on a CPU runtime."
        )
    if active_precision != precision:
        raise RuntimeError(
            f"The running Taichi runtime has precision={active_precision!r}; "
            f"precision={precision!r} was requested"
        )
    if supported_devices is not None and active not in supported_devices:
        raise RuntimeError(
            f"The running Taichi runtime on {active!r} is not supported by this "
            f"numerical method; supported devices: {sorted(supported_devices)}"
        )
    if device == "auto":
        Logging.info(f"compute device auto resolved to the running {active} runtime")
    return active


def _initialize_runtime(
    device: str,
    precision: str,
    debug: bool,
    random_seed: int,
    supported_devices: set[str] | frozenset[str] | None,
    minimum_pool_bytes: int,
) -> str:
    """Start Taichi on the one backend that ``device`` selects, or raise."""
    arch, name = _select_backend(device, precision, supported_devices)
    if device == "auto":
        Logging.info(f"compute device auto resolved to {name}")
    device_memory_fraction = 0.5
    default_fp, default_ip = _PRECISION_MAP[precision]
    init_kwargs = {
        "arch": arch,
        "default_fp": default_fp,
        "default_ip": default_ip,
        "debug": debug,
        "kernel_profiler": False,
        "offline_cache": False,
        "cpu_max_num_threads": worker_thread_count(),
    }
    # Metal manages its own memory pool and accepts no random_seed or
    # advanced_optimization.
    if name == "cpu":
        init_kwargs["device_memory_fraction"] = device_memory_fraction
    elif name != "metal":
        init_kwargs.update(
            _safe_device_memory_for_init(device_memory_fraction, name, minimum_pool_bytes)
        )
    if name != "metal":
        init_kwargs["random_seed"] = random_seed
        init_kwargs["advanced_optimization"] = True

    try:
        startup_output = StringIO()
        try:
            with contextlib.redirect_stdout(startup_output):
                ti.init(**init_kwargs)
        finally:
            for line in startup_output.getvalue().splitlines():
                if not line.startswith("[Taichi] Starting on arch="):
                    Logging.info(f"Taichi runtime: {line}")
        active_arch = ti.lang.impl.current_cfg().arch
        if active_arch != arch:
            raise RuntimeError(
                f"Taichi requested {name} but initialized {active_arch}; "
                "refusing an implicit backend fallback"
            )
        _probe_taichi_backend()
    except Exception as error:
        if ti.lang.impl.get_runtime().prog is not None:
            ti.reset()
        raise RuntimeError(
            f"Taichi could not start compute_device={device!r} on {name} at precision "
            f"{precision}; the error is chained below"
        ) from error
    Logging.info(f"backend {name} initialized at precision {precision}")
    return name
