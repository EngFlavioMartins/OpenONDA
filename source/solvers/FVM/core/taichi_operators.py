"""Taichi CPU kernels for deterministic finite-volume reductions."""

import os
from pathlib import Path
import tempfile

import numpy as np

_CACHE_DIR = Path(tempfile.gettempdir()) / f"openonda-taichi-{os.getpid()}"
os.environ.setdefault("TI_OFFLINE_CACHE_FILE_PATH", str(_CACHE_DIR))

try:
    import taichi as ti
except ImportError as error:  # pragma: no cover - validated by configuration tests
    raise RuntimeError(
        "operator_backend='taichi' requires the canonical Taichi dependency"
    ) from error


def _ensure_cpu_runtime() -> None:
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)


@ti.kernel
def _reduce_kernel(
    slots: ti.types.ndarray(dtype=ti.i64, ndim=1),
    contributions: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for index in range(contributions.shape[0]):
        ti.atomic_add(result[slots[index]], contributions[index])


@ti.kernel
def _rhs_kernel(
    flux: ti.types.ndarray(dtype=ti.f64, ndim=1),
    owners: ti.types.ndarray(dtype=ti.i64, ndim=1),
    neighbours: ti.types.ndarray(dtype=ti.i64, ndim=1),
    n_interior: ti.i64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for face in range(flux.shape[0]):
        ti.atomic_sub(result[owners[face]], flux[face])
        if face < n_interior:
            ti.atomic_add(result[neighbours[face]], flux[face])


def reduce_contributions(slots, contributions, size: int) -> np.ndarray:
    """Reduce face contributions into CSR entries on Taichi CPU."""
    _ensure_cpu_runtime()
    result = np.zeros(size, dtype=np.float64)
    _reduce_kernel(
        np.ascontiguousarray(slots, dtype=np.int64),
        np.ascontiguousarray(contributions, dtype=np.float64),
        result,
    )
    return result


def assemble_rhs(flux, owners, neighbours, n_elements: int, n_interior: int) -> np.ndarray:
    """Assemble an owned-cell RHS using Taichi CPU atomics."""
    _ensure_cpu_runtime()
    result = np.zeros(n_elements, dtype=np.float64)
    _rhs_kernel(
        np.ascontiguousarray(flux, dtype=np.float64),
        np.ascontiguousarray(owners, dtype=np.int64),
        np.ascontiguousarray(neighbours, dtype=np.int64),
        n_interior,
        result,
    )
    return result
