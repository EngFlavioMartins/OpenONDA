"""Taichi CPU kernels for deterministic finite-volume reductions."""

import os
from pathlib import Path
import tempfile

import numpy as np

_CACHE_DIR = Path(tempfile.gettempdir()) / f"openonda-taichi-{os.getpid()}"
os.environ.setdefault("TI_OFFLINE_CACHE_FILE_PATH", str(_CACHE_DIR))

import taichi as ti  # noqa: E402 - the cache path above must be set before Taichi loads

from openonda.vpm.runtime.backend import start_taichi  # noqa: E402


@ti.kernel
def _rhs_kernel(
    flux: ti.types.ndarray(dtype=ti.f64, ndim=1),  # type: ignore[invalid-annotation]
    owners: ti.types.ndarray(dtype=ti.i64, ndim=1),  # type: ignore[invalid-annotation]
    neighbours: ti.types.ndarray(dtype=ti.i64, ndim=1),  # type: ignore[invalid-annotation]
    n_interior: ti.i64,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),  # type: ignore[invalid-annotation]
) -> None:
    for face in range(flux.shape[0]):
        ti.atomic_sub(result[owners[face]], flux[face])
        if face < n_interior:
            ti.atomic_add(result[neighbours[face]], flux[face])


def assemble_rhs(
    flux: np.ndarray, owners: np.ndarray, neighbours: np.ndarray, n_cells: int, n_interior: int
) -> np.ndarray:
    """Assemble an owned-cell RHS using Taichi CPU atomics."""
    start_taichi(None, device="cpu", precision="f64")
    result = np.zeros(n_cells, dtype=np.float64)
    _rhs_kernel(
        np.ascontiguousarray(flux, dtype=np.float64),
        np.ascontiguousarray(owners, dtype=np.int64),
        np.ascontiguousarray(neighbours, dtype=np.int64),
        n_interior,
        result,
    )
    return result
