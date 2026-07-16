"""Preconditioner rebuild telemetry for static-topology solves."""

import numpy as np
from scipy.sparse import diags

from source.solvers.FVM.solve import linear_interface


def _system():
    n = 40
    matrix = diags((-np.ones(n - 1), 4.0 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1))
    return matrix.tocsr(), np.linspace(1.0, 2.0, n)


def test_ilu_rebuild_and_reuse_are_reported():
    linear_interface._ILU_CACHE.clear()
    matrix, rhs = _system()
    options = {
        "method": "bicgstab",
        "equation_type": "momentum",
        "reuse_ilu": True,
        "ilu_key": "telemetry",
        "tol": 1.0e-10,
        "return_info": True,
    }
    _, first = linear_interface.solve_linear_system(matrix, rhs, **options)
    _, second = linear_interface.solve_linear_system(matrix, rhs, **options)
    assert first.preconditioner_rebuilt is True
    assert second.preconditioner_rebuilt is False


def test_amg_rebuild_and_reuse_are_reported():
    linear_interface._AMG_CACHE.clear()
    matrix, rhs = _system()
    options = {
        "method": "amg",
        "equation_type": "pressure",
        "tol": 1.0e-10,
        "amg_tol": 1.0e-10,
        "return_info": True,
    }
    _, first = linear_interface.solve_linear_system(matrix, rhs, **options)
    _, second = linear_interface.solve_linear_system(matrix, rhs, **options)
    assert first.preconditioner_rebuilt is True
    assert second.preconditioner_rebuilt is False
