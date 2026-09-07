"""Matrix workspaces follow the topology of each MPI partition."""

import numpy as np

from source.solvers.fvm.assemble import matrix_assembly
from source.solvers.fvm.solve.simple_solver import SIMPLESolver


def test_interior_only_partition_uses_an_interior_only_workspace():
    mesh = {
        "n_cells": 2,
        "n_faces": 1,
        "n_interior_faces": 1,
        "owners": np.array([0], dtype=np.int32),
        "neighbours": np.array([1], dtype=np.int32),
        "boundary": [],
    }
    solver = SIMPLESolver(mesh, {}, [])
    try:
        assert not solver._momentum_matrix_workspace.include_boundaries
        matrix = matrix_assembly.assemble_matrix_from_fluxes_vectorized(
            {
                "flux_cf": np.array([1.0]),
                "flux_ff": np.array([-1.0]),
                "flux_vf": np.array([0.0]),
            },
            mesh,
            workspace=solver._momentum_matrix_workspace,
        )
        assert matrix.shape == (2, 2)
    finally:
        solver.close()
