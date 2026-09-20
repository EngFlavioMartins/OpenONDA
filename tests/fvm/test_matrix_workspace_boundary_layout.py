"""Matrix workspaces follow the topology of each MPI partition."""

import numpy as np

from source.solvers.fvm.assemble import matrix_assembly
from source.solvers.fvm.solve import simple_solver
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


def test_interior_only_partition_still_enters_boundary_flux_reductions(monkeypatch):
    class Parallel:
        is_partitioned = True

        def __init__(self):
            self.sum_calls = 0
            self.max_values = []

        def global_sum(self, value):
            self.sum_calls += 1
            return value

        def global_max(self, value):
            self.max_values.append(value)
            return value

        def global_all(self, value):
            return value

    parallel = Parallel()
    mesh = {
        "n_cells": 2,
        "n_faces": 1,
        "n_interior_faces": 1,
        "owners": np.array([0], dtype=np.int32),
        "neighbours": np.array([1], dtype=np.int32),
        "boundary": [],
        "_parallel_context": parallel,
    }
    geometry = {
        "cell_volume": np.ones(2),
        "face_area_vector": np.array([[1.0, 0.0, 0.0]]),
        "cell_connection_vector": np.array([[1.0, 0.0, 0.0]]),
        "face_interpolation_weight": np.array([0.5]),
    }
    monkeypatch.setattr(
        simple_solver.gradients,
        "_resolve_gradient_fn",
        lambda _geometry: lambda field, mesh, geometry: np.zeros((2, 3)),
    )

    simple_solver.assemble_pressure_correction_equation_rhie_chow(
        np.zeros((2, 3)),
        np.ones(2),
        np.zeros(2),
        1.0,
        mesh,
        geometry,
        [],
    )

    assert parallel.sum_calls == 2
    assert parallel.max_values == [0]
