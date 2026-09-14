"""Pressure gradient halo exchange must include ranks without boundary faces."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.fvm.solve import simple_solver


@pytest.mark.parametrize("remote_change", [False, True])
def test_remote_pressure_boundary_change_refreshes_local_gradient(monkeypatch, remote_change):
    calls = []

    def gradient(*args):
        calls.append(True)
        return np.full((2, 3, 1), 7.0)

    monkeypatch.setattr(simple_solver.gradients, "_resolve_gradient_fn", lambda _: gradient)
    parallel = SimpleNamespace(
        is_partitioned=True, global_max=lambda local: max(local, int(remote_change))
    )
    initial = np.zeros((2, 3))
    result = simple_solver._update_fixed_flux_pressure_boundaries(
        np.zeros(2),
        np.zeros((2, 3)),
        np.ones(2),
        {
            "n_cells": 2,
            "n_interior_faces": 0,
            "owners": np.empty(0, dtype=int),
            "_parallel_context": parallel,
        },
        {"face_area_vector": np.empty((0, 3)), "cell_connection_vector": np.empty((0, 3))},
        [],
        kinematic_pressure_gradient=initial,
    )
    assert len(calls) == int(remote_change)
    np.testing.assert_array_equal(result, np.full((2, 3), 7.0) if remote_change else initial)
