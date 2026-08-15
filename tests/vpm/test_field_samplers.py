from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.VPM.io.sampling.field_samplers import SurfaceSampler


def test_surface_sampler_can_skip_velocity_derivatives(tmp_path):
    sampler = SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[0.0, 0.1, 0.0, 0.1],
        spacing=0.1,
        include_derivatives=False,
    )

    class Solver:
        particles = SimpleNamespace(number_of_particles=1)

        def compute_target_velocities(self, points, *, include_freestream):
            assert include_freestream
            return np.ones((len(points), 3))

        def compute_target_vorticities(self, points):
            return np.full((len(points), 3), 2.0)

        def compute_target_velocity_gradients(self, points):
            raise AssertionError("disabled derivative traversal was called")

    data = sampler.sample(Solver())

    np.testing.assert_allclose(data["Ux"], 1.0)
    np.testing.assert_allclose(data["omega_x"], 2.0)
    np.testing.assert_allclose(data["dudx"], 0.0)
    np.testing.assert_allclose(data["Sxx"], 0.0)

    pyvista = pytest.importorskip("pyvista")
    output = sampler.save_vtp(Solver(), tmp_path / "surface.vts")
    fields = set(pyvista.read(output).point_data.keys())
    assert {"Velocity", "VelocityMagnitude", "Vorticity", "VorticityMagnitude"} <= fields
    assert "VelocityGradient" not in fields
    assert "StrainRate" not in fields
