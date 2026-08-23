from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.io.sampling.field_samplers import SurfaceSampler


def test_surface_sampler_can_skip_velocity_derivatives(tmp_path):
    sampler = SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[0.0, 0.1, 0.0, 0.1],
        spacing=0.1,
        include_derivatives=False,
    )

    class Solver:
        particles = SimpleNamespace(n_particles_total=1)

        def compute_velocity_and_gradient_at_points(self, points, *, particle_spacing):
            assert particle_spacing == 0.1
            gradient = np.zeros((len(points), 3, 3))
            gradient[:, 2, 1] = 2.0
            gradient[:, 0, 2] = 3.0
            gradient[:, 1, 0] = 4.0
            return np.ones((len(points), 3)), gradient

    data = sampler.sample(Solver())

    np.testing.assert_allclose(data["velocity_x"], 1.0)
    np.testing.assert_allclose(data["vorticity_x"], 2.0)
    np.testing.assert_allclose(data["vorticity_y"], 3.0)
    np.testing.assert_allclose(data["vorticity_z"], 4.0)
    np.testing.assert_allclose(data["velocity_gradient_xx"], 0.0)
    np.testing.assert_allclose(data["strain_rate_xx"], 0.0)

    pyvista = pytest.importorskip("pyvista")
    output = sampler.save_vtp(Solver(), tmp_path / "surface.vts")
    fields = set(pyvista.read(output).point_data.keys())
    assert {"velocity", "velocity_magnitude", "vorticity", "vorticity_magnitude"} <= fields
    assert "velocity_gradient" not in fields
    assert "strain_rate" not in fields
