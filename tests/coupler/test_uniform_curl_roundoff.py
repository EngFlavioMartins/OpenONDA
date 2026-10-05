"""The velocity-trace transfer preserves uniform and weak flows."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler import CouplerSetup
from source.coupler import vorticity_transfer as module


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("speed", [0.1, 1.0, 100.0])
def test_cached_buffered_transfer_preserves_uniform_flow_and_resolved_weak_curl(
    monkeypatch, axis, speed
):
    coordinates = np.array(np.meshgrid(*([np.linspace(-0.75, 0.75, 4)] * 3), indexing="ij"))
    donors = coordinates.reshape(3, -1).T
    driver = SimpleNamespace(
        setup=CouplerSetup(),
        kinematic_viscosity=0.01,
        fvm_box=np.array([-1.0, 1.0] * 3),
        vpm_core_radius_ratio=1.0,
        vpm_particle_spacing=0.25,
        vpm_time_step_size=0.01,
    )
    fvm = SimpleNamespace(
        setup=SimpleNamespace(boundaries=[]),
        get_cell_centre_coordinates=lambda: donors,
        get_cell_volume=lambda: np.full(len(donors), 0.5**3),
    )
    transfer = module.VorticityTransfer(driver)
    transfer.setup(fvm)
    captured = []

    def capture(vpm, *, lattice, fvm_vortex_strength_at_node, **kwargs):
        strengths = fvm_vortex_strength_at_node(lattice.positions)
        captured.append(strengths)
        return strengths

    monkeypatch.setattr(module, "replace_particles_from_buffered_m4_renewal", capture)
    velocity = np.zeros_like(donors)
    velocity[:, axis] = speed
    gradient = np.zeros((len(donors), 3, 3))
    transfer._transfer_buffered_m4_renewal(
        None, fvm_velocity=velocity, fvm_velocity_gradient=gradient
    )
    np.testing.assert_array_equal(captured[-1], np.zeros_like(captured[-1]))
    stencils = transfer._buffered_trace_stencils

    # A resolved linear shear must survive even with a large background flow.
    omega = 1e-6
    shear_velocity = velocity.copy()
    shear_velocity[:, 1] += omega * donors[:, 0]
    gradient[:, 0, 1] = omega
    transfer._transfer_buffered_m4_renewal(
        None, fvm_velocity=shear_velocity, fvm_velocity_gradient=gradient
    )
    assert transfer._buffered_trace_stencils is stencils
    lattice = transfer._stable_renewal_lattice
    interior = np.all(np.abs(lattice.positions) < 0.6, axis=1)
    expected = np.zeros((np.count_nonzero(interior), 3))
    expected[:, 2] = omega * driver.vpm_particle_spacing**3
    np.testing.assert_allclose(captured[-1][interior], expected, rtol=1e-7, atol=1e-15)
