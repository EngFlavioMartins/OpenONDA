"""Physics tests for stretching-aware residual viscosity."""

import numpy as np
import pytest

from source.solvers.VPM import VPMSetup, VPMSolver
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StabilizationConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)


@pytest.mark.unit
def test_stretching_viscosity_acts_only_on_positive_line_amplification(tmp_path):
    solver = VPMSolver(
        setup=VPMSetup(
            processing_unit="CPU",
            precision="f64",
            max_particles=16,
            advection=AdvectionConfig(scheme="NONE"),
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig.cs(viscosity=0.01, characteristic_distance=0.5),
            stabilization=StabilizationConfig.stretching_viscosity(0.4),
            velocity=VelocityConfig.direct(),
            backup_frequency=0,
            logging_frequency=0,
            backup_directory=str(tmp_path),
        )
    )
    count = 3
    solver.add_vortex_particles(
        position=np.zeros((count, 3)),
        velocity=np.zeros((count, 3)),
        circulation=np.eye(3),
        radius=np.full(count, 0.2),
        volume=np.full(count, 0.5**3),
        viscosity=np.full(count, 0.01),
        viscosity_turbulent=np.full(count, 0.02),
    )
    strain = np.repeat(np.diag([2.0, -1.0, -1.0])[None, :, :], count, axis=0)
    solver.particles.set_field("strain_rate", strain)

    diagnostics = solver.stabilization.operators.apply_stretching_viscosity(solver.particles, 0.4)

    expected_stabilization = np.array([0.2, 0.0, 0.0])
    np.testing.assert_allclose(
        solver.stabilization.operators.stabilization_viscosity.to_numpy()[:count],
        expected_stabilization,
        rtol=1.0e-7,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        solver.particles.viscosity_effective_cpu(),
        0.03 + expected_stabilization,
        rtol=1.0e-7,
        atol=1.0e-12,
    )
    assert diagnostics["stabilization_viscosity_mean"] == pytest.approx(0.2 / 3.0)
    assert diagnostics["stabilization_viscosity_max"] == pytest.approx(0.2)
    assert diagnostics["stabilization_viscosity_active_fraction"] == pytest.approx(1.0 / 3.0)
