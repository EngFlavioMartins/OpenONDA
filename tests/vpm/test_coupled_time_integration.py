"""Conservation tests for common-stage advection/stretching integration."""

import numpy as np
import pytest

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StabilizationConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)


def test_coupled_transposed_step_preserves_total_strength(tmp_path):
    rng = np.random.default_rng(731)
    n_particles = 12
    position = rng.uniform(-0.5, 0.5, (n_particles, 3))
    circulation = 0.05 * rng.normal(size=(n_particles, 3))
    radius = np.full(n_particles, 0.18)
    volume = np.full(n_particles, 0.18**3)

    solver = Solver(
        setup=VPMSetup(
            time_step_size=1.0e-3,
            time_integration="COUPLED",
            processing_unit="CPU",
            precision="f64",
            advection=AdvectionConfig(scheme="RK2"),
            stretching=StretchingConfig.transposed(scheme="RK2"),
            viscous=ViscousConfig.inviscid(),
            stabilization=StabilizationConfig.disabled(),
            velocity=VelocityConfig.direct(),
            backup_frequency=0,
            logging_frequency=0,
            backup_directory=str(tmp_path),
        )
    )
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        circulation=circulation,
        radius=radius,
        volume=volume,
        viscosity=np.zeros(n_particles),
    )

    strength_before = circulation.sum(axis=0)
    solver.update_state()
    strength_after = solver.particles_circulation.sum(axis=0)

    np.testing.assert_allclose(strength_after, strength_before, rtol=0.0, atol=2e-13)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "advection": AdvectionConfig(scheme="RK3"),
                "stretching": StretchingConfig.transposed(scheme="RK2"),
            },
            "matching RK2 or RK3",
        ),
    ],
)
def test_coupled_config_rejects_incompatible_physics(overrides, message):
    values = {
        "time_integration": "COUPLED",
        "advection": AdvectionConfig(scheme="RK2"),
        "stretching": StretchingConfig.transposed(scheme="RK2"),
        "viscous": ViscousConfig.inviscid(),
        "stabilization": StabilizationConfig.disabled(),
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        VPMSetup(**values)
