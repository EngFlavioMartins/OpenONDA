"""Global particle-regeneration threshold tests."""

import numpy as np
import pytest

from source.solvers.vpm.config.viscous import ViscousConfig
from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin


def test_regeneration_threshold_modes_return_one_cloud_wide_value():
    diffusion = object.__new__(_GridDiffusionMixin)
    vortex_strength_magnitude = np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2, 1)

    absolute = diffusion._select_diffusion_threshold(
        vortex_strength_magnitude,
        "absolute",
        0.25,
        4.0,
        10.0,
    )
    relative_maximum = diffusion._select_diffusion_threshold(
        vortex_strength_magnitude,
        "relative_max",
        0.2,
        4.0,
        10.0,
    )
    budget = diffusion._select_diffusion_threshold(
        vortex_strength_magnitude,
        "budget",
        0.25,
        4.0,
        10.0,
    )

    assert absolute == 0.25
    assert relative_maximum == pytest.approx(0.8)
    assert budget == 2.0
    assert all(np.isscalar(value) for value in (absolute, relative_maximum, budget))


def test_non_global_regeneration_threshold_mode_is_rejected():
    with pytest.raises(ValueError, match="gbd_threshold_mode"):
        ViscousConfig.gbd(threshold_mode="spatially_varying")


def test_budget_threshold_uses_float64_accounting_for_large_f32_grid():
    diffusion = object.__new__(_GridDiffusionMixin)
    values = np.geomspace(1.0e-10, 1.0e-3, 450_000, dtype=np.float32)
    field = values.reshape(300, 300, 5)
    total = float(field.sum(dtype=np.float64))
    budget = 1.0e-4

    threshold = diffusion._select_diffusion_threshold(
        field,
        "budget",
        budget,
        float(field.max()),
        total,
    )
    retained = float(field[field >= threshold].sum(dtype=np.float64)) / total

    assert retained >= 1.0 - budget


def test_population_cap_selects_the_global_strongest_nodes():
    vortex_strength_magnitude = np.array(
        [
            [[0.2, 9.0], [3.0, 0.1]],
            [[8.0, 2.0], [7.0, 1.0]],
        ]
    )
    ix, iy, iz = np.where(vortex_strength_magnitude > 0.0)

    ix, iy, iz, threshold, n_candidates = _GridDiffusionMixin._cap_surviving_nodes(
        vortex_strength_magnitude,
        ix,
        iy,
        iz,
        cap=3,
    )

    retained = np.sort(vortex_strength_magnitude[ix, iy, iz])
    np.testing.assert_array_equal(retained, np.array([7.0, 8.0, 9.0]))
    assert threshold == 7.0
    assert n_candidates == 8


def test_regeneration_cap_uses_declared_global_capacity():
    particles = type("Particles", (), {"capacity": 10_000})()

    assert _GridDiffusionMixin._regeneration_cap(particles, 8, None) == 10_000
    assert _GridDiffusionMixin._regeneration_cap(particles, 8, 500) == 500
