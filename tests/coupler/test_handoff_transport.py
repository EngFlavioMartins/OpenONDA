"""Interface-transport properties without physical time evolution."""

from __future__ import annotations

import numpy as np

from source.coupler.vorticity_transfer import (
    build_transfer_lattice,
    cosine_eta,
    solenoidal_velocity_correction,
)


def _vortex_ring_velocity(points: np.ndarray, centre_x: float) -> np.ndarray:
    """Smooth divergence-free manufactured velocity with closed vorticity lines."""
    point = np.asarray(points)
    x = point[:, 0] - centre_x
    y = point[:, 1]
    z = point[:, 2]
    envelope = np.exp(-6.0 * (x**2 + y**2 + z**2))
    return np.column_stack((np.zeros(len(point)), -z * envelope, y * envelope))


def test_vortex_crossing_authority_ramp_has_no_fixed_point_strength_jump():
    box = np.array([-1.0, 1.0] * 3)
    h = 0.1
    ramp_width = 0.3
    lattice = build_transfer_lattice(
        box,
        h,
    )
    populations = []
    correction_norms = []
    for centre_x in np.linspace(-1.1, 0.2, 14):

        def velocity(points, centre=centre_x):
            return _vortex_ring_velocity(points, centre)

        result = solenoidal_velocity_correction(
            lattice,
            h,
            fvm_velocity_at=velocity,
            vpm_velocity_at=velocity,
            authority_at=lambda points: cosine_eta(points, box, ramp_width, 0.1),
            core_radius_ratio=1.0,
            n_existing_particles=4096,
        )
        populations.append(result.n_total_particles)
        correction_norms.append(result.correction_vortex_strength_l1)

    np.testing.assert_array_equal(populations, np.full(14, 4096))
    np.testing.assert_array_equal(correction_norms, np.zeros(14))
