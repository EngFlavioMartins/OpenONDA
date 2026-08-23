from __future__ import annotations

import numpy as np

from source.solvers.vpm.diagnostics.resolution import discretization_health


def _planar_aligned_cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate = np.array([-0.2, 0.0, 0.2])
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    position = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    vortex_strength = np.tile(np.array([0.0, 0.0, 0.1]), (len(position), 1))
    core_radius = np.full(len(position), 0.3)
    return position, vortex_strength, core_radius


def test_compact_reconstruction_reports_alignment_without_full_field_input():
    position, vortex_strength, core_radius = _planar_aligned_cloud()

    metrics = discretization_health(position, vortex_strength, core_radius)

    assert metrics["vorticity_divergence_error"] < 1e-14
    assert metrics["vortex_strength_misalignment_degrees"] < 1e-12


def test_supplied_vorticity_still_overrides_compact_reconstruction():
    position, vortex_strength, core_radius = _planar_aligned_cloud()
    transverse_vorticity = np.tile(
        np.array([1.0, 0.0, 0.0]),
        (len(position), 1),
    )

    metrics = discretization_health(
        position,
        vortex_strength,
        core_radius,
        transverse_vorticity,
    )

    assert metrics["vortex_strength_misalignment_degrees"] == 90.0
