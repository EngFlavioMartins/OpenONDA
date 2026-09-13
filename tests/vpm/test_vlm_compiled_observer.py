"""Compiled moving-surface diagnostics against the scalar geometry oracle."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from source.solvers.vpm.boundary_elements.vlm.kernels.collision import (
    classify_finite_surface_segment,
    classify_moving_finite_surface_segment,
    swept_panel_candidates,
)
from source.solvers.vpm.boundary_elements.vlm.kernels.observer import (
    classify_relative_segment,
    observe_moving_surfaces,
)


@pytest.mark.parametrize("shape", ["quad", "triangle", "warped", "reverse"])
def test_compiled_scalar_matches_host_geometry(shape):
    """Random paths plus grazing/stationary/edge/endpoint cases retain classification."""
    rng = np.random.default_rng(123)
    corners = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]])
    if shape == "triangle":
        corners[3] = corners[0]
    if shape == "warped":
        corners[2, 2] = 0.05
    if shape == "reverse":
        corners = corners[[0, 3, 2, 1]]
    normal = np.array([0.0, 0, -2 if shape == "reverse" else 2.0])
    paths = [(rng.uniform(-0.2, 1.2, 3), rng.uniform(-0.2, 1.2, 3)) for _ in range(150)]
    paths.extend(
        [
            (np.array([0.5, 0.5, 0.0005]), np.array([0.5, 0.5, 0.0005])),
            (np.array([-0.2, 0.5, 0.01]), np.array([-0.2, 0.5, -0.01])),
            (np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.5, 0.02])),
            (np.array([0.5, 1.0005, 0.0005]), np.array([0.5, 1.0005, 0.0005])),
            (np.array([-0.1, 0.5, 0.01]), np.array([1.1, 0.5, 0.01])),
        ]
    )
    for start, end in paths:
        expected = classify_finite_surface_segment(
            start, end, 0.03, corners, normal, tolerance=1e-5
        )
        event, distance, parameter, ds, de = classify_relative_segment(
            start, end, 0.03, corners, normal, 1e-5
        )
        assert event == expected["event"]
        np.testing.assert_allclose(
            [distance, parameter, ds, de],
            [expected[k] for k in ("distance", "parameter", "signed_start", "signed_end")],
            rtol=1e-11,
            atol=2e-12,
        )


def test_compiled_rejects_degenerate_geometry():
    """Compilation does not turn invalid panel geometry into a passing diagnostic."""
    for corners, normal, message in (
        (np.zeros((4, 3)), np.array([0.0, 0, 1]), "degenerate"),
        (np.zeros((4, 3)), np.zeros(3), "zero normal"),
    ):
        with pytest.raises(ValueError, match=message):
            classify_relative_segment(np.ones(3), -np.ones(3), 0.1, corners, normal, 1e-5)


def test_compiled_moving_observer_matches_exhaustive_priority_and_positions():
    """Use independently fitted moving poses and the original broad/narrow ordering."""
    rng = np.random.default_rng(41)
    base = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]])
    corners = np.stack([base, base + [0.8, 0, 0], base + [0, 0, 0.4]])
    rot = Rotation.from_rotvec([0.4, -0.2, 0.1]).as_matrix()
    end_corners = corners @ rot.T + [0.1, -0.2, 0.3]
    normals = np.tile([0.0, 0, 1.0], (3, 1))
    end_normals = normals @ rot.T
    start = rng.uniform(-0.3, 1.5, (24, 3))
    end = start + rng.normal(0, 0.4, start.shape)
    radii = rng.uniform(0.01, 0.2, len(start))
    surfaces = ["blade", "blade", "other"]
    n_substeps, tolerance = 3, 1e-4
    lower = np.minimum(corners.min(axis=1), end_corners.min(axis=1))
    upper = np.maximum(corners.max(axis=1), end_corners.max(axis=1))
    expected = {}
    priority = [0, 3, 1, 2]
    for i, panels in swept_panel_candidates(start, end, radii, lower, upper, tolerance):
        for sub in range(n_substeps):
            a, b = sub / n_substeps, (sub + 1) / n_substeps
            p0, p1 = start[i] + a * (end[i] - start[i]), start[i] + b * (end[i] - start[i])
            for panel in panels:
                margin = 2 * (radii[i] + tolerance)
                if np.any(np.maximum(p0, p1) + margin < lower[panel]) or np.any(
                    np.minimum(p0, p1) - margin > upper[panel]
                ):
                    continue
                result = classify_moving_finite_surface_segment(
                    p0,
                    p1,
                    radii[i],
                    corners[panel] + a * (end_corners[panel] - corners[panel]),
                    corners[panel] + b * (end_corners[panel] - corners[panel]),
                    normals[panel] + a * (end_normals[panel] - normals[panel]),
                    normals[panel] + b * (end_normals[panel] - normals[panel]),
                    tolerance=tolerance,
                )
                event = result["event"]
                key = i, surfaces[panel]
                old = expected.get(key)
                if event and (
                    old is None
                    or priority[event] > priority[old[3]]
                    or (priority[event] == priority[old[3]] and result["distance"] < old[4])
                ):
                    expected[key] = (i, panel, sub, event, result["distance"], result["position"])
    actual = list(
        observe_moving_surfaces(
            start,
            end,
            radii,
            corners,
            end_corners,
            normals,
            end_normals,
            surfaces,
            n_substeps,
            tolerance,
        )
    )
    ordered = [expected[key] for key in sorted(expected)]
    assert len(actual) == len(ordered) > 0
    for got, want in zip(actual, ordered, strict=True):
        assert got[:4] == want[:4]
        np.testing.assert_allclose(got[4], want[4], atol=2e-12)
        np.testing.assert_allclose(got[5], want[5], atol=2e-12)
