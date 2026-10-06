"""Separate mixed-boundary forcing components without changing the other input."""

import numpy as np

from tests.support.cylinder.prepare_boundary_component_inputs import (
    GRADIENT_CONTROL,
    NORMAL_CONTROL,
    component_endpoint_fields,
)


def test_component_controls_preserve_mean_trend_and_untested_oscillation():
    normal = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    time = np.linspace(0.0, 12.0, 301)
    actual_un = 0.04 * np.cos(2 * np.pi * 0.18 * time)
    reference_un = 0.07 * np.cos(2 * np.pi * 0.19 * time + 0.3)
    actual_gt = 0.06 * np.sin(2 * np.pi * 0.18 * time + 0.8)
    reference_gt = 0.16 * np.sin(2 * np.pi * 0.19 * time + 0.5)
    tangent = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    sign = np.array([1.0, -1.0, 1.0, -1.0])
    dc_un = (0.1 + 0.002 * time[:, None]) * sign
    dc_u = normal[None] * dc_un[..., None] + 0.02 * tangent[None]
    actual_dc = {
        "velocity": dc_u,
        "normal_velocity": dc_un,
        "tangential_gradient": 0.03 * np.broadcast_to(tangent, dc_u.shape),
    }
    actual_oscillation = {
        "velocity": normal[None] * (actual_un[:, None] * sign)[..., None]
        + actual_gt[:, None, None] * tangent[None],
        "normal_velocity": actual_un[:, None] * sign,
        "tangential_gradient": actual_gt[:, None, None] * tangent[None],
    }
    reference_oscillation = {
        "velocity": np.zeros_like(dc_u),
        "normal_velocity": reference_un[:, None] * sign,
        "tangential_gradient": reference_gt[:, None, None] * tangent[None],
    }
    models = component_endpoint_fields(actual_dc, actual_oscillation, reference_oscillation, normal)
    a, b = models[NORMAL_CONTROL], models[GRADIENT_CONTROL]
    np.testing.assert_allclose(
        a["normal_velocity"] - dc_un, reference_oscillation["normal_velocity"], atol=1e-16
    )
    np.testing.assert_array_equal(
        a["tangential_gradient"],
        actual_dc["tangential_gradient"] + actual_oscillation["tangential_gradient"],
    )
    np.testing.assert_array_equal(
        b["normal_velocity"], actual_dc["normal_velocity"] + actual_oscillation["normal_velocity"]
    )
    np.testing.assert_array_equal(b["velocity"], dc_u + actual_oscillation["velocity"])
    np.testing.assert_allclose(
        b["tangential_gradient"] - actual_dc["tangential_gradient"],
        reference_oscillation["tangential_gradient"],
        atol=1e-16,
    )
    for model in models.values():
        np.testing.assert_allclose(
            np.einsum("tfi,fi->tf", model["velocity"], normal), model["normal_velocity"], atol=1e-16
        )
        np.testing.assert_array_equal(
            np.einsum("tfi,fi->tf", model["tangential_gradient"], normal), 0
        )
        np.testing.assert_allclose(model["normal_velocity"].sum(axis=1), 0, atol=1e-16)
    actual_tangent = (
        dc_u + actual_oscillation["velocity"] - normal[None] * b["normal_velocity"][..., None]
    )
    result_tangent = a["velocity"] - normal[None] * a["normal_velocity"][..., None]
    np.testing.assert_allclose(result_tangent, actual_tangent, atol=1e-16)
    assert np.max(abs(a["normal_velocity"] - b["normal_velocity"])) > 0.05
    assert np.max(abs(a["tangential_gradient"] - b["tangential_gradient"])) > 0.05
