"""Algebraic qualification only of the rejected residual-blend experiment.

These identities are necessary but not sufficient: the real frozen 3D cube
test showed that this candidate still needs spatial regularization.
"""

import numpy as np
import pytest

from studies.coupler_accuracy.experimental_residual_blend import blend_represented_state


def _three_dimensional_field():
    h = 0.25
    axes = [h * np.arange(-4, 5)] * 3
    position = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    # Curl of a Gaussian vector potential: nonzero components and variation
    # in all three coordinates. No invariant or periodic extrusion direction.
    strength = (
        np.exp(-np.sum(position**2, axis=1) / 0.4)[:, None]
        * np.cross(position, [1.0, 2.0, 3.0])
        * h**3
    )
    displacement = position[:, None] - position[None, :]
    gaussian = np.exp(-np.sum(displacement**2, axis=-1) / h**2) / np.pi**1.5
    return h, position, strength, gaussian


@pytest.mark.parametrize("cap", [1.0, 1.8, 2.0])
def test_exactly_represented_3d_target_does_not_change_particle_coefficients(cap):
    h, position, strength, gaussian = _three_dimensional_field()
    target = gaussian @ strength
    authority = np.clip(1.0 - np.max(np.abs(position), axis=1), 0, 1)
    result = blend_represented_state(
        strength, target, authority, (9, 9, 9), h,
        core_radius=h, amplification_cap=cap,
    )
    np.testing.assert_allclose(result.vortex_strength, strength, rtol=0, atol=2e-17)


def test_no_authority_preserves_3d_particle_field():
    h, _, strength, _ = _three_dimensional_field()
    result = blend_represented_state(
        strength, np.zeros_like(strength), np.zeros(len(strength)), (9, 9, 9), h,
        core_radius=h,
    )
    np.testing.assert_array_equal(result.vortex_strength, strength)


def test_repeated_3d_blend_reduces_a_resolved_representation_error():
    h, _, expected, gaussian = _three_dimensional_field()
    target = gaussian @ expected
    strength = np.zeros_like(expected)
    errors = []
    for _ in range(20):
        result = blend_represented_state(
            strength, target, np.ones(len(strength)), (9, 9, 9), h,
            core_radius=h, amplification_cap=1.8,
        )
        strength = result.vortex_strength
        errors.append(np.linalg.norm(gaussian @ strength - target))
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 0.03 * errors[0]
