"""Independent Gaussian image-sum checks for slab renewal."""

from __future__ import annotations

import numpy as np
import pytest

from source.coupler.stable_renewal import (
    blend_represented_state,
    gaussian_represented_vortex_strength,
)


def _oracle(
    shape: tuple[int, int, int],
    origin_z: float,
    planes: tuple[float, float],
    source_z: int,
    gamma: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Direct 3D Gaussian of translated and axially reflected point sources."""
    coordinates = np.indices(shape, dtype=np.float64)
    x, y, z = coordinates[0] - 1, coordinates[1] - 1, coordinates[2] + origin_z
    source_position = np.array([0.0, 0.0, origin_z + source_z])
    result = np.zeros((*shape, 3))
    half_width = int(np.ceil(6 * sigma))
    for shift in range(-5, 6):
        for reflected in (False, True):
            image_z = (
                2 * planes[0] - source_position[2] if reflected else source_position[2]
            ) + shift * 2 * (planes[1] - planes[0])
            distance = z - image_z
            support = (
                (np.abs(x) <= half_width)
                & (np.abs(y) <= half_width)
                & (np.abs(distance) <= half_width)
            )
            kernel = np.exp(-(x * x + y * y + distance * distance) / sigma**2)
            kernel *= support / (np.pi**1.5 * sigma**3)
            parity = np.array([-1.0, -1.0, 1.0]) if reflected else 1.0
            result += kernel[..., None] * (gamma * parity)
    return result.reshape(-1, 3)


@pytest.mark.parametrize(
    ("origin_z", "source_z", "planes"),
    [(-3.5, 4, (0.0, 6.0)), (-3.0, 3, (0.0, 6.0))],
)
def test_slip_slab_gaussian_matches_full_image_sum(origin_z, source_z, planes):
    shape = (3, 3, 14)
    gamma = np.array([0.7, -0.4, 1.3])
    strength = np.zeros((*shape, 3))
    strength[1, 1, source_z] = gamma
    sigma = 1.2
    represented = gaussian_represented_vortex_strength(
        strength.reshape(-1, 3),
        shape,
        1.0,
        core_radius=sigma,
        slip_slab_bounds=planes,
        lattice_origin_z=origin_z,
    )
    reference = _oracle(shape, origin_z, planes, source_z, gamma, sigma)
    np.testing.assert_allclose(represented, reference, rtol=2e-15, atol=2e-16)


def test_node_aligned_plane_cancels_tangential_and_doubles_normal_source():
    shape = (3, 3, 13)
    strength = np.zeros((*shape, 3))
    strength[1, 1, 3] = [1.0, 0.0, 1.0]
    represented = gaussian_represented_vortex_strength(
        strength.reshape(-1, 3),
        shape,
        1.0,
        core_radius=1.0,
        slip_slab_bounds=(0.0, 6.0),
        lattice_origin_z=-3.0,
    )
    np.testing.assert_allclose(represented[:, 0], 0.0, atol=1e-16)
    assert represented.reshape(*shape, 3)[1, 1, 3, 2] > 2 / np.pi**1.5 - 1e-12


def test_blend_corrects_only_physical_slab_and_uses_image_residual():
    shape = (3, 3, 14)
    origin_z = -3.5
    planes = (0.0, 6.0)
    vpm = np.zeros((np.prod(shape), 3))
    target = np.zeros_like(vpm)
    target.reshape(*shape, 3)[1, 1, 4] = [1.0, 0.0, 1.0]
    authority = np.ones(len(vpm))
    weight = np.zeros(len(vpm))
    weight.reshape(shape)[:, :, 4:10] = 1.0
    slab = blend_represented_state(
        vpm,
        target,
        authority,
        shape,
        1.0,
        core_radius=1.2,
        output_weight=weight,
        slip_slab_bounds=planes,
        lattice_origin_z=origin_z,
    )
    free = blend_represented_state(
        vpm,
        target,
        authority,
        shape,
        1.0,
        core_radius=1.2,
        output_weight=weight,
    )
    assert np.all(slab.vortex_strength[weight == 0.0] == 0.0)
    assert np.all(slab.physical_target[weight == 0.0] == 0.0)
    assert not np.allclose(slab.vortex_strength, free.vortex_strength)
    represented = gaussian_represented_vortex_strength(
        slab.vortex_strength,
        shape,
        1.0,
        core_radius=1.2,
        slip_slab_bounds=planes,
        lattice_origin_z=origin_z,
    )
    expected = np.linalg.norm((represented - slab.physical_target)[weight > 0.0]) / np.linalg.norm(
        slab.physical_target[weight > 0.0]
    )
    assert slab.residual_after_correction == pytest.approx(expected)
