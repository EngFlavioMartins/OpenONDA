"""Slip-slab images of lattice deposits for slabs of any width."""

import numpy as np
import pytest

from openonda.core.slip_slab_images import (
    add_slab_images,
    even_slab_field,
    physical_slab_share,
    slab_image_lattice,
)


def _m4(distance):
    q = np.abs(distance)
    return np.where(
        q <= 1.0,
        1.0 - 2.5 * q**2 + 1.5 * q**3,
        np.where(q <= 2.0, 0.5 * (2.0 - q) ** 2 * (1.0 - q), 0.0),
    )


def _deposit(z_nodes, spacing, z, strength):
    return _m4((z_nodes - z) / spacing)[:, None] * strength[None, :]


def _explicit_images(z_nodes, spacing, z, strength, z_min, z_max, reach=40):
    """M4 deposits of the source and of its images up to ``reach`` periods away."""
    total = np.zeros((len(z_nodes), 3))
    width = z_max - z_min
    reflected = strength * np.array([-1.0, -1.0, 1.0])
    for k in range(-reach, reach + 1):
        total += _deposit(z_nodes, spacing, z + 2 * k * width, strength)
        total += _deposit(z_nodes, spacing, 2 * z_min - z + 2 * k * width, reflected)
    return total


@pytest.mark.parametrize(
    "z_min, z_max, z",
    [
        (-0.24, 0.24, 0.1),  # thick slab, planes on nodes
        (-0.2, 0.2, 0.1),  # thick slab, planes on half nodes
        (-0.02, 0.02, -0.013),  # one layer between half-node planes
        (0.0, 0.08, 0.03),  # two layers between node planes
        (-0.02, 0.08, 0.05),  # one half-node plane and one node plane
    ],
)
def test_images_match_explicit_deposits_of_every_image(z_min, z_max, z):
    spacing = 0.08 if z_max - z_min > 0.2 else 0.04
    origin = -0.4
    z_nodes = origin + spacing * np.arange(21)
    strength = np.array([0.3, -0.7, 1.1])
    deposit = _deposit(z_nodes, spacing, z, strength)
    images = add_slab_images(deposit, origin, spacing, (z_min, z_max))
    expected = _explicit_images(z_nodes, spacing, z, strength, z_min, z_max)
    np.testing.assert_allclose(images, expected, rtol=0.0, atol=1e-14)


def test_one_layer_slab_makes_a_spanwise_uniform_field():
    spacing = 0.04
    z_nodes = -0.2 + spacing * np.arange(11)
    deposit = np.zeros((2, 11, 3))
    deposit[:, 5] = [[0.0, 0.0, 1.0], [0.4, -0.2, 0.0]]
    images = add_slab_images(deposit, -0.2, spacing, (-0.02, 0.02))
    assert z_nodes[5] == pytest.approx(0.0)
    # The spanwise component repeats on every node; in-plane ones alternate sign.
    np.testing.assert_allclose(images[0, :, 2], 1.0)
    alternating = [[0.4, -0.2] if k % 2 else [-0.4, 0.2] for k in range(11)]
    np.testing.assert_allclose(images[1, :, :2], alternating)


def test_node_on_a_plane_doubles_its_normal_and_cancels_its_tangential_components():
    deposit = np.zeros((9, 3))
    deposit[4] = [0.5, -0.5, 1.0]
    images = add_slab_images(deposit, -0.4, 0.1, (0.0, 0.4))
    np.testing.assert_allclose(images[4], [0.0, 0.0, 2.0], atol=1e-15)


def test_planes_must_be_increasing_and_on_nodes_or_half_nodes():
    assert slab_image_lattice(-0.2, 0.04, (-0.02, 0.02)) == (9, 2)
    with pytest.raises(ValueError, match="nodes or half nodes"):
        slab_image_lattice(0.0, 0.1, (0.03, 0.4))
    with pytest.raises(ValueError, match="increasing"):
        slab_image_lattice(0.0, 0.1, (0.4, 0.0))


def _fold_into_slab(z, z_min, z_max):
    """Physical position of a point under the slab images, by explicit reflection."""
    while z < z_min or z > z_max:
        z = 2.0 * z_min - z if z < z_min else 2.0 * z_max - z
    return z


@pytest.mark.parametrize(
    "z_min, z_max",
    [
        (-0.24, 0.24),  # planes on nodes
        (-0.2, 0.2),  # planes on half nodes
        (-0.04, 0.04),  # one layer between half-node planes
        (-0.04, 0.08),  # one half-node plane and one node plane
    ],
)
def test_even_field_takes_the_value_of_its_physical_mirror_node(z_min, z_max):
    spacing, origin = 0.08, -0.6
    z_nodes = origin + spacing * np.arange(16)
    rng = np.random.default_rng(3)
    values = rng.uniform(1.0, 2.0, size=(2, 16))
    field = even_slab_field(values, origin, spacing, (z_min, z_max))
    for k, z in enumerate(z_nodes):
        mirror = np.argmin(np.abs(z_nodes - _fold_into_slab(z, z_min, z_max)))
        np.testing.assert_allclose(field[:, k], values[:, mirror], rtol=1e-14)


def test_physical_share_counts_a_plane_node_once_with_its_reflection():
    share = physical_slab_share(9, -0.4, 0.1, (0.0, 0.25))
    np.testing.assert_allclose(share, [0, 0, 0, 0, 0.5, 1, 1, 0, 0])
    # The images of the physical share cover each node exactly once.
    covered = even_slab_field(np.ones(9), -0.4, 0.1, (0.0, 0.25))
    np.testing.assert_allclose(covered, 1.0)
