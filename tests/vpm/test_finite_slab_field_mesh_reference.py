"""Slab-commensurate auxiliary mesh tests; physical particle positions unchanged."""

import numpy as np
import pytest

from tests.vpm._finite_image_mesh_reference import direct_finite_images
from tests.vpm._finite_slab_field_mesh_reference import finite_slab_field_mesh, slab_world_images


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("zmin", [0., 1024.3])
def test_both_planes_nonintegral_width_translated_zero_velocity_and_finite_j(upper, zmin):
    zmax = zmin+.193
    x = np.array([[.013, -.017, zmax if upper else zmin], [.036, .027, zmin+.071]])
    gamma, sigma = np.array([[.2, .7, -.4], [0., 0., 0.]]), np.full(2, .04)
    descriptors = [(int(upper), True)]
    world, _ = slab_world_images(descriptors, zmin, zmax, 5)
    u, j, _ = direct_finite_images(x, gamma, sigma, x[:1], world)
    result = finite_slab_field_mesh(x, gamma, sigma, x[:1], descriptors, zmin=zmin, zmax=zmax,
                                   tau=.12, spacing=.04, order=10)
    np.testing.assert_allclose(result.velocity, u, rtol=0, atol=1e-5)
    assert np.linalg.norm(result.gradient-j)/np.linalg.norm(j) < 1e-4
    assert result.diagnostics["slab_cells"] == 5
    assert result.diagnostics["spacing"][2] == (zmax-zmin)/5
    assert not result.diagnostics["particle_positions_snapped"]
    assert not result.diagnostics["exact_coincidence_override"]


def test_nonzero_z_cancellation_and_compatible_rotation_match_direct():
    x = np.array([[-.082, .012, .023], [-.042, .012, .023], [-.002, .012, .023],
                  [.038, .012, .023], [.078, .012, .023]])
    query = np.array([[.011, .019, .018], [-.073, .058, .061], [.112, -.036, .04]])
    sigma = np.full(5, .04)
    descriptors = [(0, True), (-1, False), (1, False)]
    world, _ = slab_world_images(descriptors, 0., .96, 32)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for weights in (np.ones(5), np.array([1, -4, 6, -4, 1])):
        gamma = weights[:, None]*np.array([[.3, -.2, .7]])
        for matrix in (np.eye(3), rotation):
            xx, qq, gg = x@matrix.T, query@matrix.T, gamma@matrix.T
            u, j, _ = direct_finite_images(xx, gg, sigma, qq, world)
            result = finite_slab_field_mesh(xx, gg, sigma, qq, descriptors, zmin=0., zmax=.96,
                                           tau=.12, spacing=.03, order=10)
            for candidate, exact in ((result.velocity, u), (result.gradient, j)):
                assert np.linalg.norm(candidate-exact)/np.linalg.norm(exact) < 1e-4
            np.testing.assert_allclose(np.trace(result.gradient, axis1=1, axis2=2), 0, atol=1e-10)


def test_refined_actual_parameters_meet_independent_derivative_guard():
    x = np.array([[-.047, .018, .073], [.056, -.036, .041], [.011, .029, .096]])
    gamma = np.array([[.3, -.2, .5], [-.7, .4, .2], [.4, -.2, .3]])
    sigma = np.array([.04, .07, .09])
    query = np.array([[.012, -.013, .068], [-.021, .027, .059]])
    descriptors = [(0, True), (1, False)]
    world, _ = slab_world_images(descriptors, 0., .96, 32)
    options = {"zmin": 0., "zmax": .96, "tau": .12, "spacing": .03, "order": 10}
    result = finite_slab_field_mesh(x, gamma, sigma, query, descriptors, **options)
    finite_difference = np.empty_like(result.gradient)
    epsilon = 1e-6
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = epsilon
        plus = finite_slab_field_mesh(x, gamma, sigma, query+step, descriptors, **options)
        minus = finite_slab_field_mesh(x, gamma, sigma, query-step, descriptors, **options)
        for key in ("grid_origin_lattice", "grid_shape", "fft_shape"):
            assert plus.diagnostics[key] == minus.diagnostics[key] == result.diagnostics[key]
        finite_difference[:, :, axis] = (plus.velocity-minus.velocity)/(2*epsilon)
    _, exact_j, _ = direct_finite_images(x, gamma, sigma, query, world)
    true_error = np.linalg.norm(finite_difference-exact_j)/np.linalg.norm(exact_j)
    discrepancy = np.linalg.norm(finite_difference-result.gradient)/np.linalg.norm(exact_j)
    assert true_error < 1e-4, (true_error, discrepancy)
    assert discrepancy < 1e-4, (true_error, discrepancy)


def test_invalid_image_descriptor_and_memory_cap_cannot_publish():
    x, gamma, sigma = np.array([[0., 0., .1]]), np.ones((1, 3)), np.array([.04])
    options = {"zmin": 0., "zmax": .193, "tau": .12, "spacing": .04}
    for images in ([(.5, True)], [(True, False)], [(1, 1)]):
        with pytest.raises(ValueError, match="descriptor"):
            finite_slab_field_mesh(x, gamma, sigma, x, images, **options)
    with pytest.raises(ValueError, match="grid exceeded"):
        finite_slab_field_mesh(x, gamma, sigma, x, [(0, True)], max_grid_nodes=10, **options)
