"""Free-slip slab images remain a full three-component induction operator."""

import numpy as np
import pytest
import taichi as ti

from source.coupler.stable_renewal import (
    build_stable_renewal_lattice,
    inward_cosine_authority,
    scatter_m4_prime_to_lattice,
)
from source.solvers.vpm.config.case import Numerics
from source.solvers.vpm.config.fingerprint import _canonical_value
from source.solvers.vpm.config.viscous import ViscousConfig
from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.fmm import FMMInduction
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


def _fields():
    physics = PhysicsBase("GAUSSIAN", 2, ti.f64, max_evaluation_points=3)
    induction = SlipSlabInduction(
        DirectInduction(),
        z_min=-0.5,
        z_max=0.5,
        tail_tolerance=1e-4,
        max_shells=65,
        velocity_scale=1.0,
        gradient_scale=1.0,
    ).bind(physics)
    x = ti.Vector.field(3, ti.f64, shape=2)
    g = ti.Vector.field(3, ti.f64, shape=2)
    r = ti.field(ti.f64, shape=2)
    u = ti.Vector.field(3, ti.f64, shape=3)
    jacobian = ti.Matrix.field(3, 3, ti.f64, shape=3)
    rate = ti.Vector.field(3, ti.f64, shape=2)
    x.from_numpy(np.array([[0.1, 0.2, 0.15], [-0.2, 0.05, -0.27]]))
    g.from_numpy(np.array([[0.2, -0.3, 0.7], [-0.4, 0.15, -0.2]]))
    r.from_numpy(np.array([0.12, 0.14]))
    return induction, x, g, r, u, jacobian, rate


def test_stage_images_and_stretching_match_independent_pair_sum():
    induction, x, g, r, u, jacobian, rate = _fields()
    induction.evaluate_stage(
        position=x,
        vortex_strength=g,
        core_radius=r,
        count=2,
        velocity_out=u,
        vortex_strength_rate_out=rate,
        velocity_gradient_out=jacobian,
    )
    source = x.to_numpy()
    strength = g.to_numpy()
    radius = r.to_numpy()
    kernel = make_vortex_kernel("GAUSSIAN")
    expected_u = np.zeros((2, 3))
    expected_j = np.zeros((2, 3, 3))
    # More images than the runtime needs, independently using host kernel pairs.
    for k in range(-80, 81):
        for odd in (False, True):
            for j in range(2):
                if k == 0 and not odd:
                    image = source[j]
                    gamma = strength[j]
                elif odd:
                    image = source[j].copy()
                    image[2] = -1.0 + 2.0 * k - image[2]
                    gamma = strength[j] * np.array([-1.0, -1.0, 1.0])
                else:
                    image = source[j].copy()
                    image[2] += 2.0 * k
                    gamma = strength[j]
                for i in range(2):
                    delta = source[i] - image
                    expected_u[i] += kernel.velocity_pair(delta, gamma, radius[i], radius[j])
                    expected_j[i] += kernel.gradient_pair(delta, gamma, radius[i], radius[j])
    np.testing.assert_allclose(u.to_numpy()[:2], expected_u, atol=3e-5, rtol=2e-4)
    np.testing.assert_allclose(jacobian.to_numpy()[:2], expected_j, atol=3e-5, rtol=2e-4)
    np.testing.assert_allclose(
        rate.to_numpy(), np.einsum("nji,nj->ni", expected_j, strength), atol=3e-5, rtol=2e-4
    )
    assert induction.last_tail["relative"] <= induction.tail_tolerance


def test_slab_faces_have_negligible_normal_induced_velocity():
    induction, x, g, r, u, jacobian, _ = _fields()
    target = ti.Vector.field(3, ti.f64, shape=3)
    target.from_numpy(np.array([[0.2, -0.1, -0.5], [0.2, -0.1, 0.5], [0.2, -0.1, 0.0]]))
    induction.evaluate_targets(
        target_position=target,
        source_position=x,
        source_vortex_strength=g,
        source_core_radius=r,
        target_velocity=u,
        target_velocity_gradient=jacobian,
        target_count=3,
        source_count=2,
        include_freestream=False,
        background_velocity=induction.physics._zero_velocity,
    )
    assert np.max(np.abs(u.to_numpy()[:2, 2])) < 3e-5
    assert abs(u.to_numpy()[2, 2]) > 1e-5  # Truly three dimensional interior.


def test_nonbinary_f64_slip_planes_preserve_velocity_and_gradient_parity():
    physics = PhysicsBase("GAUSSIAN", 1, ti.f64, max_evaluation_points=4)
    slab = SlipSlabInduction(
        DirectInduction(),
        z_min=-0.48,
        z_max=0.48,
        tail_tolerance=1e-4,
        max_shells=129,
    ).bind(physics)
    source = ti.Vector.field(3, ti.f64, shape=1)
    strength = ti.Vector.field(3, ti.f64, shape=1)
    radius = ti.field(ti.f64, shape=1)
    targets = ti.Vector.field(3, ti.f64, shape=2)
    velocity = ti.Vector.field(3, ti.f64, shape=2)
    gradient = ti.Matrix.field(3, 3, ti.f64, shape=2)
    source.from_numpy(np.array([[0.01, -0.05, 0.17]]))
    strength.from_numpy(np.array([[0.2, -0.3, 1.0]]))
    radius.fill(0.12)
    targets.from_numpy(np.array([[0.2, 0.11, -0.48], [0.2, 0.11, 0.48]]))
    slab.evaluate_targets(
        target_position=targets,
        source_position=source,
        source_vortex_strength=strength,
        source_core_radius=radius,
        target_velocity=velocity,
        target_velocity_gradient=gradient,
        target_count=2,
        source_count=1,
        include_freestream=False,
        background_velocity=physics._zero_velocity,
    )
    assert np.max(np.abs(velocity.to_numpy()[:, 2])) < 5e-5
    jacobian = gradient.to_numpy()
    assert np.max(np.abs(jacobian[:, :2, 2])) < 2e-4
    assert np.max(np.abs(jacobian[:, 2, :2])) < 2e-4


def test_slab_authority_preserves_thin_span_interior():
    box = (-1.0, 1.0, -1.0, 1.0, -0.24, 0.24)
    points = np.array([[0, 0, 0], [0, 0, -0.24], [0, 0, 0.25], [0.9, 0, 0]])
    authority = inward_cosine_authority(points, box, 0.4, slip_slab=True)
    np.testing.assert_allclose(authority[:2], 1.0)
    assert authority[2] == 0.0
    assert 0.0 < authority[3] < 1.0


def test_unconverged_image_tail_fails_closed():
    physics = PhysicsBase("GAUSSIAN", 1, ti.f64, max_evaluation_points=4)
    induction = SlipSlabInduction(
        DirectInduction(), z_min=-0.24, z_max=0.24, tail_tolerance=1e-12, max_shells=3
    ).bind(physics)
    x = ti.Vector.field(3, ti.f64, shape=1)
    g = ti.Vector.field(3, ti.f64, shape=1)
    r = ti.field(ti.f64, shape=1)
    u = ti.Vector.field(3, ti.f64, shape=1)
    rate = ti.Vector.field(3, ti.f64, shape=1)
    x.from_numpy(np.array([[0.0, 0.0, 0.1]]))
    g.from_numpy(np.array([[0.2, 0.4, 1.0]]))
    r.fill(0.05)
    with pytest.raises(RuntimeError, match="tail did not converge"):
        induction.evaluate_stage(
            position=x,
            vortex_strength=g,
            core_radius=r,
            count=1,
            velocity_out=u,
            vortex_strength_rate_out=rate,
        )


def test_source_outside_physical_slab_is_rejected():
    induction = SlipSlabInduction(DirectInduction(), z_min=-0.24, z_max=0.24)
    with pytest.raises(ValueError, match="outside the slip slab"):
        induction.validate_source_arrays(np.array([[0.0, 0.0, 0.25]]), np.array([[0, 0, 1]]))


@pytest.mark.parametrize("scheme", ["CS", "RWM", "DVH"])
def test_slab_rejects_diffusion_without_reflected_boundary_support(scheme):
    slab = SlipSlabInduction(DirectInduction(), z_min=-0.24, z_max=0.24)
    with pytest.raises(ValueError, match="supports GBD or NONE"):
        Numerics(induction=slab, viscous=ViscousConfig(scheme=scheme))


def test_slab_restart_fingerprint_includes_planes_tail_and_base():
    first = SlipSlabInduction(DirectInduction(), z_min=-0.24, z_max=0.24, tail_tolerance=1e-4)
    second = SlipSlabInduction(DirectInduction(), z_min=-0.48, z_max=0.48, tail_tolerance=1e-4)
    fingerprint = _canonical_value(first)
    assert fingerprint["z_min"] == -0.24
    assert fingerprint["z_max"] == 0.24
    assert fingerprint["tail_tolerance"] == 1e-4
    assert fingerprint["base"]["method"] == "DIRECT"
    assert fingerprint != _canonical_value(second)


def test_gbd_mirror_scatter_has_axial_parity_at_slip_plane():
    @ti.data_oriented
    class Grid(_GridDiffusionMixin):
        pass

    grid = Grid()
    x = ti.Vector.field(3, ti.f32, shape=1)
    g = ti.Vector.field(3, ti.f32, shape=1)
    field = ti.Vector.field(3, ti.f32, shape=(7, 7, 9))
    x.from_numpy(np.array([[0.0, 0.0, 0.1]], dtype=np.float32))
    g.from_numpy(np.array([[0.2, -0.4, 1.0]], dtype=np.float32))
    args = (x, g, field, -0.6, -0.6, -0.4, 0.1, 7, 7, 9, 0, 1)
    grid._m4_scatter_gpu_kernel(*args)
    grid._m4_scatter_slip_image_kernel(*args, 0.0)
    nodes = field.to_numpy()
    np.testing.assert_allclose(nodes[:, :, 4, :2], 0.0, atol=3e-7)
    np.testing.assert_allclose(nodes[:, :, 3, :2], -nodes[:, :, 5, :2], atol=3e-7)
    np.testing.assert_allclose(nodes[:, :, 3, 2], nodes[:, :, 5, 2], atol=3e-7)

    # A span-even variable viscosity must preserve the axial parity through
    # repeated componentwise diffusion steps, including gradients at the face.
    viscosity = ti.field(ti.f32, shape=(7, 7, 9))
    mask = ti.field(ti.i32, shape=(7, 7, 9))
    destination = ti.Vector.field(3, ti.f32, shape=(7, 7, 9))
    zz = 0.1 * (np.arange(9) - 4)
    nu = 0.01 + 0.001 * zz[None, None, :] ** 2
    viscosity.from_numpy(np.broadcast_to(nu, (7, 7, 9)).astype(np.float32))
    for _ in range(3):
        grid._laplacian_step_variable_gpu_kernel(
            field, destination, viscosity, mask, 0.001, 0.1, 7, 7, 9
        )
        field, destination = destination, field
    diffused = field.to_numpy()
    np.testing.assert_allclose(diffused[:, :, 4, :2], 0.0, atol=5e-7)
    np.testing.assert_allclose(diffused[:, :, 3, :2], -diffused[:, :, 5, :2], atol=5e-7)
    np.testing.assert_allclose(diffused[:, :, 3, 2], diffused[:, :, 5, 2], atol=5e-7)


def test_gbd_mirror_scatter_respects_half_node_plane():
    @ti.data_oriented
    class Grid(_GridDiffusionMixin):
        pass

    grid = Grid()
    x = ti.Vector.field(3, ti.f32, shape=1)
    g = ti.Vector.field(3, ti.f32, shape=1)
    field = ti.Vector.field(3, ti.f32, shape=(7, 7, 10))
    x.from_numpy(np.array([[0.0, 0.0, 0.1]], dtype=np.float32))
    g.from_numpy(np.array([[0.2, -0.4, 1.0]], dtype=np.float32))
    args = (x, g, field, -0.6, -0.6, -0.45, 0.1, 7, 7, 10, 0, 1)
    grid._m4_scatter_gpu_kernel(*args)
    grid._m4_scatter_slip_image_kernel(*args, 0.0)
    nodes = field.to_numpy()
    np.testing.assert_allclose(nodes[:, :, 4, :2], -nodes[:, :, 5, :2], atol=3e-7)
    np.testing.assert_allclose(nodes[:, :, 4, 2], nodes[:, :, 5, 2], atol=3e-7)


def test_fmm_slab_targets_agree_with_direct_on_3d_cloud():
    position = np.array(
        [
            [-0.2, 0.1, -0.31],
            [0.15, -0.07, -0.1],
            [0.28, 0.24, 0.19],
            [-0.15, -0.19, 0.33],
        ],
        dtype=np.float32,
    )
    strength = np.array(
        [
            [0.2, -0.3, 0.4],
            [-0.12, 0.28, -0.18],
            [0.31, 0.16, 0.21],
            [-0.24, -0.08, 0.3],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [0.0, 0.0, -0.48],
            [0.04, -0.06, -0.24],
            [0.1, 0.0, 0.0],
            [-0.04, 0.12, 0.27],
            [0.0, 0.0, 0.48],
        ],
        dtype=np.float32,
    )
    x = ti.Vector.field(3, ti.f32, shape=4)
    g = ti.Vector.field(3, ti.f32, shape=4)
    r = ti.field(ti.f32, shape=4)
    t = ti.Vector.field(3, ti.f32, shape=5)
    x.from_numpy(position)
    g.from_numpy(strength)
    r.fill(0.1)
    t.from_numpy(target)

    results = []
    for backend in (DirectInduction, FMMInduction):
        physics = PhysicsBase("GAUSSIAN", 4, ti.f32, max_evaluation_points=20)
        slab = SlipSlabInduction(
            backend(),
            z_min=-0.48,
            z_max=0.48,
            tail_tolerance=2e-3,
            max_shells=33,
        ).bind(physics)
        hierarchy_builds = []
        if backend is FMMInduction:
            tree = slab.base.workspace.tree
            original_build = tree.build

            def counted_build(*args, _build=original_build, _counts=hierarchy_builds):
                _counts.append(1)
                return _build(*args)

            tree.build = counted_build
        u = ti.Vector.field(3, ti.f32, shape=5)
        jacobian = ti.Matrix.field(3, 3, ti.f32, shape=5)
        slab.evaluate_targets(
            target_position=t,
            source_position=x,
            source_vortex_strength=g,
            source_core_radius=r,
            target_velocity=u,
            target_velocity_gradient=jacobian,
            target_count=5,
            source_count=4,
            include_freestream=False,
            background_velocity=physics._zero_velocity,
        )
        results.append((u.to_numpy(), jacobian.to_numpy()))
        if backend is FMMInduction:
            assert len(hierarchy_builds) == 1
    np.testing.assert_allclose(results[1][0], results[0][0], atol=3e-3, rtol=3e-3)
    np.testing.assert_allclose(results[1][1], results[0][1], atol=1e-2, rtol=1e-2)
    assert np.max(np.abs(results[1][0][[0, -1], 2])) < 2e-3


def test_slab_renewal_ghost_nodes_have_zero_physical_weight():
    lattice = build_stable_renewal_lattice(
        (-0.4, 0.4, -0.4, 0.4, -0.2, 0.2),
        0.1,
        buffer_length=0.2,
        authority_ramp_width=0.3,
        slip_slab=True,
    )
    ghost = (lattice.positions[:, 2] < -0.2 - 1e-10) | (lattice.positions[:, 2] > 0.2 + 1e-10)
    assert np.any(ghost)
    np.testing.assert_array_equal(lattice.fluid_weight[ghost], 0.0)
    np.testing.assert_array_equal(lattice.fvm_authority[ghost], 0.0)
    assert lattice.renewal_bounds[4] == -0.2
    assert lattice.renewal_bounds[5] == 0.2
    image = np.array([[0.0, 0.0, -0.25]])
    gamma = np.array([[-0.2, 0.4, 1.0]])
    scattered = scatter_m4_prime_to_lattice(image, gamma, lattice, allow_slab_images=True)
    assert np.linalg.norm(scattered) > 0.0
