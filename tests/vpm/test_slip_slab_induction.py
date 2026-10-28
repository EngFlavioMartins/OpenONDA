"""Free-slip slab images remain a full three-component induction operator."""

import numpy as np
import pytest
import taichi as ti

from openonda.core.slip_slab_images import add_slab_images, slab_image_lattice
from openonda.coupler.transfer_lattice import (
    blending_weight,
    build_transfer_lattice,
    scatter_m4_prime_to_lattice,
)
from openonda.vpm.config.case import Numerics
from openonda.vpm.config.configuration_values import _configuration_value
from openonda.vpm.config.divergence_relaxation import DivergenceRelaxationConfig
from openonda.vpm.config.filament_refinement import FilamentRefinementConfig
from openonda.vpm.config.stabilization import StabilizationConfig
from openonda.vpm.config.viscous import ViscousConfig
from openonda.vpm.kernels.base import make_vortex_kernel
from openonda.vpm.kernels.registry import kernel_names
from openonda.vpm.physics.base import PhysicsBase
from openonda.vpm.physics.diffusion.grid import _GridDiffusionMixin
from openonda.vpm.physics.induction.direct import DirectInduction
from openonda.vpm.physics.induction.fmm import FMMInduction
from openonda.vpm.physics.induction.slip_slab import SlipSlabInduction


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


def _fields(kernel="gaussian"):
    physics = PhysicsBase(kernel, 2, ti.f64, max_evaluation_points=3)
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


@pytest.mark.parametrize("name", sorted(kernel_names()))
def test_stage_images_and_stretching_match_independent_pair_sum(name):
    induction, x, g, r, u, jacobian, rate = _fields(name)
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
    kernel = make_vortex_kernel(name)
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
                    target_core = radius[i] if k == 0 and not odd else radius[j]
                    expected_u[i] += kernel.velocity_pair(delta, gamma, target_core, radius[j])
                    expected_j[i] += kernel.gradient_pair(delta, gamma, target_core, radius[j])
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
    physics = PhysicsBase("gaussian", 1, ti.f64, max_evaluation_points=4)
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


def test_slab_blend_weight_preserves_thin_span_interior():
    box = (-1.0, 1.0, -1.0, 1.0, -0.24, 0.24)
    points = np.array([[0, 0, 0], [0, 0, -0.24], [0, 0, 0.25], [0.9, 0, 0]])
    blend_weight = blending_weight(points, box, 0.0, 0.4, slip_slab=True)
    np.testing.assert_allclose(blend_weight[:2], 1.0)
    assert blend_weight[2] == 0.0
    assert 0.0 < blend_weight[3] < 1.0


def test_unconverged_image_tail_fails_closed():
    physics = PhysicsBase("gaussian", 1, ti.f64, max_evaluation_points=4)
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
    with pytest.raises(RuntimeError, match="truncation exceeds"):
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


@pytest.mark.parametrize("scheme", ["cs", "rwm"])
def test_slab_accepts_core_spreading_and_random_walk(scheme):
    slab = SlipSlabInduction(DirectInduction(), z_min=-0.24, z_max=0.24)
    Numerics(induction=slab, viscous=ViscousConfig(scheme=scheme))


def test_slab_dvh_rejects_padding_shorter_than_its_heat_kernel_support():
    slab = SlipSlabInduction(DirectInduction(), z_min=-0.24, z_max=0.24)
    viscous = ViscousConfig.dvh(particle_spacing=0.08, padding=3.0, kinematic_viscosity=1e-3)
    with pytest.raises(ValueError, match="dvh_domain_padding >= dvh_support_radius_ratio"):
        Numerics(induction=slab, viscous=viscous)


_SLAB_GBD = ViscousConfig.gbd(particle_spacing=0.08, kinematic_viscosity=1e-3)
_KEEP = (-1.0, 1.0, -1.0, 1.0, -0.24, 0.24)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"domain_bounds": (-1, 1, -1, 1, -0.24, 0.32)}, "domain_bounds z"),
        (
            {"viscous": ViscousConfig.gbd(particle_spacing=0.1, kinematic_viscosity=1e-3)},
            "multiple of half the GBD grid spacing",
        ),
        (
            {"stabilization": StabilizationConfig.bounded_domain((*_KEEP[:4], -0.16, 0.24))},
            "would remove particles between the slip planes",
        ),
        (
            {
                "stabilization": StabilizationConfig.bounded_domain(
                    _KEEP, outflow_fringe_depth=(0, 0.5, 0, 0, 0, 0.08)
                )
            },
            "zero on the slip-plane z faces",
        ),
        (
            {
                "stabilization": StabilizationConfig(
                    regularization_interval_steps=1, regularization_grid_spacing=0.1
                )
            },
            "multiple of half the regularization grid spacing",
        ),
        (
            {
                "stabilization": StabilizationConfig(
                    divergence_relaxation=DivergenceRelaxationConfig(
                        interval_steps=1, grid_spacing=0.1
                    )
                )
            },
            "multiple of half the divergence-relaxation grid spacing",
        ),
    ],
)
def test_slab_rejects_a_mismatched_span_or_lattice(settings, message):
    slab = SlipSlabInduction(DirectInduction(), z_min=-0.24, z_max=0.24)
    Numerics(
        induction=slab,
        viscous=_SLAB_GBD,
        domain_bounds=_KEEP,
        stabilization=StabilizationConfig.bounded_domain(_KEEP),
    )
    with pytest.raises(ValueError, match=message):
        Numerics(**{"induction": slab, "viscous": _SLAB_GBD, **settings})


def test_slab_accepts_every_stabilization_operator_on_fitting_lattices():
    slab = SlipSlabInduction(DirectInduction(), z_min=-0.24, z_max=0.24)
    Numerics(
        induction=slab,
        viscous=_SLAB_GBD,
        stabilization=StabilizationConfig(
            regularization_interval_steps=1,
            regularization_grid_spacing=0.08,
            filament_refinement=FilamentRefinementConfig(interval_steps=1),
            divergence_relaxation=DivergenceRelaxationConfig(interval_steps=1, grid_spacing=0.16),
        ),
    )


def test_slab_restart_configuration_includes_planes_tail_and_base():
    first = SlipSlabInduction(DirectInduction(), z_min=-0.24, z_max=0.24, tail_tolerance=1e-4)
    second = SlipSlabInduction(DirectInduction(), z_min=-0.48, z_max=0.48, tail_tolerance=1e-4)
    configuration = _configuration_value(first)
    assert configuration["z_min"] == -0.24
    assert configuration["z_max"] == 0.24
    assert configuration["tail_tolerance"] == 1e-4
    assert configuration["base"]["method"] == "direct"
    assert configuration != _configuration_value(second)


def test_gbd_slab_images_have_axial_parity_at_slip_plane():
    @ti.data_oriented
    class Grid(_GridDiffusionMixin):
        pass

    grid = Grid()
    x = ti.Vector.field(3, ti.f32, shape=1)
    g = ti.Vector.field(3, ti.f32, shape=1)
    deposit = ti.Vector.field(3, ti.f32, shape=(7, 7, 9))
    field = ti.Vector.field(3, ti.f32, shape=(7, 7, 9))
    x.from_numpy(np.array([[0.0, 0.0, 0.1]], dtype=np.float32))
    g.from_numpy(np.array([[0.2, -0.4, 1.0]], dtype=np.float32))
    grid._m4_scatter_gpu_kernel(x, g, deposit, -0.6, -0.6, -0.4, 0.1, 7, 7, 9, 0, 1)
    # Slab [0, 0.4]: its planes are nodes 4 and 8 of this lattice
    grid._add_slab_images_kernel(
        deposit, field, 7, 7, 9, *slab_image_lattice(-0.4, 0.1, (0.0, 0.4))
    )
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


def test_gbd_slab_images_respect_half_node_plane():
    @ti.data_oriented
    class Grid(_GridDiffusionMixin):
        pass

    grid = Grid()
    x = ti.Vector.field(3, ti.f32, shape=1)
    g = ti.Vector.field(3, ti.f32, shape=1)
    deposit = ti.Vector.field(3, ti.f32, shape=(7, 7, 10))
    field = ti.Vector.field(3, ti.f32, shape=(7, 7, 10))
    x.from_numpy(np.array([[0.0, 0.0, 0.1]], dtype=np.float32))
    g.from_numpy(np.array([[0.2, -0.4, 1.0]], dtype=np.float32))
    grid._m4_scatter_gpu_kernel(x, g, deposit, -0.6, -0.6, -0.45, 0.1, 7, 7, 10, 0, 1)
    # Slab [0, 0.4]: its planes lie halfway between nodes 4|5 and 8|9
    grid._add_slab_images_kernel(
        deposit, field, 7, 7, 10, *slab_image_lattice(-0.45, 0.1, (0.0, 0.4))
    )
    nodes = field.to_numpy()
    np.testing.assert_allclose(nodes[:, :, 4, :2], -nodes[:, :, 5, :2], atol=3e-7)
    np.testing.assert_allclose(nodes[:, :, 4, 2], nodes[:, :, 5, 2], atol=3e-7)


@pytest.mark.parametrize("name", sorted(kernel_names()))
def test_fmm_slab_targets_agree_with_direct_on_3d_cloud(name):
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
        physics = PhysicsBase(name, 4, ti.f32, max_evaluation_points=20)
        slab = SlipSlabInduction(
            backend(),
            z_min=-0.48,
            z_max=0.48,
            tail_tolerance=2e-3,
            max_shells=33,
        ).bind(physics)
        hierarchy_builds = []
        if backend is FMMInduction:
            slab.base._ensure_workspace(4)
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
            assert len(hierarchy_builds) == 0  # Mesh queries include their own primary field.
    np.testing.assert_allclose(results[1][0], results[0][0], atol=3e-3, rtol=3e-3)
    np.testing.assert_allclose(results[1][1], results[0][1], atol=1e-2, rtol=1e-2)
    assert np.max(np.abs(results[1][0][[0, -1], 2])) < 2e-3


def test_slab_transfer_lattice_ghost_nodes_have_zero_physical_weight():
    lattice = build_transfer_lattice(
        (-0.4, 0.4, -0.4, 0.4, -0.2, 0.2),
        0.1,
        remeshing_extension_length=0.2,
        blend_end_depth=0.3,
        slip_slab=True,
    )
    ghost = (lattice.positions[:, 2] < -0.2 - 1e-10) | (lattice.positions[:, 2] > 0.2 + 1e-10)
    assert np.any(ghost)
    np.testing.assert_array_equal(lattice.fluid_weight[ghost], 0.0)
    np.testing.assert_array_equal(lattice.blending_weight[ghost], 0.0)
    assert lattice.transfer_region_bounds[4] == -0.2
    assert lattice.transfer_region_bounds[5] == 0.2


def test_one_layer_transfer_lattice_holds_every_slab_image_of_its_source_particles():
    h = 0.08
    planes = (-0.5 * h, 0.5 * h)
    lattice = build_transfer_lattice(
        (-0.4, 0.4, -0.4, 0.4, *planes),
        h,
        remeshing_extension_length=0.2,
        blend_end_depth=0.3,
        lattice_anchor=(0.0, 0.0, 0.0),
        slip_slab=True,
    )
    source = np.array([[0.03, -0.02, 0.013]])
    gamma = np.array([[0.2, -0.4, 1.0]])
    deposit = scatter_m4_prime_to_lattice(source, gamma, lattice).reshape(*lattice.shape, 3)
    images = add_slab_images(deposit, lattice.origin[2], h, planes)

    # Every image's M4 deposit, explicitly: one layer repeats every 2 h in z
    def m4(distance):
        q = np.abs(distance)
        return np.where(
            q <= 1, 1 - 2.5 * q**2 + 1.5 * q**3, np.where(q <= 2, 0.5 * (2 - q) ** 2 * (1 - q), 0)
        )

    axes = [lattice.origin[d] + h * np.arange(lattice.shape[d]) for d in range(3)]
    expected = np.zeros_like(images)
    for k in range(-8, 9):
        for z, sign in (
            (source[0, 2] + 2 * k * h, 1.0),
            (2 * planes[0] - source[0, 2] + 2 * k * h, -1.0),
        ):
            weight = np.einsum(
                "i,j,k->ijk",
                m4((axes[0] - source[0, 0]) / h),
                m4((axes[1] - source[0, 1]) / h),
                m4((axes[2] - z) / h),
            )
            expected += weight[..., None] * gamma[0] * np.array([sign, sign, 1.0])
    np.testing.assert_allclose(images, expected, rtol=0.0, atol=1e-12)
