"""Particle operators between free-slip planes act on the field of every image.

Each operator is compared with the same operator applied in free space to the
particles together with their explicit images, or with a planar closed form.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import exp1

from openonda.core.slip_slab_images import AXIAL_PARITY, fold_into_slab, slab_image_particles
from openonda.vpm.diagnostics.resolution import particle_resolution_metrics
from openonda.vpm.numerics.fourier_integrals import (
    _grid_for_particles,
    gaussian_fourier_integrals,
)
from openonda.vpm.stabilization.divergence_relaxation import (
    GaussianParticleGridOperator,
    constrained_divergence_relaxation,
    gaussian_invariant_rows,
)
from openonda.vpm.stabilization.filament_refinement import (
    gaussian_particle_moments,
    split_stretched_filaments,
)
from openonda.vpm.stabilization.remeshing import gaussian_core_remesh

H = 0.1
# Slip planes on half nodes of the h lattice through the origin, so free-space
# and slab lattices share their nodes.
PLANES = (-0.25, 0.25)


def _blob_vorticity(targets, position, strength, sigma):
    """Gaussian blob vorticity sum, exp(-r^2/s^2)/(pi^1.5 s^3) per particle."""
    d = targets[:, None, :] - position[None, :, :]
    weight = np.exp(-(d**2).sum(-1) / sigma**2) / (np.pi**1.5 * sigma**3)
    return weight @ strength


def _with_images(position, strength, sigma, reach):
    image_position, image_strength, owner = slab_image_particles(position, strength, PLANES, reach)
    return (
        np.concatenate((position, image_position)),
        np.concatenate((strength, image_strength)),
        np.concatenate((sigma, sigma[owner])),
    )


def _slab_cloud(count=40, seed=3):
    rng = np.random.default_rng(seed)
    position = np.column_stack(
        (
            rng.uniform(-0.3, 0.3, count),
            rng.uniform(-0.3, 0.3, count),
            rng.uniform(*PLANES, count),
        )
    )
    return position, rng.normal(size=(count, 3)) * 1e-3, np.full(count, 1.5 * H)


def test_folded_particles_carry_the_same_image_field():
    rng = np.random.default_rng(1)
    position = np.column_stack(
        (rng.uniform(-0.2, 0.2, 30), rng.uniform(-0.2, 0.2, 30), rng.uniform(-2.0, 2.0, 30))
    )
    strength = rng.normal(size=(30, 3))
    sigma = np.full(30, 0.15)
    folded, folded_strength = fold_into_slab(position, strength, PLANES)
    assert np.all((folded[:, 2] >= PLANES[0]) & (folded[:, 2] <= PLANES[1]))
    np.testing.assert_array_equal(np.abs(folded_strength), np.abs(strength))
    probes = np.column_stack((rng.uniform(-0.2, 0.2, 20), np.zeros(20), rng.uniform(*PLANES, 20)))
    # Both clouds with all their images within reach of the slab.
    fields = [
        _blob_vorticity(probes, *_with_images(p, g, sigma, reach=8.0)[:2], sigma=0.15)
        for p, g in ((position, strength), (folded, folded_strength))
    ]
    # The unfolded cloud's own members outside the slab are images of themselves.
    np.testing.assert_allclose(fields[1], fields[0], rtol=1e-10, atol=1e-12)


def test_slab_moment_rows_are_the_gradient_of_the_slab_moments():
    position, strength, sigma = _slab_cloud()
    rows = gaussian_invariant_rows(position, sigma, slip_slab=True)
    moments = gaussian_particle_moments(position, strength, sigma, slip_slab=True)
    linear = np.einsum("mpa,pa->m", rows, strength)
    np.testing.assert_allclose(linear, np.concatenate(moments[::2] + moments[3:]), atol=1e-15)
    # Only spanwise strength enters; the in-plane totals, I_z, A_x and A_y vanish.
    assert not np.any(rows[..., :2])
    np.testing.assert_array_equal(moments[0][:2], 0.0)
    assert moments[2][2] == 0.0 and not np.any(moments[3][:2])


def test_slab_fourier_integrals_match_the_planar_closed_form():
    # One particle layer of a slab one spacing wide is a planar field: the
    # slab integrals are the width times the 2-D Gaussian-blob integrals.
    h, sigma = 0.05, 0.1
    planes = (-0.5 * h, 0.5 * h)
    x, y = np.meshgrid(np.arange(-0.6, 0.6001, h), np.arange(-0.4, 0.4001, h), indexing="ij")
    x, y = x.ravel(), y.ravel()
    gamma = (
        (np.exp(-((x + 0.2) ** 2 + y**2) / 0.1**2) - np.exp(-((x - 0.2) ** 2 + y**2) / 0.1**2))
        * h
        * h
    )
    position = np.column_stack((x, y, np.zeros_like(x)))
    strength = np.zeros_like(position)
    strength[:, 2] = gamma * h
    grid = _grid_for_particles(position, h, padding=80, slip_slab=planes)
    result = gaussian_fourier_integrals(
        position, strength, np.full(len(x), sigma), np.full(len(x), h**3), grid=grid
    )
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    s2 = 2 * sigma**2
    with np.errstate(divide="ignore"):
        stream = np.where(
            d2 > 0,
            -(np.log(d2) + exp1(np.maximum(d2, 1e-300) / s2)) / (4 * np.pi),
            -(np.log(s2) - np.euler_gamma) / (4 * np.pi),
        )
    energy = 0.5 * h * gamma @ stream @ gamma
    enstrophy = h * gamma @ (np.exp(-d2 / s2) / (np.pi * s2)) @ gamma
    assert result.total_enstrophy == pytest.approx(enstrophy, rel=1e-10)
    # The x-y box is periodic with zero padding, as for free-space audits.
    assert result.total_kinetic_energy == pytest.approx(energy, rel=2e-3)
    assert abs(result.total_helicity) <= 1e-12 * result.total_enstrophy


def test_slab_remesh_equals_the_free_space_remesh_of_all_images():
    position, strength, sigma = _slab_cloud(count=12)
    particles = SimpleNamespace(
        position_cpu=lambda: position,
        vortex_strength_cpu=lambda: strength,
        core_radius_cpu=lambda: sigma,
        kinematic_viscosity_cpu=lambda: np.full(len(position), 1e-3),
        eddy_viscosity_cpu=lambda: np.zeros(len(position)),
        group_id_cpu=lambda: np.zeros(len(position), dtype=np.int32),
        zone_id_cpu=lambda: np.zeros(len(position), dtype=np.int32),
    )
    settings = {"spacing": H, "core_radius": H, "tail_budget": 1e-14, "max_particles": None}
    slab = gaussian_core_remesh(particles, slip_slab=PLANES, **settings)
    # Free-space remesh of the particles and every image within the deposit,
    # convolution and slab reach, keeping the nodes between the planes.
    all_position, all_strength, all_sigma = _with_images(position, strength, sigma, reach=1.5)
    images = SimpleNamespace(
        **{
            name: (lambda value=value: value)
            for name, value in {
                "position_cpu": all_position,
                "vortex_strength_cpu": all_strength,
                "core_radius_cpu": all_sigma,
                "kinematic_viscosity_cpu": np.full(len(all_position), 1e-3),
                "eddy_viscosity_cpu": np.zeros(len(all_position)),
                "group_id_cpu": np.zeros(len(all_position), dtype=np.int32),
                "zone_id_cpu": np.zeros(len(all_position), dtype=np.int32),
            }.items()
        }
    )
    free = gaussian_core_remesh(images, **settings)
    inside = (free["position"][:, 2] > PLANES[0]) & (free["position"][:, 2] < PLANES[1])
    expected = dict(
        zip(
            map(tuple, np.round(free["position"][inside] / H).astype(int)),
            free["vortex_strength"][inside],
            strict=True,
        )
    )
    actual = dict(
        zip(
            map(tuple, np.round(slab["position"] / H).astype(int)),
            slab["vortex_strength"],
            strict=True,
        )
    )
    assert np.all(np.abs(slab["position"][:, 2]) < PLANES[1])
    scale = np.abs(slab["vortex_strength"]).max()
    for node in set(expected) | set(actual):
        np.testing.assert_allclose(
            actual.get(node, np.zeros(3)), expected.get(node, np.zeros(3)), atol=1e-11 * scale
        )
    # M4' deposits keep the strength and impulse; the sampled convolution
    # changes the second moment slightly, which regularization then restores.
    before = gaussian_particle_moments(position, strength, sigma, slip_slab=True)
    after = gaussian_particle_moments(
        slab["position"], slab["vortex_strength"], slab["core_radius"], slip_slab=True
    )
    for index in (0, 2):
        np.testing.assert_allclose(after[index], before[index], atol=1e-15, rtol=1e-9)
    np.testing.assert_allclose(after[3], before[3], rtol=1e-3)


def test_slab_grid_operator_equals_the_free_space_operator_of_all_images():
    position, strength, sigma = _slab_cloud(count=12)
    slab = GaussianParticleGridOperator(
        position, sigma, np.ones(len(position)), spacing=H, slip_slab=PLANES
    )
    all_position, all_strength, all_sigma = _with_images(position, strength, sigma, reach=1.5)
    free = GaussianParticleGridOperator(
        all_position, all_sigma, np.ones(len(all_position)), spacing=H
    )
    np.testing.assert_allclose(
        slab.apply(strength),
        free.apply(all_strength)[: len(position)],
        atol=1e-12 * np.abs(slab.apply(strength)).max(),
    )


def test_planar_layers_are_solenoidal_only_with_their_images():
    # Two layers of a z-independent field in a slab two spacings wide.
    x, y = np.meshgrid(H * np.arange(-8, 9), H * np.arange(-8, 9), indexing="ij")
    layers = [np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z))) for z in (-H / 2, H / 2)]
    position = np.concatenate(layers)
    strength = np.zeros_like(position)
    # Centred between nodes: at a sampled extremum the metric is 0/0.
    centre = (position[:, 0] - 0.03) ** 2 + (position[:, 1] - 0.02) ** 2
    strength[:, 2] = np.exp(-centre / 0.3**2)
    sigma = np.full(len(position), 1.5 * H)
    planes = (-H, H)
    with_images = particle_resolution_metrics(
        position, strength, sigma, sample_all=True, slip_slab=planes
    )
    without = particle_resolution_metrics(position, strength, sigma, sample_all=True)
    assert with_images["vorticity_divergence_error"] < 1e-6
    assert without["vorticity_divergence_error"] > 1e-2
    operator = GaussianParticleGridOperator(
        position, sigma, np.ones(len(position)), spacing=H, slip_slab=planes
    )
    residual, _, _ = operator.relaxation_residual(strength)
    assert np.abs(residual).max() <= 1e-10 * np.abs(operator.apply(strength)).max()


def test_slab_divergence_relaxation_keeps_the_slab_moments():
    rng = np.random.default_rng(7)
    x, y, z = np.meshgrid(
        H * np.arange(-5, 6),
        H * np.arange(-5, 6),
        np.array([-0.2, -0.1, 0.0, 0.1, 0.2]),
        indexing="ij",
    )
    position = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    radius = np.hypot(position[:, 0], position[:, 1])
    strength = np.zeros_like(position)
    strength[:, 2] = np.exp(-(radius**2) / 0.2**2) * H**3
    strength += 0.05 * rng.normal(size=strength.shape) * np.abs(strength[:, 2:]).max()
    sigma = np.full(len(position), 1.5 * H)
    volume = np.full(len(position), H**3)
    operator = GaussianParticleGridOperator(
        position, sigma, np.ones(len(position)), spacing=H, slip_slab=PLANES
    )
    before_residual = np.linalg.norm(operator.relaxation_residual(strength)[0])
    result = constrained_divergence_relaxation(
        position,
        strength,
        sigma,
        volume,
        grid_spacing=H,
        max_correction_norm=0.5,
        total_kinetic_energy_tolerance=1e-3,
        total_enstrophy_tolerance=1e-2,
        total_helicity_tolerance=0.5,
        variation_tolerance=0.1,
        spectral_convergence_fraction=1.0,
        slip_slab=PLANES,
    )
    after_residual = np.linalg.norm(operator.relaxation_residual(result.vortex_strength)[0])
    assert after_residual < 0.9 * before_residual
    before = gaussian_particle_moments(position, strength, sigma, slip_slab=True)
    after = gaussian_particle_moments(position, result.vortex_strength, sigma, slip_slab=True)
    for index in (0, 2, 3):
        np.testing.assert_allclose(after[index], before[index], rtol=1e-9, atol=1e-15)


def test_refined_children_are_folded_back_with_axial_parity():
    # A tilted, stretched filament element next to the upper plane.
    position = np.array([[0.0, 0.0, 0.22], [0.3, 0.0, 0.0]])
    strength = np.array([[0.6, 0.0, 0.8], [0.0, 0.0, 1.0]])
    sigma = np.full(2, 0.1)
    result = split_stretched_filaments(
        position,
        strength,
        sigma,
        np.full(2, 1e-3),
        reference_vortex_strength=np.array([0.25, 1.0]),
        reference_length=np.array([0.1, 0.1]),
        max_stretch_factor=2.0,
        slip_slab=PLANES,
    )
    assert result.refined_particles == 1
    children = result.position[1:]
    assert np.all((children[:, 2] >= PLANES[0]) & (children[:, 2] <= PLANES[1]))
    crossed = np.flatnonzero(result.vortex_strength[1:, 0] < 0)
    assert len(crossed) == 1
    np.testing.assert_allclose(
        result.vortex_strength[1 + crossed[0]], 0.5 * strength[0] * AXIAL_PARITY
    )
    before = gaussian_particle_moments(position, strength, sigma, slip_slab=True)
    after = gaussian_particle_moments(
        result.position, result.vortex_strength, result.core_radius, slip_slab=True
    )
    np.testing.assert_allclose(after[0], before[0], atol=1e-15)
    np.testing.assert_allclose(after[2], before[2], atol=1e-15)


def test_random_walk_continues_across_the_planes_as_the_image_walk():
    import taichi as ti

    from openonda.vpm.particles.container import Particles
    from openonda.vpm.physics.diffusion.random_walk import apply_random_walk
    from openonda.vpm.physics.diffusion.schemes import DiffusionPhysics

    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    try:
        count = 64
        rng = np.random.default_rng(2)
        position = np.column_stack(
            (rng.uniform(-0.1, 0.1, count), np.zeros(count), np.full(count, 0.2))
        )
        strength = rng.normal(size=(count, 3))
        walks = []
        for slab in (None, PLANES):
            physics = DiffusionPhysics(max_n_particles=count, accumulator_dtype=ti.f64)
            physics._slip_slab_bounds = slab
            particles = Particles(max_n_particles=count, float_dtype="f64")
            particles.add_vortex_particles(
                position=position,
                velocity=np.zeros((count, 3)),
                vortex_strength=strength,
                core_radius=np.full(count, 0.1),
                particle_volume=np.full(count, 1e-3),
                kinematic_viscosity=np.full(count, 0.5),
            )
            particles.set_field("effective_viscosity", np.full(count, 0.5))
            # Steps of about one slab width cross the planes repeatedly.
            states = [particles.position_cpu().copy()]
            for step in range(4):
                apply_random_walk(physics, particles, 0.25, random_seed=11, accepted_step=step)
                states.append(particles.position_cpu().copy())
            walks.append((states, particles.vortex_strength_cpu()))
    finally:
        ti.reset()
    free_states, slab = walks[0][0], walks[1]
    assert np.any(np.abs(free_states[-1][:, 2]) > PLANES[1])
    # The image walk: the same counter-based increments, folded after each step.
    expected_position, expected_strength = position, strength
    for before, after in zip(free_states[:-1], free_states[1:], strict=True):
        expected_position, expected_strength = fold_into_slab(
            expected_position + (after - before), expected_strength, PLANES
        )
    assert np.all((slab[0][-1][:, 2] >= PLANES[0]) & (slab[0][-1][:, 2] <= PLANES[1]))
    np.testing.assert_allclose(slab[0][-1], expected_position, atol=1e-12)
    np.testing.assert_array_equal(slab[1], expected_strength)


def _planar_solver(directory, viscous, stabilization):
    from openonda import vpm

    planes = (-0.5 * H, 0.5 * H)
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=directory,
            name="test",
            numerics=vpm.Numerics(
                time_step_size=0.1,
                compute_device="cpu",
                precision="f64",
                max_n_particles=2000,
                integrator=vpm.RK2(),
                induction=vpm.SlipSlabInduction(
                    vpm.DirectInduction(), z_min=planes[0], z_max=planes[1]
                ),
                viscous=viscous,
                stabilization=stabilization,
            ),
        )
    )
    # A Gaussian vortex on the single particle layer of the slab.
    x, y = np.meshgrid(H * np.arange(-6, 7), H * np.arange(-6, 7), indexing="ij")
    count = x.size
    strength = np.zeros((count, 3))
    strength[:, 2] = np.exp(-(x**2 + y**2).ravel() / 0.25**2) / (np.pi * 0.25**2) * H**3
    solver.add_vortex_particles(
        np.column_stack((x.ravel(), y.ravel(), np.zeros(count))),
        np.zeros((count, 3)),
        strength,
        np.full(count, 1.5 * H),
        np.full(count, H**3),
    )
    return solver, strength[:, 2].sum()


@pytest.mark.parametrize("scheme", ["cs", "rwm"])
def test_planar_core_spreading_and_random_walk_stay_in_the_slab(tmp_path, scheme):
    from openonda import vpm
    from openonda.vpm.config.stabilization import StabilizationConfig

    viscous = (vpm.ViscousConfig.cs if scheme == "cs" else vpm.ViscousConfig.rwm)(
        kinematic_viscosity=2e-3, particle_spacing=H, core_radius_ratio=1.5
    )
    # Core spreading is paired with the regularization that resets its cores.
    stabilization = (
        StabilizationConfig(
            regularization_interval_steps=1,
            regularization_grid_spacing=H,
            regularization_core_radius=1.5 * H,
            regularization_core_radius_trigger=1.55 * H,
            regularization_divergence_trigger=None,
            regularization_misalignment_trigger=None,
        )
        if scheme == "cs"
        else StabilizationConfig()
    )
    solver, circulation = _planar_solver(tmp_path, viscous, stabilization)
    try:
        for _ in range(3):
            solver.advance()
        position = solver.particles.position_cpu()
        strength = solver.particles.vortex_strength_cpu()
        assert np.all(np.abs(position[:, 2]) <= 0.5 * H)
        assert strength[:, 2].sum() == pytest.approx(circulation, rel=1e-12)
        assert not np.any(strength[:, :2])
        if scheme == "cs":
            assert solver.stabilization.regularization_events >= 1
            np.testing.assert_array_equal(position[:, 2], 0.0)
    finally:
        solver.close()


def test_thick_slab_ring_runs_every_stabilization_operator(tmp_path):
    from openonda import vpm
    from openonda.vpm.config.divergence_relaxation import DivergenceRelaxationConfig
    from openonda.vpm.config.filament_refinement import FilamentRefinementConfig
    from openonda.vpm.config.stabilization import StabilizationConfig

    planes = (-0.5, 0.5)
    stabilization = StabilizationConfig(
        regularization_interval_steps=3,
        regularization_grid_spacing=H,
        regularization_divergence_trigger=0.0,
        max_vortex_strength_growth=0.05,
        max_vorticity_growth=1.0,
        divergence_relaxation=DivergenceRelaxationConfig(
            interval_steps=2,
            grid_spacing=H,
            max_correction_norm=0.2,
            total_kinetic_energy_tolerance=1e-2,
            total_enstrophy_tolerance=5e-2,
            total_helicity_tolerance=0.5,
            variation_tolerance=0.2,
            spectral_convergence_fraction=1.0,
        ),
        filament_refinement=FilamentRefinementConfig(
            interval_steps=1, max_vortex_strength_factor=1.02
        ),
    )
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            name="test",
            numerics=vpm.Numerics(
                time_step_size=0.02,
                compute_device="cpu",
                precision="f64",
                max_n_particles=4000,
                integrator=vpm.RK2(),
                induction=vpm.SlipSlabInduction(
                    vpm.DirectInduction(), z_min=planes[0], z_max=planes[1]
                ),
                viscous=vpm.ViscousConfig.inviscid(particle_spacing=H, core_radius_ratio=1.5),
                stabilization=stabilization,
            ),
        )
    )
    # A ring in the x-z plane next to the upper plane, so its in-plane and
    # spanwise strength both meet their images.
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    position = np.array([0.0, 0.0, 0.05]) + 0.3 * np.column_stack(
        (np.cos(theta), np.zeros_like(theta), np.sin(theta))
    )
    strength = (
        0.6
        * np.pi
        / len(theta)
        * np.column_stack((-np.sin(theta), np.zeros_like(theta), np.cos(theta)))
    )
    count = len(theta)
    solver.add_vortex_particles(
        position,
        np.zeros((count, 3)),
        strength,
        np.full(count, 0.12),
        np.full(count, H**3),
        kinematic_viscosity=np.zeros(count),
    )
    try:
        mechanisms = set()
        for _ in range(5):
            solver.advance()
            mechanisms.add(solver.stabilization.last_mechanism)
            position = solver.particles.position_cpu()
            spanwise = solver.particles.vortex_strength_cpu()[:, 2].sum()
            assert np.all((position[:, 2] >= planes[0]) & (position[:, 2] <= planes[1]))
            assert abs(spanwise) <= 1e-14
        assert mechanisms >= {
            "divergence relaxation",
            "conservative regularization",
            "filament refinement",
        }
    finally:
        solver.close()


def test_slab_moment_restoration_leaves_in_plane_totals_free():
    from openonda.vpm.stabilization.divergence_relaxation import restore_particle_moments

    position, strength, sigma = _slab_cloud()
    rotated = strength + 1e-4 * np.random.default_rng(4).normal(size=strength.shape)
    restored, _ = restore_particle_moments(
        position,
        rotated,
        sigma,
        np.full(len(position), H**3),
        strength,
        angular_core_coefficient=1.0 / 3.0,
        slip_slab=True,
    )
    target = gaussian_particle_moments(position, strength, sigma, slip_slab=True)
    after = gaussian_particle_moments(position, restored, sigma, slip_slab=True)
    for index in (0, 2, 3):
        np.testing.assert_allclose(after[index], target[index], atol=1e-15)
    # Only spanwise strength is corrected.
    np.testing.assert_array_equal(restored[:, :2], rotated[:, :2])
