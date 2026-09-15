"""Field-level qualification of core reset, independently of its moment gates."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.stabilization.remeshing import gaussian_core_remesh


def test_grid_projection_removes_gradient_preserves_solenoidal_field_and_mean():
    from source.solvers.vpm.stabilization.remeshing import project_grid_strength

    n = 16
    x = np.arange(n) * 2 * np.pi / n
    field = np.zeros((n, n, n, 3))
    field[..., 0] = np.cos(x)[:, None, None] + 0.3
    field[..., 1] = np.sin(x)[:, None, None]
    projected = project_grid_strength(field, 2 * np.pi / n)
    np.testing.assert_allclose(projected[..., 0], 0.3, atol=1e-14)
    np.testing.assert_allclose(projected[..., 1], field[..., 1], atol=1e-14)
    np.testing.assert_allclose(projected[..., 2], 0, atol=1e-14)


def _particles():
    arrays = {
        "position": np.array([[-0.11, 0.03, 0.02], [0.13, -0.04, 0.01]]),
        "vortex_strength": np.array([[0.3, -0.1, 1.0], [-0.3, 0.1, -0.8]]),
        "core_radius": np.array([0.14, 0.18]),
        "kinematic_viscosity": np.array([0.001, 0.001]),
        "eddy_viscosity": np.array([0.0, 0.003]),
        "group_id": np.array([0, 1]),
        "zone_id": np.array([0, 0]),
    }
    return arrays, SimpleNamespace(
        **{name + "_cpu": lambda values=values: values.copy() for name, values in arrays.items()}
    )


def _field(points, arrays):
    d = points[:, None, :] - arrays["position"][None, :, :]
    sigma = arrays["core_radius"]
    weight = np.exp(-np.sum(d * d, axis=2) / sigma**2) / (np.pi**1.5 * sigma**3)
    return weight @ arrays["vortex_strength"]


def test_variable_core_reset_preserves_resolved_gaussian_field_and_second_moments():
    source, particles = _particles()
    remapped = gaussian_core_remesh(
        particles, spacing=0.035, core_radius=0.07, tail_budget=1e-6, max_particles=100000
    )
    points = np.random.default_rng(15).uniform(-0.25, 0.25, (80, 3))
    error = np.linalg.norm(_field(points, remapped) - _field(points, source)) / np.linalg.norm(
        _field(points, source)
    )
    assert error < 0.003
    # Crossing the 10,000-particle diagnostic threshold must not change the
    # quadratic form used to decide whether this representation change passes.
    from source.solvers.vpm.numerics.fourier_integrals import _grid_for_particles
    from source.solvers.vpm.stabilization.regularization import _transfer_integrals

    grid = _grid_for_particles(np.vstack((source["position"], remapped["position"])), 0.035)
    before = _transfer_integrals(
        source["position"], source["vortex_strength"], source["core_radius"], np.ones(2), grid
    )
    after = _transfer_integrals(
        remapped["position"],
        remapped["vortex_strength"],
        remapped["core_radius"],
        remapped["particle_volume"],
        grid,
    )
    assert len(remapped["position"]) > 10_000
    for quantity in ("total_kinetic_energy", "total_enstrophy"):
        assert abs(after[quantity] / before[quantity] - 1) < 0.005
    pruning_bound = 1e-6 * np.linalg.norm(source["vortex_strength"], axis=1).sum()
    np.testing.assert_allclose(
        remapped["vortex_strength"].sum(axis=0),
        source["vortex_strength"].sum(axis=0),
        atol=pruning_bound,
    )
    # Gaussian second moment in one coordinate is sigma²/2. No nu*dt term.
    for axis in range(3):
        before = (source["position"][:, axis] ** 2 + source["core_radius"] ** 2 / 2) @ source[
            "vortex_strength"
        ]
        after = (remapped["position"][:, axis] ** 2 + remapped["core_radius"] ** 2 / 2) @ remapped[
            "vortex_strength"
        ]
        np.testing.assert_allclose(after, before, atol=2e-6)


def test_capacity_does_not_silently_override_the_tail_budget():
    _, particles = _particles()
    with pytest.raises(ValueError, match="capacity is 10"):
        gaussian_core_remesh(
            particles, spacing=0.04, core_radius=0.07, tail_budget=1e-3, max_particles=10
        )


def test_group_remap_preserves_overlapping_contributions_and_material_properties():
    source, particles = _particles()
    # Coincident, opposing contributions would cancel if merged before tagging.
    source["position"][:] = 0
    source["core_radius"][:] = 0.14
    source["vortex_strength"][1] = -source["vortex_strength"][0]
    mapped = gaussian_core_remesh(
        particles,
        spacing=0.04,
        core_radius=0.07,
        tail_budget=1e-6,
        max_particles=100000,
        preserve_groups=True,
    )
    points = np.random.default_rng(42).uniform(-0.2, 0.2, (40, 3))
    for group in (0, 1):
        old = {key: value[source["group_id"] == group] for key, value in source.items()}
        new = {key: value[mapped["group_id"] == group] for key, value in mapped.items()}
        assert np.all(new["eddy_viscosity"] == source["eddy_viscosity"][group])
        error = np.linalg.norm(_field(points, new) - _field(points, old)) / np.linalg.norm(
            _field(points, old)
        )
        assert error < 0.005
    np.testing.assert_allclose(_field(points, mapped), 0, atol=1e-10)
    with pytest.raises(ValueError, match="capacity"):
        gaussian_core_remesh(
            particles,
            spacing=0.04,
            core_radius=0.07,
            tail_budget=1e-6,
            max_particles=len(mapped["position"]) - 1,
            preserve_groups=True,
        )


def test_group_remap_cannot_enable_unconstrained_projection():
    from source.solvers.vpm.config.stabilization import StabilizationConfig

    with pytest.raises(ValueError, match="transfer-only"):
        StabilizationConfig(regularization_preserve_groups=True)


def test_core_enlargement_requires_explicit_filtering_instead_of_negative_variance():
    _, particles = _particles()
    with pytest.raises(ValueError, match="cannot enlarge"):
        gaussian_core_remesh(
            particles, spacing=0.04, core_radius=0.2, tail_budget=1e-3, max_particles=1000
        )


def test_reset_to_the_configured_core_accepts_float32_storage_roundoff():
    _, particles = _particles()
    particles.core_radius_cpu = lambda: np.full(2, 0.08, dtype=np.float32)
    remapped = gaussian_core_remesh(
        particles, spacing=0.04, core_radius=0.08, tail_budget=1e-4, max_particles=1000
    )
    np.testing.assert_array_equal(remapped["core_radius"], 0.08)


def test_gaussian_reconstruction_is_not_silently_used_for_other_kernels():
    from source.solvers.vpm.config.case import Numerics
    from source.solvers.vpm.config.stabilization import StabilizationConfig

    with pytest.raises(ValueError, match="require GAUSSIAN"):
        Numerics(
            particle_kernel="WINCKELMANS",
            stabilization=StabilizationConfig(
                regularization_interval_steps=10, regularization_grid_spacing=0.1
            ),
        )


@pytest.mark.parametrize("transfer", [0.002, 0.02])
@pytest.mark.parametrize("preserve_groups", [False, True])
def test_transfer_only_remap_keeps_cores_and_rolls_back_excess_transfer(
    monkeypatch, transfer, preserve_groups
):
    from source.solvers.vpm.config.stabilization import StabilizationConfig
    from source.solvers.vpm.stabilization import regularization

    arrays, _ = _particles()
    arrays.update(particle_volume=np.ones(2), velocity=np.zeros((2, 3)))
    original = {
        key: value.astype(np.float32) if value.dtype.kind == "f" else value.copy()
        for key, value in arrays.items()
    }
    current = {key: value.copy() for key, value in original.items()}
    particles = SimpleNamespace(
        **{key + "_cpu": lambda key=key: current[key].copy() for key in current}
    )
    replacements = []

    def replace(**data):
        replacements.append(data)
        current.update({key: np.asarray(value).copy() for key, value in data.items()})

    calls = []

    def integrals(*args):
        calls.append(args)
        value = 1.0 if len(calls) == 1 else 1.0 + transfer
        return {"total_kinetic_energy": value, "total_enstrophy": value, "total_helicity": 0.0}

    monkeypatch.setattr(regularization, "_transfer_integrals", integrals)
    monkeypatch.setattr(
        regularization,
        "discretization_health",
        lambda *args: {
            "vorticity_divergence_error": 1.0,
            "vortex_strength_misalignment_degrees": 90.0,
        },
    )
    monkeypatch.setattr(
        "source.solvers.vpm.stabilization.divergence_relaxation.constrained_divergence_relaxation",
        lambda *args, **kwargs: pytest.fail("transfer-only mode must not project"),
    )
    cfg = StabilizationConfig(
        regularization_interval_steps=1,
        regularization_grid_spacing=0.04,
        regularization_core_radius=0.07,
        regularization_max_particles=100000,
        regularization_transfer_only=True,
        regularization_preserve_groups=preserve_groups,
        regularization_total_kinetic_energy_dissipation_limit=0.01,
        regularization_total_enstrophy_dissipation_limit=0.01,
    )
    ctx = SimpleNamespace(
        particles=particles,
        np_dtype=np.float32,
        mutations=SimpleNamespace(replace=replace),
        state=SimpleNamespace(particles_removed=0, vortex_strength_removed=np.zeros(3)),
        metrics=SimpleNamespace(kinetic_energy_rate=0.0),
    )
    if transfer > 0.01:
        with pytest.raises(RuntimeError, match="declared transfer interval"):
            regularization.regularize(ctx, cfg)
        for key in original:
            np.testing.assert_array_equal(current[key], original[key])
    else:
        outcome = regularization.regularize(ctx, cfg)
        np.testing.assert_array_equal(current["core_radius"], np.float32(0.07))
        assert not outcome.projected
        assert outcome.total_kinetic_energy_change_relative == pytest.approx(transfer)
        assert outcome.total_enstrophy_change_relative == pytest.approx(transfer)
        assert len(replacements) == 1  # No adaptive broadening or enstrophy adjustment.
        assert len(calls) == 2
        if preserve_groups:
            from source.solvers.vpm.stabilization.filament_refinement import (
                gaussian_particle_moments,
            )

            for group in (0, 1):
                old = original["group_id"] == group
                new = current["group_id"] == group
                before = gaussian_particle_moments(
                    original["position"][old],
                    original["vortex_strength"][old],
                    original["core_radius"][old],
                )
                after = gaussian_particle_moments(
                    current["position"][new],
                    current["vortex_strength"][new],
                    current["core_radius"][new],
                )
                for index in (0, 2, 3):
                    np.testing.assert_allclose(after[index], before[index], atol=1e-7)
