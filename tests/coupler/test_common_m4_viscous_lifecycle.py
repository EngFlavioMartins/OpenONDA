"""Fast lifecycle gates for common-M4 renewal and VPM diffusion schemes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

H = 0.2
DT = 0.01
NU = 1.0e-3
TRANSFER_BOX = np.array([-0.4, 0.4, -0.4, 0.4, -0.4, 0.4])


def _viscous_config(scheme: str):
    from source.solvers.vpm import ViscousConfig

    if scheme == "CS":
        return ViscousConfig.cs(
            kinematic_viscosity=NU,
            particle_spacing=H,
            core_radius_ratio=1.25,
        )
    if scheme == "RWM":
        return ViscousConfig.rwm(
            kinematic_viscosity=NU,
            particle_spacing=H,
            core_radius_ratio=1.25,
        )
    if scheme == "DVH":
        return ViscousConfig.dvh(
            particle_spacing=H,
            padding=2.0,
            threshold=1.0e-14,
            threshold_mode="absolute",
            dvh_support_radius_ratio=3,
            kinematic_viscosity=NU,
            max_nodes=4096,
            core_radius_ratio=1.25,
        )
    if scheme == "GBD":
        return ViscousConfig.gbd(
            particle_spacing=H,
            padding=2.0,
            threshold=1.0e-14,
            threshold_mode="absolute",
            kinematic_viscosity=NU,
            max_nodes=4096,
            core_radius_ratio=1.25,
        )
    if scheme == "NONE":
        return ViscousConfig.inviscid(particle_spacing=H, core_radius_ratio=1.25)
    raise AssertionError(f"unsupported test scheme {scheme}")


def _make_solver(case_dir: Path, scheme: str):
    from source.solvers.vpm import Numerics, VelocityConfig, VPMCase, VPMSolver

    return VPMSolver(
        VPMCase(
            directory=case_dir,
            numerics=Numerics(
                time_step_size=DT,
                compute_device="CPU",
                precision="f64",
                max_n_particles=4096,
                domain_bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
                velocity=VelocityConfig.direct(),
                viscous=_viscous_config(scheme),
                verbose=False,
            ),
        ),
    )


def _renew(solver, *, donor_position: np.ndarray | None = None, donor_gamma=None):
    from source.coupler.vorticity_transfer import replace_particles_from_lattice_blend

    position = (
        np.array([[0.0, 0.0, 0.0]])
        if donor_position is None
        else np.asarray(donor_position, dtype=np.float64).reshape(-1, 3)
    )
    gamma = (
        np.array([[0.0, 0.0, H**3]])
        if donor_gamma is None
        else np.asarray(donor_gamma, dtype=np.float64).reshape(-1, 3)
    )
    volume = np.full(len(position), H**3)
    return replace_particles_from_lattice_blend(
        solver,
        transfer_box=TRANSFER_BOX,
        eta_blend_width=H,
        fvm_position=position,
        fvm_cell_volume=volume,
        fvm_vorticity=gamma / volume[:, None],
        lattice_anchor=np.zeros(3),
        particle_spacing=H,
        core_radius_ratio=1.25,
        kinematic_viscosity=NU,
        compute_divergence_diagnostic=False,
    )


def _particle_state(solver) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        solver.particles.position_cpu(use_cache=False).astype(np.float64),
        solver.particles.vortex_strength_cpu(use_cache=False).astype(np.float64),
        solver.particles.core_radius_cpu(use_cache=False).astype(np.float64),
    )


@pytest.mark.parametrize("scheme", ["NONE", "CS", "RWM", "DVH", "GBD"])
def test_common_m4_api_is_scheme_agnostic_and_same_state_is_idempotent(
    tmp_path: Path, scheme: str
) -> None:
    """The coupler accepts every public viscous scheme without a GBD-only path."""
    solver = _make_solver(tmp_path / scheme.lower(), scheme)
    first = _renew(solver)
    state = _particle_state(solver)

    second = _renew(solver)

    assert first.n_particles_after > 0
    assert second.n_particles_removed == 0
    assert second.n_particles_injected == 0
    assert second.n_particles_after == first.n_particles_after
    for actual, expected in zip(_particle_state(solver), state, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_gbd_and_common_m4_are_stable_across_repeated_physical_lifecycles(
    tmp_path: Path,
) -> None:
    """GBD may regenerate once per step; renewal must not create a second churn cycle."""
    solver = _make_solver(tmp_path / "gbd_lifecycle", "GBD")
    _renew(solver)
    counts: list[int] = []

    for _ in range(3):
        solver.stepper._apply_viscous_diffusion(DT)
        renewed = _renew(solver)
        once = _particle_state(solver)
        repeated = _renew(solver)

        assert renewed.n_particles_after > 0
        assert repeated.n_particles_removed == 0
        assert repeated.n_particles_injected == 0
        for actual, expected in zip(_particle_state(solver), once, strict=True):
            np.testing.assert_array_equal(actual, expected)
        positions, strength, _radius = once
        assert np.isfinite(strength).all()
        assert len(np.unique(positions, axis=0)) == len(positions)
        counts.append(len(positions))

    # The first cycle converts GBD's independently phased lattice to the fixed
    # coupler lattice.  Subsequent physical cycles are the steady lifecycle.
    assert max(counts[1:]) <= 2 * min(counts[1:])
    assert max(counts) < solver.particles.capacity // 4


def test_cs_diffusion_then_common_m4_preserves_gaussian_blob_moments(tmp_path: Path) -> None:
    """M4' carries CS diffusion age and all Gaussian moments through renewal."""
    from source.solvers.vpm.stabilization.filament_refinement import gaussian_particle_moments

    solver = _make_solver(tmp_path / "cs_lifecycle", "CS")
    position = np.array([[0.3, 0.03, -0.02]])
    strength = np.array([[0.0, H**3, 0.0]])
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros((1, 3)),
        vortex_strength=strength,
        core_radius=np.array([1.25 * H]),
        particle_volume=np.array([H**3]),
        kinematic_viscosity=np.array([NU]),
    )
    for _ in range(4):
        solver.physics.core_spreading_diffusion(solver.particles, DT)
    before = _particle_state(solver)
    moments_before = gaussian_particle_moments(*before)

    # FVM and VPM describe the same synchronized circulation state. Therefore
    # this operation is a representation change only, not a physical update.
    _renew(solver, donor_position=before[0], donor_gamma=before[1])
    after = _particle_state(solver)
    moments_after = gaussian_particle_moments(*after)

    np.testing.assert_allclose(moments_after[0], moments_before[0], atol=2.0e-14)
    np.testing.assert_allclose(moments_after[2], moments_before[2], atol=2.0e-14)
    np.testing.assert_allclose(moments_after[3], moments_before[3], atol=2.0e-14)
