"""
Audit tests for the grid-based viscous diffusion schemes (GBD and DVH).

These verify the numerical guarantees the schemes are built on:

GBD (Cottet & Koumoutsakos 2000) — explicit FTCS Laplacian on an M4'-remesh:
  * exact conservation of total circulation (clamped-edge Neumann stencil
    telescopes to zero);
  * exact discrete diffusion rate: the second moment of the field grows by
    6·nu·dt per step (the discrete identity Σ r²(∇²_h u) = 6 Σ u holds away
    from boundaries);
  * the documented stability bound alpha = nu·dt/h² ≤ 1/6 (the checkerboard
    mode amplifies as |1 − 12·alpha|).

GBD scales with dt directly → any dt below the CFL bound is valid.

DVH (Durante et al. 2024) — heat-kernel scatter with Shepard normalization:
  * exact conservation of total circulation (Shepard weights sum to 1 per
    particle);
  * the diffusive increment per application is FIXED at Δt_d = β·R_d²/(4·nu)
    (the kernel width is β·R_d², independent of the dt argument) — verified
    via the scattered cloud's second moment ⟨r²⟩ ≈ 6·nu·Δt_d.

DVH therefore requires dt = Δt_d — the solver pins dt to Δt_d (one firing
per step, the diffusion operator acting every step) and the coupler adopts
the VPM dt for the FVM (see source/coupler/core/solver.py initialize()).
"""

from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

from source.solvers.VPM.config.types import ViscousConfig
from source.solvers.VPM.physics.diffusion import _DVH_BETA, DiffusionPhysics


@pytest.fixture(scope="module")
def physics():
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    return DiffusionPhysics(particles_kernel="GAUSSIAN", max_particles=10_000)


N = 32  # lattice nodes per axis
H = 0.05  # grid spacing [m]


def _gaussian_blob_grid(sigma: float = 4.0 * H) -> np.ndarray:
    """ω_z Gaussian blob centered on the lattice, well inside the boundary."""
    ax = (np.arange(N) - (N - 1) / 2.0) * H
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    field = np.zeros((N, N, N, 3), dtype=np.float32)
    field[..., 2] = np.exp(-(X**2 + Y**2 + Z**2) / (2.0 * sigma**2))
    return field


def _laplacian_step(physics, field: np.ndarray, alpha: float) -> np.ndarray:
    """One FTCS step through the production GPU kernel."""
    nx, ny, nz = field.shape[:3]
    enx, eny, enz = physics._ensure_grid_capacity(nx, ny, nz)
    assert (enx, eny, enz) == (nx, ny, nz)
    buf = np.zeros((*physics._grid_shape, 3), dtype=np.float32)
    buf[:nx, :ny, :nz, :] = field
    physics._current_grid.from_numpy(buf)
    physics._other_grid.fill(0.0)
    physics._body_mask_grid.fill(0)
    physics._laplacian_step_gpu_kernel(
        physics._current_grid,
        physics._other_grid,
        physics._body_mask_grid,
        alpha,
        nx,
        ny,
        nz,
    )
    return physics._other_grid.to_numpy()[:nx, :ny, :nz, :]


def _second_moment(field: np.ndarray) -> float:
    """⟨r²⟩ of the ω_z component about the lattice center."""
    ax = (np.arange(N) - (N - 1) / 2.0) * H
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    w = field[..., 2].astype(np.float64)
    return float(np.sum((X**2 + Y**2 + Z**2) * w) / np.sum(w))


# ─────────────────────────────────────────────────────────────────────────────
# GBD
# ─────────────────────────────────────────────────────────────────────────────
def test_gbd_laplacian_conserves_total_circulation(physics):
    field = _gaussian_blob_grid()
    alpha = 0.15  # < 1/6
    out = _laplacian_step(physics, field, alpha)
    g0 = field[..., 2].astype(np.float64).sum()
    g1 = out[..., 2].astype(np.float64).sum()
    assert abs(g1 - g0) < 1e-4 * abs(g0), f"Γ drift {abs(g1 - g0) / g0:.2e}"


def test_gbd_box_mask_is_solid_free_and_zero_flux(physics):
    """The solid mask must suppress regeneration in the body without making
    the wall an absorbing circulation sink."""
    nx = ny = nz = 20
    grid_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    physics._ensure_grid_capacity(nx, ny, nz)
    physics.configure_body_box([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1])
    physics._prepare_body_mask_current_grid(grid_min, H, nx, ny, nz)
    mask = physics._body_mask_grid.to_numpy()[:nx, :ny, :nz].astype(bool)

    field = np.zeros((*physics._grid_shape, 3), dtype=np.float32)
    field[:nx, :ny, :nz, 2] = 1.0
    field[:nx, :ny, :nz, 2][mask] = 0.0
    physics._current_grid.from_numpy(field)
    physics._other_grid.fill(0.0)
    physics._laplacian_step_gpu_kernel(
        physics._current_grid,
        physics._other_grid,
        physics._body_mask_grid,
        0.12,
        nx,
        ny,
        nz,
    )
    out = physics._other_grid.to_numpy()[:nx, :ny, :nz, 2]
    assert np.all(out[mask] == 0.0)
    np.testing.assert_allclose(out.sum(dtype=np.float64), (~mask).sum(), atol=1e-5)

    # Restore the module-scoped fixture for the remaining mask-free audits.
    physics._body_box_bounds = None
    physics._body_mask_active = False
    physics._body_mask_grid.fill(0)


def test_gbd_moment_growth_equals_6_nu_dt(physics):
    """Discrete identity: ⟨r²⟩ grows by exactly 6·alpha·h² = 6·nu·dt per step."""
    nu, dt = 1e-3, 0.3
    alpha = nu * dt / H**2  # 0.12 < 1/6
    field = _gaussian_blob_grid()
    out = _laplacian_step(physics, field, alpha)
    growth = _second_moment(out) - _second_moment(field)
    expected = 6.0 * nu * dt
    assert abs(growth - expected) < 0.01 * expected, (
        f"moment growth {growth:.6e} vs 6·nu·dt = {expected:.6e}"
    )


def test_gbd_stability_bound_alpha_one_sixth(physics):
    """Checkerboard mode amplifies by |1 − 12 alpha|: decays for alpha < 1/6,
    explodes above — confirming dt ≤ h²/(6 nu) is the right (and only)
    time-step requirement for GBD."""
    ax = np.arange(N)
    gi, gj, gk = np.meshgrid(ax, ax, ax, indexing="ij")
    checker = ((-1.0) ** (gi + gj + gk)).astype(np.float32)
    field = np.zeros((N, N, N, 3), dtype=np.float32)
    field[..., 2] = checker

    for alpha, should_grow in [(0.15, False), (0.25, True)]:
        out = field
        for _ in range(10):
            out = _laplacian_step(physics, out, alpha)
        amp = float(np.abs(out[2:-2, 2:-2, 2:-2, 2]).max())
        if should_grow:
            assert amp > 10.0, f"alpha={alpha}: expected instability, amp={amp:.3f}"
        else:
            assert amp < 1.0, f"alpha={alpha}: expected decay, amp={amp:.3f}"


def test_gbd_max_dt_formula():
    vc = ViscousConfig.gbd(h=0.05, viscosity=0.001)
    assert np.isclose(vc.gbd_max_dt(), 0.05**2 / (6 * 0.001))


def test_particle_cap_protects_circulation_and_local_wake(physics):
    values = np.array([40.0, 30.0, 20.0, 10.0, 1.0, 1.0])
    grid = values.reshape(6, 1, 1)
    importance = np.array([1.0, 1.0, 1.0, 1.0, 100.0, 90.0]).reshape(6, 1, 1)
    ix = np.arange(6)
    iy = np.zeros(6, dtype=int)
    iz = np.zeros(6, dtype=int)

    kept_x, _, _, _, old_count = physics._cap_surviving_nodes(
        grid,
        ix,
        iy,
        iz,
        cap=4,
        importance=importance,
        min_abs_fraction=0.8,
    )

    assert old_count == 6
    assert set(kept_x) == {0, 1, 2, 4}
    assert values[kept_x].sum() / values.sum() >= 0.8


def test_particle_cap_falls_back_to_strongest_when_budget_is_infeasible(physics):
    grid = np.array([40.0, 30.0, 20.0, 10.0]).reshape(4, 1, 1)
    ix = np.arange(4)
    zeros = np.zeros(4, dtype=int)
    kept_x, _, _, _, _ = physics._cap_surviving_nodes(
        grid,
        ix,
        zeros,
        zeros,
        cap=2,
        importance=np.ones_like(grid),
        min_abs_fraction=0.99,
    )

    assert set(kept_x) == {0, 1}
    assert np.isclose(grid[kept_x, 0, 0].sum() / grid.sum(), 0.7)


def test_particle_cap_preserves_equal_vortex_groups(physics):
    grid = np.ones((20, 1, 1))
    grid[10:] *= 0.999
    ix = np.arange(20)
    zeros = np.zeros(20, dtype=int)
    labels = np.repeat([0, 1], 10).reshape(20, 1, 1)

    kept_x, _, _, _, _ = physics._cap_surviving_nodes(
        grid,
        ix,
        zeros,
        zeros,
        cap=10,
        labels=labels,
    )

    assert np.count_nonzero(kept_x < 10) == 5
    assert np.count_nonzero(kept_x >= 10) == 5


def test_regenerated_group_ids_fill_empty_diffusion_nodes(physics):
    positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    circulations = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    labels = physics._scatter_id_field(
        positions,
        circulations,
        np.array([0, 1]),
        np.zeros(3),
        1.0,
        5,
        1,
        1,
        propagate_to=(np.arange(5), np.zeros(5, dtype=int), np.zeros(5, dtype=int)),
    )

    assert labels[:, 0, 0].tolist() == [0, 0, 0, 1, 1]


def test_viscous_config_carries_particle_cap():
    vc = ViscousConfig.gbd(
        h=0.05,
        viscosity=0.001,
        max_nodes=350_000,
        cap_abs_fraction=0.99,
    )
    assert vc.gbd_max_nodes == 350_000
    assert vc.regen_cap_abs_fraction == 0.99


# ─────────────────────────────────────────────────────────────────────────────
# DVH
# ─────────────────────────────────────────────────────────────────────────────
def _dvh_scatter(physics, pos, circ, rd_ratio=4):
    """Scatter through the production DVH kernel; return the active grid."""
    nx = ny = nz = N
    physics._ensure_grid_capacity(nx, ny, nz)
    grid_min = np.array([-(N - 1) / 2.0 * H] * 3)
    physics._dvh_scatter_circ(
        np.asarray(pos, dtype=np.float64),
        np.asarray(circ, dtype=np.float64),
        grid_min,
        H,
        1e-3,  # nu   (unused by the kernel width — that is the point)
        0.05,  # dt   (explicitly unused)
        nx,
        ny,
        nz,
        rd_ratio=rd_ratio,
    )
    return physics._current_grid.to_numpy()[:nx, :ny, :nz, :]


def test_dvh_scatter_conserves_total_circulation(physics):
    """Shepard normalization conserves each particle's Γ exactly."""
    rng = np.random.default_rng(7)
    pos = (rng.random((50, 3)) - 0.5) * (N - 12) * H  # interior particles
    circ = rng.normal(size=(50, 3)) * 1e-3
    grid = _dvh_scatter(physics, pos, circ)
    for c in range(3):
        g_in = circ[:, c].sum()
        g_out = float(grid[..., c].astype(np.float64).sum())
        assert abs(g_out - g_in) < 1e-6 * (abs(g_in) + 1e-9), f"component {c}"


@pytest.mark.parametrize("rd_ratio", [3, 4, 5])
def test_dvh_diffusive_width_is_fixed_at_dt_d(physics, rd_ratio):
    """A single particle's scattered cloud has ⟨r²⟩ ≈ 6·nu·Δt_d with
    Δt_d = β·R_d²/(4·nu) — i.e. the diffusion per application is set by
    β·R_d², NOT by the dt argument.  This is why DVH dictates the time step
    while GBD merely bounds it."""
    pos = np.array([[0.0, 0.0, 0.0]])  # exactly on the lattice center node
    circ = np.array([[0.0, 0.0, 1e-3]])
    grid = _dvh_scatter(physics, pos, circ, rd_ratio=rd_ratio)

    ax = (np.arange(N) - (N - 1) / 2.0) * H
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    w = grid[..., 2].astype(np.float64)
    r2_mean = float(np.sum((X**2 + Y**2 + Z**2) * w) / np.sum(w))

    # Continuous heat kernel e^{-r²/(4 nu Δt_d)} has ⟨r²⟩ = 6 nu Δt_d = 1.5 β R_d²
    expected = 1.5 * _DVH_BETA * (rd_ratio * H) ** 2
    assert abs(r2_mean - expected) < 0.15 * expected, (
        f"rd_ratio={rd_ratio}: ⟨r²⟩={r2_mean:.4e} vs 1.5·β·R_d²={expected:.4e}"
    )


def test_dvh_required_dt_formula():
    vc = ViscousConfig.dvh(h=0.05, viscosity=0.001, dvh_rd_ratio=3)
    expected = _DVH_BETA * (3 * 0.05) ** 2 / (4 * 0.001)
    assert np.isclose(vc.dvh_required_dt(), expected)
    # cubeFlow numbers: this is ≈ 0.433 s — far above the convective dt,
    # which is exactly why the coupler must adopt the VPM dt.
    assert 0.4 < expected < 0.45


# ─────────────────────────────────────────────────────────────────────────────
# DVH with per-particle effective viscosity (LES coupling)
# ─────────────────────────────────────────────────────────────────────────────
def _dvh_scatter_nu_eff(physics, pos, circ, nu, nu_eff, rd_ratio=4):
    """Scatter through the production DVH kernel with per-particle nu_eff."""
    nx = ny = nz = N
    physics._ensure_grid_capacity(nx, ny, nz)
    grid_min = np.array([-(N - 1) / 2.0 * H] * 3)
    physics._dvh_scatter_circ(
        np.asarray(pos, dtype=np.float64),
        np.asarray(circ, dtype=np.float64),
        grid_min,
        H,
        nu,
        0.05,  # dt (explicitly unused)
        nx,
        ny,
        nz,
        rd_ratio=rd_ratio,
        nu_eff_np=None if nu_eff is None else np.asarray(nu_eff, dtype=np.float64),
    )
    return physics._current_grid.to_numpy()[:nx, :ny, :nz, :]


def test_dvh_nu_eff_uniform_equals_baseline(physics):
    """nu_eff = nu for every particle must reproduce the constant-nu scatter."""
    rng = np.random.default_rng(11)
    pos = (rng.random((30, 3)) - 0.5) * (N - 12) * H
    circ = rng.normal(size=(30, 3)) * 1e-3
    nu = 1e-3
    base = _dvh_scatter_nu_eff(physics, pos, circ, nu, None).copy()
    les = _dvh_scatter_nu_eff(physics, pos, circ, nu, np.full(30, nu))
    np.testing.assert_allclose(les, base, atol=1e-12)


def test_dvh_nu_eff_scales_diffusive_width(physics):
    """A particle with nu_eff = q·nu must spread with ⟨r²⟩ = q·(1.5·β·R_d²).

    This is the exact split-step heat kernel for that particle's effective
    viscosity — the mechanism by which the Smagorinsky nu_t acts in DVH runs.
    """
    pos = np.array([[0.0, 0.0, 0.0]])
    circ = np.array([[0.0, 0.0, 1e-3]])
    nu = 1e-3
    ax = (np.arange(N) - (N - 1) / 2.0) * H
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")

    for q in (1.0, 2.0, 3.0):
        grid = _dvh_scatter_nu_eff(physics, pos, circ, nu, np.array([q * nu]))
        w = grid[..., 2].astype(np.float64)
        r2_mean = float(np.sum((X**2 + Y**2 + Z**2) * w) / np.sum(w))
        expected = q * 1.5 * _DVH_BETA * (4 * H) ** 2
        assert abs(r2_mean - expected) < 0.15 * expected, (
            f"q={q}: ⟨r²⟩={r2_mean:.4e} vs q·1.5·β·R_d²={expected:.4e}"
        )


def test_dvh_nu_eff_width_cap(physics):
    """nu_eff/nu beyond q_max is clipped: width saturates at q_max·β·R_d²."""
    pos = np.array([[0.0, 0.0, 0.0]])
    circ = np.array([[0.0, 0.0, 1e-3]])
    nu = 1e-3
    capped = _dvh_scatter_nu_eff(physics, pos, circ, nu, np.array([100.0 * nu])).copy()
    at_cap = _dvh_scatter_nu_eff(physics, pos, circ, nu, np.array([4.0 * nu]))
    np.testing.assert_allclose(capped, at_cap, atol=1e-12)


def test_dvh_nu_eff_conserves_circulation(physics):
    """Shepard normalization keeps per-particle Γ conservation for any width."""
    rng = np.random.default_rng(13)
    pos = (rng.random((40, 3)) - 0.5) * (N - 12) * H
    circ = rng.normal(size=(40, 3)) * 1e-3
    nu = 1e-3
    nu_eff = nu * (1.0 + 3.0 * rng.random(40))
    grid = _dvh_scatter_nu_eff(physics, pos, circ, nu, nu_eff)
    for c in range(3):
        g_in = circ[:, c].sum()
        g_out = float(grid[..., c].astype(np.float64).sum())
        assert abs(g_out - g_in) < 1e-6 * (abs(g_in) + 1e-9), f"component {c}"


# ─────────────────────────────────────────────────────────────────────────────
# Regenerated-particle core radius (radius consistency for coupled runs)
# ─────────────────────────────────────────────────────────────────────────────
def test_regen_radius_respects_configured_ratio(physics):
    """DVH/GBD regen assigns σ = regen_radius_ratio·h to every rebuilt particle.

    Regression: the ratio was a hardcoded 2.5, silently overriding the
    coupler hand-off radii (overlap_radius_ratio·h) every step — the Beale
    strength correction then deconvolved with the wrong kernel width
    (measured ~4× in-box velocity error at 2.5h vs the corrected-for 1.5h).
    """
    ix = np.array([1, 2, 3])
    iy = np.array([1, 1, 1])
    iz = np.array([2, 2, 2])
    grid_np = np.zeros((8, 8, 8, 3), dtype=np.float32)
    grid_np[ix, iy, iz, 2] = 1e-3
    zone = np.zeros((8, 8, 8), dtype=np.int32)
    group = np.zeros((8, 8, 8), dtype=np.int32)

    default_ratio = physics.regen_radius_ratio
    try:
        for ratio in (2.5, 1.5, 1.2):
            physics.regen_radius_ratio = ratio
            out = physics._build_diffusion_particle_arrays(
                ix, iy, iz, grid_np, np.zeros(3), H, 1e-3, 0.01, None, 3, zone, group
            )
            np.testing.assert_allclose(out["radius"], ratio * H, rtol=1e-6)
    finally:
        physics.regen_radius_ratio = default_ratio


def test_viscous_config_carries_regen_radius_ratio():
    """ViscousConfig exposes the standard regen_radius_ratio default of 2.5."""
    vc = ViscousConfig.gbd(h=0.05, viscosity=1e-3)
    assert vc.regen_radius_ratio == 2.5
    tuned = ViscousConfig.gbd(h=0.05, viscosity=1e-3, regen_radius_ratio=1.5)
    assert tuned.regen_radius_ratio == 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Turbulent-viscosity carry through regen (Bug B) + per-node α (Bug A)
# ─────────────────────────────────────────────────────────────────────────────
def test_regen_carries_viscosity_turbulent(physics):
    """``_build_diffusion_particle_arrays`` must emit and carry ν_t (Bug B).

    Without ``nu_t_grid`` the regenerated particles get ν_t = 0 (molecular
    fallback).  With a ``nu_t_grid`` the new particles inherit the |Γ|-weighted
    ν_t of their grid node, so ν_t survives the DVH/GBD rebuild and reaches
    backup instead of being silently wiped every step.
    """
    ix = np.array([1, 2, 3, 4])
    iy = np.array([1, 1, 2, 2])
    iz = np.array([2, 2, 3, 3])
    grid_np = np.zeros((8, 8, 8, 3), dtype=np.float32)
    grid_np[ix, iy, iz, 2] = 1e-3
    zone = np.zeros((8, 8, 8), dtype=np.int32)
    group = np.zeros((8, 8, 8), dtype=np.int32)

    # Without nu_t_grid → ν_t = 0 (backward-compatible default)
    out_no_nut = physics._build_diffusion_particle_arrays(
        ix, iy, iz, grid_np, np.zeros(3), H, 1e-3, 0.01, None, 4, zone, group
    )
    assert "viscosity_turbulent" in out_no_nut
    np.testing.assert_allclose(out_no_nut["viscosity_turbulent"], 0.0, atol=1e-12)

    # With nu_t_grid → ν_t carried from grid node
    nu_t_grid = np.zeros((8, 8, 8), dtype=np.float32)
    expected_nu_t = np.array([1e-4, 2e-4, 3e-4, 5e-4], dtype=np.float32)
    nu_t_grid[ix, iy, iz] = expected_nu_t
    out_nut = physics._build_diffusion_particle_arrays(
        ix,
        iy,
        iz,
        grid_np,
        np.zeros(3),
        H,
        1e-3,
        0.01,
        None,
        4,
        zone,
        group,
        nu_t_grid=nu_t_grid,
    )
    np.testing.assert_allclose(out_nut["viscosity_turbulent"], expected_nu_t, rtol=1e-6)

    # Molecular viscosity is still stamped unchanged
    np.testing.assert_allclose(out_nut["viscosity"], 1e-3, rtol=1e-6)


def test_scatter_scalar_weighted_averages_by_circulation(physics):
    """``_scatter_scalar_weighted`` returns the |Γ|-weighted average per node."""
    nx = ny = nz = 8
    grid_min = np.zeros(3, dtype=np.float32)
    # Two particles on the same node, different weights & scalars
    pos = np.array([[H, H, H], [H, H, H]], dtype=np.float32)
    circ = np.array([[0.0, 0.0, 1e-3], [0.0, 0.0, 3e-3]], dtype=np.float32)
    scalar = np.array([1e-4, 5e-4], dtype=np.float32)

    grid = physics._scatter_scalar_weighted(pos, circ, scalar, grid_min, H, nx, ny, nz)
    node = (1, 1, 1)
    w_total = 1e-3 + 3e-3
    expected = (1e-3 * 1e-4 + 3e-3 * 5e-4) / w_total
    np.testing.assert_allclose(grid[node], expected, rtol=1e-6)
    # Unpopulated nodes stay zero
    assert grid[0, 0, 0] == 0.0


def _laplacian_step_variable(physics, field, nu_eff_field, dt, h):
    """One variable-coefficient FTCS step through the production GPU kernel."""
    nx, ny, nz = field.shape[:3]
    enx, eny, enz = physics._ensure_grid_capacity(nx, ny, nz)
    assert (enx, eny, enz) == (nx, ny, nz)
    buf = np.zeros((*physics._grid_shape, 3), dtype=np.float32)
    buf[:nx, :ny, :nz, :] = field
    physics._current_grid.from_numpy(buf)
    nu_eff_buf = np.zeros(physics._grid_shape, dtype=np.float32)
    nu_eff_buf[:nx, :ny, :nz] = nu_eff_field
    physics._nu_eff_grid.from_numpy(nu_eff_buf)
    physics._other_grid.fill(0.0)
    physics._body_mask_grid.fill(0)
    physics._laplacian_step_variable_gpu_kernel(
        physics._current_grid,
        physics._other_grid,
        physics._nu_eff_grid,
        physics._body_mask_grid,
        float(dt),
        float(h),
        nx,
        ny,
        nz,
    )
    return physics._other_grid.to_numpy()[:nx, :ny, :nz, :]


def test_gbd_variable_laplacian_scales_moment_with_nu_eff(physics):
    """Per-node α (Bug A): ⟨r²⟩ grows by 6·ν_eff·dt, not 6·ν·dt.

    A uniform ν_eff = q·ν must produce q× the moment growth of the molecular
    scalar-α kernel — the mechanism by which Smagorinsky ν_t acts in GBD runs.
    """
    nu, dt = 1e-3, 0.3
    field = _gaussian_blob_grid()

    # Molecular baseline (scalar kernel, α = ν·dt/h²)
    alpha_mol = nu * dt / H**2
    out_mol = _laplacian_step(physics, field, alpha_mol)
    growth_mol = _second_moment(out_mol) - _second_moment(field)

    # Variable kernel with uniform ν_eff = 3·ν → 3× the growth
    q = 3.0
    nu_eff_field = np.full((N, N, N), q * nu, dtype=np.float32)
    out_var = _laplacian_step_variable(physics, field, nu_eff_field, dt, H)
    growth_var = _second_moment(out_var) - _second_moment(field)

    assert abs(growth_var - q * growth_mol) < 0.02 * q * growth_mol, (
        f"q={q}: variable growth {growth_var:.6e} vs {q}×molecular {q * growth_mol:.6e}"
    )


def test_gbd_variable_laplacian_matches_scalar_when_nu_eff_equals_nu(physics):
    """ν_eff = ν everywhere must reproduce the scalar-α kernel exactly."""
    nu, dt = 1e-3, 0.2
    field = _gaussian_blob_grid()

    alpha_mol = nu * dt / H**2
    out_scalar = _laplacian_step(physics, field, alpha_mol)

    nu_eff_field = np.full((N, N, N), nu, dtype=np.float32)
    out_variable = _laplacian_step_variable(physics, field, nu_eff_field, dt, H)

    np.testing.assert_allclose(out_variable, out_scalar, atol=1e-6)


def test_gbd_variable_laplacian_conerves_total_circulation(physics):
    """Variable-coefficient Laplacian with Neumann BC still conserves Γ."""
    nu, dt = 1e-3, 0.15
    field = _gaussian_blob_grid()
    nu_eff_field = np.full((N, N, N), 3.0 * nu, dtype=np.float32)
    out = _laplacian_step_variable(physics, field, nu_eff_field, dt, H)
    g0 = field[..., 2].astype(np.float64).sum()
    g1 = out[..., 2].astype(np.float64).sum()
    assert abs(g1 - g0) < 1e-4 * abs(g0), f"Γ drift {abs(g1 - g0) / g0:.2e}"
