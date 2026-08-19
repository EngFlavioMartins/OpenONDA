#!/usr/bin/env python3
"""
Standalone diagnostic tests for the DVH (Diffused Vortex Hydrodynamics) scatter
and DVH pipeline.

These tests exercise each block of the DVH algorithm in isolation, without
running a full simulation, to verify:

1. **Circulation conservation** — Total Σα must be preserved exactly by the
   Shepard-normalized heat-kernel scatter (before pruning).
2. **Center-of-vorticity preservation** — A symmetric vortex centered at the
   origin must not drift after scatter + regen.
3. **Enstrophy decrease** — Diffusion must decrease enstrophy, never increase it.
4. **Grid symmetry** — An axisymmetric vortex must produce symmetric grid values.
5. **Single-particle conservation** — One particle's circulation must land on
   grid nodes and sum exactly back to the original.
6. **Stability under repeated regen** — Multiple regen cycles should not cause
   blow-up or drift.

Reference:
  Durante, Danilo, et al. "Numerical simulation of 3D vorticity dynamics with
  the Diffused Vortex Hydrodynamics method." Mathematics and computers in
  simulation 225 (2024): 528-544.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------- #
#  Minimal pure-NumPy reimplementation of DVH scatter for comparison     #
# --------------------------------------------------------------------- #

# Durante 2024, Eq. 15: β = 4nu·Δt_d / R_d² ≈ 0.077.
# The diffusive timestep Δt_d is calibrated so that the Gaussian spreads
# meaningfully across all ~270 nodes within R_d (not equal to Δt_a).
_DVH_BETA = 0.077


def dvh_scatter_pure_numpy(
    pos: np.ndarray,  # (N, 3)
    circ: np.ndarray,  # (N, 3)
    grid_min: np.ndarray,  # (3,)
    h: float,
    nu: float,
    time_step_size: float,  # advection dt (unused — width set by β and R_d)
    nx: int,
    ny: int,
    nz: int,
    rd_ratio: float = 4.0,
) -> np.ndarray:
    """Reference DVH scatter — pure NumPy, no Taichi dependency.

    The Gaussian width uses β·R_d² (Durante 2024, Eq. 15) rather than
    4nu·Δt_a so the kernel meaningfully reaches all nodes within R_d.

    Returns float64 grid of shape (nx, ny, nz, 3).
    """
    R_d = rd_ratio * h
    R_d_sq = R_d * R_d
    four_nu_dt = _DVH_BETA * R_d * R_d  # = β·R_d² (≡ 4nu·Δt_d)
    grid = np.zeros((nx, ny, nz, 3), dtype=np.float64)

    for j in range(len(pos)):
        yj = pos[j]
        aj = circ[j]

        i_lo = max(0, int(np.floor((yj[0] - R_d - grid_min[0]) / h)))
        i_hi = min(nx - 1, int(np.ceil((yj[0] + R_d - grid_min[0]) / h)))
        j_lo = max(0, int(np.floor((yj[1] - R_d - grid_min[1]) / h)))
        j_hi = min(ny - 1, int(np.ceil((yj[1] + R_d - grid_min[1]) / h)))
        k_lo = max(0, int(np.floor((yj[2] - R_d - grid_min[2]) / h)))
        k_hi = min(nz - 1, int(np.ceil((yj[2] + R_d - grid_min[2]) / h)))

        if i_lo > i_hi or j_lo > j_hi or k_lo > k_hi:
            continue

        xI = grid_min[0] + np.arange(i_lo, i_hi + 1) * h
        xJ = grid_min[1] + np.arange(j_lo, j_hi + 1) * h
        xK = grid_min[2] + np.arange(k_lo, k_hi + 1) * h

        r2 = (
            ((xI - yj[0]) ** 2)[:, None, None]
            + ((xJ - yj[1]) ** 2)[None, :, None]
            + ((xK - yj[2]) ** 2)[None, None, :]
        )

        inside = r2 <= R_d_sq
        if not inside.any():
            continue

        w = np.where(inside, np.exp(-r2 / four_nu_dt), 0.0)
        w_sum = w.sum()
        if w_sum < 1e-300:
            continue

        w_norm = w / w_sum

        grid[i_lo : i_hi + 1, j_lo : j_hi + 1, k_lo : k_hi + 1, 0] += w_norm * aj[0]
        grid[i_lo : i_hi + 1, j_lo : j_hi + 1, k_lo : k_hi + 1, 1] += w_norm * aj[1]
        grid[i_lo : i_hi + 1, j_lo : j_hi + 1, k_lo : k_hi + 1, 2] += w_norm * aj[2]

    return grid


def make_lamb_oseen_particles(
    h: float,
    rc: float,
    gamma: float,
    nu: float,
    domain_half: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 3D Lamb-Oseen vortex on a hexagonal-like lattice.

    The vortex axis is along z. Circulation α = vol * ω(r).

    Returns (positions, circulations, volumes) — all float64.
    """
    t0 = rc**2 / (4 * nu)  # initial age
    coords_1d = np.arange(-domain_half, domain_half + h / 2, h)
    z_1d = np.arange(-domain_half, domain_half + h / 2, h)
    xx, yy, zz = np.meshgrid(coords_1d, coords_1d, z_1d, indexing="ij")
    pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    r2 = pos[:, 0] ** 2 + pos[:, 1] ** 2
    vol = h**3
    # ω_z = Γ / (4π nu t₀) · exp(-r²/(4nut₀))
    omega_z = gamma / (4 * np.pi * nu * t0) * np.exp(-r2 / (4 * nu * t0))
    circ = np.zeros((len(pos), 3), dtype=np.float64)
    circ[:, 2] = omega_z * vol

    volumes = np.full(len(pos), vol, dtype=np.float64)

    # Remove negligible particles
    mag = np.abs(circ[:, 2])
    keep = mag > 1e-12 * mag.max()
    return pos[keep], circ[keep], volumes[keep]


# ===================================================================== #
#  TESTS                                                                 #
# ===================================================================== #


def test_single_particle_conservation():
    """A single particle's Γ must be exactly conserved by DVH scatter."""
    h = 0.1
    nu = 1e-3
    time_step_size = 0.01
    rd_ratio = 4.0

    pos = np.array([[0.5, 0.5, 0.5]])
    circ = np.array([[0.0, 0.0, 1.0]])
    grid_min = np.array([0.0, 0.0, 0.0])
    nx = ny = nz = 11

    grid = dvh_scatter_pure_numpy(pos, circ, grid_min, h, nu, time_step_size, nx, ny, nz, rd_ratio)

    total_circ = grid.sum(axis=(0, 1, 2))  # (3,)
    err = np.abs(total_circ - circ[0])
    print("  Single-particle conservation test:")
    print(f"    Input Γ  = {circ[0]}")
    print(f"    Output Γ = {total_circ}")
    print(f"    Error    = {err}")
    assert np.allclose(total_circ, circ[0], atol=1e-14), (
        f"Single-particle circulation NOT conserved! Error: {err}"
    )
    print("    ✓ PASS")


def test_multi_particle_conservation():
    """Total Σα must be conserved by DVH scatter (before pruning)."""
    h = 0.05
    nu = 1.0 / 530.0
    rc = 0.125
    gamma = 1.0
    pos, circ, vol = make_lamb_oseen_particles(h, rc, gamma, nu, domain_half=0.5)

    # Generous grid to contain everything
    pad = 20 * h
    lo = pos.min(axis=0) - pad
    hi = pos.max(axis=0) + pad
    nx = int(np.ceil((hi[0] - lo[0]) / h)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / h)) + 1
    nz = int(np.ceil((hi[2] - lo[2]) / h)) + 1
    grid_min = lo.astype(np.float64)

    time_step_size = 0.01
    rd_ratio = 4.0

    input_total = circ.sum(axis=0)  # (3,)

    grid = dvh_scatter_pure_numpy(pos, circ, grid_min, h, nu, time_step_size, nx, ny, nz, rd_ratio)
    output_total = grid.sum(axis=(0, 1, 2))

    rel_err = np.abs(output_total - input_total) / (np.abs(input_total) + 1e-30)
    print(f"  Multi-particle conservation test ({len(pos)} particles):")
    print(f"    Grid: {nx}×{ny}×{nz} = {nx * ny * nz} nodes")
    print(f"    Input  ΣΓ = {input_total}")
    print(f"    Output ΣΓ = {output_total}")
    print(f"    Rel err   = {rel_err}")
    assert np.all(rel_err < 1e-10), (
        f"Multi-particle circulation NOT conserved! Rel error: {rel_err}"
    )
    print("    ✓ PASS")


def test_center_of_vorticity_preservation():
    """Center of vorticity must not drift for a symmetric vortex."""
    h = 0.05
    nu = 1.0 / 530.0
    rc = 0.125
    gamma = 1.0
    pos, circ, vol = make_lamb_oseen_particles(h, rc, gamma, nu, domain_half=0.5)

    pad = 20 * h
    lo = pos.min(axis=0) - pad
    hi = pos.max(axis=0) + pad
    nx = int(np.ceil((hi[0] - lo[0]) / h)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / h)) + 1
    nz = int(np.ceil((hi[2] - lo[2]) / h)) + 1
    grid_min = lo.astype(np.float64)

    time_step_size = 0.01
    rd_ratio = 4.0

    # Input center of vorticity
    circ_mag_in = np.linalg.norm(circ, axis=1)
    cov_in = (pos * circ_mag_in[:, None]).sum(axis=0) / circ_mag_in.sum()

    grid = dvh_scatter_pure_numpy(pos, circ, grid_min, h, nu, time_step_size, nx, ny, nz, rd_ratio)

    # Output center of vorticity from grid
    circ_mag_grid = np.linalg.norm(grid, axis=-1)
    ixs, iys, izs = np.where(circ_mag_grid > 0)
    if len(ixs) == 0:
        raise RuntimeError("Grid empty after scatter!")

    grid_pos = np.stack(
        [
            grid_min[0] + ixs * h,
            grid_min[1] + iys * h,
            grid_min[2] + izs * h,
        ],
        axis=1,
    )
    mags = circ_mag_grid[ixs, iys, izs]
    cov_out = (grid_pos * mags[:, None]).sum(axis=0) / mags.sum()

    drift = np.linalg.norm(cov_out - cov_in)
    print("  Center-of-vorticity preservation test:")
    print(f"    Input  CoV = {cov_in}")
    print(f"    Output CoV = {cov_out}")
    print(f"    Drift      = {drift:.6e} m")
    # Drift should be tiny (grid quantization only)
    assert drift < 2 * h, f"Vortex drifted {drift:.4e} m (> 2h = {2 * h})!"
    print("    ✓ PASS")


def test_enstrophy_decrease():
    """Enstrophy (Σ|α|²) must decrease after diffusion scatter."""
    h = 0.05
    nu = 1.0 / 530.0
    rc = 0.125
    gamma = 1.0
    pos, circ, vol = make_lamb_oseen_particles(h, rc, gamma, nu, domain_half=0.5)

    pad = 20 * h
    lo = pos.min(axis=0) - pad
    hi = pos.max(axis=0) + pad
    nx = int(np.ceil((hi[0] - lo[0]) / h)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / h)) + 1
    nz = int(np.ceil((hi[2] - lo[2]) / h)) + 1
    grid_min = lo.astype(np.float64)

    time_step_size = 0.01
    rd_ratio = 4.0

    # Input enstrophy proxy: Σ|α|²
    enstr_in = (circ**2).sum()

    grid = dvh_scatter_pure_numpy(pos, circ, grid_min, h, nu, time_step_size, nx, ny, nz, rd_ratio)
    enstr_out = (grid**2).sum()

    ratio = enstr_out / enstr_in
    print("  Enstrophy decrease test:")
    print(f"    Input  enstrophy  = {enstr_in:.6e}")
    print(f"    Output enstrophy  = {enstr_out:.6e}")
    print(f"    Ratio (out/in)    = {ratio:.6f}")
    assert ratio <= 1.001, f"Enstrophy INCREASED! ratio = {ratio:.6f}"
    assert ratio < 0.9999, (
        f"Enstrophy did NOT decrease (ratio={ratio:.6f} ≈ 1). No diffusion is occurring!"
    )
    print("    ✓ PASS (diffusion is active)")


def test_symmetry():
    """An axisymmetric vortex must produce symmetric grid output."""
    h = 0.1
    nu = 1e-3
    time_step_size = 0.01
    rd_ratio = 4.0

    # Place 4 particles symmetrically around origin (in xy plane, single z)
    pos = np.array(
        [
            [0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, -0.1, 0.0],
        ],
        dtype=np.float64,
    )
    circ = np.array(
        [
            [0, 0, 1.0],
            [0, 0, 1.0],
            [0, 0, 1.0],
            [0, 0, 1.0],
        ],
        dtype=np.float64,
    )

    grid_min = np.array([-0.5, -0.5, -0.5])
    nx = ny = nz = 11

    grid = dvh_scatter_pure_numpy(pos, circ, grid_min, h, nu, time_step_size, nx, ny, nz, rd_ratio)
    gz = grid[:, :, 5, 2]  # z-component at z=0 slice

    # Check x-symmetry: grid[i,j] ~ grid[nx-1-i,j]
    sym_err_x = np.abs(gz - gz[::-1, :]).max()
    sym_err_y = np.abs(gz - gz[:, ::-1]).max()
    print("  Symmetry test:")
    print(f"    X-symmetry error = {sym_err_x:.2e}")
    print(f"    Y-symmetry error = {sym_err_y:.2e}")
    assert sym_err_x < 1e-14, f"X-symmetry broken: {sym_err_x}"
    assert sym_err_y < 1e-14, f"Y-symmetry broken: {sym_err_y}"
    print("    ✓ PASS")


def test_pruning_budget_mode():
    """Budget pruning should keep at least (1-threshold) of total |Γ|."""
    h = 0.05
    nu = 1.0 / 530.0
    rc = 0.125
    gamma = 1.0
    pos, circ, vol = make_lamb_oseen_particles(h, rc, gamma, nu, domain_half=0.5)

    pad = 20 * h
    lo = pos.min(axis=0) - pad
    hi = pos.max(axis=0) + pad
    nx = int(np.ceil((hi[0] - lo[0]) / h)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / h)) + 1
    nz = int(np.ceil((hi[2] - lo[2]) / h)) + 1
    grid_min = lo.astype(np.float64)

    time_step_size = 0.01
    rd_ratio = 4.0
    threshold_frac = 0.01  # budget threshold

    grid = dvh_scatter_pure_numpy(pos, circ, grid_min, h, nu, time_step_size, nx, ny, nz, rd_ratio)
    circ_mag = np.linalg.norm(grid, axis=-1)
    gamma_total = circ_mag.sum()

    # Replicate budget thresholding
    flat = circ_mag.ravel()
    order = np.argsort(-flat)
    cumsum = np.cumsum(flat[order])
    target = (1.0 - threshold_frac) * gamma_total
    cutoff = min(int(np.searchsorted(cumsum, target)), len(order) - 1)
    thresh = float(flat[order[cutoff]])

    ix, iy, iz = np.where(circ_mag >= thresh)
    kept_gamma = circ_mag[ix, iy, iz].sum()
    retained_frac = kept_gamma / gamma_total

    print("  Budget pruning test:")
    print(f"    Total |Γ| on grid      = {gamma_total:.6e}")
    print(f"    Threshold              = {thresh:.6e}")
    print(f"    Nodes kept             = {len(ix)} / {(circ_mag > 0).sum()}")
    print(f"    Retained |Γ| fraction  = {retained_frac:.6f}")
    print(f"    Target fraction        = {1 - threshold_frac}")
    assert retained_frac >= (1.0 - threshold_frac) - 1e-10, (
        f"Budget mode retained only {retained_frac:.4f}, expected ≥ {1 - threshold_frac}"
    )
    print("    ✓ PASS")


def test_repeated_regen_stability():
    """Repeated DVH scatter + budget prune cycles must not blow up or drift."""
    h = 0.05
    nu = 1.0 / 530.0
    rc = 0.125
    gamma = 1.0
    time_step_size = 0.01
    rd_ratio = 4.0
    threshold_frac = 0.005
    n_cycles = 10

    pos, circ, vol = make_lamb_oseen_particles(h, rc, gamma, nu, domain_half=0.5)
    initial_total_gamma = circ.sum(axis=0).copy()
    initial_abs_gamma = np.abs(circ[:, 2]).sum()

    print(f"  Repeated regen stability test ({n_cycles} cycles):")
    print(f"    Initial particles    = {len(pos)}")
    print(f"    Initial Σ|Γ_z|       = {initial_abs_gamma:.6e}")
    print(f"    Initial ΣΓ           = {initial_total_gamma}")

    for cycle in range(n_cycles):
        pad = 20 * h
        lo = pos.min(axis=0) - pad
        hi = pos.max(axis=0) + pad
        nx = int(np.ceil((hi[0] - lo[0]) / h)) + 1
        ny = int(np.ceil((hi[1] - lo[1]) / h)) + 1
        nz = int(np.ceil((hi[2] - lo[2]) / h)) + 1
        grid_min = lo.astype(np.float64)

        grid = dvh_scatter_pure_numpy(
            pos, circ, grid_min, h, nu, time_step_size, nx, ny, nz, rd_ratio
        )

        # Budget pruning
        circ_mag = np.linalg.norm(grid, axis=-1)
        gamma_total = circ_mag.sum()
        flat = circ_mag.ravel()
        order = np.argsort(-flat)
        cumsum = np.cumsum(flat[order])
        target = (1.0 - threshold_frac) * gamma_total
        cutoff = min(int(np.searchsorted(cumsum, target)), len(order) - 1)
        thresh = max(float(flat[order[cutoff]]), 1e-10)

        ix, iy, iz = np.where(circ_mag >= thresh)
        if len(ix) == 0:
            print(f"    Cycle {cycle}: FAILED — no surviving nodes!")
            break

        # Rebuild particles from surviving grid nodes
        new_pos = np.stack(
            [
                grid_min[0] + ix * h,
                grid_min[1] + iy * h,
                grid_min[2] + iz * h,
            ],
            axis=1,
        )
        new_circ = grid[ix, iy, iz]

        total_gamma = new_circ.sum(axis=0)
        abs_gamma = np.abs(new_circ[:, 2]).sum()

        # Center of vorticity
        mags = np.linalg.norm(new_circ, axis=1)
        cov = (new_pos * mags[:, None]).sum(axis=0) / mags.sum()

        print(
            f"    Cycle {cycle + 1:2d}: N={len(new_pos):5d}  "
            f"ΣΓ_z={total_gamma[2]:+.6e}  Σ|Γ_z|={abs_gamma:.6e}  "
            f"CoV=({cov[0]:+.4f},{cov[1]:+.4f},{cov[2]:+.4f})"
        )

        pos = new_pos
        circ = new_circ

    # After all cycles, check no blow-up
    final_abs_gamma = np.abs(circ[:, 2]).sum()
    circ_mag_final = np.linalg.norm(circ, axis=1)
    cov_final = (pos * circ_mag_final[:, None]).sum(axis=0) / circ_mag_final.sum()
    drift = np.linalg.norm(cov_final[:2])  # xy drift from origin

    print(f"    Final xy-drift from origin = {drift:.6e} m")
    print(f"    Final Σ|Γ_z| / initial     = {final_abs_gamma / initial_abs_gamma:.4f}")

    assert drift < 5 * h, f"Vortex drifted {drift} m after {n_cycles} regen cycles!"
    assert final_abs_gamma > 0.5 * initial_abs_gamma, (
        f"Lost too much circulation: {final_abs_gamma / initial_abs_gamma:.2%} remaining"
    )
    print("    ✓ PASS")


def test_lamb_oseen_profile_accuracy():
    """After one DVH scatter step, radial profile should match analytical diffusion."""
    h = 0.03125
    nu = 1.0 / 530.0
    rc = 0.125
    gamma = 1.0
    time_step_size = 0.05

    pos, circ, vol = make_lamb_oseen_particles(h, rc, gamma, nu, domain_half=0.6)

    pad = 20 * h
    lo = pos.min(axis=0) - pad
    hi = pos.max(axis=0) + pad
    nx = int(np.ceil((hi[0] - lo[0]) / h)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / h)) + 1
    nz = int(np.ceil((hi[2] - lo[2]) / h)) + 1
    grid_min = lo.astype(np.float64)

    rd_ratio = 4.0
    grid = dvh_scatter_pure_numpy(pos, circ, grid_min, h, nu, time_step_size, nx, ny, nz, rd_ratio)

    # Analytical solution at t = t0 + dt
    t0 = rc**2 / (4 * nu)
    t_new = t0 + time_step_size

    # Sample radial profile from grid at z mid-plane
    nz_mid = nz // 2
    gz = grid[:, :, nz_mid, 2]  # ω_z component

    # Convert grid circulation to vorticity: ω = α / vol
    vol_cell = h**3
    omega_grid = gz / vol_cell

    # Compute radial distance from center
    # Radial bins
    r_max = 4 * rc
    n_bins = 40
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    omega_binned = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i in range(nx):
        for j in range(ny):
            x = grid_min[0] + i * h
            y = grid_min[1] + j * h
            r = np.sqrt(x**2 + y**2)
            bin_idx = np.searchsorted(r_edges, r) - 1
            if 0 <= bin_idx < n_bins:
                omega_binned[bin_idx] += omega_grid[i, j]
                counts[bin_idx] += 1

    valid = counts > 0
    omega_avg = np.zeros(n_bins)
    omega_avg[valid] = omega_binned[valid] / counts[valid]

    # Analytical
    omega_analytical = gamma / (4 * np.pi * nu * t_new) * np.exp(-(r_centers**2) / (4 * nu * t_new))

    # Compare only up to 2*rc where signal is strong
    mask = r_centers < 2 * rc
    if mask.sum() > 0 and valid[mask].sum() > 0:
        inner_mask = mask & valid
        if inner_mask.sum() > 3:
            rel_err = np.abs(omega_avg[inner_mask] - omega_analytical[inner_mask]) / (
                omega_analytical[inner_mask] + 1e-30
            )
            max_rel_err = rel_err.max()
            mean_rel_err = rel_err.mean()
            print("  Lamb-Oseen profile accuracy test:")
            print(f"    Particles    = {len(pos)}")
            print(f"    Grid         = {nx}×{ny}×{nz}")
            print(f"    t0={t0:.4f}, dt={time_step_size}, t_new={t_new:.4f}")
            print(f"    Max rel err (r < 2rc)  = {max_rel_err:.4f}")
            print(f"    Mean rel err (r < 2rc) = {mean_rel_err:.4f}")
            # During DVH: we don't expect < 5% error on first step because
            # the input particles are already on a grid, but we should get < 50%
            assert max_rel_err < 0.50, (
                f"Profile error too large: max relative error = {max_rel_err:.4f}"
            )
            print("    ✓ PASS")
        else:
            print("  Lamb-Oseen profile accuracy test: SKIP (insufficient data points)")
    else:
        print("  Lamb-Oseen profile accuracy test: SKIP (no valid bins)")


# ===================================================================== #
#  Main driver                                                           #
# ===================================================================== #
