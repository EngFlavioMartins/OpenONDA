"""
WS1: Scheme-independent M4' remeshing for the FVM-VPM coupler.

Implements the M4' (Monaghan 1985; Cottet–Koumoutsakos Λ₂,₃) interpolating
kernel and a P2M (particle-to-mesh) scatter onto a regular lattice.

Why M4' and not the cubic-spline M4?
-------------------------------------
The existing ``kernels.m4_kernel`` is the *smoothing* cubic spline
(W(1) = 0.25 ≠ 0).  Applied to remeshing it leaks circulation to neighbouring
nodes every pass, acting as an extra diffusion step.  M4' is *interpolating*:
W(0)=1, W(n)=0 for all non-zero integers, so the scatter is exact at the nodes
and conserves the 0th (circulation) and 1st (linear impulse) moments by the
partition-of-unity property.

Reference: Monaghan (1985); Cottet & Koumoutsakos (2000) §7.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
"""

from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only on minimal installs
    njit = None


def m4p(q: np.ndarray | float) -> np.ndarray:
    """M4' interpolating kernel (Monaghan 1985).

    Piece-wise cubic, C¹, support [-2, 2]:

        0 ≤ |q| < 1:  1 - 5/2 q² + 3/2 |q|³
        1 ≤ |q| < 2:  1/2 (1 - |q|)(2 - |q|)²
        |q| ≥ 2:      0

    Key properties:
    - M4'(0) = 1  — interpolating at the origin
    - M4'(n) = 0  — zero at all non-zero integers (no cross-node leakage)
    - Σ_n M4'(x−n) = 1  — partition of unity (circulation conservation)
    - Σ_n (x−n) M4'(x−n) = 0  — 1st moment (linear impulse conservation)

    Parameters
    ----------
    q : normalized distance |x - x_node| / h  (scalar or array)

    Returns
    -------
    w : kernel weights, same shape as q
    """
    q = np.abs(np.asarray(q, dtype=float))
    w = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    w[m1] = 1.0 - 2.5 * q[m1] ** 2 + 1.5 * q[m1] ** 3
    w[m2] = 0.5 * (1.0 - q[m2]) * (2.0 - q[m2]) ** 2
    return w


def _grid_positions(origin: np.ndarray, h: float, shape: tuple[int, int, int]) -> np.ndarray:
    gx, gy, gz = np.meshgrid(
        *[origin[d] + h * np.arange(shape[d]) for d in range(3)],
        indexing="ij",
    )
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _scatter_m4p_numpy(
    rel: np.ndarray,
    base: np.ndarray,
    circ_f: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    G = np.zeros((*shape, 3))

    for ox in range(4):
        wx = m4p(rel[:, 0] - (base[:, 0] + ox))
        ix = base[:, 0] + ox
        for oy in range(4):
            wy = m4p(rel[:, 1] - (base[:, 1] + oy))
            iy = base[:, 1] + oy
            for oz in range(4):
                wz = m4p(rel[:, 2] - (base[:, 2] + oz))
                iz = base[:, 2] + oz
                w = (wx * wy * wz)[:, None]  # (N, 1)
                ok = (
                    (ix >= 0)
                    & (ix < shape[0])
                    & (iy >= 0)
                    & (iy < shape[1])
                    & (iz >= 0)
                    & (iz < shape[2])
                )
                if ok.any():
                    np.add.at(G, (ix[ok], iy[ok], iz[ok]), (w * circ_f)[ok])

    return G


if njit is not None:

    @njit(fastmath=False)
    def _m4p_scalar(q: float) -> float:
        q = abs(q)
        if q < 1.0:
            return 1.0 - 2.5 * q * q + 1.5 * q * q * q
        if q < 2.0:
            return 0.5 * (1.0 - q) * (2.0 - q) * (2.0 - q)
        return 0.0

    @njit(fastmath=False)
    def _scatter_m4p_numba_impl(
        rel: np.ndarray,
        base: np.ndarray,
        circ_f: np.ndarray,
        nx: int,
        ny: int,
        nz: int,
    ) -> np.ndarray:
        G = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        n = rel.shape[0]
        for ox in range(4):
            for oy in range(4):
                for oz in range(4):
                    for p in range(n):
                        ix = base[p, 0] + ox
                        iy = base[p, 1] + oy
                        iz = base[p, 2] + oz
                        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                            continue
                        wx = _m4p_scalar(rel[p, 0] - ix)
                        wy = _m4p_scalar(rel[p, 1] - iy)
                        wz = _m4p_scalar(rel[p, 2] - iz)
                        w = wx * wy * wz
                        G[ix, iy, iz, 0] += w * circ_f[p, 0]
                        G[ix, iy, iz, 1] += w * circ_f[p, 1]
                        G[ix, iy, iz, 2] += w * circ_f[p, 2]
        return G

else:
    _scatter_m4p_numba_impl = None


def _scatter_m4p(
    rel: np.ndarray,
    base: np.ndarray,
    circ_f: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    if (
        _scatter_m4p_numba_impl is not None
        and os.environ.get("OPENONDA_DISABLE_NUMBA_REMESH", "0") != "1"
    ):
        return _scatter_m4p_numba_impl(rel, base, circ_f, shape[0], shape[1], shape[2])
    return _scatter_m4p_numpy(rel, base, circ_f, shape)


def remesh_to_grid(
    pos: np.ndarray,
    circ: np.ndarray,
    origin: np.ndarray,
    h: float,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product M4' P2M scatter onto a regular lattice.

    Each particle scatters its circulation onto the surrounding 4×4×4 = 64
    grid nodes using the M4' kernel.  Because M4' satisfies partition of unity
    the total circulation is *exactly* conserved.

    Parameters
    ----------
    pos    : (N, 3) particle positions
    circ   : (N, 3) particle circulations Γ_i
    origin : (3,)   position of grid node [0, 0, 0]
    h      : scalar uniform grid spacing
    shape  : (Nx, Ny, Nz) grid dimensions (number of nodes per axis)

    Returns
    -------
    grid_pos  : (Nx*Ny*Nz, 3)  node positions
    grid_circ : (Nx*Ny*Nz, 3)  accumulated circulations Σ W_ip Γ_p
    """
    if len(pos) == 0:
        grid_pos = _grid_positions(np.asarray(origin, dtype=float), h, shape)
        return grid_pos, np.zeros((len(grid_pos), 3))

    rel = (np.asarray(pos, dtype=float) - np.asarray(origin, dtype=float)) / float(h)
    # Leftmost node index in the 4-node stencil: floor(q) - 1 → nodes at offsets 0..3
    base = np.floor(rel).astype(int) - 1  # (N, 3)

    circ_f = np.asarray(circ, dtype=float)
    G = _scatter_m4p(rel, base, circ_f, shape)
    grid_pos = _grid_positions(np.asarray(origin, dtype=float), h, shape)
    return grid_pos, G.reshape(-1, 3)


def grid_vorticity(grid_circ: np.ndarray, cell_volume: float) -> np.ndarray:
    """Convert grid circulation to vorticity: ω_m = Γ_m / V_cell."""
    return grid_circ / (cell_volume + 1e-30)


def particles_from_grid(
    grid_pos: np.ndarray,
    grid_circ: np.ndarray,
    h: float,
    threshold_abs: float = 0.0,
    threshold_rel: float = 0.0,
    radius_ratio: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create new particles from grid nodes whose circulation exceeds threshold.

    Parameters
    ----------
    grid_pos      : (M, 3) grid node positions
    grid_circ     : (M, 3) accumulated circulations on grid
    h             : grid spacing (used to set particle volume and radius)
    threshold_abs : discard nodes with |Γ| < threshold_abs
    threshold_rel : discard nodes with |Γ| < threshold_rel * max|Γ|
    radius_ratio  : new_sigma = h * radius_ratio

    Returns
    -------
    new_pos    : (K, 3) positions of kept nodes
    new_circ   : (K, 3) circulations
    new_vol    : (K,) volumes h³
    new_radius : (K,) core radii h * radius_ratio
    """
    mag = np.linalg.norm(grid_circ, axis=1)
    max_mag = float(np.max(mag)) if len(mag) > 0 else 0.0
    threshold = max(threshold_abs, threshold_rel * max_mag)
    keep = mag >= threshold
    vol = h**3
    return (
        grid_pos[keep],
        grid_circ[keep],
        np.full(keep.sum(), vol),
        np.full(keep.sum(), h * radius_ratio),
    )
