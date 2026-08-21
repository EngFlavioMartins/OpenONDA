"""Conservative M4-prime particle-to-lattice remeshing."""

from __future__ import annotations

from numba import njit
import numpy as np


def m4p(q: np.ndarray | float) -> np.ndarray:
    """M4' interpolating kernel (Monaghan 1985).

    Piece-wise cubic, C¹, support [-2, 2]:

        0 ≤ |q| < 1:  1 - 5/2 q² + 3/2 |q|³
        1 ≤ |q| < 2:  1/2 (1 - |q|)(2 - |q|)²
        |q| ≥ 2:      0

    Key properties:
    - M4'(0) = 1  — interpolating at the origin
    - M4'(n) = 0  — zero at all non-zero integers (no cross-node leakage)
    - Σ_n M4'(x−n) = 1  — partition of unity (vortex-strength conservation)
    - Σ_n (x−n) M4'(x−n) = 0  — 1st moment (linear impulse conservation)

    Parameters
    ----------
    q : normalized distance |x - x_node| / particle_spacing  (scalar or array)

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


def grid_positions(
    origin: np.ndarray, particle_spacing: float, shape: tuple[int, int, int]
) -> np.ndarray:
    """Return regular-lattice node coordinates in remesh storage order."""
    gx, gy, gz = np.meshgrid(
        *[origin[d] + particle_spacing * np.arange(shape[d]) for d in range(3)],
        indexing="ij",
    )
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


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


def _scatter_m4p(
    rel: np.ndarray,
    base: np.ndarray,
    circ_f: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    return _scatter_m4p_numba_impl(rel, base, circ_f, shape[0], shape[1], shape[2])


def remesh_to_grid(
    pos: np.ndarray,
    vortex_strength: np.ndarray,
    origin: np.ndarray,
    particle_spacing: float,
    shape: tuple[int, int, int],
    grid_positions_cache: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product M4' P2M scatter onto a regular lattice.

    Each particle scatters its vortex strength onto the surrounding 4×4×4 = 64
    grid nodes using the M4' kernel.  Because M4' satisfies partition of unity
    the total vortex strength is *exactly* conserved.

    Parameters
    ----------
    pos    : (N, 3) particle positions
    vortex_strength : (N, 3) particle strengths α_i
    origin : (3,)   position of grid node [0, 0, 0]
    particle_spacing      : scalar uniform grid spacing
    shape  : (Nx, Ny, Nz) grid dimensions (number of nodes per axis)

    Returns
    -------
    grid_pos  : (Nx*Ny*Nz, 3)  node positions
    grid_vortex_strength : (Nx*Ny*Nz, 3)  accumulated Σ W_ip α_p
    """
    grid_pos = (
        grid_positions(np.asarray(origin, dtype=float), particle_spacing, shape)
        if grid_positions_cache is None
        else np.asarray(grid_positions_cache, dtype=np.float64).reshape(-1, 3)
    )
    expected_nodes = int(np.prod(shape))
    if len(grid_pos) != expected_nodes:
        raise ValueError(
            f"grid_positions_cache has {len(grid_pos)} nodes, expected {expected_nodes}"
        )
    if len(pos) == 0:
        return grid_pos, np.zeros((len(grid_pos), 3))

    rel = (np.asarray(pos, dtype=float) - np.asarray(origin, dtype=float)) / float(particle_spacing)
    circ_f = np.asarray(vortex_strength, dtype=float)

    nearest = np.rint(rel).astype(np.int64)
    aligned = np.max(np.abs(rel - nearest), axis=1) <= 1.0e-5
    aligned &= np.all((nearest >= 0) & (nearest < np.asarray(shape)), axis=1)
    G = np.zeros((*shape, 3), dtype=np.float64)
    if aligned.any():
        index = nearest[aligned]
        np.add.at(G, (index[:, 0], index[:, 1], index[:, 2]), circ_f[aligned])
    if (~aligned).any():
        rel_free = rel[~aligned]
        base_free = np.floor(rel_free).astype(int) - 1
        G += _scatter_m4p(rel_free, base_free, circ_f[~aligned], shape)
    return grid_pos, G.reshape(-1, 3)
