from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from source.coupler.core.helpers.interior_bs import (
    gaussian_bs_velocity,
    gaussian_bs_velocity_treecode,
)

try:
    from numba import njit, prange

    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

U_INF = 1.0
D = 1.0
EPS = 0.026
CM = 1.0 / 2.54


if _HAVE_NUMBA:

    @njit(parallel=True, fastmath=True, cache=True)
    def _gaussian_vorticity(targets, pos, circ, rad):
        n_t = targets.shape[0]
        n_s = pos.shape[0]
        out = np.zeros((n_t, 3))
        pi15 = np.pi**1.5
        for i in prange(n_t):
            tx, ty, tz = targets[i, 0], targets[i, 1], targets[i, 2]
            ax = ay = az = 0.0
            for j in range(n_s):
                dx = tx - pos[j, 0]
                dy = ty - pos[j, 1]
                dz = tz - pos[j, 2]
                s = rad[j]
                r2 = (dx * dx + dy * dy + dz * dz) / (s * s)
                if r2 < 25.0:
                    w = np.exp(-r2) / (pi15 * s * s * s)
                    ax += w * circ[j, 0]
                    ay += w * circ[j, 1]
                    az += w * circ[j, 2]
            out[i, 0] = ax
            out[i, 1] = ay
            out[i, 2] = az
        return out


def vpm_velocity(particles, pts):
    targets = np.asarray(pts, np.float64)
    position = np.asarray(particles["position"], np.float64)
    circulation = np.asarray(particles["circulation"], np.float64)
    radius = np.asarray(particles["radius"], np.float64)
    evaluator = gaussian_bs_velocity_treecode if len(position) > 20_000 else gaussian_bs_velocity
    kwargs = {"theta": 0.3} if evaluator is gaussian_bs_velocity_treecode else {}
    u = evaluator(targets, position, circulation, radius, **kwargs)
    return u + np.array([U_INF, 0.0, 0.0])[None, :]


def vpm_vorticity(particles, pts):
    if not _HAVE_NUMBA:
        return np.full((len(pts), 3), np.nan)
    pos = np.asarray(particles["position"], np.float64)
    if pos.shape[0] == 0:
        return np.zeros((len(pts), 3))
    circ = np.asarray(particles["circulation"], np.float64)
    rad = np.asarray(particles["radius"], np.float64)
    zs = np.asarray(pts)[:, 2]
    slab = np.abs(pos[:, 2] - zs.mean()) < 5.0 * rad.max() + 0.1
    return _gaussian_vorticity(
        np.ascontiguousarray(pts, np.float64), pos[slab], circ[slab], rad[slab]
    )


def inside_box(points, box):
    return (
        (points[:, 0] >= box["xmin"])
        & (points[:, 0] <= box["xmax"])
        & (points[:, 1] >= box["ymin"])
        & (points[:, 1] <= box["ymax"])
        & (points[:, 2] >= box["zmin"])
        & (points[:, 2] <= box["zmax"])
    )


def hybrid_velocity(hyb_vtu, particles, pts, box, sample_vtu):
    pts = np.asarray(pts, dtype=float)
    inside = inside_box(pts, box)
    result = np.full((len(pts), 3), np.nan)
    if inside.any():
        result[inside] = sample_vtu(hyb_vtu, pts[inside])["U"]
    if (~inside).any():
        result[~inside] = vpm_velocity(particles, pts[~inside])
    return result, inside


def body_mask(X, Y, margin=0.05):
    return (np.abs(X) <= 0.5 + margin) & (np.abs(Y) <= 0.5 + margin)


def _add_body(ax, colors):
    ax.add_patch(
        plt.Rectangle(
            (-0.5, -0.5),
            D,
            D,
            fc=colors["background_light"],
            ec=colors.get("DarkText", "#333333"),
            lw=0.3,
            zorder=5,
        )
    )
