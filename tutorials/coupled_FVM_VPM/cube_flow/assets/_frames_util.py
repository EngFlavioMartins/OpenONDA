from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from source.solvers.VPM.config.backend import initialize_taichi_backend

initialize_taichi_backend("CPU", precision="f32")

try:
    from numba import njit, prange

    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

FREESTREAM_SPEED = 1.0
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
    from source.solvers.VPM.acceleration.treecode_gpu import TaichiTreecode

    targets = np.asarray(pts, np.float32)
    position = np.asarray(particles["position"], np.float32)
    circulation = np.asarray(particles["circulation"], np.float32)
    radius = np.asarray(particles["radius"], np.float32)
    if not len(position):
        return np.tile([FREESTREAM_SPEED, 0.0, 0.0], (len(targets), 1))
    capacity = max(len(position), len(targets))
    tree = TaichiTreecode(
        max_particles=capacity,
        max_nodes=2 * len(position),
        theta=0.3,
        kernel_type="GAUSSIAN",
        multipole_order=2,
    )
    tree.build(position, circulation, radius)
    return tree.compute_target_velocities(
        targets,
        freestream_velocity=np.array([FREESTREAM_SPEED, 0.0, 0.0], dtype=np.float32),
    )


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
