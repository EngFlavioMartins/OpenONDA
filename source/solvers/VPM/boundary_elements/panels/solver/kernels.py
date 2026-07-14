"""
Panel geometry-update kernels wrappers.

This module provides a small API layer used by panel kinematics to apply
translation and rotation updates on a per-body panel range.
"""

from __future__ import annotations

import numpy as np
import taichi as ti


def panel_update_translation_kernel(
    lattice, body_range, displacement: np.ndarray, velocity: np.ndarray
) -> None:
    """Apply a rigid translation to a contiguous range of panels.

    Updates both panel geometry (vertex positions) and panel velocity using
    the panel solver's Taichi kernels.

    Parameters
    ----------
    lattice
        Panel lattice instance with ``_update_geometry_kernel`` and
        ``_update_velocity_kernel`` methods.
    body_range : tuple[int, int]
        ``(start_idx, end_idx)`` defining the panel range to update.
    displacement : np.ndarray
        Translation vector with shape ``(3,)``.
    velocity : np.ndarray
        Translational velocity vector with shape ``(3,)``.

    Examples
    --------
    >>> panel_update_translation_kernel(lattice, (0, 100), disp, vel)
    """
    start_idx, end_idx = body_range
    count = end_idx - start_idx
    disp = np.asarray(displacement, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    lattice._update_geometry_kernel(
        start_idx,
        count,
        ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ti.Vector(disp.tolist()),
    )
    lattice._update_velocity_kernel(
        start_idx,
        count,
        ti.Vector(vel.tolist()),
        ti.Vector([0.0, 0.0, 0.0]),
        ti.Vector(disp.tolist()),
    )


def panel_update_rotation_kernel(
    lattice, body_range, rotation: np.ndarray, omega: np.ndarray, center: np.ndarray
) -> None:
    """Apply a rigid rotation to a contiguous range of panels.

    Updates both panel geometry (vertex positions) and panel velocity using
    the panel solver's Taichi kernels. The rotation is applied about the
    given ``center`` point.

    Parameters
    ----------
    lattice
        Panel lattice instance with ``_update_geometry_kernel`` and
        ``_update_velocity_kernel`` methods.
    body_range : tuple[int, int]
        ``(start_idx, end_idx)`` defining the panel range to update.
    rotation : np.ndarray
        3x3 rotation matrix.
    omega : np.ndarray
        Angular velocity vector with shape ``(3,)``.
    center : np.ndarray
        Center of rotation with shape ``(3,)``.

    Examples
    --------
    >>> panel_update_rotation_kernel(lattice, (0, 100), R, omega, center)
    """
    start_idx, end_idx = body_range
    count = end_idx - start_idx
    R = np.asarray(rotation, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    w = np.asarray(omega, dtype=np.float64)
    T = c - R @ c
    lattice._update_geometry_kernel(start_idx, count, ti.Matrix(R.tolist()), ti.Vector(T.tolist()))
    lattice._update_velocity_kernel(
        start_idx,
        count,
        ti.Vector([0.0, 0.0, 0.0]),
        ti.Vector(w.tolist()),
        ti.Vector(c.tolist()),
    )
