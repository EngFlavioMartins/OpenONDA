"""
Data module for VPM panel solver (Lattice structure).
==================
GPU-resident data structure for 3D panel geometry, doublet_strength, and topology.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import taichi as ti

from ....config.constants import PANEL_EPSILON

if TYPE_CHECKING:
    from ..coupling.kinematics import PanelKinematics

logger = logging.getLogger("vpm")


@dataclass
class PanelBody:
    """Dataclass storing metadata for a collection of panels belonging to a single body."""

    uid: str
    start_idx: int
    count: int
    kinematics: "PanelKinematics"
    group_id: int = 0
    reference_area: float = 1.0


@ti.data_oriented
class PanelLattice:
    """
    GPU-resident data structure representing the 3D panel lattice.
    Equivalent to VLMLattice but for arbitrary 3D triangular panel meshes.
    """

    def __init__(self, max_n_panels: int = 20000, float_dtype: str = "f32"):
        # Guard: Taichi must already be initialised before creating fields,
        # otherwise ti.field() triggers an auto-init with wrong precision.
        if ti.lang.impl.get_runtime().prog is None:
            raise RuntimeError(
                "PanelLattice must be created after ti.init(). "
                "Ensure the VPM Solver (which calls ti.init) is "
                "constructed before any PanelLattice instance."
            )
        self.max_n_panels = max_n_panels
        self.float_dtype = float_dtype
        self.ti_dtype = ti.f32 if float_dtype == "f32" else ti.f64
        self.n_panels = 0

        # Metadata
        self.bodies: list[PanelBody] = []

        self._init_fields()

    def _init_fields(self):
        dtype = self.ti_dtype
        N = self.max_n_panels

        # Geometry
        self.vertex_position = ti.Vector.field(3, dtype=dtype, shape=(N, 3))
        self.panel_centre = ti.Vector.field(3, dtype=dtype, shape=N)
        self.normal = ti.Vector.field(3, dtype=dtype, shape=N)
        self.area = ti.field(dtype=dtype, shape=N)
        self.local_vertex_position = ti.Vector.field(3, dtype=dtype, shape=(N, 3))

        # Kinematics
        self.velocity = ti.Vector.field(3, dtype=dtype, shape=N)

        # Strengths (Doublet / Potential)
        self.doublet_strength = ti.field(dtype=dtype, shape=N)
        self.doublet_strength_old = ti.field(dtype=dtype, shape=N)
        self.doublet_strength_older = ti.field(dtype=dtype, shape=N)
        self.source_strength = ti.field(dtype=dtype, shape=N)

        # Cumulative doublet_strength for time-history integration
        self.cumulative_doublet_strength = ti.field(dtype=dtype, shape=N)
        self.cumulative_doublet_strength_old = ti.field(dtype=dtype, shape=N)
        self.cumulative_doublet_strength_older = ti.field(dtype=dtype, shape=N)

        # Physics diagnostics
        self.potential_time_derivative = ti.field(dtype=dtype, shape=N)
        self.pressure_coefficient = ti.field(dtype=dtype, shape=N)

        # Panel identity
        self.group_id = ti.field(ti.i32, shape=N)

    @ti.kernel
    def save_old_doublet_strength(self):
        """Advance the timestep history of panel doublet_strength."""
        for i in range(self.n_panels):
            self.doublet_strength_older[i] = self.doublet_strength_old[i]
            self.doublet_strength_old[i] = self.doublet_strength[i]
            self.cumulative_doublet_strength_older[i] = self.cumulative_doublet_strength_old[i]
            self.cumulative_doublet_strength_old[i] = self.cumulative_doublet_strength[i]

    @ti.kernel
    def _flip_normals_kernel(
        self, start_idx: int, count: int, ref_point: ti.types.vector(3, float)
    ):  # type: ignore
        for i in range(count):
            idx = start_idx + i
            r = self.panel_centre[idx] - ref_point
            if r.dot(self.normal[idx]) < 0:
                self.normal[idx] = -self.normal[idx]
                v1, v2 = self.vertex_position[idx, 1], self.vertex_position[idx, 2]
                self.vertex_position[idx, 1], self.vertex_position[idx, 2] = v2, v1
                v1_loc, v2_loc = (
                    self.local_vertex_position[idx, 1],
                    self.local_vertex_position[idx, 2],
                )
                self.local_vertex_position[idx, 1], self.local_vertex_position[idx, 2] = (
                    v2_loc,
                    v1_loc,
                )

    def flip_normal(self, start_idx: int, count: int, reference_point: np.ndarray):
        self._flip_normals_kernel(start_idx, count, ti.Vector(reference_point.tolist()))

    def update_geometry(self, time: float = 0.0):
        """Apply identity transform to initialise geometry from local_vertex_position."""
        n = self.n_panels
        if n == 0:
            return
        eye = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        zero = ti.Vector([0.0, 0.0, 0.0])
        self._update_geometry_kernel(0, n, eye, zero)

    @ti.kernel
    def _update_geometry_kernel(
        self,
        start_idx: int,
        count: int,
        R: ti.types.matrix(3, 3, float),
        T: ti.types.vector(3, float),
    ):  # type: ignore
        for i in range(count):
            idx = start_idx + i
            v0_l, v1_l, v2_l = (
                self.local_vertex_position[idx, 0],
                self.local_vertex_position[idx, 1],
                self.local_vertex_position[idx, 2],
            )
            v0, v1, v2 = R @ v0_l + T, R @ v1_l + T, R @ v2_l + T
            (
                self.vertex_position[idx, 0],
                self.vertex_position[idx, 1],
                self.vertex_position[idx, 2],
            ) = v0, v1, v2

            cp = (v1 - v0).cross(v2 - v0)
            area = cp.norm() * 0.5
            self.area[idx] = area
            if area > PANEL_EPSILON:
                self.normal[idx] = cp / (2.0 * area)
            else:
                self.normal[idx] = ti.Vector([0.0, 0.0, 1.0])
            self.panel_centre[idx] = (v0 + v1 + v2) / 3.0

    @ti.kernel
    def _update_velocity_kernel(
        self,
        start_idx: int,
        count: int,
        v_lin: ti.types.vector(3, float),
        angular_velocity: ti.types.vector(3, float),
        rotation_centre: ti.types.vector(3, float),
    ):  # type: ignore
        for i in range(count):
            idx = start_idx + i
            pos = self.panel_centre[idx]
            self.velocity[idx] = v_lin + angular_velocity.cross(pos - rotation_centre)

    def add_body(
        self,
        uid: str,
        vertex_position: np.ndarray,
        kinematics: Optional["PanelKinematics"] = None,
        group_id: int = 0,
    ):
        from ..coupling.kinematics import StaticPanel

        if kinematics is None:
            kinematics = StaticPanel()

        num_new = vertex_position.shape[0]
        if self.n_panels + num_new > self.max_n_panels:
            raise RuntimeError(
                f"Max panels exceeded: {self.n_panels + num_new} > {self.max_n_panels}"
            )

        start_idx = self.n_panels
        self.bodies.append(
            PanelBody(
                uid=uid,
                start_idx=start_idx,
                count=num_new,
                kinematics=kinematics,
                group_id=group_id,
            )
        )

        # Upload
        v_flat = vertex_position.reshape(-1, 3).astype(
            np.float32 if self.float_dtype == "f32" else np.float64
        )
        self.local_vertex_position.from_numpy(v_flat.reshape(num_new, 3, 3))

        # Set initial group ID
        group_id = np.full(num_new, group_id, dtype=np.int32)
        group_id_np = self.group_id.to_numpy()
        group_id_np[start_idx : start_idx + num_new] = group_id
        self.group_id.from_numpy(group_id_np)

        self.n_panels += num_new
        logger.info(f"Added PanelBody '{uid}' with {num_new} panels at index {start_idx}.")
