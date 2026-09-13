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
    from ..coupling.kinematics import BodyPose, PanelKinematics

logger = logging.getLogger("vpm")


@dataclass
class PanelBody:
    """Dataclass storing metadata for a collection of panels belonging to a single body."""

    uid: str
    start_idx: int
    count: int
    kinematics: "PanelKinematics"
    group_id: int = 0
    reference_area: float | None = None
    pose: "BodyPose | None" = None
    geometry_revision: int = 0


@ti.data_oriented
class PanelLattice:
    """GPU-resident storage for arbitrary triangular panel bodies.

    Parameters
    ----------
    max_n_panels : int, default=20000
        Fixed capacity for panel geometry, solution fields, and the dense
        influence matrix.  Only the prefix ``[:n_panels]`` is active.
    float_dtype : {"f32", "f64"}, default="f32"
        Precision used by Taichi fields and geometry/velocity kernels.

    Attributes
    ----------
    vertex_position : taichi.Vector.field
        Active triangle vertices, shape ``(max_n_panels, 3, 3)``; positions in
        metres.  The last two vertices determine the outward normal through
        ``cross(v1-v0, v2-v0)``.
    panel_centre : taichi.Vector.field
        Triangle centroids, shape ``(max_n_panels, 3)``; metres.
    normal : taichi.Vector.field
        Unit panel normals, shape ``(max_n_panels, 3)``; dimensionless.
    area : taichi.field
        Triangle areas, shape ``(max_n_panels,)``; m².
    doublet_strength, source_strength : taichi.field
        Panel potential/source unknowns.  Doublet strength has potential
        units m²/s; source strength has velocity units m/s.
    body_velocity, incident_velocity : taichi.Vector.field
        Rigid-body and incident-fluid velocities at panel centres, shape
        ``(max_n_panels, 3)``; m/s.  They are intentionally separate fields.
    bodies : list[PanelBody]
        Host-side metadata mapping body ids to contiguous panel ranges.

    Notes
    -----
    The fields are fixed-capacity and are allocated immediately, so Taichi
    must already be initialized.  ``n_panels`` is the authoritative active
    count; unused capacity may contain stale data and must not be sampled.
    """

    def __init__(self, max_n_panels: int = 20000, float_dtype: str = "f32"):
        """Allocate a panel lattice after Taichi has been initialized.

        Parameters
        ----------
        max_n_panels : int, default=20000
            Maximum number of triangular panels that may be uploaded.
        float_dtype : {"f32", "f64"}, default="f32"
            Storage and kernel precision.  ``"f64"`` requires a Taichi
            backend with double-precision support.

        Raises
        ------
        RuntimeError
            If Taichi has not been initialized before construction.

        Notes
        -----
        Allocation is eager and can be large because the influence matrix has
        ``max_n_panels²`` entries.  Geometry is not uploaded until
        :meth:`add_body` is called.
        """
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

        # Distinct physical velocity fields.  VPM writes the incident field;
        # rigid-body kinematics writes body_velocity.  Neither is permitted to
        # overwrite the other.
        self.body_velocity = ti.Vector.field(3, dtype=dtype, shape=N)
        self.incident_velocity = ti.Vector.field(3, dtype=dtype, shape=N)

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
        self, start_idx: int, count: int, ref_point: ti.types.vector(3, ti.f64)
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
        """Orient a contiguous panel range away from a reference point.

        Parameters
        ----------
        start_idx : int
            First panel index in the active lattice.
        count : int
            Number of panels to inspect.
        reference_point : array_like, shape (3,)
            Point used to determine orientation, in metres.  A panel is
            flipped when its normal points toward this point.

        Notes
        -----
        Flipping swaps the second and third triangle vertices as well as the
        normal, preserving the panel area and reversing its orientation in
        the geometry field.
        """
        self._flip_normals_kernel(start_idx, count, ti.Vector(reference_point.tolist(), dt=ti.f64))

    def update_geometry(self, time: float = 0.0, start_idx: int = 0, count: int | None = None):
        """Apply identity geometry update to one panel range.

        ``local_vertex_position`` is the immutable reference geometry for each
        body.  Updating only the newly uploaded range is essential when bodies
        are appended after an earlier body has already moved kinematically.
        """
        if count is None:
            count = self.n_panels - start_idx
        if count <= 0:
            return
        if start_idx < 0 or start_idx + count > self.n_panels:
            raise ValueError(
                f"panel geometry range [{start_idx}, {start_idx + count}) "
                f"is outside [0, {self.n_panels})"
            )
        dtype = ti.f64 if self.float_dtype == "f64" else ti.f32
        eye = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dt=dtype)
        zero = ti.Vector([0.0, 0.0, 0.0], dt=dtype)
        if self.float_dtype == "f64":
            self._update_geometry_f64_kernel(start_idx, count, eye, zero)
        else:
            self._update_geometry_kernel(start_idx, count, eye, zero)

    @ti.kernel
    def _update_geometry_kernel(
        self,
        start_idx: int,
        count: int,
        R: ti.template(),
        T: ti.template(),
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
                fallback = v0_l * 0.0
                fallback[2] = 1.0
                self.normal[idx] = fallback
            self.panel_centre[idx] = (v0 + v1 + v2) / 3.0

    @ti.kernel
    def _update_geometry_f64_kernel(
        self,
        start_idx: int,
        count: int,
        R: ti.types.matrix(3, 3, ti.f64),
        T: ti.types.vector(3, ti.f64),
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
                fallback = v0_l * 0.0
                fallback[2] = 1.0
                self.normal[idx] = fallback
            self.panel_centre[idx] = (v0 + v1 + v2) / 3.0

    @ti.kernel
    def _update_body_velocity_kernel(
        self,
        start_idx: int,
        count: int,
        v_lin: ti.template(),
        angular_velocity: ti.template(),
        rotation_centre: ti.template(),
    ):  # type: ignore
        for i in range(count):
            idx = start_idx + i
            pos = self.panel_centre[idx]
            self.body_velocity[idx] = v_lin + angular_velocity.cross(pos - rotation_centre)

    @ti.kernel
    def _update_body_velocity_f64_kernel(
        self,
        start_idx: int,
        count: int,
        v_lin: ti.types.vector(3, ti.f64),
        angular_velocity: ti.types.vector(3, ti.f64),
        rotation_centre: ti.types.vector(3, ti.f64),
    ):  # type: ignore
        for i in range(count):
            idx = start_idx + i
            pos = self.panel_centre[idx]
            self.body_velocity[idx] = v_lin + angular_velocity.cross(pos - rotation_centre)

    # Kept for callers of the older low-level wrapper module.  New code uses
    # ``apply_body_pose`` so geometry and velocity are transformed together.
    @ti.kernel
    def _update_velocity_kernel(
        self,
        start_idx: int,
        count: int,
        v_lin: ti.template(),
        angular_velocity: ti.template(),
        rotation_centre: ti.template(),
    ):  # type: ignore
        for i in range(count):
            idx = start_idx + i
            pos = self.panel_centre[idx]
            self.body_velocity[idx] = v_lin + angular_velocity.cross(pos - rotation_centre)

    def apply_body_pose(self, body: PanelBody, pose: "BodyPose") -> None:
        """Apply one complete pose to a body geometry range and velocity field."""
        rotation = np.asarray(pose.rotation, dtype=np.float64)
        translation = np.asarray(pose.translation, dtype=np.float64)
        centre = np.asarray(pose.rotation_centre, dtype=np.float64)
        linear_velocity = np.asarray(pose.linear_velocity, dtype=np.float64)
        angular_velocity = np.asarray(pose.angular_velocity, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("BodyPose.rotation must have shape (3, 3)")
        if any(
            vector.shape != (3,)
            for vector in (translation, centre, linear_velocity, angular_velocity)
        ):
            raise ValueError("BodyPose vector values must have shape (3,)")

        # x = R @ (x0 - c) + c + T = R @ x0 + (c + T - R @ c)
        transform_translation = centre + translation - rotation @ centre
        dtype = ti.f64 if self.float_dtype == "f64" else ti.f32
        rotation_arg = ti.Matrix(rotation.tolist(), dt=dtype)
        translation_arg = ti.Vector(transform_translation.tolist(), dt=dtype)
        linear_velocity_arg = ti.Vector(linear_velocity.tolist(), dt=dtype)
        angular_velocity_arg = ti.Vector(angular_velocity.tolist(), dt=dtype)
        centre_arg = ti.Vector((centre + translation).tolist(), dt=dtype)
        if self.float_dtype == "f64":
            self._update_geometry_f64_kernel(
                body.start_idx, body.count, rotation_arg, translation_arg
            )
            self._update_body_velocity_f64_kernel(
                body.start_idx,
                body.count,
                linear_velocity_arg,
                angular_velocity_arg,
                centre_arg,
            )
        else:
            self._update_geometry_kernel(body.start_idx, body.count, rotation_arg, translation_arg)
            self._update_body_velocity_kernel(
                body.start_idx,
                body.count,
                linear_velocity_arg,
                angular_velocity_arg,
                centre_arg,
            )

    def add_body(
        self,
        uid: str,
        vertex_position: np.ndarray,
        kinematics: Optional["PanelKinematics"] = None,
        group_id: int = 0,
        reference_area: float | None = None,
    ) -> int:
        """Append one body and its triangle vertices to the active lattice.

        Parameters
        ----------
        uid : str
            Unique host-side body identifier.
        vertex_position : array_like, shape (N, 3, 3)
            Triangle vertices in metres.  Vertex order defines the initial
            normal orientation through the right-hand rule.
        kinematics : PanelKinematics, optional
            Pose model used by the panel solver.  ``None`` selects a static
            body.
        group_id : int, default=0
            Integer grouping label copied to every uploaded panel.
        reference_area : float, optional
            Body reference area in m² for coefficient reporting.

        Returns
        -------
        int
            Number of panels appended.  Their contiguous range is recorded in
            :attr:`bodies`.

        Raises
        ------
        ValueError
            If ``uid`` is already present.
        RuntimeError
            If the new body would exceed ``max_n_panels``.

        Notes
        -----
        The input vertices are copied into ``local_vertex_position``.  The
        active count is incremented only after the upload succeeds; geometry
        and derived fields are generated later by the panel solver.
        """
        from ..coupling.kinematics import BodyPose, StaticPanel

        if any(body.uid == uid for body in self.bodies):
            raise ValueError(f"Duplicate panel body uid: {uid}")
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
                reference_area=reference_area,
                pose=BodyPose(),
            )
        )

        # Upload
        v_flat = vertex_position.reshape(-1, 3).astype(
            np.float32 if self.float_dtype == "f32" else np.float64
        )
        local_vertex_np = self.local_vertex_position.to_numpy()
        local_vertex_np[start_idx : start_idx + num_new] = v_flat.reshape(num_new, 3, 3)
        self.local_vertex_position.from_numpy(local_vertex_np)

        # Set initial group ID
        group_id = np.full(num_new, group_id, dtype=np.int32)
        group_id_np = self.group_id.to_numpy()
        group_id_np[start_idx : start_idx + num_new] = group_id
        self.group_id.from_numpy(group_id_np)

        self.n_panels += num_new
        logger.info(f"Added PanelBody '{uid}' with {num_new} panels at index {start_idx}.")
        return num_new
