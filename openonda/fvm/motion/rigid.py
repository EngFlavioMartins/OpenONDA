"""Arbitrary Lagrangian–Eulerian (ALE) rigid motion of a finite-volume mesh.

The whole mesh moves as one rigid body, so every cell keeps its volume and
the mesh velocity ``u_mesh = V + Omega x (x - c)`` is divergence free. The
discrete geometric conservation law then reduces to the closure of each cell:
the mesh face fluxes ``phi_mesh = u_mesh(x_f) . S_f`` sum to zero round-off in
every cell with planar faces.

Momentum is convected by the relative flux ``phi - phi_mesh``; continuity
keeps the absolute flux ``phi``. Velocity and pressure stay absolute-frame
quantities, as in OpenFOAM's moving-mesh ``pimpleFoam``.

Geometry is not recomputed: positions are mapped through the pose, vectors
and the least-squares gradient stencil are rotated, and scalars (volumes,
areas, interpolation weights, wall distances) are invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from ..fields.mixed_velocity_boundary import (
    update_normal_velocity_tangential_gradient_boundary,
)
from ..schemes.boundaries import BOUNDARIES, BoundaryStrategy
from .kinematics import MeshMotion, RigidPose, advance_pose

if TYPE_CHECKING:
    from ..core.solver import FVMSolver

# Geometry entries mapped through the pose (points) or rotated (vectors).
_GEOMETRY_POINTS = ("face_centre", "cell_centre")
_GEOMETRY_VECTORS = (
    "face_area_vector",
    "cell_connection_vector",
    "boundary_translation",
    "least_squares_neighbour_weighted_displacement",
)
_LSQ_INVERSE = "least_squares_normal_matrix_inverse"
_LSQ_OPERATOR = "least_squares_gradient_operator"
_BOUNDARY_STENCIL_CACHES = ("_boundary_owner_lsq_with_ghosts", "_boundary_owner_lsq_real")


class RigidMeshMotion:
    """Move a solver mesh rigidly and provide its mesh flux and wall velocities.

    Parameters
    ----------
    motion : MeshMotion
        Placement and kinematics of the mesh.
    solver : FVMSolver
        Solver whose ``mesh_data``/``geo_data`` arrays are updated in place.

    Attributes
    ----------
    pose : RigidPose
        Current map from the placed reference mesh to the current mesh.
    time : float
        Physical time in s of the current geometry.
    accepted_pose : RigidPose
        Pose at the last accepted solver time.
    mesh_face_flux : ndarray, shape (n_faces,)
        ``u_mesh . S_f`` in m³/s on the current geometry.
    """

    def __init__(self, motion: MeshMotion, solver: FVMSolver) -> None:
        self.motion = motion
        self.kinematics = motion.kinematics
        self._mesh_data = solver.mesh_data
        self._geo_data = solver.geo_data
        self._boundaries = solver.boundaries
        self._n_cells = int(solver.mesh_data["n_cells"])
        self._n_interior = int(solver.mesh_data["n_interior_faces"])

        # The constructed mesh identifies a restart; the placed mesh is the
        # reference configuration of the kinematics.
        mesh_data = self._mesh_data
        mesh_data["reference_vertex_position"] = np.array(mesh_data["vertex_position"])
        placement = motion.placement()
        self._reference: dict[str, Any] = {
            key: np.array(self._geo_data[key])
            for key in _GEOMETRY_POINTS + _GEOMETRY_VECTORS
            if key in self._geo_data
        }
        self._reference["vertex_position"] = np.array(mesh_data["vertex_position"])
        visualization = mesh_data.get("_visualization_mesh")
        if visualization is not None and visualization is not mesh_data:
            self._reference["visualization_vertex_position"] = np.array(
                visualization["vertex_position"]
            )
        if _LSQ_INVERSE in self._geo_data:
            self._reference[_LSQ_INVERSE] = np.array(self._geo_data[_LSQ_INVERSE])
        if _LSQ_OPERATOR in self._geo_data:
            self._reference[_LSQ_OPERATOR] = self._geo_data[_LSQ_OPERATOR].copy()
        self._place_reference(placement)

        # Wall patches carry their configured velocity relative to the mesh.
        # Slip planes may need a prescribed normal velocity when the mesh
        # moves through them.
        self._wall_velocity: dict[str, np.ndarray] = {}
        self._empty_patches: list[dict] = []
        for boundary in self._boundaries:
            strategy = BOUNDARIES.strategy(boundary.get("velocity_type"), "velocity", "ghost")
            if strategy is BoundaryStrategy.FIXED_VALUE and boundary.get("type") == "wall":
                configured = boundary.get("velocity_value_field")
                if configured is None:
                    configured = boundary.get("velocity_value", np.zeros(3))
                self._wall_velocity[boundary["name"]] = np.array(configured, dtype=np.float64)
            elif strategy is BoundaryStrategy.EMPTY:
                self._empty_patches.append(boundary)
        # Every rank sees the configured slip planes, also those it has no
        # faces of, so their switch to a moving plane is decided collectively.
        self._slip_type = {
            item.name: item.velocity_type
            for item in sorted(solver._resolved_setup.boundaries, key=lambda item: item.name)
            if BOUNDARIES.strategy(item.velocity_type, "velocity", "ghost")
            in (BoundaryStrategy.SLIP, BoundaryStrategy.SYMMETRY)
        }
        self._parallel = solver.parallel

        self.time = float(solver.time)
        self.accepted_time = self.time
        self.accepted_pose = RigidPose()
        self.pose = RigidPose()
        self.mesh_face_flux = np.zeros(int(mesh_data["n_faces"]))
        self._written_rotation = None
        self._write_geometry(self.pose)
        self._update_mesh_flux()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _place_reference(self, placement: RigidPose) -> None:
        """Move the stored reference arrays to the initial placement."""
        if placement.is_identity():
            return
        rotation = placement.rotation
        for key in _GEOMETRY_POINTS + ("vertex_position", "visualization_vertex_position"):
            if key in self._reference:
                self._reference[key] = placement.apply(self._reference[key])
        for key in _GEOMETRY_VECTORS:
            if key in self._reference:
                self._reference[key] = self._reference[key] @ rotation.T
        if _LSQ_INVERSE in self._reference:
            self._reference[_LSQ_INVERSE] = rotation @ self._reference[_LSQ_INVERSE] @ rotation.T
        if _LSQ_OPERATOR in self._reference:
            self._reference[_LSQ_OPERATOR] = self._rotated_gradient_operator(
                self._reference[_LSQ_OPERATOR], rotation
            )

    @staticmethod
    def _rotated_gradient_operator(operator: sparse.csr_matrix, rotation: np.ndarray):
        """Rotate the three spatial rows of every cell in a gradient operator."""
        n_owned = operator.shape[0] // 3
        block = sparse.kron(sparse.identity(n_owned, format="csr"), sparse.csr_matrix(rotation))
        return sparse.csr_matrix(block @ operator)

    def _write_geometry(self, pose: RigidPose) -> None:
        """Write the reference geometry mapped by ``pose`` into the solver arrays."""
        geo_data = self._geo_data
        rotation = pose.rotation
        for key in _GEOMETRY_POINTS:
            if key in self._reference:
                geo_data[key][...] = pose.apply(self._reference[key])
        for key in _GEOMETRY_VECTORS:
            if key in self._reference:
                geo_data[key][...] = self._reference[key] @ rotation.T
        self._mesh_data["vertex_position"][...] = pose.apply(self._reference["vertex_position"])
        if "visualization_vertex_position" in self._reference:
            self._mesh_data["_visualization_mesh"]["vertex_position"][...] = pose.apply(
                self._reference["visualization_vertex_position"]
            )
        if _LSQ_INVERSE in self._reference:
            geo_data[_LSQ_INVERSE][...] = rotation @ self._reference[_LSQ_INVERSE] @ rotation.T
        if _LSQ_OPERATOR in self._reference:
            geo_data[_LSQ_OPERATOR] = self._rotated_gradient_operator(
                self._reference[_LSQ_OPERATOR], rotation
            )
        # Boundary least-squares stencils are cached for a fixed orientation.
        if not np.array_equal(rotation, self._written_rotation):
            for key in _BOUNDARY_STENCIL_CACHES:
                geo_data.pop(key, None)
            self._written_rotation = rotation

    def pose_at(self, time: float) -> RigidPose:
        """Advance the accepted pose to ``time`` with one kinematic increment."""
        time_step_size = time - self.accepted_time
        if time_step_size == 0.0:
            return self.accepted_pose
        return advance_pose(self.kinematics, self.accepted_pose, self.accepted_time, time_step_size)

    def rotation_centre(self) -> np.ndarray:
        """Current position of the kinematic rotation centre in m."""
        centre = np.asarray(self.kinematics.rotation_centre, dtype=np.float64)
        return self.pose.apply(centre[np.newaxis])[0]

    def current_position(self, points: np.ndarray) -> np.ndarray:
        """Map points given in the placed reference frame to the current mesh."""
        array = np.asarray(points, dtype=np.float64)
        return self.pose.apply(array.reshape(-1, 3)).reshape(array.shape)

    def mesh_velocity(self, points: np.ndarray) -> np.ndarray:
        """Mesh velocity ``V + Omega x (x - c)`` in m/s at current points."""
        velocity = self.kinematics.get_velocity(self.time)
        angular_velocity = self.kinematics.get_angular_velocity(self.time)
        arm = np.asarray(points, dtype=np.float64) - self.rotation_centre()
        return velocity + np.cross(angular_velocity, arm)

    def _update_mesh_flux(self) -> None:
        """Recompute ``phi_mesh = u_mesh(x_f) . S_f`` on the current geometry."""
        surface = self._geo_data["face_area_vector"]
        velocity = self.kinematics.get_velocity(self.time)
        angular_velocity = self.kinematics.get_angular_velocity(self.time)
        arm = self._geo_data["face_centre"] - self.rotation_centre()
        self.mesh_face_flux[:] = surface @ velocity + np.cross(arm, surface) @ angular_velocity

    def move_to(self, time: float, pose: RigidPose | None = None) -> np.ndarray:
        """Place the mesh at ``time`` and return the previous face area vectors.

        ``pose`` defaults to the accepted pose advanced to ``time``.
        """
        previous_surface = np.array(self._geo_data["face_area_vector"])
        self.pose = self.pose_at(time) if pose is None else pose
        self.time = float(time)
        self._write_geometry(self.pose)
        self._update_mesh_flux()
        self._check_empty_patches()
        return previous_surface

    def accept(self, time: float) -> None:
        """Commit the current pose as the accepted state at solver time ``time``."""
        self.time = float(time)
        self.accepted_pose = self.pose
        self.accepted_time = self.time

    def restore(self, time: float, rotation: np.ndarray, translation: np.ndarray) -> None:
        """Place the mesh at a restarted accepted pose."""
        self.accepted_time = float(time)
        self.accepted_pose = RigidPose(
            np.array(rotation, dtype=np.float64), np.array(translation, dtype=np.float64)
        )
        self.move_to(self.accepted_time, self.accepted_pose)

    def _check_empty_patches(self) -> None:
        """Reject motion through the span planes of a two-dimensional model."""
        for boundary in self._empty_patches:
            faces = slice(boundary["start_face"], boundary["start_face"] + boundary["n_faces"])
            if np.any(self._normal_speed(faces) > 1.0e-9 * max(self._speed_scale(faces), 1.0)):
                raise ValueError(
                    f"Mesh motion moves the empty patch {boundary['name']!r} out of its "
                    "plane; a two-dimensional mesh may only translate in its plane and "
                    "rotate about the span axis"
                )

    def _normal_speed(self, faces: slice) -> np.ndarray:
        """Magnitude of the mesh velocity normal to the selected faces in m/s."""
        area = np.linalg.norm(self._geo_data["face_area_vector"][faces], axis=1)
        return np.abs(self.mesh_face_flux[faces]) / np.maximum(area, 1.0e-300)

    def _speed_scale(self, faces: slice) -> float:
        """Largest mesh speed on the selected faces in m/s."""
        speed = np.linalg.norm(self.mesh_velocity(self._geo_data["face_centre"][faces]), axis=1)
        return float(np.max(speed, initial=0.0))

    # ------------------------------------------------------------------
    # Fields and boundaries
    # ------------------------------------------------------------------
    def face_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Linear face interpolation of a cell field, boundary faces from ghosts."""
        mesh_data = self._mesh_data
        n_interior = self._n_interior
        owners = mesh_data["owners"]
        neighbours = mesh_data["neighbours"][:n_interior]
        weights = self._geo_data["face_interpolation_weight"][:n_interior, np.newaxis]
        faces = np.empty((int(mesh_data["n_faces"]), 3))
        faces[:n_interior] = (
            weights * velocity[neighbours] + (1.0 - weights) * velocity[owners[:n_interior]]
        )
        faces[n_interior:] = velocity[self._n_cells : self._n_cells + len(faces) - n_interior]
        return faces

    def reexpress_flux(
        self, flux: np.ndarray, velocity: np.ndarray, previous_surface: np.ndarray
    ) -> None:
        """Carry a stored face flux onto the rotated faces.

        The interpolated part ``U_f . S_f`` follows the new face orientation;
        the Rhie–Chow part ``phi - U_f . S_f`` is a scalar of the face and is
        kept. A pure translation leaves every flux unchanged.
        """
        change = self._geo_data["face_area_vector"] - previous_surface
        flux += np.einsum("ij,ij->i", self.face_velocity(velocity), change)

    def update_boundaries(self, solver: FVMSolver) -> None:
        """Impose the current mesh motion on moving walls and slip planes.

        Collective: every rank enters, since a slip plane moving through
        itself changes the momentum operator of all ranks.
        """
        surface = self._geo_data["face_area_vector"]
        patches = {boundary["name"]: boundary for boundary in self._boundaries}
        moving = np.zeros(len(self._slip_type))
        for index, name in enumerate(self._slip_type):
            boundary = patches.get(name)
            if boundary is not None and boundary["n_faces"]:
                faces = slice(boundary["start_face"], boundary["start_face"] + boundary["n_faces"])
                tolerance = 1.0e-12 * max(self._speed_scale(faces), 1.0e-300)
                moving[index] = float(np.any(self._normal_speed(faces) > tolerance))
        moving = {
            name: self._parallel.global_max(float(flag))
            for name, flag in zip(self._slip_type, moving, strict=True)
        }

        for boundary in self._boundaries:
            name = boundary["name"]
            start = int(boundary["start_face"])
            faces = slice(start, start + int(boundary["n_faces"]))
            if name in self._wall_velocity:
                relative = self._wall_velocity[name] @ self.pose.rotation.T
                wall_velocity = self.mesh_velocity(self._geo_data["face_centre"][faces]) + relative
                boundary["velocity_value_field"] = wall_velocity
                ghosts = slice(
                    self._n_cells + start - self._n_interior,
                    self._n_cells + start - self._n_interior + int(boundary["n_faces"]),
                )
                solver.velocity[ghosts] = wall_velocity
                solver.volumetric_face_flux[faces] = np.einsum(
                    "ij,ij->i", wall_velocity, surface[faces]
                )
            elif name in self._slip_type:
                # A slip plane moving along its normal keeps zero shear and
                # takes the mesh normal velocity.
                area = np.linalg.norm(surface[faces], axis=1)
                normal_velocity = self.mesh_face_flux[faces] / area
                if moving[name]:
                    boundary["velocity_type"] = "normalValueTangentialGradient"
                    boundary["normal_velocity_field"] = normal_velocity
                    boundary["tangential_gradient_field"] = np.zeros((len(area), 3))
                    boundary["max_removed_tangential_gradient_normal_component"] = 0.0
                    update_normal_velocity_tangential_gradient_boundary(
                        solver.velocity, boundary, self._mesh_data, self._geo_data
                    )
                elif boundary["velocity_type"] != self._slip_type[name]:
                    boundary["velocity_type"] = self._slip_type[name]
                    for key in (
                        "normal_velocity_field",
                        "tangential_gradient_field",
                        "max_removed_tangential_gradient_normal_component",
                    ):
                        boundary.pop(key, None)
                solver.volumetric_face_flux[faces] = self.mesh_face_flux[faces]
