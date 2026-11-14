"""
Vortex-lattice-method solver (VLMSolver): assembles and solves the panel
circulation system and evaluates forces.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import copy
import time
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import ArrayLike
import taichi as ti

from openonda.core.kinematics import (
    RigidMotion,
    RigidPose,
    RotatingMotion,
    StaticMotion,
    advance_pose,
)
from openonda.core.metadata import metadata_value

from ....config.constants import VLM_SMALL_VELOCITY
from ....io.logging import Logging
from ....kernels import make_vortex_kernel
from ..config import VLMConfig, VLMSurface
from ..geometry.aircraft import Aircraft, Wing
from ..geometry.surface_io import load_surface as _load_surface
from .circulation_solver import CirculationSolver
from .influence import (
    accumulate_bound_transport,
    add_induced_velocity_at_targets,
    add_stage_rates_with_bound_exchange,
    compute_aerodynamic_influence_coefficient_matrix,
    compute_coupled_right_hand_side,
    compute_induced_velocities,
    compute_induced_velocities_at_bound,
    compute_panel_force_coupled,
    initialize_bound_transport,
)
from .kernels import shed_wake_particles_kernel
from .lattice import VLMLattice
from .mesh import (
    generate_vlm_mesh,
    update_trailing_directions_local,
    update_trailing_edge_directions,
)
from .unsteady import add_unsteady_pressure_loads

if TYPE_CHECKING:
    from ....particles.container import Particles
    from ....physics.base import PhysicsBase
    from ....physics.induction.base import StageRates, StageState

EPSILON = VLM_SMALL_VELOCITY

# One near-wake source: (position, core radius, strip-circulation
# coefficients, constant vortex strength).
NearWakeSource: TypeAlias = tuple[np.ndarray, float, np.ndarray, np.ndarray]


class SurfaceTransform(TypedDict):
    """Initial placement of one surface applied after mesh generation."""

    translation: tuple[float, float, float] | None
    rotation_degrees: tuple[float, float, float] | None
    rotation_centre: tuple[float, float, float] | None


# Lattice fields that move rigidly with their surface.
_GEOMETRY_FIELDS = (
    "panel_corner_position",
    "vortex_point_position",
    "collocation_point",
    "bound_vortex_midpoint",
    "normal",
)


def _map_rigidly(geometry: dict[str, np.ndarray], pose: RigidPose, panels: slice) -> None:
    """Map one panel slice of float64 lattice geometry through ``pose`` in place."""
    for name in _GEOMETRY_FIELDS[:-1]:
        values = geometry[name][panels]
        geometry[name][panels] = pose.apply(values.reshape(-1, 3)).reshape(values.shape)
    geometry["normal"][panels] = geometry["normal"][panels] @ pose.rotation.T


def _padded(values: np.ndarray, lattice: VLMLattice) -> np.ndarray:
    """Return active-panel values padded to the lattice capacity in lattice precision."""
    padded = np.zeros((lattice.max_n_panels, *values.shape[1:]), dtype=lattice.np_dtype)
    padded[: len(values)] = values
    return padded


class VLMSolver:
    """
    Vortex Lattice Method solver with VPM coupling support.

    Solves for circulation distribution on thin lifting surfaces using
    horseshoe vortex elements and the zero-normal-flow boundary condition.

    Geometry, kinematics, transforms, mesh spacing, and fluid data are all
    declared in :class:`VLMConfig`. The runtime solver does not expose
    configuration mutation methods.
    """

    def __init__(self, config: VLMConfig):
        """Initialize from one complete, immutable VLM definition."""
        # Solver configuration
        self.config = config

        # Multi-body storage: Dict[uid -> (Aircraft, kinematics)]
        self.surfaces: dict[str, tuple[Aircraft, RigidMotion]] = {}
        # Accumulated rigid pose of each surface relative to its placement;
        # the motions are immutable, so the moving state lives here.
        self.surface_poses: dict[str, RigidPose] = {}
        # Placed lattice geometry in float64, set by generate_mesh; the
        # current geometry is this reference mapped through surface_poses.
        self._placed_geometry: dict[str, np.ndarray] = {}
        # Area-weighted centroid of the placed lattice [m], the moment reference.
        self._placed_centroid = np.zeros(3)

        # Combined aircraft for mesh generation (set by _register_surface)
        self.aircraft: Aircraft

        # Lattice will be created when first needed (after Taichi init)
        self._lattice: VLMLattice | None = None
        self._lattice_initialized = False
        # Accepted physical time, set during advance operations from parent solver
        self._current_time: float | None = None

        # Flight condition / Fluid properties
        self.freestream_velocity = (
            None
            if config.freestream_velocity is None
            else np.array(config.freestream_velocity, dtype=np.float64)
        )
        self.density = config.density
        # Density of the latest force evaluation, restored from backups.
        self._force_density = config.density
        self.kinematic_viscosity = config.kinematic_viscosity
        self.wake_core_overlap = config.wake_core_overlap
        self._wake_kernel = make_vortex_kernel("gaussian")

        # solution state
        self._mesh_generated = False
        self._solved = False
        self._coupled_mode = False
        self._circulation_solver = CirculationSolver()
        # Stage geometry fields; Taichi fields are dynamic DSL objects.
        self._stage_geometry_fields: tuple[Any, Any, Any, Any, Any] | None = None
        self._stage_geometry_time: float | None = None

        # Force evaluation configuration
        self.force = config.force
        self._last_forces: dict[str, Any] | None = None
        self._last_reference_velocity: np.ndarray | None = None
        self._previous_panel_corners: np.ndarray | None = None

        # Explicit initialization of optional attributes to avoid AttributeError
        self._surface_transforms: dict[str, SurfaceTransform] = {}
        self._surface_group_ids: dict[str, int] = {}
        # Contiguous lattice panels of each surface, set by generate_mesh.
        self._surface_panels: dict[str, slice] = {}

        for surface in config.surfaces:
            self._register_surface(surface)

        # The lattice holds exactly the panels of the declared surfaces.
        self.max_n_panels = self.aircraft.total_n_panels()
        self.dtype: Literal["f32", "f64"]

        Logging.record("VLM initialized", ("panels", self.max_n_panels))

    @property
    def lattice(self) -> VLMLattice:
        """Panel lattice fields, allocated after the VPM solver initialises Taichi."""
        if self._lattice is None:
            raise RuntimeError(
                "VLM lattice is not allocated; generate the mesh after the VPM "
                "solver has initialised Taichi"
            )
        return self._lattice

    def _ensure_lattice_initialized(self) -> None:
        """Ensure lattice is created (lazy initialization after Taichi init).

        VPM is required to be the Taichi master: the VPM Solver must call
        ``start_taichi`` (i.e. ``VPMSolver.__init__``) *before* any
        VLM lattice fields are allocated.  The lattice takes the precision that
        Taichi was initialised with.
        """
        if not self._lattice_initialized:
            # Guard: Taichi must already be initialised by the VPM Solver.
            if ti.lang.impl.get_runtime().prog is None:
                raise RuntimeError(
                    "VLMSolver._ensure_lattice_initialized called before Taichi "
                    "is initialised.  Always construct the VPM Solver first so "
                    "that start_taichi() runs before any VLM fields "
                    "are created."
                )

            ti_dtype = ti.lang.impl.get_runtime().default_fp
            self.dtype = "f32" if ti_dtype == ti.f32 else "f64"
            self._lattice = VLMLattice(self.max_n_panels, ti_dtype)
            self._transported_bound = ti.Vector.field(3, ti_dtype, shape=self.max_n_panels)
            self._bound_exchange_rate = ti.Vector.field(3, ti_dtype, shape=self.max_n_panels)
            self._bound_transport_ready = False
            self._lattice_initialized = True

    def generate_mesh(self) -> None:
        """Generate the VLM mesh using the distribution declared in ``config``."""
        self._ensure_lattice_initialized()

        if self._mesh_generated:
            return

        t0 = time.time()

        generate_vlm_mesh(
            self.aircraft,
            self.lattice,
            spanwise_spacing=self.config.mesh.spacing,
            spanwise_spacing_ratio=self.config.mesh.ratio,
            spanwise_spacing_region=self.config.mesh.region,
        )

        # The mesh generator emits the surfaces in registration order.
        start = 0
        for name, (aircraft, _) in self.surfaces.items():
            stop = start + aircraft.total_n_panels()
            self._surface_panels[name] = slice(start, stop)
            start = stop
        self._place_surfaces()

        # Populate panel group IDs
        group_id = self.get_panel_group_ids()

        self.lattice.group_id.from_numpy(group_id)

        from .restart import restart_configuration_hash

        self._restart_geometry_references = metadata_value(self.aircraft.refs)
        self._restart_configuration_hash = restart_configuration_hash(self)
        self._mesh_generated = True
        self._solved = False

        elapsed = time.time() - t0
        total_area = float(self.lattice.area.to_numpy()[: self.lattice.n_panels].sum())
        Logging.record(
            "VLM mesh generated",
            ("panels", self.lattice.n_panels),
            ("wings", len(self.aircraft.wings)),
            ("area", total_area, "m²"),
            ("elapsed", elapsed, "s"),
        )

    def _place_surfaces(self) -> None:
        """Store the placed lattice as the float64 reference of the surface motion.

        Every later geometry is this reference mapped through the surface
        pose, so the lattice stays a rigid image of the placed surface for
        any number of steps.
        """
        n_panels = self.lattice.n_panels
        geometry = {
            name: getattr(self.lattice, name).to_numpy()[:n_panels].astype(np.float64)
            for name in _GEOMETRY_FIELDS
        }
        for surface_name, transform in self._surface_transforms.items():
            if transform["rotation_degrees"] is None and transform["translation"] is None:
                continue
            rotation = self._build_rotation_matrix(transform["rotation_degrees"])
            translation = (
                np.zeros(3) if transform["translation"] is None else transform["translation"]
            )
            centre = (
                np.zeros(3)
                if transform["rotation_centre"] is None
                else np.asarray(transform["rotation_centre"], dtype=np.float64)
            )
            # x -> R (x - c) + c + t as one rigid pose.
            placement = RigidPose(rotation, centre + translation - rotation @ centre)
            _map_rigidly(geometry, placement, self._surface_panels[surface_name])
        self._placed_geometry = geometry
        self._placed_centroid = np.average(
            geometry["panel_corner_position"].mean(axis=1),
            axis=0,
            weights=self.lattice.area.to_numpy()[:n_panels],
        )
        self._write_geometry(geometry)

    def _surface_geometry(self, poses: dict[str, RigidPose]) -> dict[str, np.ndarray]:
        """Return the float64 lattice geometry with each surface at ``poses``.

        The far trailing points are not body geometry: the last solve aligns
        them with the flow. Each far leg of the accepted lattice is turned
        with its surface from the accepted pose to the requested one.
        """
        n_panels = self.lattice.n_panels
        geometry = {name: values.copy() for name, values in self._placed_geometry.items()}
        vortex = self.lattice.vortex_point_position.to_numpy()[:n_panels].astype(np.float64)
        far_legs = vortex[:, [0, 3]] - vortex[:, [1, 2]]
        for name, pose in poses.items():
            panels = self._surface_panels[name]
            _map_rigidly(geometry, pose, panels)
            turn = pose.rotation @ self.surface_poses[name].rotation.T
            far_legs[panels] = far_legs[panels] @ turn.T
        bound = geometry["vortex_point_position"][:, [1, 2]]
        geometry["vortex_point_position"][:, [0, 3]] = bound + far_legs
        return geometry

    def _write_geometry(self, geometry: dict[str, np.ndarray]) -> None:
        """Upload float64 geometry arrays of the active panels to the lattice fields."""
        for name, values in geometry.items():
            getattr(self.lattice, name).from_numpy(_padded(values, self.lattice))

    def _build_rotation_matrix(self, rotation_degrees: Sequence[float] | None) -> np.ndarray:
        """Build 3x3 rotation matrix from [rx, ry, rz] angles in degrees."""
        rotation_matrix = np.eye(3)
        if rotation_degrees is None:
            return rotation_matrix

        angles = np.array(rotation_degrees, dtype=np.float64)

        # Rotation about X
        if abs(angles[0]) > 1e-10:
            rotation_x_radians = np.radians(angles[0])
            rotation_matrix_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(rotation_x_radians), -np.sin(rotation_x_radians)],
                    [0, np.sin(rotation_x_radians), np.cos(rotation_x_radians)],
                ]
            )
            rotation_matrix = rotation_matrix_x @ rotation_matrix

        # Rotation about Y
        if abs(angles[1]) > 1e-10:
            rotation_y_radians = np.radians(angles[1])
            rotation_matrix_y = np.array(
                [
                    [np.cos(rotation_y_radians), 0, np.sin(rotation_y_radians)],
                    [0, 1, 0],
                    [-np.sin(rotation_y_radians), 0, np.cos(rotation_y_radians)],
                ]
            )
            rotation_matrix = rotation_matrix_y @ rotation_matrix

        # Rotation about Z
        if abs(angles[2]) > 1e-10:
            rotation_z_radians = np.radians(angles[2])
            rotation_matrix_z = np.array(
                [
                    [np.cos(rotation_z_radians), -np.sin(rotation_z_radians), 0],
                    [np.sin(rotation_z_radians), np.cos(rotation_z_radians), 0],
                    [0, 0, 1],
                ]
            )
            rotation_matrix = rotation_matrix_z @ rotation_matrix

        return rotation_matrix

    def _register_surface(self, surface: VLMSurface) -> str:
        """Load one declared surface before any Taichi fields are allocated."""
        aircraft = (
            _load_surface(surface.surface)
            if isinstance(surface.surface, str)
            else copy.deepcopy(surface.surface)
        )
        surface_name = surface.name or aircraft.uid
        if surface_name in self.surfaces:
            raise ValueError(f"Duplicate VLM surface name: {surface_name}")
        kinematics = surface.kinematics if surface.kinematics is not None else StaticMotion()

        self.surfaces[surface_name] = (aircraft, kinematics)
        self.surface_poses[surface_name] = RigidPose()
        self._surface_group_ids[surface_name] = surface.group_id
        self._surface_transforms[surface_name] = {
            "translation": surface.translation,
            "rotation_degrees": surface.rotation_degrees,
            "rotation_centre": surface.rotation_centre,
        }

        self._update_combined_aircraft()
        Logging.record(
            "VLM surface",
            ("name", surface_name),
            ("panels", aircraft.total_n_panels()),
            ("group", surface.group_id),
        )
        return surface_name

    def get_panel_group_ids(self) -> np.ndarray:
        """
        Get array of group IDs for each panel in the lattice.

        Returns:
             np.ndarray: Array of shape (n_panels,) with group ID for each panel.
        """
        group_id = np.zeros(self.lattice.n_panels, dtype=np.int32)
        for surface_name, gid in self._surface_group_ids.items():
            group_id[self._surface_panels[surface_name]] = gid
        return group_id

    def ensure_mesh_generated(self) -> None:
        """
        Ensure mesh is generated. Called by VPMSolver after Taichi is initialized.

        This deferred initialization ensures Taichi fields are created with the
        correct backend (GPU/Vulkan) as configured by VPMSolver.
        """
        if self._mesh_generated:
            return

        if not self.aircraft:
            return

        self.generate_mesh()

    def _update_combined_aircraft(self) -> None:
        """Update combined aircraft from all loaded surfaces."""
        if len(self.surfaces) == 1:
            # Single surface mode
            name, (aircraft, _) = next(iter(self.surfaces.items()))
            self.aircraft = aircraft
        else:
            # Multi-surface: merge into combined aircraft
            combined = Aircraft(uid="combined")
            for name, (aircraft, _) in self.surfaces.items():
                for wing_uid, wing in aircraft.wings.items():
                    combined_uid = f"{name}_{wing_uid}"
                    new_wing = Wing(
                        uid=combined_uid, segments=wing.segments.copy(), symmetry=wing.symmetry
                    )
                    combined.wings[combined_uid] = new_wing
            combined.compute_default_refs()
            self.aircraft = combined

    def surface_rotation_centre(self, surface_name: str) -> np.ndarray:
        """Return the current position in m of a surface's rotation centre.

        The rotation centre of the surface motion is given in the placed
        coordinates and moves with the accumulated surface pose.
        """
        _, motion = self.surfaces[surface_name]
        centre = np.asarray(motion.rotation_centre, dtype=np.float64)
        return self.surface_poses[surface_name].apply(centre[np.newaxis])[0]

    def _compute_rotor_tip_speed(
        self, surface_name: str, kinematics: RigidMotion
    ) -> tuple[np.ndarray, float]:
        """Return (velocity_vector, tip_speed) for a RotatingMotion surface, or (zeros, 0)."""
        if not isinstance(kinematics, RotatingMotion) or not self._mesh_generated:
            return np.zeros(3), 0.0
        angular_speed_magnitude = abs(kinematics.angular_speed)
        if angular_speed_magnitude < 1e-10:
            return np.zeros(3), 0.0
        rotation_centre = self.surface_rotation_centre(surface_name)
        rotation_axis = np.asarray(kinematics.rotation_axis)
        collocation_point = self.lattice.get_collocation_points()
        radial_position = collocation_point[self._surface_panels[surface_name]] - rotation_centre
        radial_position -= np.outer(radial_position @ rotation_axis, rotation_axis)
        maximum_tip_speed = angular_speed_magnitude * float(
            np.max(np.linalg.norm(radial_position, axis=1), initial=0.0)
        )
        return rotation_axis * maximum_tip_speed, maximum_tip_speed

    def _get_active_kinematic_velocity(self) -> np.ndarray:
        """
        Get a representative kinematic velocity from active surfaces.

        For translating surfaces, returns the translational velocity.
        For rotating surfaces, returns a velocity vector whose magnitude
        equals the maximum tip speed (omega × R_max) across all rotors,
        directed along the rotation axis (for reference scaling only).

        Returns:
            Velocity vector [vx, vy, vz] or zeros if no active kinematics
        """
        current_time = self._current_time if self._current_time is not None else 0.0
        representative_velocity = np.zeros(3)
        maximum_speed = 0.0
        for surface_name, (_, kinematics) in self.surfaces.items():
            if isinstance(kinematics, StaticMotion):
                continue
            translational_velocity = kinematics.get_velocity(current_time)
            translational_speed = float(np.linalg.norm(translational_velocity))
            if translational_speed > maximum_speed:
                representative_velocity = translational_velocity
                maximum_speed = translational_speed
            rotor_tip_velocity, tip_speed = self._compute_rotor_tip_speed(surface_name, kinematics)
            if tip_speed > maximum_speed:
                representative_velocity = rotor_tip_velocity
                maximum_speed = tip_speed
        return representative_velocity

    def _get_max_kinematic_speed(self) -> float:
        """
        Get maximum kinematic speed across all surfaces.

        For rotating surfaces this is the tip speed (omega × R_max).
        For translating surfaces this is the translational speed.

        Returns:
            Maximum kinematic speed [m/s], or 0.0 if no active kinematics
        """
        kinematic_velocity = self._get_active_kinematic_velocity()
        return float(np.linalg.norm(kinematic_velocity))

    def update_trailing_directions(self, transport_velocity: np.ndarray) -> None:
        """
        Update trailing edge directions based on transport velocity.

        The wake trails in the direction of the local relative velocity.

        Args:
           transport_velocity: Velocity vector/field for wake transport (N, 3) or (3,)
                        Ideally this is external_velocity - V_skinematic at trailing edge.
        """
        if not self._mesh_generated:
            self.generate_mesh()

        # If scalar vector provided, apply uniformly
        if transport_velocity.ndim == 1:
            update_trailing_edge_directions(self.lattice, transport_velocity)
        else:
            # Use local velocity field
            update_trailing_directions_local(self.lattice, transport_velocity)

    def _compute_panel_kinematic_velocity(
        self,
        kinematics: RigidMotion,
        rotation_centre: np.ndarray,
        time: float,
        collocation_point: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> np.ndarray:
        """Return the rigid-body velocity ``V + Omega x (x - c)`` of a panel slice."""
        angular_velocity = kinematics.get_angular_velocity(time)
        translational_velocity = kinematics.get_velocity(time)
        radial_position = collocation_point[start_idx:end_idx] - rotation_centre
        return translational_velocity + np.cross(angular_velocity, radial_position)

    def advance_time(self, time_step_size: float, current_time: float) -> None:
        """Advance the surface poses to ``current_time`` and map the geometry through them."""
        if not self._mesh_generated:
            self.generate_mesh()
        self._previous_panel_corners = self.lattice.panel_corner_position.to_numpy()[
            : self.lattice.n_panels
        ].copy()
        self._stage_geometry_time = None
        self._current_time = float(current_time)
        time = self._current_time
        n_panels = self.lattice.n_panels
        kinematic_velocity = np.zeros((n_panels, 3), dtype=np.float64)
        moving = {
            name: motion
            for name, (_, motion) in self.surfaces.items()
            if not isinstance(motion, StaticMotion)
        }
        if moving:
            poses = dict(self.surface_poses)
            for name, motion in moving.items():
                poses[name] = advance_pose(
                    motion, poses[name], time - time_step_size, time_step_size
                )
            geometry = self._surface_geometry(poses)
            self.surface_poses.update(poses)
            self._write_geometry(geometry)
            for name, motion in moving.items():
                panels = self._surface_panels[name]
                kinematic_velocity[panels] = self._compute_panel_kinematic_velocity(
                    motion,
                    self.surface_rotation_centre(name),
                    time,
                    geometry["collocation_point"],
                    panels.start,
                    panels.stop,
                )
        self.lattice.set_kinematic_velocity(kinematic_velocity)
        if moving:
            self._solved = False

    def _minimum_panel_chord(self) -> float:
        """Return the shortest chordwise panel edge in the geometry [m]."""
        panel_chords = []
        for wing in self.aircraft.wings.values():
            for segment in wing.segments.values():
                root_chord = np.linalg.norm(
                    segment.vertex_position["d"] - segment.vertex_position["a"]
                )
                tip_chord = np.linalg.norm(
                    segment.vertex_position["c"] - segment.vertex_position["b"]
                )
                min_panel_chord = min(root_chord, tip_chord) / segment.n_chordwise_panels
                if min_panel_chord > EPSILON:
                    panel_chords.append(float(min_panel_chord))
        if not panel_chords:
            raise ValueError("VLM geometry has no positive panel chord length")
        return min(panel_chords)

    def check_coupling_stability(
        self, time_step_size: float, freestream_velocity: ArrayLike | None = None
    ) -> dict[str, float | bool]:
        """
        Check the convective time-step resolution of VLM-VPM coupling.

        The wake-convection Courant number is

        ``C_wake = U_characteristic * dt / min_panel_chord``.

        ``C_wake <= 1`` keeps one explicit wake displacement within the
        shortest chordwise panel. The characteristic speed is a conservative
        upper bound: background-flow magnitude plus the maximum surface speed.

        Args:
            time_step_size: Time step size [s]
            freestream_velocity: Background flow velocity [vx, vy, vz] [m/s].
                When omitted, the solver freestream is used.

        Returns:
            Diagnostic values ``stable``, ``courant``, ``max_dt``,
            ``characteristic_speed``, and ``min_panel_chord``.

        Warns:
            RuntimeWarning: If ``C_wake > 1``.
        """
        if not np.isfinite(time_step_size) or time_step_size <= 0:
            raise ValueError(f"time_step_size must be finite and positive, got {time_step_size}")

        if freestream_velocity is None:
            background = (
                self.freestream_velocity if self.freestream_velocity is not None else np.zeros(3)
            )
        else:
            background = np.asarray(freestream_velocity, dtype=float)
        if np.shape(background) != (3,) or not np.all(np.isfinite(background)):
            raise ValueError("freestream_velocity must contain three finite components")

        min_panel_chord = self._minimum_panel_chord()
        characteristic_speed = float(np.linalg.norm(background)) + self._get_max_kinematic_speed()
        courant = characteristic_speed * time_step_size / min_panel_chord
        max_time_step_size = (
            min_panel_chord / characteristic_speed
            if characteristic_speed > EPSILON
            else float("inf")
        )
        stable = courant <= 1.0
        result: dict[str, float | bool] = {
            "stable": stable,
            "courant": courant,
            "max_dt": max_time_step_size,
            "characteristic_speed": characteristic_speed,
            "min_panel_chord": min_panel_chord,
        }
        if not stable:
            Logging.warning(
                "VLM-VPM wake convection is under-resolved: "
                f"wake_courant_number={courant:.3g} > 1. Reduce time_step_size to "
                f"<= {max_time_step_size:.3g} s."
            )
        return result

    def solve(
        self,
        external_velocity: np.ndarray | None = None,
        coupled: bool = False,
    ) -> np.ndarray:
        """
        Solve VLM system for circulation.

        Args:
            external_velocity: Total external velocity field at collocation_point points (N x 3).
                       If None, assumes data is already in lattice.external_velocity (GPU-resident).
            coupled: If True, uses bound-only aerodynamic_influence_coefficient for coupling with VPM particles.

        Returns:
            Computed circulation (circulation) array
        """
        self._stage_geometry_time = None
        self._coupled_mode = bool(coupled)
        solve_start_time = time.time()
        if not self._mesh_generated:
            self.generate_mesh()
        n_panels = self.lattice.n_panels
        if external_velocity is not None:
            if external_velocity.ndim == 1 and external_velocity.shape[0] == 3:
                external_velocity = np.tile(external_velocity, (n_panels, 1))
            if external_velocity.shape[0] != n_panels:
                raise ValueError(
                    f"external_velocity shape {external_velocity.shape} does not match panels {n_panels}"
                )
            self.lattice.set_external_velocity(external_velocity)
            external_velocity_array = external_velocity
        else:
            external_velocity_array = self.lattice.external_velocity.to_numpy()[:n_panels]
        relative_velocity = external_velocity_array - self.lattice.get_kinematic_velocity()
        velocity_upload_end_time = time.time()
        self.update_trailing_directions(relative_velocity)
        trailing_direction_end_time = time.time()
        compute_aerodynamic_influence_coefficient_matrix(
            self.lattice.collocation_point,
            self.lattice.vortex_point_position,
            self.lattice.panel_corner_position,
            self.lattice.normal,
            self.lattice.is_trailing_edge,
            self.lattice.trailing_edge_index,
            self.lattice.aerodynamic_influence_coefficient,
            self.lattice.n_panels,
            self.lattice.filament_core_radius,
            coupled_mode=1 if coupled else 0,
        )
        influence_matrix_end_time = time.time()
        compute_coupled_right_hand_side(
            self.lattice.collocation_point,
            self.lattice.normal,
            self.lattice.external_velocity,
            self.lattice.kinematic_velocity,
            self.lattice.right_hand_side,
            self.lattice.n_panels,
        )
        right_hand_side_end_time = time.time()
        # The unsteady load differences against the circulation of the previous step.
        self.lattice.save_old_circulation()
        if coupled:
            self._augment_near_wake_particles()
        self._circulation_solver.solve(
            self.lattice.aerodynamic_influence_coefficient,
            self.lattice.right_hand_side,
            self.lattice.circulation,
            n_panels,
        )
        circulation_np = self.lattice.circulation.to_numpy()[:n_panels]
        linear_solve_end_time = time.time()
        total_time = linear_solve_end_time - solve_start_time
        if total_time > 1.0:
            Logging.record(
                "VLM solve timing",
                ("velocity upload", velocity_upload_end_time - solve_start_time, "s"),
                (
                    "trailing directions",
                    trailing_direction_end_time - velocity_upload_end_time,
                    "s",
                ),
                ("influence matrix", influence_matrix_end_time - trailing_direction_end_time, "s"),
                ("right-hand side", right_hand_side_end_time - influence_matrix_end_time, "s"),
                ("linear solve", linear_solve_end_time - right_hand_side_end_time, "s"),
                ("total", total_time, "s"),
            )
        self._solved = True
        return circulation_np

    def _stage_geometry_arrays(self, stage_time: float) -> tuple[np.ndarray, ...]:
        """Return temporary geometry and surface velocity at ``stage_time``.

        Each moving surface is placed at its pose advanced from the accepted
        time to the stage time; no accepted lattice coordinate, pose or clock
        is changed. Stage boundary solves and stage velocities use the same
        geometry for one stage time.
        """
        reference_time = self._current_time if self._current_time is not None else 0.0
        duration = float(stage_time) - float(reference_time)
        kinematic = self.lattice.kinematic_velocity.to_numpy().copy()
        moving = {
            name: motion
            for name, (_, motion) in self.surfaces.items()
            if not isinstance(motion, StaticMotion)
        }
        if not moving:
            return (
                self.lattice.vortex_point_position.to_numpy().copy(),
                self.lattice.panel_corner_position.to_numpy().copy(),
                self.lattice.collocation_point.to_numpy().copy(),
                self.lattice.normal.to_numpy().copy(),
                kinematic,
            )
        poses = dict(self.surface_poses)
        if abs(duration) >= 1.0e-14:
            for name, motion in moving.items():
                poses[name] = advance_pose(motion, poses[name], reference_time, duration)
        geometry = self._surface_geometry(poses)
        for name, motion in moving.items():
            panels = self._surface_panels[name]
            centre = poses[name].apply(np.asarray(motion.rotation_centre)[np.newaxis])[0]
            kinematic[panels] = self._compute_panel_kinematic_velocity(
                motion,
                centre,
                float(stage_time),
                geometry["collocation_point"],
                panels.start,
                panels.stop,
            )
        return (
            *(
                _padded(geometry[name], self.lattice)
                for name in (
                    "vortex_point_position",
                    "panel_corner_position",
                    "collocation_point",
                    "normal",
                )
            ),
            kinematic,
        )

    def _ensure_stage_geometry_fields(
        self, stage_time: float
    ) -> tuple[ti.Field, ti.Field, ti.Field, ti.Field, ti.Field]:
        """Upload one pure stage geometry state and return its device fields."""
        fields = self._stage_geometry_fields
        if fields is None:
            fields = (
                ti.Vector.field(3, self.lattice.dtype, shape=(self.max_n_panels, 4)),
                ti.Vector.field(3, self.lattice.dtype, shape=(self.max_n_panels, 4)),
                ti.Vector.field(3, self.lattice.dtype, shape=self.max_n_panels),
                ti.Vector.field(3, self.lattice.dtype, shape=self.max_n_panels),
                ti.field(dtype=self.lattice.dtype, shape=(self.max_n_panels, 3)),
            )
            self._stage_geometry_fields = fields
        elif self._stage_geometry_time == stage_time:
            return fields
        arrays = self._stage_geometry_arrays(stage_time)
        np_dtype = self.lattice.np_dtype
        for field, array in zip(fields, arrays, strict=True):
            field.from_numpy(np.asarray(array, dtype=np_dtype))
        self._stage_geometry_time = float(stage_time)
        return fields

    def _stage_geometry(self, stage_time: float) -> tuple[ti.Field, ti.Field]:
        """Return pure stage source geometry while holding circulation fixed."""
        fields = self._ensure_stage_geometry_fields(float(stage_time))
        return fields[0], fields[1]

    def add_stage_velocity(
        self, target_position: ti.Field, target_velocity: ti.Field, count: int, stage_time: float
    ) -> None:
        """Accumulate the latest solved VLM velocity at temporary VPM targets.

        ``stage_time`` is accepted explicitly at the stage boundary. The
        current VLM coupling is lagged to the latest accepted-step solve, so
        solved circulation is held fixed; prescribed source motion and target
        positions are evaluated at the requested RK stage time.
        """
        if not self._solved or self._lattice is None or count <= 0:
            return
        vortex_points, corners = self._stage_geometry(stage_time)
        add_induced_velocity_at_targets(
            target_position,
            target_velocity,
            vortex_points,
            corners,
            self.lattice.trailing_edge_index,
            self.lattice.circulation,
            int(count),
            int(self.lattice.n_panels),
            int(self._coupled_mode),
            self.lattice.filament_core_radius,
        )

    def _initialize_bound_transport(self) -> None:
        """Seed each trailing strip's integrated bound strength before RK exchange."""
        initialize_bound_transport(
            self.lattice.panel_corner_position,
            self.lattice.cumulative_circulation,
            self.lattice.is_trailing_edge,
            self._transported_bound,
            self.lattice.n_panels,
        )

    @contextmanager
    def particle_transport_step(self) -> Iterator[None]:
        """Publish strip exchange only after all particle RK stages succeed.

        This bound-vorticity exchange is consumed by the next accepted wake emission. It is not
        persistent restart state: checkpoints are written after that emission.
        """
        transported_before = self._transported_bound.to_numpy().copy()
        self._bound_transport_ready = False
        self._initialize_bound_transport()
        try:
            yield
        except BaseException:
            # A rejected trial must not publish a partially accumulated
            # reaction increments.  Restore the pre-trial accepted exchange and
            # leave the next accepted step responsible for reinitialisation.
            self._transported_bound.from_numpy(transported_before)
            self._bound_exchange_rate.fill(0.0)
            self._bound_transport_ready = False
            raise
        else:
            self._bound_transport_ready = True

    def add_stage_rates(
        self,
        state: "StageState",
        stage_time: float,
        rates: "StageRates",
        mode: int,
        weighted_dt: float,
    ) -> None:
        """Accumulate the bound field and its stage-weighted strip reaction."""
        if not self._solved or state.count <= 0:
            return
        vortex_points, corners = self._stage_geometry(stage_time)
        lattice = self.lattice
        if weighted_dt != 0.0:
            self._bound_exchange_rate.fill(0.0)
        add_stage_rates_with_bound_exchange(
            state.position,
            state.vortex_strength,
            state.core_radius,
            rates.velocity,
            rates.vortex_strength_rate,
            rates.velocity_gradient,
            vortex_points,
            corners,
            lattice.trailing_edge_index,
            lattice.is_trailing_edge,
            lattice.circulation,
            self._bound_exchange_rate,
            state.count,
            lattice.n_panels,
            int(self._coupled_mode),
            mode,
            weighted_dt,
            rates.velocity_gradient is not None,
            lattice.filament_core_radius,
        )
        if weighted_dt != 0.0:
            accumulate_bound_transport(
                self._transported_bound,
                self._bound_exchange_rate,
                weighted_dt,
                lattice.n_panels,
            )

    def compute_postprocess(
        self,
        external_velocity: np.ndarray,
        reference_velocity: np.ndarray,
        density: float,
        time_step_size: float | None = None,
        coupled: bool = False,
        bound_external_velocity: np.ndarray | None = None,
    ) -> None:
        """
        Compute derived quantities (velocity, pressures, forces).

        Args:
           external_velocity: External velocity (N, 3).
           reference_velocity: Reference velocity vector [ux, uy, uz] (m/s).
           density: Fluid density
           time_step_size: Time step size
           coupled: Whether in coupled mode (bound only aerodynamic_influence_coefficient)
        """
        if self.force.unsteady and (
            time_step_size is None or not np.isfinite(time_step_size) or time_step_size <= 0
        ):
            raise ValueError("Unsteady VLM pressure requires a finite positive time_step_size")
        self._force_density = density
        self.lattice.force_density = float(density)
        self.lattice.set_external_velocity(external_velocity)

        reference_velocity_magnitude = float(np.linalg.norm(reference_velocity))
        if reference_velocity_magnitude < 1e-10:
            reference_velocity_magnitude = 1.0

        # 0. Keep cumulative circulation current for wake diagnostics.
        self._compute_cumulative_circulation_cpu()

        # 1. Compute velocity at collocation points.
        compute_induced_velocities(
            self.lattice.n_panels,
            self.lattice.collocation_point,
            self.lattice.vortex_point_position,
            self.lattice.panel_corner_position,
            self.lattice.trailing_edge_index,
            self.lattice.circulation,
            self.lattice.velocity,
            self.lattice.external_velocity,
            int(coupled),
            self.lattice.filament_core_radius,
        )
        velocity = self.lattice.velocity.to_numpy()[: self.lattice.n_panels]
        kinematic_velocity = self.lattice.get_kinematic_velocity()
        relative_velocity = np.zeros((self.max_n_panels, 3), dtype=self.lattice.np_dtype)
        relative_velocity[: self.lattice.n_panels] = velocity - kinematic_velocity
        self.lattice.relative_velocity.from_numpy(relative_velocity)

        # 2. Compute velocity at BOUND VORTEX midpoints (for Forces)
        incident_at_bound = (
            external_velocity if bound_external_velocity is None else bound_external_velocity
        )
        padded = np.zeros((self.max_n_panels, 3), dtype=self.lattice.np_dtype)
        padded[: self.lattice.n_panels] = incident_at_bound
        self.lattice.bound_external_velocity.from_numpy(padded)
        bound_kinematics = np.zeros_like(padded)
        points = self.lattice.bound_vortex_midpoint.to_numpy()[: self.lattice.n_panels]
        for name, (_, motion) in self.surfaces.items():
            panels = self._surface_panels[name]
            bound_kinematics[panels] = self._compute_panel_kinematic_velocity(
                motion,
                self.surface_rotation_centre(name),
                self._current_time or 0.0,
                points,
                panels.start,
                panels.stop,
            )
        self.lattice.bound_kinematic_velocity.from_numpy(bound_kinematics)
        self.lattice.reference_speed = float(reference_velocity_magnitude)
        compute_induced_velocities_at_bound(
            self.lattice.n_panels,
            self.lattice.bound_vortex_midpoint,
            self.lattice.vortex_point_position,
            self.lattice.panel_corner_position,
            self.lattice.trailing_edge_index,
            self.lattice.circulation,
            self.lattice.bound_vortex_velocity,
            self.lattice.bound_external_velocity,
            1 if coupled else 0,
            self.lattice.filament_core_radius,
        )
        bound_velocity = self.lattice.bound_vortex_velocity.to_numpy()[: self.lattice.n_panels]
        bound_relative_velocity = np.zeros((self.max_n_panels, 3), dtype=self.lattice.np_dtype)
        bound_relative_velocity[: self.lattice.n_panels] = (
            bound_velocity - bound_kinematics[: self.lattice.n_panels]
        )
        self.lattice.bound_relative_velocity.from_numpy(bound_relative_velocity)

        compute_panel_force_coupled(
            self.lattice.bound_vortex_velocity,
            self.lattice.vortex_point_position,
            self.lattice.circulation,
            self.lattice.bound_kinematic_velocity,
            self.lattice.panel_force,
            self.lattice.n_panels,
            density,
        )
        self.lattice.unsteady_panel_force.fill(0.0)
        self.lattice.panel_moment_correction.fill(0.0)
        self.lattice.unsteady_pressure_jump_coefficient.fill(0.0)
        if self.force.unsteady and time_step_size is not None:
            add_unsteady_pressure_loads(
                self.lattice.panel_corner_position,
                self.lattice.vortex_point_position,
                self.lattice.circulation,
                self.lattice.circulation_old,
                self.lattice.cumulative_circulation,
                self.lattice.cumulative_circulation_old,
                self.lattice.panel_force,
                self.lattice.unsteady_panel_force,
                self.lattice.panel_moment_correction,
                self.lattice.unsteady_pressure_jump_coefficient,
                self.lattice.n_panels,
                density / time_step_size,
                2.0 / (density * reference_velocity_magnitude**2),
            )

    def _resolve_reference_velocity(
        self, reference_velocity: np.ndarray | None, n_panels: int
    ) -> np.ndarray:
        """Return a valid reference velocity, auto-computed if not provided."""
        if reference_velocity is not None:
            return np.asarray(reference_velocity, dtype=float)
        if (
            self.freestream_velocity is not None
            and np.linalg.norm(self.freestream_velocity) > 1e-10
        ):
            return self.freestream_velocity
        kinematic_velocity = self._get_active_kinematic_velocity()
        background_velocity = (
            np.mean(self.lattice.external_velocity.to_numpy()[:n_panels], axis=0)
            if n_panels > 0
            else np.zeros(3)
        )
        resolved_velocity = background_velocity - kinematic_velocity
        if np.linalg.norm(resolved_velocity) < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return resolved_velocity

    def _decompose_wind_axes(
        self,
        total_force: np.ndarray,
        reference_velocity_magnitude: float,
        reference_velocity: np.ndarray,
    ) -> tuple[float, float, float]:
        """Decompose total force into lift, drag, side-force in wind axes."""
        if reference_velocity_magnitude > 1e-10:
            reference_direction = reference_velocity / reference_velocity_magnitude
            vertical_direction = np.array([0.0, 0.0, 1.0])
            lift_direction = (
                vertical_direction
                - np.dot(vertical_direction, reference_direction) * reference_direction
            )
            lift_direction_magnitude = np.linalg.norm(lift_direction)
            lift_direction = (
                lift_direction / lift_direction_magnitude
                if lift_direction_magnitude > 1e-10
                else vertical_direction
            )
            side_force_direction = np.cross(reference_direction, lift_direction)
            return (
                float(np.dot(total_force, lift_direction)),
                float(np.dot(total_force, reference_direction)),
                float(np.dot(total_force, side_force_direction)),
            )
        force_x, force_y, force_z = total_force
        return -float(force_z), float(force_x), float(force_y)

    def _compute_force_moments(
        self,
        panel_force: np.ndarray,
        reference_chord: float | None,
        panel_moment_correction: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], np.ndarray]:
        """Compute moments about reference center and quarter-chord."""
        bound_vortex_midpoint = self.lattice.bound_vortex_midpoint.to_numpy()[
            : self.lattice.n_panels
        ]
        # A single surface carries its placed centroid; several surfaces share
        # the fixed placed centroid of the whole lattice.
        reference_point = self._placed_centroid
        if len(self.surfaces) == 1:
            pose = next(iter(self.surface_poses.values()))
            reference_point = pose.apply(reference_point[np.newaxis])[0]
        total_moment = np.sum(
            np.cross(bound_vortex_midpoint - reference_point, panel_force)
            + panel_moment_correction,
            axis=0,
        )
        if reference_chord is None:
            reference_chord = float(self.aircraft.refs.get("chord", 1.0))
        # Derive the quarter-chord point from the current physical strips, so
        # prescribed placement, pitch and translation all transform it with the wing.
        corners = self.lattice.panel_corner_position.to_numpy()[: self.lattice.n_panels]
        leading = self.lattice.is_leading_edge.to_numpy()[: self.lattice.n_panels] == 1
        trailing = self.lattice.trailing_edge_index.to_numpy()[: self.lattice.n_panels][leading]
        leading_midpoint = 0.5 * (corners[leading, 0] + corners[leading, 1])
        trailing_midpoint = 0.5 * (corners[trailing, 2] + corners[trailing, 3])
        strip_chord = np.linalg.norm(trailing_midpoint - leading_midpoint, axis=1)
        strip_span = np.linalg.norm(corners[leading, 1] - corners[leading, 0], axis=1)
        quarter_chord_reference_point = np.average(
            0.75 * leading_midpoint + 0.25 * trailing_midpoint,
            axis=0,
            weights=strip_chord * strip_span,
        )
        quarter_chord_total_moment = np.sum(
            np.cross(bound_vortex_midpoint - quarter_chord_reference_point, panel_force)
            + panel_moment_correction,
            axis=0,
        )
        return (
            tuple(total_moment),
            tuple(quarter_chord_total_moment),
            reference_point,
        )

    def _build_force_coefficients(  # noqa: PLR0913
        self,
        lift: float,
        drag: float,
        side_force: float,
        force_x: float,
        force_y: float,
        force_z: float,
        moment: tuple[float, float, float],
        quarter_chord_moment: tuple[float, float, float],
        reference_point: np.ndarray,
        density: float,
        reference_velocity_magnitude: float,
        reference_area: float | None,
        reference_chord: float | None,
        reference_span: float | None,
    ) -> dict[str, float | np.ndarray]:
        """Build normalised coefficient dict from raw forces/moments."""
        moment_x, moment_y, moment_z = moment
        quarter_chord_moment_x, quarter_chord_moment_y, quarter_chord_moment_z = (
            quarter_chord_moment
        )
        dynamic_pressure = 0.5 * density * reference_velocity_magnitude**2
        if reference_area is None:
            reference_area = float(self.aircraft.refs.get("area", 1.0))
        if reference_span is None:
            reference_span = float(self.aircraft.refs.get("span", 1.0))
        if reference_chord is None:
            reference_chord = float(self.aircraft.refs.get("chord", 1.0))
        force_normalization = dynamic_pressure * reference_area
        chord_moment_normalization = force_normalization * reference_chord
        span_moment_normalization = force_normalization * reference_span
        if dynamic_pressure > 1e-10 and force_normalization > 1e-10:
            lift_coefficient = lift / force_normalization
            drag_coefficient = drag / force_normalization
            side_force_coefficient = side_force / force_normalization
            force_coefficient_x = force_x / force_normalization
            force_coefficient_y = force_y / force_normalization
            force_coefficient_z = force_z / force_normalization
            rolling_moment_coefficient = moment_x / span_moment_normalization
            pitching_moment_coefficient = moment_y / chord_moment_normalization
            yawing_moment_coefficient = moment_z / span_moment_normalization
            rolling_moment_coefficient_quarter_chord = (
                quarter_chord_moment_x / span_moment_normalization
            )
            pitching_moment_coefficient_quarter_chord = (
                quarter_chord_moment_y / chord_moment_normalization
            )
            yawing_moment_coefficient_quarter_chord = (
                quarter_chord_moment_z / span_moment_normalization
            )
        else:
            lift_coefficient = drag_coefficient = side_force_coefficient = force_coefficient_x = (
                force_coefficient_y
            ) = force_coefficient_z = 0.0
            rolling_moment_coefficient = 0.0
            pitching_moment_coefficient = 0.0
            yawing_moment_coefficient = 0.0
            rolling_moment_coefficient_quarter_chord = 0.0
            pitching_moment_coefficient_quarter_chord = 0.0
            yawing_moment_coefficient_quarter_chord = 0.0
        return {
            "force_x": force_x,
            "force_y": force_y,
            "force_z": force_z,
            "lift": lift,
            "drag": drag,
            "side_force": side_force,
            "moment_x": moment_x,
            "moment_y": moment_y,
            "moment_z": moment_z,
            "force_coefficient_x": force_coefficient_x,
            "force_coefficient_y": force_coefficient_y,
            "force_coefficient_z": force_coefficient_z,
            "lift_coefficient": lift_coefficient,
            "drag_coefficient": drag_coefficient,
            "side_force_coefficient": side_force_coefficient,
            "rolling_moment_coefficient": rolling_moment_coefficient,
            "pitching_moment_coefficient": pitching_moment_coefficient,
            "yawing_moment_coefficient": yawing_moment_coefficient,
            "rolling_moment_coefficient_quarter_chord": rolling_moment_coefficient_quarter_chord,
            "pitching_moment_coefficient_quarter_chord": pitching_moment_coefficient_quarter_chord,
            "yawing_moment_coefficient_quarter_chord": yawing_moment_coefficient_quarter_chord,
            "dynamic_pressure": dynamic_pressure,
            "reference_area": reference_area,
            "reference_point": reference_point,
            "reference_chord": reference_chord,
            "reference_span": reference_span,
        }

    def compute_forces(
        self,
        density: float,
        reference_velocity: np.ndarray | None = None,
        reference_area: float | None = None,
        reference_chord: float | None = None,
        reference_span: float | None = None,
    ) -> dict[str, float | np.ndarray]:
        """
        Compute integrated aerodynamic forces and moments.

        lift_coefficient, drag_coefficient, side_force_coefficient are normalized by the dynamic pressure and reference area:
        lift_coefficient = L / (0.5 * rho * reference_area * reference_velocity²)

        Args:
           density: Fluid density (kg/m³)
           reference_velocity: Reference velocity vector [ux, uy, uz] (for coefficients and L/D axes).
                  If None, auto-computed from freestream_velocity or kinematics.
           reference_area: Reference area (m²). If None, uses aircraft defaults.
           reference_chord: Reference chord (m). If None, uses aircraft defaults.
           reference_span: Reference span (m). If None, uses aircraft defaults.

        Returns:
            Dictionary with force components and coefficients
        """
        if not self._solved:
            raise RuntimeError("Must solve system before computing forces")
        panel_force = self.lattice.get_forces() * (density / self._force_density)
        total_force = np.sum(panel_force, axis=0)
        force_x, force_y, force_z = total_force
        reference_velocity = self._resolve_reference_velocity(
            reference_velocity, self.lattice.n_panels
        )
        reference_velocity_magnitude = float(np.linalg.norm(reference_velocity))
        lift, drag, side_force = self._decompose_wind_axes(
            total_force, reference_velocity_magnitude, reference_velocity
        )
        moment, quarter_chord_moment, reference_point = self._compute_force_moments(
            panel_force,
            reference_chord,
            self.lattice.panel_moment_correction.to_numpy()[: self.lattice.n_panels]
            * (density / self._force_density),
        )
        result = self._build_force_coefficients(
            lift,
            drag,
            side_force,
            force_x,
            force_y,
            force_z,
            moment,
            quarter_chord_moment,
            reference_point,
            density,
            reference_velocity_magnitude,
            reference_area,
            reference_chord,
            reference_span,
        )
        unsteady = self.lattice.unsteady_panel_force.to_numpy()[: self.lattice.n_panels].sum(axis=0)
        unsteady *= density / self._force_density
        result.update(
            {
                f"unsteady_force_{axis}": float(value)
                for axis, value in zip("xyz", unsteady, strict=True)
            }
        )
        return result

    def compute_total_bound_vortex_strength(self) -> np.ndarray:
        """
        Compute the integrated vector strength of the bound VLM field.

        A coupled horseshoe contains all three on-wing legs, from its strip's
        left trailing edge to its right trailing edge. Their vectors telescope
        to that endpoint difference, including sweep, taper and twist. The
        standalone diagnostic retains the quarter-chord bound-leg convention.

        Returns:
            np.ndarray: Integrated bound vortex-strength vector [m³/s].

        """
        if not self._solved:
            return np.zeros(3)

        n_panels = self.lattice.n_panels
        circulation = self.lattice.circulation.to_numpy()[:n_panels]
        if self._coupled_mode:
            trailing_edges = self.lattice.trailing_edge_index.to_numpy()[:n_panels]
            corners = self.lattice.panel_corner_position.to_numpy()[trailing_edges]
            l_vec = corners[:, 2] - corners[:, 3]
        else:
            vortex_pts = self.lattice.vortex_point_position.to_numpy()[:n_panels]
            l_vec = vortex_pts[:, 2] - vortex_pts[:, 1]

        net_vortex_strength = np.sum(circulation[:, np.newaxis] * l_vec, axis=0)

        return net_vortex_strength

    def compute_bound_linear_impulse(self) -> np.ndarray:
        """Integrate half of ``x cross omega`` over the finite bound field [m⁴/s].

        Coupled mode includes the three on-wing legs, ending at the actual
        trailing edge. Standalone mode uses the same quarter-chord-only
        convention as the bound-strength diagnostic. Density is applied by
        the caller; the bound contribution alone is origin-dependent.
        """
        if not self._solved:
            return np.zeros(3)
        n = self.lattice.n_panels
        gamma = self.lattice.circulation.to_numpy()[:n].astype(np.float64)
        vortex = self.lattice.vortex_point_position.to_numpy()[:n].astype(np.float64)
        crosses = np.cross(vortex[:, 1], vortex[:, 2])
        if self._coupled_mode:
            trailing = self.lattice.trailing_edge_index.to_numpy()[:n]
            corners = self.lattice.panel_corner_position.to_numpy()[trailing].astype(np.float64)
            crosses += np.cross(corners[:, 3], vortex[:, 1])
            crosses += np.cross(vortex[:, 2], corners[:, 2])
        return 0.5 * np.sum(gamma[:, None] * crosses, axis=0)

    def _compute_one_surface_forces(
        self,
        surface_name: str,
        panel_force: np.ndarray,
        reference_direction: np.ndarray,
        force_normalization: float,
        panel_moment_correction: np.ndarray,
    ) -> dict[str, float]:
        """Compute forces, moment about the current pivot, and fluid-on-body power."""
        panels = self._surface_panels[surface_name]
        _, motion = self.surfaces[surface_name]
        time = self._current_time if self._current_time is not None else 0.0
        centre = self.surface_rotation_centre(surface_name)
        positions = self.lattice.bound_vortex_midpoint.to_numpy()[panels]
        forces = panel_force[panels]
        panel_centers = self.lattice.panel_corner_position.to_numpy()[panels].mean(axis=1)
        areas = self.lattice.area.to_numpy()[panels]
        centroid = np.average(panel_centers, axis=0, weights=areas)
        surface_force = forces.sum(axis=0)
        moment = (np.cross(positions - centre, forces) + panel_moment_correction[panels]).sum(
            axis=0
        )
        translation_velocity = motion.get_velocity(time)
        angular_velocity = motion.get_angular_velocity(time)
        rotational_power = float(moment @ angular_velocity)
        translational_power = float(surface_force @ translation_velocity)
        lift, drag, _ = self._decompose_wind_axes(surface_force, 1.0, reference_direction)
        result = {
            "lift": lift,
            "drag": drag,
            "lift_coefficient": lift / force_normalization if force_normalization > 1e-10 else 0.0,
            "drag_coefficient": drag / force_normalization if force_normalization > 1e-10 else 0.0,
            "panel_count": panels.stop - panels.start,
            "rotational_power": rotational_power,
            "translational_power": translational_power,
            "power": rotational_power + translational_power,
        }
        for label, vector in (
            ("force", surface_force),
            ("moment", moment),
            ("rotation_centre", centre),
            ("translation_velocity", translation_velocity),
            ("angular_velocity", angular_velocity),
            ("position", self.surface_poses[surface_name].translation),
            ("centroid", centroid),
        ):
            result.update(
                {f"{label}_{axis}": float(value) for axis, value in zip("xyz", vector, strict=True)}
            )
        return result

    def compute_per_surface_forces(
        self,
        density: float,
        reference_velocity: np.ndarray | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute forces for each individual surface.

        All coefficients are normalized by dynamic pressure and the aircraft
        reference area: lift_coefficient = L / (0.5 * rho * S_ref * reference_velocity²)

        Args:
            density: Fluid density (kg/m³)
            reference_velocity: Reference velocity vector [ux, uy, uz]

        Returns:
            Dictionary mapping surface name to force dictionary
        """
        if not self._solved:
            return {}
        panel_force = self.lattice.get_forces() * (density / self._force_density)
        reference_velocity = self._resolve_reference_velocity(
            reference_velocity, self.lattice.n_panels
        )
        reference_velocity_magnitude = float(np.linalg.norm(reference_velocity))
        dynamic_pressure = 0.5 * density * reference_velocity_magnitude**2
        force_normalization = dynamic_pressure * float(self.aircraft.refs.get("area", 1.0))
        reference_direction = (
            reference_velocity / reference_velocity_magnitude
            if reference_velocity_magnitude > 1e-10
            else np.array([1.0, 0.0, 0.0])
        )
        moment_correction = self.lattice.panel_moment_correction.to_numpy()[: self.lattice.n_panels]
        moment_correction *= density / self._force_density
        return {
            name: self._compute_one_surface_forces(
                name,
                panel_force,
                reference_direction,
                force_normalization,
                moment_correction,
            )
            for name in self.surfaces
        }

    def save_results(self, filename: str, time: float = 0.0) -> None:
        """
        Save VLM results to VTK file.

        Args:
            filename: Output filename (without extension)
            time: Physical simulation time stored in the output field.
        """
        if not self._solved:
            Logging.warning("VLM is unsolved; saving geometry without solved loads")

        self.lattice.save_vtk(filename, time=time)

    def _fill_segment_cumulative(
        self,
        circulation: np.ndarray,
        circulation_cumulative: np.ndarray,
        panel_index: int,
        n_chordwise_panels: int,
        n_spanwise_panels: int,
    ) -> int:
        """Fill cumulative circulation for one segment and return the next panel index."""
        n_segment_panels = n_chordwise_panels * n_spanwise_panels
        if panel_index + n_segment_panels <= len(circulation_cumulative):
            segment_circulation = circulation[panel_index : panel_index + n_segment_panels]
            segment_cumulative_circulation = np.cumsum(
                segment_circulation.reshape((n_spanwise_panels, n_chordwise_panels)),
                axis=1,
            )
            circulation_cumulative[panel_index : panel_index + n_segment_panels] = (
                segment_cumulative_circulation.ravel()
            )
        return panel_index + n_segment_panels

    def _compute_cumulative_circulation_cpu(self) -> None:
        """
        Compute cumulative circulation for all panels using segment information.

        This replaces the Taichi kernel version which relies on neighbor_indices
        that may be incorrectly set for tapered/swept wings.

        For each spanwise station, the cumulative circulation is the sum of all
        chordwise panel circulations. This is what gets shed as trailing vortices.
        """
        n_panels = self.lattice.n_panels
        circulation = self.lattice.circulation.to_numpy()[:n_panels]
        circulation_cumulative = np.zeros(n_panels, dtype=np.float64)

        panel_index = 0
        for _wing_uid, wing in self.aircraft.wings.items():
            for _segment_uid, segment in wing.segments.items():
                n_chordwise_panels = segment.n_chordwise_panels
                n_spanwise_panels = segment.n_spanwise_panels
                panel_index = self._fill_segment_cumulative(
                    circulation,
                    circulation_cumulative,
                    panel_index,
                    n_chordwise_panels,
                    n_spanwise_panels,
                )
                if wing.symmetry > 0:
                    panel_index = self._fill_segment_cumulative(
                        circulation,
                        circulation_cumulative,
                        panel_index,
                        n_chordwise_panels,
                        n_spanwise_panels,
                    )

        # Upload to Taichi field for kernel access
        dtype_np = np.float32 if self.lattice.dtype == ti.f32 else np.float64
        circulation_full = np.zeros(self.lattice.max_n_panels, dtype=dtype_np)
        circulation_full[:n_panels] = circulation_cumulative
        self.lattice.cumulative_circulation.from_numpy(circulation_full)

    def _prepare_near_wake(
        self, time_step_size: float, physics: "PhysicsBase", particles: "Particles"
    ) -> None:
        """Construct one row from previous TE positions convected to the new clock."""
        n = self.lattice.n_panels
        corners = self.lattice.panel_corner_position.to_numpy()[:n]
        old = corners if self._previous_panel_corners is None else self._previous_panel_corners
        indices = np.flatnonzero(self.lattice.is_trailing_edge.to_numpy()[:n])
        points = corners[indices][:, [3, 2]].reshape(-1, 3)
        target_velocity = physics.compute_target_velocity(
            particles, points, include_freestream=True
        )
        fluid = np.zeros_like(points) if target_velocity is None else target_velocity
        fluid = fluid.reshape(-1, 2, 3)
        offset = self.lattice.wake_offset.to_numpy()
        velocity = self.lattice.trailing_edge_velocity.to_numpy()
        offset[indices] = (
            old[indices][:, [3, 2]] - corners[indices][:, [3, 2]] + fluid * time_step_size
        )
        velocity[indices] = fluid
        self.lattice.wake_offset.from_numpy(offset)
        self.lattice.trailing_edge_velocity.from_numpy(velocity)

    def _compute_wake_particles(self) -> None:
        """Discretize the completed near-wake row at edge midpoints and its far edge."""
        self.lattice.reset_wake_buffer()
        self._compute_cumulative_circulation_cpu()
        # Antisymmetric signed root circulations cancel to solver roundoff.
        # Do not turn frame-dependent last-bit noise into extra wake particles.
        dtype = np.float32 if self.lattice.dtype == ti.f32 else np.float64
        shed_wake_particles_kernel(
            self.lattice.n_panels,
            self.wake_core_overlap,
            32.0 * np.finfo(dtype).eps,
            self.lattice.cumulative_circulation,
            self.lattice.panel_corner_position,
            self.lattice.neighbor_indices,
            self.lattice.is_trailing_edge,
            self.lattice.is_mirrored,
            self.lattice.group_id,
            self.lattice.wake_offset,
            self.lattice.trailing_edge_velocity,
            self.lattice.wake_position,
            self.lattice.wake_velocity,
            self.lattice.wake_vortex_strength,
            self.lattice.wake_core_radius,
            self.lattice.wake_volume,
            self.lattice.wake_group_id,
            self.lattice.n_wake_particles,
            self._transported_bound,
        )

        n_particles_shed = self.lattice.n_wake_particles[None]
        wake_buffer_capacity = self.lattice.wake_position.shape[0]
        if n_particles_shed > wake_buffer_capacity:
            raise RuntimeError(
                f"VLM wake buffer overflow: {n_particles_shed} particles exceed "
                f"capacity {wake_buffer_capacity}"
            )

    # Implicit starting-vortex aerodynamic_influence_coefficient augmentation
    @staticmethod
    def _fill_trailing_edge_strip_segment(
        trailing_edge_indices: list[int],
        strip_index_by_panel: np.ndarray,
        panel_offset: int,
        n_chordwise_panels: int,
        n_spanwise_panels: int,
    ) -> int:
        """Register trailing-edge strips and return the next panel offset."""
        for spanwise_index in range(n_spanwise_panels):
            strip_index = len(trailing_edge_indices)
            trailing_edge_indices.append(
                panel_offset + spanwise_index * n_chordwise_panels + n_chordwise_panels - 1
            )
            strip_index_by_panel[
                panel_offset + spanwise_index * n_chordwise_panels : panel_offset
                + (spanwise_index + 1) * n_chordwise_panels
            ] = strip_index
        return panel_offset + n_chordwise_panels * n_spanwise_panels

    def _build_trailing_edge_strip_map(self) -> tuple[list[int], np.ndarray]:
        """Map every panel to its trailing-edge strip."""
        n_panels = self.lattice.n_panels
        trailing_edge_indices: list[int] = []
        strip_index_by_panel = np.zeros(n_panels, dtype=np.intp)
        panel_offset = 0
        for _wing_uid, wing in self.aircraft.wings.items():
            for _segment_uid, segment in wing.segments.items():
                n_chordwise_panels = segment.n_chordwise_panels
                n_spanwise_panels = segment.n_spanwise_panels
                panel_offset = self._fill_trailing_edge_strip_segment(
                    trailing_edge_indices,
                    strip_index_by_panel,
                    panel_offset,
                    n_chordwise_panels,
                    n_spanwise_panels,
                )
                if (
                    wing.symmetry > 0
                    and panel_offset + n_chordwise_panels * n_spanwise_panels <= n_panels
                ):
                    panel_offset = self._fill_trailing_edge_strip_segment(
                        trailing_edge_indices,
                        strip_index_by_panel,
                        panel_offset,
                        n_chordwise_panels,
                        n_spanwise_panels,
                    )
        return trailing_edge_indices, strip_index_by_panel

    def _near_wake_row(
        self, corners: np.ndarray, offsets: np.ndarray, old_bound: np.ndarray
    ) -> tuple[list[NearWakeSource], np.ndarray, int]:
        """Return affine native-row sources in cumulative strip circulation.

        Each source is (position, radius, strip-vector coefficients, constant
        strength). Strength units are m³/s; circulation coefficients have
        length units. A strip's old closing vector is kept in the inertial
        frame, including transported reaction.
        """
        edges, strip_by_panel = self._build_trailing_edge_strip_map()
        n = self.lattice.n_panels
        neighbors = self.lattice.neighbor_indices.to_numpy()[:n]
        mirrored = self.lattice.is_mirrored.to_numpy()[:n]
        sources: list[NearWakeSource] = []
        for strip, panel in enumerate(edges):
            left, right = corners[panel, 3], corners[panel, 2]
            dl, dr = offsets[panel]
            span = np.linalg.norm(right - left)
            length = 0.5 * (np.linalg.norm(dl) + np.linalg.norm(dr))
            if span <= 1e-12 or length <= 1e-12:
                continue
            left_radius = float(self.wake_core_overlap * max(np.linalg.norm(dl), span))
            right_radius = float(self.wake_core_overlap * max(np.linalg.norm(dr), span))
            transverse_radius = float(self.wake_core_overlap * max(length, span))
            left_index, right_index = neighbors[panel, :2]
            shared_root = left_index != -1 and mirrored[panel] != mirrored[left_index]
            if not shared_root or panel < left_index:
                coefficients = np.zeros((len(edges), 3))
                coefficients[strip] -= dl
                if left_index != -1:
                    coefficients[strip_by_panel[left_index]] += -dl if shared_root else dl
                sources.append((left + 0.5 * dl, left_radius, coefficients, np.zeros(3)))
            if right_index == -1:
                coefficients = np.zeros((len(edges), 3))
                coefficients[strip] = dr
                sources.append((right + 0.5 * dr, right_radius, coefficients, np.zeros(3)))
            far_left, far_right = left + dl, right + dr
            coefficients = np.zeros((len(edges), 3))
            coefficients[strip] = far_left - far_right
            sources.append(
                (
                    0.5 * (far_left + far_right),
                    transverse_radius,
                    coefficients,
                    old_bound[panel],
                )
            )
        return sources, strip_by_panel, len(edges)

    def _near_wake_row_influence(
        self,
        sources: list[NearWakeSource],
        targets: np.ndarray,
        normals: np.ndarray,
        n_strips: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project an affine native row onto all receiving collocation normals."""
        matrix, old_velocity = np.zeros((len(targets), n_strips)), np.zeros(len(targets))
        for position, radius, coefficients, constant in sources:
            displacement = targets - position
            for strip in np.flatnonzero(np.any(coefficients != 0, axis=1)):
                velocity = self._wake_kernel.velocity_pair(
                    displacement, coefficients[strip], radius, radius
                )
                matrix[:, strip] += np.einsum("ij,ij->i", velocity, normals)
            if np.any(constant):
                velocity = self._wake_kernel.velocity_pair(displacement, constant, radius, radius)
                old_velocity += np.einsum("ij,ij->i", velocity, normals)
        return matrix, old_velocity

    def _near_wake_particle_influence(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the accepted row matrix, old-strength RHS and panel/strip map."""
        lattice, n = self.lattice, self.lattice.n_panels
        corners = lattice.panel_corner_position.to_numpy()[:n]
        offsets = lattice.wake_offset.to_numpy()[:n]
        old_bound = self._transported_bound.to_numpy()[:n]
        sources, strip_map, n_strips = self._near_wake_row(corners, offsets, old_bound)
        matrix, old_velocity = self._near_wake_row_influence(
            sources,
            lattice.collocation_point.to_numpy()[:n],
            lattice.normal.to_numpy()[:n],
            n_strips,
        )
        return matrix, old_velocity, strip_map

    def _augment_near_wake_particles(self) -> None:
        """Include the actual newborn particle row in the circulation solve."""
        matrix, old_velocity, strip_by_panel = self._near_wake_particle_influence()
        n = self.lattice.n_panels
        influence = self.lattice.aerodynamic_influence_coefficient.to_numpy()
        influence[:n, :n] += matrix[:, strip_by_panel]
        self.lattice.aerodynamic_influence_coefficient.from_numpy(influence)
        rhs = self.lattice.right_hand_side.to_numpy()
        rhs[:n] -= old_velocity
        self.lattice.right_hand_side.from_numpy(rhs)

    def _resolve_coupling_reference_velocity(
        self, freestream_velocity: tuple[float, float, float]
    ) -> np.ndarray:
        """Resolve the reference velocity used by coupled advance."""
        if (
            self.freestream_velocity is not None
            and np.linalg.norm(self.freestream_velocity) > 1e-10
        ):
            return self.freestream_velocity
        background_velocity = np.array(freestream_velocity)
        if np.linalg.norm(background_velocity) > 1e-10:
            return background_velocity
        kinematic_velocity = self._get_active_kinematic_velocity()
        kinematic_speed = np.linalg.norm(kinematic_velocity)
        return -kinematic_velocity if kinematic_speed > 1e-10 else np.array([1.0, 0.0, 0.0])

    def advance_coupled(
        self,
        particles: "Particles",
        physics: "PhysicsBase",
        freestream_velocity: tuple[float, float, float],
        time_step_size: float,
        time: float | None = None,
    ) -> None:
        """Complete the VLM solve and wake row at the newly accepted particle clock.

        The VPM stepper has already transported the old wake. Move the geometry,
        construct the local near-wake offsets, solve for circulation, and deposit
        the completed trailing/starting row once. Refresh force targets against
        that accepted wake. Newborn particles are transported on the next step.
        """
        if time is None:
            raise ValueError("Coupled VLM advance requires the accepted physical time")
        if not self._mesh_generated:
            self.generate_mesh()

        n_panels = self.lattice.n_panels

        if not self._bound_transport_ready:
            # The initial empty wake skips RK; retain the old bound vector
            # before moving the body even when no particles were transported.
            self._initialize_bound_transport()
            self._bound_transport_ready = True
        self.advance_time(time_step_size, current_time=time)

        reference_velocity = self._resolve_coupling_reference_velocity(freestream_velocity)
        self._last_reference_velocity = reference_velocity

        # The old wake has already moved; new particles are shed after this solve.
        physics.compute_target_velocity(
            particles,
            self.lattice.collocation_point,
            self.lattice.external_velocity,
            include_freestream=True,
        )

        self._prepare_near_wake(time_step_size, physics, particles)
        self.solve(external_velocity=None, coupled=True)

        self._compute_wake_particles()
        n_particles_shed = self.lattice.n_wake_particles[None]
        if n_particles_shed > 0:
            added = particles.add_vortex_particles_from_fields_grouped(
                n_particles_shed,
                self.lattice.wake_position,
                self.lattice.wake_velocity,
                self.lattice.wake_vortex_strength,
                self.lattice.wake_core_radius,
                self.lattice.wake_volume,
                self.lattice.wake_group_id,
                kinematic_viscosity=self.kinematic_viscosity,
            )
            if not added:
                raise ValueError(
                    f"VLM wake insertion exceeds particle capacity: {particles.n_particles_total}"
                    f" + {n_particles_shed} > {particles.capacity}. Increase max_n_particles."
                )

        self._bound_transport_ready = False
        physics.compute_target_velocity(
            particles,
            self.lattice.collocation_point,
            self.lattice.external_velocity,
            include_freestream=True,
        )
        external_velocity = self.lattice.external_velocity.to_numpy()[:n_panels]
        bound_external_velocity = physics.compute_target_velocity(
            particles,
            self.lattice.bound_vortex_midpoint.to_numpy()[:n_panels],
            include_freestream=True,
        )
        self.compute_postprocess(
            external_velocity,
            reference_velocity,
            self.density,
            time_step_size=time_step_size,
            coupled=True,
            bound_external_velocity=bound_external_velocity,
        )
        self._last_forces = self.compute_forces(self.density, self._last_reference_velocity)
