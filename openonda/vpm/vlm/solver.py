"""
Vortex-lattice-method solver (VLMSolver): assembles and solves the panel
circulation system and evaluates forces.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import copy
from typing import TYPE_CHECKING, Any, Literal

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
from openonda.core.log_style import StepLog
from openonda.core.metadata import metadata_value
from openonda.core.profiling import Stopwatch

from ..config.constants import VLM_SMALL_VELOCITY
from ..kernels import make_vortex_kernel
from ..runtime.backend import active_precision
from .config import VLMConfig, VLMSurface
from .filament import (
    CirculationSolver,
    accumulate_bound_transport,
    add_induced_velocity_at_targets,
    add_stage_rates_with_bound_exchange,
    compute_aerodynamic_influence_coefficient_matrix,
    compute_coupled_right_hand_side,
    initialize_bound_transport,
)
from .geometry import Aircraft, Wing
from .geometry import load_surface as _load_surface
from .lattice import VLMLattice
from .loads import (
    compute_bound_linear_impulse,
    compute_forces,
    compute_per_surface_forces,
    compute_postprocess,
    compute_total_bound_vortex_strength,
    panel_kinematic_velocity,
    resolve_reference_velocity,
)
from .mesh import (
    generate_vlm_mesh,
    update_trailing_directions_local,
    update_trailing_edge_directions,
)
from .wake import augment_near_wake_particles, prepare_near_wake, shed_wake_particles

if TYPE_CHECKING:
    from ..particles.container import Particles
    from ..physics.base import PhysicsBase
    from ..physics.induction.base import StageRates, StageState

EPSILON = VLM_SMALL_VELOCITY

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

    def __init__(self, config: VLMConfig, logger: StepLog | None = None):
        """Initialize from one complete, immutable VLM definition.

        ``logger`` receives the solver's records and warnings; without it the
        solver reports nothing.
        """
        # Solver configuration
        self.config = config
        self.logger = logger or StepLog.disabled()

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
        self._surface_placements: dict[str, RigidPose] = {}
        self._surface_group_ids: dict[str, int] = {}
        # Contiguous lattice panels of each surface, set by generate_mesh.
        self._surface_panels: dict[str, slice] = {}

        for surface in config.surfaces:
            self._register_surface(surface)

        # The lattice holds exactly the panels of the declared surfaces.
        self.max_n_panels = self.aircraft.total_n_panels()
        self.dtype: Literal["f32", "f64"]

        self.logger.section("VLM initialized", ("panels", self.max_n_panels))

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

            self.dtype = active_precision()
            ti_dtype = ti.f32 if self.dtype == "f32" else ti.f64
            self._lattice = VLMLattice(self.max_n_panels, ti_dtype)
            self._transported_bound = ti.Vector.field(3, ti_dtype, shape=self.max_n_panels)
            self._bound_exchange_rate = ti.Vector.field(3, ti_dtype, shape=self.max_n_panels)
            self._bound_transport_ready = False
            self._lattice_initialized = True

    def generate_mesh(self) -> None:
        """Generate the VLM mesh using the distribution declared in ``config``.

        The mesh is generated once; later calls return at once, so every entry
        point that needs it calls this method.
        """
        self._ensure_lattice_initialized()

        if self._mesh_generated:
            return

        mesh_clock = Stopwatch()

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

        total_area = float(self.lattice.area.to_numpy()[: self.lattice.n_panels].sum())
        self.logger.section(
            "VLM mesh generated",
            ("panels", self.lattice.n_panels),
            ("wings", len(self.aircraft.wings)),
            ("area", total_area, "m²"),
            ("elapsed", mesh_clock.seconds, "s"),
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
        for surface_name, placement in self._surface_placements.items():
            if not placement.is_identity():
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
        self._surface_placements[surface_name] = surface.placement()

        self._update_combined_aircraft()
        self.logger.section(
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

    def update_trailing_directions(self, transport_velocity: np.ndarray) -> None:
        """
        Update trailing edge directions based on transport velocity.

        The wake trails in the direction of the local relative velocity.

        Args:
           transport_velocity: Velocity vector/field for wake transport (N, 3) or (3,)
                        Ideally this is external_velocity - V_skinematic at trailing edge.
        """
        self.generate_mesh()

        # If scalar vector provided, apply uniformly
        if transport_velocity.ndim == 1:
            update_trailing_edge_directions(self.lattice, transport_velocity)
        else:
            # Use local velocity field
            update_trailing_directions_local(self.lattice, transport_velocity)

    def advance_time(self, time_step_size: float, current_time: float) -> None:
        """Advance the surface poses to ``current_time`` and map the geometry through them."""
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
                kinematic_velocity[panels] = panel_kinematic_velocity(
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
        characteristic_speed = float(np.linalg.norm(background)) + float(
            np.linalg.norm(self._get_active_kinematic_velocity())
        )
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
            self.logger.warning(
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
        solve_clock = Stopwatch()
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
        velocity_upload_seconds = solve_clock.lap()
        self.update_trailing_directions(relative_velocity)
        trailing_direction_seconds = solve_clock.lap()
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
        influence_matrix_seconds = solve_clock.lap()
        compute_coupled_right_hand_side(
            self.lattice.collocation_point,
            self.lattice.normal,
            self.lattice.external_velocity,
            self.lattice.kinematic_velocity,
            self.lattice.right_hand_side,
            self.lattice.n_panels,
        )
        right_hand_side_seconds = solve_clock.lap()
        # The unsteady load differences against the circulation of the previous step.
        self.lattice.save_old_circulation()
        if coupled:
            augment_near_wake_particles(self)
        self._circulation_solver.solve(
            self.lattice.aerodynamic_influence_coefficient,
            self.lattice.right_hand_side,
            self.lattice.circulation,
            n_panels,
        )
        circulation_np = self.lattice.circulation.to_numpy()[:n_panels]
        linear_solve_seconds = solve_clock.lap()
        total_time = (
            velocity_upload_seconds
            + trailing_direction_seconds
            + influence_matrix_seconds
            + right_hand_side_seconds
            + linear_solve_seconds
        )
        if total_time > 1.0:
            self.logger.section(
                "VLM solve timing",
                ("velocity upload", velocity_upload_seconds, "s"),
                ("trailing directions", trailing_direction_seconds, "s"),
                ("influence matrix", influence_matrix_seconds, "s"),
                ("right-hand side", right_hand_side_seconds, "s"),
                ("linear solve", linear_solve_seconds, "s"),
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
            kinematic[panels] = panel_kinematic_velocity(
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
        vortex_points, corners, *_ = self._ensure_stage_geometry_fields(float(stage_time))
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
        vortex_points, corners, *_ = self._ensure_stage_geometry_fields(float(stage_time))
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

    # Implicit starting-vortex aerodynamic_influence_coefficient augmentation

    def compute_postprocess(
        self,
        external_velocity: np.ndarray,
        reference_velocity: np.ndarray,
        density: float,
        time_step_size: float | None = None,
        coupled: bool = False,
        bound_external_velocity: np.ndarray | None = None,
    ) -> None:
        """Compute velocities and panel loads; see :func:`loads.compute_postprocess`."""
        compute_postprocess(
            self,
            external_velocity,
            reference_velocity,
            density,
            time_step_size,
            coupled,
            bound_external_velocity,
        )

    def compute_forces(
        self,
        density: float,
        reference_velocity: np.ndarray | None = None,
        reference_area: float | None = None,
        reference_chord: float | None = None,
        reference_span: float | None = None,
    ) -> dict[str, float | np.ndarray]:
        """Return the integrated forces and moments; see :func:`loads.compute_forces`."""
        return compute_forces(
            self, density, reference_velocity, reference_area, reference_chord, reference_span
        )

    def compute_per_surface_forces(
        self, density: float, reference_velocity: np.ndarray | None = None
    ) -> dict[str, dict[str, float]]:
        """Return the forces of each surface; see :func:`loads.compute_per_surface_forces`."""
        return compute_per_surface_forces(self, density, reference_velocity)

    def compute_total_bound_vortex_strength(self) -> np.ndarray:
        """Return the integrated bound vector strength [m³/s]."""
        return compute_total_bound_vortex_strength(self)

    def compute_bound_linear_impulse(self) -> np.ndarray:
        """Return the bound linear impulse per unit density [m⁴/s]."""
        return compute_bound_linear_impulse(self)

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
        self.generate_mesh()

        n_panels = self.lattice.n_panels

        if not self._bound_transport_ready:
            # The initial empty wake skips RK; retain the old bound vector
            # before moving the body even when no particles were transported.
            self._initialize_bound_transport()
            self._bound_transport_ready = True
        self.advance_time(time_step_size, current_time=time)

        reference_velocity = resolve_reference_velocity(
            self, background=np.array(freestream_velocity)
        )
        self._last_reference_velocity = reference_velocity

        # The old wake has already moved; new particles are shed after this solve.
        physics.compute_target_velocity(
            particles,
            self.lattice.collocation_point,
            self.lattice.external_velocity,
            include_freestream=True,
        )

        prepare_near_wake(self, time_step_size, physics, particles)
        self.solve(external_velocity=None, coupled=True)

        shed_wake_particles(self)
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
