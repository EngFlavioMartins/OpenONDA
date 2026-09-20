"""
Vortex-lattice-method solver (VLMSolver): assembles and solves the panel
circulation system and evaluates forces.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from contextlib import contextmanager
import copy
import time

import numpy as np
from numpy.typing import ArrayLike
import taichi as ti

from ....config.constants import VLM_EPSILON, VLM_SMALL_VELOCITY
from ....io.logging import Logging
from ....io.manifest import _manifest_value
from ....kernels import make_vortex_kernel
from ..config import VLMSetup, VLMSurfaceSetup
from ..coupling.kinematics import RotatingVLM, StaticVLM
from ..geometry.aircraft import Aircraft, Wing
from ..geometry.surface_io import load_surface as _load_surface
from ..kernels.virtual_wake import make_virtual_wake_kernel
from .field import BoundSurfaceFieldContract
from .influence import (
    accumulate_bound_transport,
    add_induced_velocity_and_gradient_at_targets,
    add_induced_velocity_at_targets,
    add_stage_rates_with_bound_exchange,
    apply_circulation_smoothing,
    compute_aerodynamic_influence_coefficient_matrix,
    compute_coupled_right_hand_side,
    compute_induced_velocities,
    compute_induced_velocities_at_bound,
    compute_panel_force_coupled,
    compute_pressure_coefficients,
    initialize_bound_transport,
)
from .kernels import shed_wake_particles_kernel
from .lattice import VLMLattice
from .linear_solvers import get_linear_solver
from .mesh import (
    generate_vlm_mesh,
    update_trailing_directions_local,
    update_trailing_edge_directions,
)
from .unsteady import add_unsteady_pressure_loads

EPSILON = VLM_SMALL_VELOCITY


class VLMSolver:
    """
    Vortex Lattice Method solver with VPM coupling support.

    Solves for circulation distribution on thin lifting surfaces using
    horseshoe vortex elements and the zero-normal-flow boundary condition.

    **Linear Solver Options:**

    - ``'SCIPY'``: CPU direct solver, selected by default below 1000 panels
    - ``'BICGSTAB_GPU'``: GPU iterative solver for non-symmetric systems

    Geometry, kinematics, transforms, mesh spacing, and fluid data are all
    declared in :class:`VLMSetup`. The runtime solver does not expose
    configuration mutation methods.
    """

    def __init__(self, setup: VLMSetup):
        """Initialize from one complete, immutable VLM definition."""
        # Solver configuration
        self.setup = setup
        self.dtype = setup.dtype
        self.epsilon = VLM_EPSILON
        self.circulation_relaxation = setup.circulation_relaxation

        # Multi-body storage: Dict[uid -> (Aircraft, kinematics)]
        self.surfaces = {}

        # Combined aircraft for mesh generation (populated by generate_mesh)
        self.aircraft = None
        self.surface = None

        # Lattice will be created when first needed (after Taichi init)
        self.lattice = None
        self._lattice_initialized = False
        self._current_time = None  # Temporary: set during advance operations from parent solver

        # Flight condition / Fluid properties
        self.freestream_velocity = (
            None
            if setup.freestream_velocity is None
            else np.array(setup.freestream_velocity, dtype=np.float64)
        )
        self.logging_interval_steps = setup.logging_interval_steps
        self.density = setup.density
        self.kinematic_viscosity = setup.kinematic_viscosity
        self.sigma_factor = setup.sigma_factor
        self.wake_core_overlap = setup.wake_core_overlap
        # The present source representation uses one point/trace radius for
        # the bound operator.  It is named separately from particle radii in
        # the field contract so a future distributed sheet can change it
        # deliberately without changing the transport rule.
        self.bound_source_radius = float(VLM_EPSILON)
        self.boundary_response = setup.boundary_response
        self.field_contract = BoundSurfaceFieldContract(
            numerical_epsilon=float(VLM_EPSILON),
            bound_source_radius=self.bound_source_radius,
            near_wake_policy=(
                "affine_row_history_transport_v2"
                if setup.boundary_response == "responsive"
                else "partial_newborn_row_stage_responsive"
            ),
        )
        self._wake_kernel = make_vortex_kernel("GAUSSIAN")
        self.alpha_rad = 0.0
        self.beta_rad = 0.0

        # Transverse shedding threshold
        # Minimum |ΔΓ| for emitting a transverse (closure) wake particle.
        # Default 0.0 means always emit, which prevents binary wake-topology
        # changes near steady state.  Set to a finite value (e.g. 1e-3) to
        # reproduce the threshold artefact as a diagnostic comparison.
        self.transverse_shedding_threshold: float = 0.0

        # solution state
        self._mesh_generated = False
        self._aerodynamic_influence_coefficient_computed = False
        self._solved = False
        self._coupled_mode = False
        self._linear_solver_instance = None
        self._stage_circulation = None
        self._stage_external_velocity = None
        self._stage_kinematic_velocity = None
        self._stage_influence = None
        self._stage_rhs = None
        self._last_stage_boundary_residual = 0.0
        self._last_stage_near_wake_elapsed = 0.0
        self._last_stage_near_wake_matrix_norm = 0.0
        self._stage_geometry_fields = None
        self._stage_geometry_numpy = None
        self._stage_response_active = False

        # Force evaluation configuration
        self.force = setup.force

        # Explicit initialization of optional attributes to avoid AttributeError
        self._surface_transforms: dict[str, dict] = {}
        self._surface_group_ids: dict[str, int] = {}
        self._surface_sampling: dict[str, bool] = {}
        self.sample_surface_forces = setup.sample_surface_forces

        for surface_setup in setup.surfaces:
            self._register_surface(surface_setup)

        n_panels = self.aircraft.total_n_panels()
        self.max_n_panels = setup.max_n_panels if setup.max_n_panels is not None else n_panels
        if self.max_n_panels < n_panels:
            raise ValueError(
                f"VLM max_n_panels={self.max_n_panels} is smaller than the "
                f"{n_panels} panels declared by the surfaces"
            )
        self.linear_solver = setup.linear_solver
        if self.linear_solver is None:
            self.linear_solver = "SCIPY" if n_panels < 1000 else "BICGSTAB_GPU"

        Logging.record(
            "VLM initialized",
            ("panel capacity", self.max_n_panels),
            ("precision", self.dtype),
            ("linear solver", self.linear_solver),
        )

    def _ensure_lattice_initialized(self) -> None:
        """Ensure lattice is created (lazy initialization after Taichi init).

        VPM is required to be the Taichi master: the VPM Solver must call
        ``initialize_taichi_backend`` (i.e. ``Solver.__init__``) *before* any
        VLM lattice fields are allocated.  This prevents a mismatch between the
        precision that Taichi was initialised with and the dtype requested here.
        """
        if not self._lattice_initialized:
            # Guard: Taichi must already be initialised by the VPM Solver.
            if ti.lang.impl.get_runtime().prog is None:
                raise RuntimeError(
                    "VLMSolver._ensure_lattice_initialized called before Taichi "
                    "is initialised.  Always construct the VPM Solver first so "
                    "that initialize_taichi_backend() runs before any VLM fields "
                    "are created."
                )

            # Warn if the requested VLM dtype does not match the Taichi runtime
            # default floating-point type.  Mismatches cause silent precision
            # conversions (f32 ↔ f64) that can degrade accuracy.
            runtime_fp = ti.lang.impl.get_runtime().default_fp
            vlm_fp = ti.f32 if self.dtype == "f32" else ti.f64
            if runtime_fp != vlm_fp:
                runtime_name = "f32" if runtime_fp == ti.f32 else "f64"
                Logging.warning(
                    f"VLMSolver dtype='{self.dtype}' does not match the Taichi "
                    f"runtime precision '{runtime_name}' set by the VPM Solver.  "
                    f"Overriding VLM dtype to '{runtime_name}' to keep VPM as the "
                    f"precision master and avoid f32/f64 conversion errors."
                )
                vlm_fp = runtime_fp
                self.dtype = runtime_name

            ti_dtype = vlm_fp
            self.lattice = VLMLattice(self.max_n_panels, ti_dtype)
            self._transported_bound = ti.Vector.field(3, ti_dtype, shape=self.max_n_panels)
            self._bound_exchange_rate = ti.Vector.field(3, ti_dtype, shape=self.max_n_panels)
            self._bound_transport_ready = False
            self._lattice_initialized = True

    def generate_mesh(self) -> None:
        """Generate the VLM mesh using the distribution declared in ``setup``."""
        self._ensure_lattice_initialized()

        if self._mesh_generated:
            return

        t0 = time.time()

        generate_vlm_mesh(
            self.aircraft,
            self.lattice,
            spanwise_spacing=self.setup.mesh.spacing,
            spanwise_spacing_ratio=self.setup.mesh.ratio,
            spanwise_spacing_region=self.setup.mesh.region,
        )

        # Apply per-surface transformations AFTER mesh generation
        self._apply_surface_transforms()

        # Populate panel group IDs
        group_id = self.get_panel_group_ids()

        # Pad group_id to match max_n_panels for Taichi field assignment
        if group_id.size < self.max_n_panels:
            padded_ids = np.zeros(self.max_n_panels, dtype=np.int32)
            padded_ids[: group_id.size] = group_id
            self.lattice.group_id.from_numpy(padded_ids)
        else:
            self.lattice.group_id.from_numpy(group_id)

        from .restart import restart_identity

        self._restart_geometry_references = _manifest_value(self.aircraft.refs)
        self._restart_identity = restart_identity(self)
        from .restart import restart_physics_identity

        self._restart_physics_identity = restart_physics_identity(self)
        self._mesh_generated = True
        self._aerodynamic_influence_coefficient_computed = False
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

    def _build_wing_panel_ranges(self) -> dict[str, tuple[int, int]]:
        """Build mapping from wing UID to panel range indices."""
        panel_idx = 0
        wing_panel_ranges = {}

        for wing_uid, wing in self.aircraft.wings.items():
            n_panels_wing = 0
            for seg in wing.segments.values():
                n_panels_wing += seg.n_chordwise_panels * seg.n_spanwise_panels
                if wing.symmetry > 0:
                    n_panels_wing += seg.n_chordwise_panels * seg.n_spanwise_panels
            wing_panel_ranges[wing_uid] = (panel_idx, panel_idx + n_panels_wing)
            panel_idx += n_panels_wing

        return wing_panel_ranges

    def _transform_panel_points(
        self,
        panel_corner_position,
        vortex_point_position,
        collocation_point,
        bound_vortex_midpoint,
        normal,
        start_idx,
        end_idx,
        rotation_matrix,
        translation,
        rotation_centre,
    ) -> None:
        """Apply transformation to a range of panels (vectorized)."""
        # Transform panel_corner_position and vortex points (shape: [n_panels, 4, 3])
        for j in range(4):
            # Vectorized transformation
            panel_corner_offset = panel_corner_position[start_idx:end_idx, j] - rotation_centre
            panel_corner_position[start_idx:end_idx, j] = (
                (rotation_matrix @ panel_corner_offset.T).T + rotation_centre + translation
            )

            vortex_point_offset = vortex_point_position[start_idx:end_idx, j] - rotation_centre
            vortex_point_position[start_idx:end_idx, j] = (
                (rotation_matrix @ vortex_point_offset.T).T + rotation_centre + translation
            )

        # Transform collocation_point and bound midpoints
        collocation_offset = collocation_point[start_idx:end_idx] - rotation_centre
        collocation_point[start_idx:end_idx] = (
            (rotation_matrix @ collocation_offset.T).T + rotation_centre + translation
        )

        bound_vortex_offset = bound_vortex_midpoint[start_idx:end_idx] - rotation_centre
        bound_vortex_midpoint[start_idx:end_idx] = (
            (rotation_matrix @ bound_vortex_offset.T).T + rotation_centre + translation
        )

        # Rotate normal (no translation)
        normal[start_idx:end_idx] = (rotation_matrix @ normal[start_idx:end_idx].T).T

    def _write_lattice_arrays(
        self,
        n_panels,
        panel_corner_position,
        vortex_point_position,
        collocation_point,
        bound_vortex_midpoint,
        normal,
    ) -> None:
        """Write numpy arrays back to GPU lattice."""
        for field, values in (
            (self.lattice.panel_corner_position, panel_corner_position),
            (self.lattice.vortex_point_position, vortex_point_position),
            (self.lattice.collocation_point, collocation_point),
            (self.lattice.bound_vortex_midpoint, bound_vortex_midpoint),
            (self.lattice.normal, normal),
        ):
            padded = field.to_numpy()
            padded[:n_panels] = values
            field.from_numpy(padded)

    def _apply_transform_to_surface(
        self,
        surface_name,
        transform,
        wing_panel_ranges,
        panel_corner_position,
        vortex_point_position,
        collocation_point,
        bound_vortex_midpoint,
        normal,
    ) -> None:
        """Apply a single surface transform to matching wings."""
        if transform["rotation_degrees"] is None and transform["translation"] is None:
            return

        rotation_matrix = self._build_rotation_matrix(transform["rotation_degrees"])
        translation = (
            np.array(transform["translation"])
            if transform["translation"] is not None
            else np.zeros(3)
        )
        rotation_centre = (
            np.array(transform["rotation_centre"])
            if transform["rotation_centre"] is not None
            else np.zeros(3)
        )

        # Find which wings belong to this surface and transform them
        for wing_uid, (start_idx, end_idx) in wing_panel_ranges.items():
            if len(self.surfaces) == 1 or wing_uid.startswith(surface_name + "_"):
                self._transform_panel_points(
                    panel_corner_position,
                    vortex_point_position,
                    collocation_point,
                    bound_vortex_midpoint,
                    normal,
                    start_idx,
                    end_idx,
                    rotation_matrix,
                    translation,
                    rotation_centre,
                )

    def _apply_surface_transforms(self) -> None:
        """
        Apply stored transformations to lattice points per-surface.

        This is called after mesh generation to position each surface correctly.
        """
        if not any(
            value["rotation_degrees"] is not None or value["translation"] is not None
            for value in self._surface_transforms.values()
        ):
            return

        wing_panel_ranges = self._build_wing_panel_ranges()

        # Get lattice arrays (must be numpy for modification)
        n_panels = self.lattice.n_panels
        panel_corner_position = self.lattice.panel_corner_position.to_numpy()[:n_panels]
        vortex_point_position = self.lattice.vortex_point_position.to_numpy()[:n_panels]
        collocation_point = self.lattice.collocation_point.to_numpy()[:n_panels]
        bound_vortex_midpoint = self.lattice.bound_vortex_midpoint.to_numpy()[:n_panels]
        normal = self.lattice.normal.to_numpy()[:n_panels]

        # Apply each surface transformation
        for surface_name, transform in self._surface_transforms.items():
            self._apply_transform_to_surface(
                surface_name,
                transform,
                wing_panel_ranges,
                panel_corner_position,
                vortex_point_position,
                collocation_point,
                bound_vortex_midpoint,
                normal,
            )

        self._write_lattice_arrays(
            n_panels,
            panel_corner_position,
            vortex_point_position,
            collocation_point,
            bound_vortex_midpoint,
            normal,
        )

    def _build_rotation_matrix(self, rotation_degrees) -> np.ndarray:
        """Build 3x3 rotation matrix from [rx, ry, rz] angles in degrees."""
        rotation_matrix = np.eye(3)
        if rotation_degrees is None:
            return rotation_matrix

        rotation_degrees = np.array(rotation_degrees, dtype=np.float64)

        # Rotation about X
        if abs(rotation_degrees[0]) > 1e-10:
            rotation_x_radians = np.radians(rotation_degrees[0])
            rotation_matrix_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(rotation_x_radians), -np.sin(rotation_x_radians)],
                    [0, np.sin(rotation_x_radians), np.cos(rotation_x_radians)],
                ]
            )
            rotation_matrix = rotation_matrix_x @ rotation_matrix

        # Rotation about Y
        if abs(rotation_degrees[1]) > 1e-10:
            rotation_y_radians = np.radians(rotation_degrees[1])
            rotation_matrix_y = np.array(
                [
                    [np.cos(rotation_y_radians), 0, np.sin(rotation_y_radians)],
                    [0, 1, 0],
                    [-np.sin(rotation_y_radians), 0, np.cos(rotation_y_radians)],
                ]
            )
            rotation_matrix = rotation_matrix_y @ rotation_matrix

        # Rotation about Z
        if abs(rotation_degrees[2]) > 1e-10:
            rotation_z_radians = np.radians(rotation_degrees[2])
            rotation_matrix_z = np.array(
                [
                    [np.cos(rotation_z_radians), -np.sin(rotation_z_radians), 0],
                    [np.sin(rotation_z_radians), np.cos(rotation_z_radians), 0],
                    [0, 0, 1],
                ]
            )
            rotation_matrix = rotation_matrix_z @ rotation_matrix

        return rotation_matrix

    def _register_surface(self, setup: VLMSurfaceSetup) -> str:
        """Load one declared surface before any Taichi fields are allocated."""
        aircraft = (
            _load_surface(setup.surface)
            if isinstance(setup.surface, str)
            else copy.deepcopy(setup.surface)
        )
        surface_name = setup.name or aircraft.uid
        if surface_name in self.surfaces:
            raise ValueError(f"Duplicate VLM surface name: {surface_name}")
        kinematics = (
            copy.deepcopy(setup.kinematics) if setup.kinematics is not None else StaticVLM()
        )

        self.surfaces[surface_name] = (aircraft, kinematics)
        self._surface_group_ids[surface_name] = setup.group_id
        self._surface_sampling[surface_name] = (
            self.sample_surface_forces if setup.sample_forces is None else setup.sample_forces
        )
        self._surface_transforms[surface_name] = {
            "translation": setup.translation,
            "rotation_degrees": setup.rotation_degrees,
            "rotation_centre": setup.rotation_centre,
        }

        self._update_combined_aircraft()
        Logging.record(
            "VLM surface",
            ("name", surface_name),
            ("panels", aircraft.total_n_panels()),
            ("group", setup.group_id),
        )
        return surface_name

    def get_panel_group_ids(self) -> np.ndarray:
        """
        Get array of group IDs for each panel in the lattice.

        Returns:
             np.ndarray: Array of shape (n_panels,) with group ID for each panel.
        """
        if self.lattice is None:
            return np.zeros(0, dtype=np.int32)

        n_panels = self.lattice.n_panels
        group_id = np.zeros(n_panels, dtype=np.int32)

        if not self._surface_group_ids:
            return group_id

        wing_ranges = self._build_wing_panel_ranges()

        # Map panels to surfaces
        # This relies on the convention that combined wings are named "{surface}_{wing}"
        # or exactly "{surface}" if single wing.

        for surface_name, gid in self._surface_group_ids.items():
            # Find all wings belonging to this surface
            for wing_uid, (start, end) in wing_ranges.items():
                # Check for match.
                # Wing UID in combined aircraft is constructed as f"{name}_{wing_uid}" in _update_combined_aircraft
                # But careful: if user names surface "wing" and original wing is "wing_0", combined is "wing_wing_0".

                # More robust check:
                # If single surface mode: keys are just original wing UIDs.
                if len(self.surfaces) == 1:
                    # All panels belong to the single surface
                    group_id[start:end] = gid
                else:
                    # Multi-surface mode: keys are f"{surface_name}_{original_wing_uid}"
                    if wing_uid.startswith(f"{surface_name}_"):
                        group_id[start:end] = gid

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
        if not self.surfaces:
            self.aircraft = None
            self.surface = None
            return

        if len(self.surfaces) == 1:
            # Single surface mode
            name, (aircraft, _) = next(iter(self.surfaces.items()))
            self.aircraft = aircraft
            self.surface = aircraft
        else:
            # Multi-surface: merge into combined aircraft
            combined = Aircraft(uid="combined")
            for name, (aircraft, _) in self.surfaces.items():
                for wing_uid, wing in aircraft.wings.items():
                    # Create unique wing name
                    combined_uid = f"{name}_{wing_uid}"
                    # Copy wing with new uid
                    new_wing = Wing(
                        uid=combined_uid, segments=wing.segments.copy(), symmetry=wing.symmetry
                    )
                    combined.wings[combined_uid] = new_wing
            combined.compute_default_refs()
            self.aircraft = combined
            self.surface = combined

    @property
    def kinematics(self):
        """Get kinematics for single-surface mode."""
        if len(self.surfaces) == 1:
            _, (_, kin) = next(iter(self.surfaces.items()))
            return kin
        return None

    @kinematics.setter
    def kinematics(self, value):
        """Set kinematics for first surface."""
        if self.surfaces:
            first_name = next(iter(self.surfaces.keys()))
            aircraft, _ = self.surfaces[first_name]
            self.surfaces[first_name] = (aircraft, value)

    def _compute_rotor_tip_speed(self, surface_name: str, kinematics) -> tuple[np.ndarray, float]:
        """Return (velocity_vector, tip_speed) for a RotatingVLM surface, or (zeros, 0)."""
        if not isinstance(kinematics, RotatingVLM) or not self._mesh_generated:
            return np.zeros(3), 0.0
        angular_speed_magnitude = abs(kinematics.angular_speed)
        if angular_speed_magnitude < 1e-10:
            return np.zeros(3), 0.0
        rotation_centre = kinematics.rotation_centre
        rotation_axis = getattr(kinematics, "axis", np.array([0.0, 0.0, 1.0]))
        collocation_point = self.lattice.get_collocation_points()
        wing_panel_ranges = self._build_wing_panel_ranges()
        surface_prefix = f"{surface_name}_" if len(self.surfaces) > 1 else ""
        maximum_tip_speed = 0.0
        for wing_uid, (panel_start, panel_end) in wing_panel_ranges.items():
            if not (
                len(self.surfaces) == 1
                or wing_uid.startswith(surface_prefix)
                or wing_uid == surface_name
            ):
                continue
            radial_position = collocation_point[panel_start:panel_end] - rotation_centre
            axial_position = np.outer(radial_position @ rotation_axis, rotation_axis)
            radial_position -= axial_position
            max_radial_distance = (
                np.max(np.linalg.norm(radial_position, axis=1)) if len(radial_position) > 0 else 0.0
            )
            maximum_tip_speed = max(
                maximum_tip_speed, angular_speed_magnitude * max_radial_distance
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
            if kinematics is None or isinstance(kinematics, StaticVLM):
                continue
            translational_velocity = kinematics.get_velocity(current_time)
            translational_speed = np.linalg.norm(translational_velocity)
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

    def _get_active_kinematics(self):
        """
        Get the active kinematics object.

        For multi-surface, returns the first non-StaticVLM kinematics.
        """
        if len(self.surfaces) == 1:
            return self.kinematics

        # Multi-surface mode
        for _name, (_, kinematics) in self.surfaces.items():
            if kinematics is not None and not isinstance(kinematics, StaticVLM):
                return kinematics

        return None

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

        self._aerodynamic_influence_coefficient_computed = False

    def _find_surface_panel_range(
        self, surface_name: str, wing_ranges: dict, n_panels: int
    ) -> tuple[int, int] | None:
        """Return (start, end) panel indices for a surface, or None if not found."""
        if len(self.surfaces) == 1:
            return 0, n_panels
        prefix = f"{surface_name}_"
        min_start, max_end, found = n_panels, 0, False
        for w_uid, (w_start, w_end) in wing_ranges.items():
            if w_uid.startswith(prefix) or w_uid == surface_name:
                found = True
                min_start = min(min_start, w_start)
                max_end = max(max_end, w_end)
        return (min_start, max_end) if found else None

    def _compute_panel_kinematic_velocity(
        self, kinematics, time: float, collocation_point, start_idx: int, end_idx: int
    ) -> np.ndarray:
        """Return (end-start, 3) kinematic velocity array for a panel slice."""
        angular_velocity = kinematics.get_angular_velocity(time)
        translational_velocity = kinematics.get_velocity(time)
        rotation_centre = getattr(kinematics, "rotation_centre", np.zeros(3))
        radial_position = collocation_point[start_idx:end_idx] - rotation_centre
        return translational_velocity + np.cross(angular_velocity, radial_position)

    def advance_time(self, time_step_size: float, current_time: float) -> None:
        """Advance kinematics state and geometry for all surfaces."""
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
        has_kinematics = False
        wing_ranges = self._build_wing_panel_ranges()
        for surface_name, (_aircraft_obj, kinematics) in self.surfaces.items():
            if kinematics is None or isinstance(kinematics, StaticVLM):
                continue
            has_kinematics = True
            panel_range = self._find_surface_panel_range(surface_name, wing_ranges, n_panels)
            if panel_range is None:
                continue
            start_idx, end_idx = panel_range
            kinematics.update(
                self, time - time_step_size, time_step_size, panel_range=(start_idx, end_idx)
            )
            collocation_point = self.lattice.get_collocation_points()
            kinematic_velocity[start_idx:end_idx] = self._compute_panel_kinematic_velocity(
                kinematics, time, collocation_point, start_idx, end_idx
            )
        self.lattice.set_kinematic_velocity(kinematic_velocity)
        if has_kinematics:
            self._aerodynamic_influence_coefficient_computed = False
            self._solved = False

    def compute_aerodynamic_influence_coefficient_matrix(self, force: bool = False) -> None:
        """Compute aerodynamic_influence_coefficient matrix."""
        if self._aerodynamic_influence_coefficient_computed and not force:
            return

        if not self._mesh_generated:
            self.generate_mesh()

        compute_aerodynamic_influence_coefficient_matrix(
            self.lattice.collocation_point,
            self.lattice.vortex_point_position,
            self.lattice.panel_corner_position,
            self.lattice.normal,
            self.lattice.is_trailing_edge,
            self.lattice.trailing_edge_index,
            self.lattice.aerodynamic_influence_coefficient,
            self.lattice.n_panels,
            self.bound_source_radius,
            0,  # coupled_mode = 0 (standalone)
        )
        self._aerodynamic_influence_coefficient_computed = True

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

    def _run_linear_solver(self, n_panels: int) -> np.ndarray:
        """Solve aerodynamic_influence_coefficient@circulation=right_hand_side and return circulation numpy array."""
        if self._linear_solver_instance is None:
            self._linear_solver_instance = get_linear_solver(
                self.linear_solver, max_n_panels=self.max_n_panels, use_preconditioner=True
            )
        solver = self._linear_solver_instance
        if solver.is_gpu:
            # 1e-10 is pathologically tight for iterative solvers; 1e-6 is
            # sufficient for VLM engineering accuracy and avoids hundreds of
            # kernel-launch-bound iterations on small systems.
            solver.solve(
                self.lattice.aerodynamic_influence_coefficient,
                self.lattice.right_hand_side,
                self.lattice.circulation,
                self.lattice.n_panels,
                max_iterations=1000,
                tolerance=1e-6,
            )
        else:
            solver.solve(
                self.lattice.aerodynamic_influence_coefficient,
                self.lattice.right_hand_side,
                self.lattice.circulation,
                self.lattice.n_panels,
            )
        if self.circulation_relaxation < 1.0:
            self.lattice.apply_relaxation(self.circulation_relaxation)
        return self.lattice.circulation.to_numpy()[:n_panels]

    def solve(
        self,
        external_velocity: np.ndarray | None = None,
        time_step_size: float | None = None,
        save_old: bool = True,
        coupled: bool = False,
    ) -> np.ndarray:
        """
        Solve VLM system for circulation.

        Args:
            external_velocity: Total external velocity field at collocation_point points (N x 3).
                       If None, assumes data is already in lattice.external_velocity (GPU-resident).
            time_step_size: Time step size (optional, used for wake shedding updates if needed)
            save_old: If True (default), saves current circulation to circulation_old
                     before solving. Set to False for re-solves within the same step.
            coupled: If True, uses bound-only aerodynamic_influence_coefficient for coupling with VPM particles.

        Returns:
            Computed circulation (circulation) array
        """
        # A native accepted solve starts a new coupling epoch; never let a
        # temporary responsive-stage circulation leak into later evaluations.
        self._stage_response_active = False
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
        if self.lattice.has_kinematic_velocity():
            relative_velocity = external_velocity_array - self.lattice.get_kinematic_velocity()
        else:
            relative_velocity = external_velocity_array
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
            self.bound_source_radius,
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
        if save_old:
            self.lattice.save_old_circulation()
            self._vlm_step_count = getattr(self, "_vlm_step_count", 0) + 1
        if coupled and save_old:
            self._augment_near_wake_particles()
        circulation_np = self._run_linear_solver(n_panels)
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

        The arrays are copies of the last accepted geometry.  Rigid motion is
        integrated only for the requested stage interval; no accepted lattice
        coordinate, motion history, or clock is changed.  The same helper is
        used by stage boundary solves and returns the same moving geometry for
        every stage query.
        """
        reference_time = self._current_time if self._current_time is not None else 0.0
        duration = float(stage_time) - float(reference_time)
        arrays = [
            self.lattice.vortex_point_position.to_numpy().copy(),
            self.lattice.panel_corner_position.to_numpy().copy(),
            self.lattice.collocation_point.to_numpy().copy(),
            self.lattice.normal.to_numpy().copy(),
        ]
        kinematic = self.lattice.kinematic_velocity.to_numpy().copy()
        moving = any(not isinstance(motion, StaticVLM) for _, motion in self.surfaces.values())
        if abs(duration) < 1.0e-14 and not moving:
            return (*arrays, kinematic)

        ranges = self._build_wing_panel_ranges()
        for name, (_, motion) in self.surfaces.items():
            panel_range = self._find_surface_panel_range(name, ranges, self.lattice.n_panels)
            if panel_range is None:
                continue
            start, stop = panel_range
            if not isinstance(motion, StaticVLM) and abs(duration) >= 1.0e-14:
                midpoint = reference_time + 0.5 * duration
                translation = np.asarray(motion.get_velocity(midpoint), dtype=np.float64) * duration
                omega = np.asarray(motion.get_angular_velocity(midpoint), dtype=np.float64)
                angle = float(np.linalg.norm(omega) * duration)
                if isinstance(motion, RotatingVLM):
                    signed_angle = motion.rotation_angle(stage_time) - motion.rotation_angle(
                        reference_time
                    )
                    omega = np.asarray(motion.axis, dtype=np.float64) * np.sign(signed_angle)
                    angle = abs(float(signed_angle))
                rotation = np.eye(3)
                if abs(angle) > 1.0e-14 and np.linalg.norm(omega) > 1.0e-14:
                    axis = omega / np.linalg.norm(omega)
                    x, y, z = axis
                    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
                    rotation += np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
                centre = np.asarray(getattr(motion, "rotation_centre", np.zeros(3)))
                for array in arrays[:3]:
                    array[start:stop] = (
                        (array[start:stop] - centre) @ rotation.T + centre + translation
                    )
                arrays[3][start:stop] = (rotation @ arrays[3][start:stop].T).T

            if isinstance(motion, StaticVLM) or motion is None:
                continue
            kinematic[start:stop] = self._compute_panel_kinematic_velocity(
                motion,
                float(stage_time),
                arrays[2],
                start,
                stop,
            )
        return (*arrays, kinematic)

    def _ensure_stage_geometry_fields(self, stage_time: float):
        """Upload one pure stage geometry state and return its device fields."""
        if getattr(self, "_stage_geometry_time", None) == stage_time:
            return self._stage_geometry_fields
        if self._stage_geometry_fields is None:
            self._stage_geometry_fields = (
                ti.Vector.field(3, self.lattice.dtype, shape=(self.max_n_panels, 4)),
                ti.Vector.field(3, self.lattice.dtype, shape=(self.max_n_panels, 4)),
                ti.Vector.field(3, self.lattice.dtype, shape=self.max_n_panels),
                ti.Vector.field(3, self.lattice.dtype, shape=self.max_n_panels),
                ti.field(dtype=self.lattice.dtype, shape=(self.max_n_panels, 3)),
            )
        arrays = self._stage_geometry_arrays(stage_time)
        np_dtype = self.lattice.np_dtype
        for field, array in zip(self._stage_geometry_fields, arrays, strict=True):
            field.from_numpy(np.asarray(array, dtype=np_dtype))
        self._stage_geometry_numpy = tuple(np.asarray(array, dtype=np_dtype) for array in arrays)
        self._stage_geometry_time = float(stage_time)
        return self._stage_geometry_fields

    def _stage_geometry(self, stage_time):
        """Return pure stage source geometry while holding circulation fixed."""
        fields = self._ensure_stage_geometry_fields(float(stage_time))
        return fields[0], fields[1]

    def _ensure_stage_workspace(self) -> None:
        """Allocate temporary boundary-system fields used by responsive stages."""
        if self._stage_circulation is not None:
            return
        dtype = self.lattice.dtype
        self._stage_circulation = ti.field(dtype=dtype, shape=self.max_n_panels)
        self._stage_external_velocity = ti.Vector.field(3, dtype=dtype, shape=self.max_n_panels)
        self._stage_kinematic_velocity = ti.field(dtype=dtype, shape=(self.max_n_panels, 3))
        self._stage_influence = ti.field(dtype=dtype, shape=(self.max_n_panels, self.max_n_panels))
        self._stage_rhs = ti.field(dtype=dtype, shape=self.max_n_panels)

    def solve_stage_boundary(self, stage_state, stage_time: float, physics, particles) -> None:
        """Solve a pure temporary VLM response for one particle RK stage.

        The stage response uses stage particles, prescribed stage geometry,
        and an affine partial newborn row including its old closing vector.
        Stage reaction uses the same RK stage coefficients as the particles.
        The virtual row contributes velocity and stretching but is never
        inserted into the accepted cloud. Only the accepted solve publishes
        the completed row and circulation history.
        """
        self._stage_response_active = False
        self._stage_wake_sources = []
        # Production RK stages carry an explicit identity.  Field-only
        # diagnostic/health refreshes construct a ``StageState`` with
        # ``stage_index=None``; those queries still solve
        # the temporary boundary system but must not erase the telemetry from
        # the accepted step's last real RK stage.
        record_stage_telemetry = getattr(stage_state, "stage_index", None) is not None
        if record_stage_telemetry:
            self._last_stage_near_wake_elapsed = 0.0
            self._last_stage_near_wake_matrix_norm = 0.0
        if self.boundary_response != "responsive" or stage_state.count <= 0:
            return
        self._ensure_stage_workspace()
        geometry = self._ensure_stage_geometry_fields(float(stage_time))
        vortex_points, corners, collocation, normal, kinematic = geometry
        n = self.lattice.n_panels
        collocation_np = collocation.to_numpy()[:n].astype(np.float64)
        incident = physics.compute_target_velocity(
            particles,
            collocation_np,
            include_freestream=True,
        )
        if incident is None:
            incident = np.zeros((n, 3), dtype=np.float64)
        incident = np.asarray(incident, dtype=np.float64)
        external = np.zeros((self.max_n_panels, 3), dtype=self.lattice.np_dtype)
        external[:n] = incident.astype(self.lattice.np_dtype)
        self._stage_external_velocity.from_numpy(external)
        self._stage_kinematic_velocity.from_numpy(
            np.asarray(kinematic.to_numpy(), dtype=self.lattice.np_dtype)
        )
        compute_aerodynamic_influence_coefficient_matrix(
            collocation,
            vortex_points,
            corners,
            normal,
            self.lattice.is_trailing_edge,
            self.lattice.trailing_edge_index,
            self._stage_influence,
            n,
            self.bound_source_radius,
            coupled_mode=1,
        )
        compute_coupled_right_hand_side(
            collocation,
            normal,
            self._stage_external_velocity,
            self._stage_kinematic_velocity,
            self._stage_rhs,
            n,
        )
        accepted_time = float(self._current_time if self._current_time is not None else stage_time)
        elapsed = max(float(stage_time) - accepted_time, 0.0)
        if elapsed > 1.0e-14:
            # The temporary matrix contains the partial newborn row for this
            # stage.  Its coefficients depend on the actual stage-particle
            # incident velocity; no accepted wake field is inserted or
            # overwritten.  This is the explicit coupled-substep temporal
            # formulation used by ``boundary_response='responsive'``.
            partial_matrix, old_velocity = self._near_wake_stage_influence(
                geometry,
                particles,
                physics,
                elapsed,
            )
            stage_influence = self._stage_influence.to_numpy()
            stage_influence[:n, :n] += partial_matrix
            self._stage_influence.from_numpy(stage_influence)
            rhs = self._stage_rhs.to_numpy()
            rhs[:n] -= old_velocity
            self._stage_rhs.from_numpy(rhs)
            if record_stage_telemetry:
                self._last_stage_near_wake_elapsed = elapsed
                self._last_stage_near_wake_matrix_norm = float(
                    np.linalg.norm(partial_matrix, ord=np.inf)
                )
        if self._linear_solver_instance is None:
            self._linear_solver_instance = get_linear_solver(
                self.linear_solver,
                max_n_panels=self.max_n_panels,
                use_preconditioner=True,
            )
        if self._linear_solver_instance.is_gpu:
            self._linear_solver_instance.solve(
                self._stage_influence,
                self._stage_rhs,
                self._stage_circulation,
                n,
                max_iterations=1000,
                tolerance=1.0e-6,
            )
        else:
            self._linear_solver_instance.solve(
                self._stage_influence,
                self._stage_rhs,
                self._stage_circulation,
                n,
            )
        self._stage_response_active = True
        matrix = self._stage_influence.to_numpy()[:n, :n]
        rhs = self._stage_rhs.to_numpy()[:n]
        trial_gamma = self._stage_circulation.to_numpy()[:n]
        if record_stage_telemetry:
            self._last_stage_boundary_residual = float(
                np.linalg.norm(matrix @ trial_gamma - rhs, ord=np.inf) if n else 0.0
            )

    def add_stage_velocity(
        self, target_position, target_velocity, count: int, stage_time: float
    ) -> None:
        """Accumulate the latest solved VLM velocity at temporary VPM targets.

        ``stage_time`` is accepted explicitly at the stage boundary. The
        current VLM coupling is lagged to the latest accepted-step solve, so
        solved circulation is held fixed; prescribed source motion and target
        positions are evaluated at the requested RK stage time.
        """
        if not self._solved or self.lattice is None or count <= 0:
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
        )

    def add_stage_velocity_and_gradient(
        self,
        target_position,
        target_velocity,
        target_gradient,
        count: int,
        stage_time: float,
        *,
        target_core_radius,
    ) -> None:
        """Accumulate stage velocity and the Jacobian of the same VLM field."""
        if not self._solved or self.lattice is None or count <= 0:
            return
        vortex_points, corners = self._stage_geometry(stage_time)
        add_induced_velocity_and_gradient_at_targets(
            target_position,
            target_velocity,
            target_gradient,
            target_core_radius,
            vortex_points,
            corners,
            self.lattice.trailing_edge_index,
            self.lattice.circulation,
            int(count),
            int(self.lattice.n_panels),
            int(self._coupled_mode),
        )

    def _initialize_bound_transport(self):
        """Seed each trailing strip's integrated bound strength before RK exchange."""
        initialize_bound_transport(
            self.lattice.panel_corner_position,
            self.lattice.cumulative_circulation,
            self.lattice.is_trailing_edge,
            self._transported_bound,
            self.lattice.n_panels,
        )

    @contextmanager
    def particle_transport_step(self, tableau=None, time_step_size=0.0):
        """Publish strip exchange only after all particle RK stages succeed.

        This ledger is consumed by the next accepted wake emission. It is not
        persistent restart state: checkpoints are written after that emission.
        """
        transported_before = self._transported_bound.to_numpy().copy()
        self._bound_transport_ready = False
        self._initialize_bound_transport()
        self._transport_tableau = tableau
        self._transport_dt = float(time_step_size)
        self._stage_exchange_history = {}
        self._stage_bound_initial = self._transported_bound.to_numpy().copy()
        try:
            yield
        except BaseException:
            # A rejected trial must not publish a partially accumulated
            # reaction ledger.  Restore the pre-trial accepted exchange and
            # leave the next accepted step responsible for reinitialisation.
            self._transported_bound.from_numpy(transported_before)
            self._bound_exchange_rate.fill(0.0)
            self._bound_transport_ready = False
            self._stage_response_active = False
            raise
        else:
            self._bound_transport_ready = True
        finally:
            self._transport_tableau = None
            self._stage_exchange_history = {}

    def add_stage_rates(self, state, stage_time, rates, mode, weighted_dt):
        """Accumulate the bound field and its stage-weighted strip reaction."""
        if (not self._solved and not self._stage_response_active) or state.count <= 0:
            return
        vortex_points, corners = self._stage_geometry(stage_time)
        lattice = self.lattice
        circulation = (
            self._stage_circulation if self._stage_response_active else lattice.circulation
        )
        coupled_mode = int(self._coupled_mode or self._stage_response_active)
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
            circulation,
            self._bound_exchange_rate,
            state.count,
            lattice.n_panels,
            coupled_mode,
            mode,
            weighted_dt,
            rates.velocity_gradient is not None,
        )
        if self._stage_response_active and self._stage_wake_sources:
            if not hasattr(self, "_virtual_wake_kernel"):
                self._virtual_wake_kernel = make_virtual_wake_kernel(
                    self._wake_kernel.name, lattice.dtype
                )
            sources = self._stage_wake_sources
            gamma = self._stage_circulation.to_numpy()[: lattice.n_panels]
            strip_gamma = np.bincount(self._stage_wake_strip_map, weights=gamma)
            positions = np.ascontiguousarray(
                [source[0] for source in sources], dtype=lattice.np_dtype
            )
            strengths = np.ascontiguousarray(
                [strip_gamma @ source[2] + source[3] for source in sources], dtype=lattice.np_dtype
            )
            radii = np.ascontiguousarray([source[1] for source in sources], dtype=lattice.np_dtype)
            owners = np.ascontiguousarray([source[4] for source in sources], dtype=np.int32)
            self._virtual_wake_kernel(
                state.position,
                state.vortex_strength,
                state.core_radius,
                rates.velocity,
                rates.vortex_strength_rate,
                rates.velocity_gradient,
                positions,
                strengths,
                radii,
                owners,
                self._bound_exchange_rate,
                state.count,
                len(sources),
                mode,
                int(weighted_dt != 0.0),
                rates.velocity_gradient is not None,
            )
        if weighted_dt != 0.0 and getattr(self, "_transport_tableau", None) is not None:
            stage_index = getattr(state, "stage_index", None)
            if stage_index is not None:
                self._stage_exchange_history[int(stage_index)] = (
                    self._bound_exchange_rate.to_numpy().copy()
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

        reference_velocity_magnitude = np.linalg.norm(reference_velocity)
        if reference_velocity_magnitude < 1e-10:
            reference_velocity_magnitude = 1.0

        # 0. Keep cumulative circulation current for wake diagnostics.
        self._compute_cumulative_circulation_cpu()

        # 1. Compute velocity at collocation points for the pressure coefficient.
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
        )
        velocity = self.lattice.velocity.to_numpy()[: self.lattice.n_panels]
        kinematic_velocity = self.lattice.get_kinematic_velocity()
        relative_velocity = np.zeros((self.max_n_panels, 3), dtype=self.lattice.np_dtype)
        relative_velocity[: self.lattice.n_panels] = velocity - kinematic_velocity
        self.lattice.relative_velocity.from_numpy(relative_velocity)

        # 2. Compute pressure coefficients (using collocation_point velocity)
        compute_pressure_coefficients(
            self.lattice.velocity,
            self.lattice.pressure_coefficient,
            self.lattice.n_panels,
            float(reference_velocity_magnitude**2),
        )

        # 3. Compute panel forces
        # apply_kutta_joukowski_smoothing=1 in coupled mode cancels the 2Δt oscillation introduced
        # by the explicit VPM-VLM coupling (γ alternates every step because
        # the near-field particle geometry alternates).
        apply_kutta_joukowski_smoothing = (
            1 if (coupled and getattr(self.force, "kj_smoothing", True)) else 0
        )

        # 4. Compute velocity at BOUND VORTEX midpoints (for Forces)
        # When kj_smoothing is active, V_bound must use the same smoothed circulation
        # as the force kernel (0.5*(γ + γ_old)); otherwise the 2Δt oscillation
        # in the raw circulation propagates through V_bound into the KJ forces even
        # though the force kernel smooths circulation itself.
        if apply_kutta_joukowski_smoothing:
            apply_circulation_smoothing(
                self.lattice.circulation,
                self.lattice.circulation_old,
                self.lattice.smoothed_circulation,
                self.lattice.n_panels,
            )
            bound_circulation_for_velocity = self.lattice.smoothed_circulation
        else:
            bound_circulation_for_velocity = self.lattice.circulation
        incident_at_bound = (
            external_velocity if bound_external_velocity is None else bound_external_velocity
        )
        padded = np.zeros((self.max_n_panels, 3), dtype=self.lattice.np_dtype)
        padded[: self.lattice.n_panels] = incident_at_bound
        self.lattice.bound_external_velocity.from_numpy(padded)
        bound_kinematics = np.zeros_like(padded)
        wing_ranges = self._build_wing_panel_ranges()
        points = self.lattice.bound_vortex_midpoint.to_numpy()[: self.lattice.n_panels]
        for name, (_, motion) in self.surfaces.items():
            start, end = self._find_surface_panel_range(name, wing_ranges, self.lattice.n_panels)
            bound_kinematics[start:end] = self._compute_panel_kinematic_velocity(
                motion,
                self._current_time or 0.0,
                points,
                start,
                end,
            )
        self.lattice.bound_kinematic_velocity.from_numpy(bound_kinematics)
        self.lattice.reference_speed = float(reference_velocity_magnitude)
        compute_induced_velocities_at_bound(
            self.lattice.n_panels,
            self.lattice.bound_vortex_midpoint,
            self.lattice.vortex_point_position,
            self.lattice.panel_corner_position,
            self.lattice.trailing_edge_index,
            bound_circulation_for_velocity,
            self.lattice.bound_vortex_velocity,
            self.lattice.bound_external_velocity,
            1 if coupled else 0,
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
            self.lattice.circulation_old,
            self.lattice.bound_kinematic_velocity,
            self.lattice.panel_force,
            self.lattice.n_panels,
            density,
            apply_kutta_joukowski_smoothing,
        )
        self.lattice.unsteady_panel_force.fill(0.0)
        self.lattice.panel_moment_correction.fill(0.0)
        self.lattice.unsteady_pressure_jump_coefficient.fill(0.0)
        if self.force.unsteady:
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
        kinematics = self._get_active_kinematics()
        current_position = np.zeros(3)
        current_orientation = np.eye(3)
        if kinematics is not None and hasattr(kinematics, "current_position"):
            current_position = kinematics.current_position
            if hasattr(kinematics, "current_orientation"):
                current_orientation = kinematics.current_orientation
        local_reference_point = np.array(self.aircraft.refs.get("reference_point", [0.0, 0.0, 0.0]))
        reference_point = current_position + current_orientation @ local_reference_point
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
        )  # type: ignore[return-value]

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
    ) -> dict[str, float]:
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
    ) -> dict[str, float]:
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
        panel_force = self.lattice.get_forces() * (
            density / getattr(self, "_force_density", self.density)
        )
        total_force = np.sum(panel_force, axis=0)
        force_x, force_y, force_z = total_force
        reference_velocity = self._resolve_reference_velocity(
            reference_velocity, self.lattice.n_panels
        )
        reference_velocity_magnitude = np.linalg.norm(reference_velocity)
        lift, drag, side_force = self._decompose_wind_axes(
            total_force, reference_velocity_magnitude, reference_velocity
        )
        moment, quarter_chord_moment, reference_point = self._compute_force_moments(
            panel_force,
            reference_chord,
            self.lattice.panel_moment_correction.to_numpy()[: self.lattice.n_panels]
            * (density / getattr(self, "_force_density", self.density)),
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
        unsteady *= density / getattr(self, "_force_density", self.density)
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
        wing_panel_ranges: dict[str, tuple[int, int]],
        reference_direction: np.ndarray,
        force_normalization: float,
        panel_moment_correction: np.ndarray,
    ) -> dict[str, float]:
        """Compute forces, moment about the current pivot, and fluid-on-body power."""
        indices = np.concatenate(
            [
                np.arange(start, stop)
                for wing, (start, stop) in wing_panel_ranges.items()
                if len(self.surfaces) == 1 or wing.startswith(surface_name + "_")
            ]
        )
        _, motion = self.surfaces[surface_name]
        time = self._current_time if self._current_time is not None else 0.0
        centre = np.asarray(getattr(motion, "rotation_centre", motion.current_position))
        positions = self.lattice.bound_vortex_midpoint.to_numpy()[indices]
        forces = panel_force[indices]
        panel_centers = self.lattice.panel_corner_position.to_numpy()[indices].mean(axis=1)
        areas = self.lattice.area.to_numpy()[indices]
        centroid = np.average(panel_centers, axis=0, weights=areas)
        surface_force = forces.sum(axis=0)
        moment = (np.cross(positions - centre, forces) + panel_moment_correction[indices]).sum(
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
            "panel_count": len(indices),
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
            ("position", motion.current_position),
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
        reference_area: float | None = None,
        reference_chord: float | None = None,
        reference_span: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute forces for each individual surface.

        All coefficients are normalized by dynamic pressure and reference area:
        lift_coefficient = L / (0.5 * rho * reference_area * reference_velocity²)

        Args:
            density: Fluid density (kg/m³)
            reference_velocity: Reference velocity vector [ux, uy, uz]
            reference_area: Reference area (m²). If None, uses aircraft defaults.
            reference_chord: Reference chord (m). If None, uses aircraft defaults.
            reference_span: Reference span (m). If None, uses aircraft defaults.

        Returns:
            Dictionary mapping surface name to force dictionary
        """
        if not self._solved:
            return {}
        panel_force = self.lattice.get_forces() * (
            density / getattr(self, "_force_density", self.density)
        )
        reference_velocity = self._resolve_reference_velocity(
            reference_velocity, self.lattice.n_panels
        )
        reference_velocity_magnitude = np.linalg.norm(reference_velocity)
        wing_panel_ranges = self._build_wing_panel_ranges()
        dynamic_pressure = 0.5 * density * reference_velocity_magnitude**2
        force_normalization = dynamic_pressure * (
            reference_area
            if reference_area is not None
            else float(self.aircraft.refs.get("area", 1.0))
        )
        reference_direction = (
            reference_velocity / reference_velocity_magnitude
            if reference_velocity_magnitude > 1e-10
            else np.array([1.0, 0.0, 0.0])
        )
        moment_correction = self.lattice.panel_moment_correction.to_numpy()[: self.lattice.n_panels]
        moment_correction *= density / getattr(self, "_force_density", self.density)
        return {
            name: self._compute_one_surface_forces(
                name,
                panel_force,
                wing_panel_ranges,
                reference_direction,
                force_normalization,
                moment_correction,
            )
            for name in self.surfaces
        }

    def log_forces_table(
        self, density: float, reference_velocity: np.ndarray | None = None
    ) -> dict[str, float | np.ndarray]:
        """Compute boundary loads and report them through the owning logger.

        Parameters
        ----------
        density : float
            Fluid density in kg/m³.
        reference_velocity : numpy.ndarray or None
            Wind-axis reference vector, shape (3,), in m/s. None uses the
            owning force model's reference velocity.

        Returns
        -------
        dict[str, float or numpy.ndarray]
            Native scalar coefficients and force/reference vectors. Loads are
            in N, moments in N m, and reference positions in m. Coefficients
            are dimensionless.
        """
        totals = self.compute_forces(density, reference_velocity)
        surfaces = self.compute_per_surface_forces(density, reference_velocity)
        Logging.boundary_forces("VLM", totals, surfaces)
        return totals

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

    def advance(
        self,
        time_step_size: float,
        external_velocity: np.ndarray,
        density: float = 1.0,
        reference_velocity: np.ndarray | None = None,
        logging_interval_steps: int | None = None,
        step: int = 0,
        time: float | None = None,
    ) -> dict[str, np.ndarray] | None:
        """
        Advance VLM-VPM coupled simulation by one time step.

        Args:
            time_step_size: Time step size (s)
            external_velocity: Total external velocity field at collocation_point points (N, 3)
            density: Fluid density
            reference_velocity: Reference velocity vector (defaults to auto-computed)
            logging_interval_steps: Print forces every N steps (None=use solver default)
            step: Current time step number
            time: Current simulation time (s) - prevents drift if provided

        Returns:
             Dictionary with new wake particles or None
        """
        if not self._mesh_generated:
            self.generate_mesh()

        # 1. Advance kinematics (handles both single and multi-surface)
        # advance_time() internally handles multi-surface kinematics
        self.advance_time(time_step_size, current_time=time)

        # 2. Solve VLM system
        self.lattice.set_external_velocity(external_velocity)
        self._prepare_near_wake(time_step_size)
        self.solve(external_velocity, time_step_size)

        # Determine reference values if not provided
        if reference_velocity is None:
            kinematic_velocity = self._get_active_kinematic_velocity()
            background_velocity = (
                np.mean(external_velocity, axis=0) if len(external_velocity) > 0 else np.zeros(3)
            )
            reference_velocity = background_velocity - kinematic_velocity
            if np.linalg.norm(reference_velocity) < 1e-10:
                reference_velocity = np.array([1.0, 0.0, 0.0])

        # Compute postprocess (velocity, forces) to enable logging
        self.compute_postprocess(
            external_velocity, reference_velocity, density, time_step_size=time_step_size
        )

        # Automatically compute and cache forces
        self._last_forces = self.compute_forces(density, reference_velocity)
        self._last_reference_velocity = reference_velocity

        # 3. Log forces if requested
        log_freq = (
            self.logging_interval_steps
            if logging_interval_steps is None
            else int(logging_interval_steps)
        )
        if log_freq > 0 and step % log_freq == 0:
            self.log_forces_table(density, reference_velocity)

        return self._compute_wake_particles()

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

    def _prepare_near_wake(self, time_step_size, physics=None, particles=None):
        """Construct one row from previous TE positions convected to the new clock."""
        n = self.lattice.n_panels
        corners = self.lattice.panel_corner_position.to_numpy()[:n]
        old = getattr(self, "_previous_panel_corners", corners)
        indices = np.flatnonzero(self.lattice.is_trailing_edge.to_numpy()[:n])
        points = corners[indices][:, [3, 2]].reshape(-1, 3)
        if physics is None:
            fluid = np.repeat(self.lattice.external_velocity.to_numpy()[indices], 2, axis=0)
        else:
            fluid = physics.compute_target_velocity(particles, points, include_freestream=True)
        fluid = fluid.reshape(-1, 2, 3)
        offset = self.lattice.wake_offset.to_numpy()
        velocity = self.lattice.trailing_edge_velocity.to_numpy()
        offset[indices] = (
            old[indices][:, [3, 2]] - corners[indices][:, [3, 2]] + fluid * time_step_size
        )
        velocity[indices] = fluid
        self.lattice.wake_offset.from_numpy(offset)
        self.lattice.trailing_edge_velocity.from_numpy(velocity)

    def _compute_wake_particles(
        self,
        *,
        reset_buffer: bool = True,
    ) -> dict[str, np.ndarray] | None:
        """Discretize the completed near-wake row at edge midpoints and its far edge."""
        if self.lattice.n_panels == 0:
            return None
        if reset_buffer:
            self.lattice.reset_wake_buffer()
        self._compute_cumulative_circulation_cpu()
        # Antisymmetric signed root circulations cancel to solver roundoff.
        # Do not turn frame-dependent last-bit noise into extra wake particles.
        dtype = np.float32 if self.lattice.dtype == ti.f32 else np.float64
        shed_wake_particles_kernel(
            self.lattice.n_panels,
            self.sigma_factor,
            self.wake_core_overlap if self.wake_core_overlap is not None else 0.0,
            float(self.transverse_shedding_threshold),
            32.0 * np.finfo(dtype).eps,
            self.lattice.cumulative_circulation,
            self.lattice.cumulative_circulation_old,
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
            self._bound_transport_ready,
        )

        n_particles_shed = self.lattice.n_wake_particles[None]
        wake_buffer_capacity = self.lattice.wake_position.shape[0]
        if n_particles_shed > wake_buffer_capacity:
            raise RuntimeError(
                f"VLM wake buffer overflow: {n_particles_shed} particles exceed "
                f"capacity {wake_buffer_capacity}"
            )

        # We return a specific marker to indicate GPU data is ready
        return {"_gpu_transfer_ready": True}

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

    def _near_wake_row(self, corners, offsets, old_bound):
        """Return affine native-row sources in cumulative strip circulation.

        Each source is (position, radius, strip-vector coefficients, constant
        strength, owner TE panel). The same geometry and strengths are used by
        boundary assembly and virtual-row transport. Strength units are m³/s;
        circulation coefficients have length units. A strip's old closing
        vector is kept in the inertial frame, including transported reaction.
        """
        edges, strip_by_panel = self._build_trailing_edge_strip_map()
        n = self.lattice.n_panels
        neighbors = self.lattice.neighbor_indices.to_numpy()[:n]
        mirrored = self.lattice.is_mirrored.to_numpy()[:n]
        sources = []
        for strip, panel in enumerate(edges):
            left, right = corners[panel, 3], corners[panel, 2]
            dl, dr = offsets[panel]
            span = np.linalg.norm(right - left)
            length = 0.5 * (np.linalg.norm(dl) + np.linalg.norm(dr))
            if span <= 1e-12 or length <= 1e-12:
                continue
            left_radius, right_radius = max(np.linalg.norm(dl), span), max(np.linalg.norm(dr), span)
            transverse_radius = max(self.sigma_factor * length, span / 3)
            if self.wake_core_overlap is not None:
                left_radius *= self.wake_core_overlap
                right_radius *= self.wake_core_overlap
                transverse_radius = self.wake_core_overlap * max(length, span)
            left_index, right_index = neighbors[panel, :2]
            shared_root = left_index != -1 and mirrored[panel] != mirrored[left_index]
            if not shared_root or panel < left_index:
                coefficients = np.zeros((len(edges), 3))
                coefficients[strip] -= dl
                if left_index != -1:
                    coefficients[strip_by_panel[left_index]] += -dl if shared_root else dl
                sources.append((left + 0.5 * dl, left_radius, coefficients, np.zeros(3), panel))
            if right_index == -1:
                coefficients = np.zeros((len(edges), 3))
                coefficients[strip] = dr
                sources.append((right + 0.5 * dr, right_radius, coefficients, np.zeros(3), panel))
            far_left, far_right = left + dl, right + dr
            coefficients = np.zeros((len(edges), 3))
            coefficients[strip] = far_left - far_right
            sources.append(
                (
                    0.5 * (far_left + far_right),
                    transverse_radius,
                    coefficients,
                    old_bound[panel],
                    panel,
                )
            )
        return sources, strip_by_panel, len(edges)

    def _near_wake_row_influence(self, sources, targets, normals, n_strips):
        """Project an affine native row onto all receiving collocation normals."""
        matrix, old_velocity = np.zeros((len(targets), n_strips)), np.zeros(len(targets))
        for position, radius, coefficients, constant, _ in sources:
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

    def _near_wake_particle_influence(self):
        """Return the accepted row matrix, old-strength RHS and panel/strip map."""
        lattice, n = self.lattice, self.lattice.n_panels
        corners = lattice.panel_corner_position.to_numpy()[:n]
        offsets = lattice.wake_offset.to_numpy()[:n]
        old_bound = (
            self._transported_bound.to_numpy()[:n]
            if self._bound_transport_ready
            else (
                lattice.cumulative_circulation_old.to_numpy()[:n, None]
                * (corners[:, 2] + offsets[:, 1] - corners[:, 3] - offsets[:, 0])
            )
        )
        sources, strip_map, n_strips = self._near_wake_row(corners, offsets, old_bound)
        matrix, old_velocity = self._near_wake_row_influence(
            sources,
            lattice.collocation_point.to_numpy()[:n],
            lattice.normal.to_numpy()[:n],
            n_strips,
        )
        return matrix, old_velocity, strip_map

    def _near_wake_stage_influence(self, geometry, stage_particles, physics, elapsed: float):
        """Return the virtual row matrix AND its old-strength normal velocity.

        Convect the accepted TE pose to the stage time, then subtract the
        stage TE pose. This includes translation and rotation of the body.
        The row convection is a first-order endpoint approximation, shared
        with accepted emission; a higher-order particle RK tableau does not
        make the complete boundary/birth formulation the same order.
        """
        n = self.lattice.n_panels
        edges, strip_map = self._build_trailing_edge_strip_map()
        self._stage_wake_sources = []
        if elapsed <= 1e-14 or not edges:
            return np.zeros((n, n)), np.zeros(n)
        corners = geometry[1].to_numpy()[:n].astype(np.float64)
        accepted = self.lattice.panel_corner_position.to_numpy()[:n].astype(np.float64)
        edge_points = corners[edges][:, [3, 2]].reshape(-1, 3)
        fluid = physics.compute_target_velocity(
            stage_particles, edge_points, include_freestream=True
        )
        if fluid is None:
            fluid = np.zeros_like(edge_points)
        offsets = np.zeros((n, 2, 3))
        offsets[edges] = (
            accepted[edges][:, [3, 2]]
            - corners[edges][:, [3, 2]]
            + np.asarray(fluid).reshape(-1, 2, 3) * elapsed
        )
        old_bound = self.lattice.cumulative_circulation.to_numpy()[:n, None] * (
            accepted[:, 2] - accepted[:, 3]
        )
        stage_index = getattr(stage_particles, "stage_index", None)
        tableau = getattr(self, "_transport_tableau", None)
        if tableau is not None and stage_index is not None:
            old_bound = self._stage_bound_initial[:n].copy()
            for j, coefficient in enumerate(tableau.a[int(stage_index)]):
                if coefficient:
                    old_bound += (
                        self._transport_dt * coefficient * self._stage_exchange_history[j][:n]
                    )
        sources, strip_map, n_strips = self._near_wake_row(corners, offsets, old_bound)
        self._stage_wake_sources = sources
        self._stage_wake_strip_map = strip_map
        matrix, old_velocity = self._near_wake_row_influence(
            sources,
            geometry[2].to_numpy()[:n].astype(np.float64),
            geometry[3].to_numpy()[:n].astype(np.float64),
            n_strips,
        )
        return matrix[:, strip_map], old_velocity

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

    def _resolve_coupling_reference_velocity(self, config) -> np.ndarray:
        """Resolve the reference velocity used by coupled advance."""
        if (
            hasattr(self, "freestream_velocity")
            and self.freestream_velocity is not None
            and np.linalg.norm(self.freestream_velocity) > 1e-10
        ):
            return self.freestream_velocity
        if hasattr(config, "freestream_velocity") and config.freestream_velocity is not None:
            background_velocity = np.array(config.freestream_velocity)
            if np.linalg.norm(background_velocity) > 1e-10:
                return background_velocity
        kinematic_velocity = self._get_active_kinematic_velocity()
        kinematic_speed = np.linalg.norm(kinematic_velocity)
        return -kinematic_velocity if kinematic_speed > 1e-10 else np.array([1.0, 0.0, 0.0])

    def advance_coupled(
        self,
        particles,
        physics,
        config,
        time_step_size: float,
        step: int,
        time: float | None = None,
        release_wake: bool = True,
    ) -> dict[str, np.ndarray] | None:
        """Complete the VLM solve and wake row at the newly accepted particle clock.

        The VPM stepper has already transported the old wake. Move the geometry,
        construct the local near-wake offsets, solve for circulation, and deposit
        the completed trailing/starting row once. Refresh force targets against
        that accepted wake. Newborn particles are transported on the next step.
        """
        if not self._mesh_generated:
            self.generate_mesh()

        n_panels = self.lattice.n_panels

        # 1. Advance kinematics (move geometry to new position)
        if not self._bound_transport_ready:
            # The initial empty wake skips RK; retain the old bound vector
            # before moving the body even when no particles were transported.
            self._initialize_bound_transport()
            self._bound_transport_ready = True
        self.advance_time(time_step_size, current_time=time)

        # 2. Determine the force normalization velocity
        reference_velocity = self._resolve_coupling_reference_velocity(config)
        self._last_reference_velocity = reference_velocity

        # 3. Compute VPM-induced velocity at collocation_point points.
        #    Particles from previous steps are already convected downstream,
        #    providing spatial separation for the explicit coupling.
        physics.compute_target_velocity(
            particles,
            self.lattice.collocation_point,
            self.lattice.external_velocity,
            include_freestream=True,
        )

        # 4. Solve VLM system (coupled aerodynamic_influence_coefficient — bound horseshoe + near-wake)
        self._prepare_near_wake(time_step_size, physics, particles)
        self.solve(external_velocity=None, time_step_size=time_step_size, coupled=True)

        # 5. Optionally shed the TE near-wake row from the clean post-solve
        # cumulative Γ.
        result = None
        if release_wake:
            self.lattice.reset_wake_buffer()
            result = self._compute_wake_particles(reset_buffer=False)

        # 6. Transfer the completed row to the free VPM wake.
        if release_wake and result and result.get("_gpu_transfer_ready"):
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

        # 7. Post-process forces from the accepted circulation and wake.
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

        return None
