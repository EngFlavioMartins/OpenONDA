"""
Upgraded PanelSolver orchestration.
==================
Main solver class managing PanelLattice, linear solvers, and force computation.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
import json
import logging
import os
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import taichi as ti

from source import log_style

from ..coupling import kinematics as kin_module
from ..coupling.kinematics import BodyPose
from ..kernels.far_field import (
    accumulate_source_panel_velocity_with_far_field_on_field,
    build_far_field_bodies,
    compute_source_panel_velocity_with_far_field,
    far_field_interaction_fraction,
)
from ..kernels.induced_velocity import (
    accumulate_doublet_panel_velocity_on_field,
    accumulate_source_panel_velocity_on_field,
    compute_induced_velocity_kernel,
    compute_source_induced_velocity_kernel,
)
from .influence import (
    build_dirichlet_aerodynamic_influence_coefficient_matrix,
    build_source_aerodynamic_influence_coefficient_matrix,
    compute_dirichlet_right_hand_side_with_sources,
    compute_forces_bernoulli,
    compute_forces_kutta_joukowski,
    compute_pressure_bernoulli,
    compute_relative_surface_velocity,
    compute_surface_velocity_with_sources,
)
from .lattice import PanelLattice
from .linear_solvers import (
    EqualityConstrainedLeastSquaresFactorization,
    PanelBiCGSTABSolver,
    PanelScipySolver,
    constrained_least_squares_metrics,
    default_residual_tolerance,
)
from .mesh import load_and_audit_body_stl, upload_body_to_lattice
from .vtk_export import panel_mesh_to_vtp

logger = logging.getLogger("vpm")

if TYPE_CHECKING:
    from source.solvers.vpm.config.case import Numerics
    from source.solvers.vpm.particles.container import Particles
    from source.solvers.vpm.physics.engine import PhysicsEngine


@dataclass
class ForceConfig:
    """Configuration for aerodynamic force evaluation on panel methods.

    Supports two methodologies with different trade-offs in accuracy,
    unsteadiness handling, and wake-truncation sensitivity.

    **1. Steady Bernoulli (default):**
       Integrates surface pressure from the steady Bernoulli equation:

           pressure_coefficient = 1 - (surface_velocity / freestream_velocity)²

       Pros:
       - Accurate for inviscid, irrotational flow
       - Well-established in panel-method literature

       Cons:
       - Does not include the unsteady ``dphi/dt`` term
       - Sensitive to panel discretisation quality

    **2. Kutta-Joukowski:**
       Applies K-J theorem per panel:  F = ρ Γ × V_local

       Pros:
       - Direct evaluation, no pressure integration
       - Fast and robust for thin lifting surfaces

       Cons:
       - Less accurate for thick bodies
       - Misses unsteady added-mass effects

    Examples
    --------
    .. code-block:: python

        # Bernoulli (default)
        force = ForceConfig.bernoulli()

        # Kutta-Joukowski
        force = ForceConfig.kutta_joukowski()
    """

    method: Literal["BERNOULLI", "KUTTA_JOUKOWSKI"] = "BERNOULLI"

    @classmethod
    def bernoulli(cls):
        return cls(method="BERNOULLI")

    @classmethod
    def kutta_joukowski(cls):
        return cls(method="KUTTA_JOUKOWSKI")


class PanelSolver:
    """Boundary-element panel solver, optionally coupled to a live VPM particle set.

    ``coupling_scope`` is the single authoritative switch for how the panel
    solver participates in a VPM step. Wherever a scope solves the panel, it
    solves from ``particles.n_particles_total`` — every active particle;
    there is no injected/retained distinction anywhere in this solver. The
    scopes differ in *who* solves the panel, and in what is done with the
    result:

    - ``"full"``: :meth:`advance` runs every VPM step from
      :class:`~source.solvers.vpm.coupling.stepper.CouplingStepper`, updating
      kinematics and solving.  For static bodies it also computes steady
      Bernoulli forces and surface pressure; moving-body unsteady forces are
      not implemented. The
      panel's induced velocity is added to every active particle's
      trajectory at every Runge-Kutta stage, via the
      ``physics.body_velocity_field`` hook
      (:meth:`accumulate_induced_velocity_on_field`) installed by
      :class:`~source.solvers.vpm.core.solver.VPMSolver` and applied in
      ``PhysicsEngine._AdvectionHandler._vel`` on top of the self-induced
      Biot-Savart velocity, entirely on device. Because that solve happens at the
      top of the step, an external coupler that replaces the particle cloud
      at fixed physical time must re-solve against the replaced state (see
      :meth:`refresh_coupled_solution`) before the next advection step or
      boundary trace uses it.
    - ``"vpm_boundary_condition"``: :meth:`advance` is skipped entirely by
      ``CouplingStepper.advance_panel``, so this scope never advances
      kinematics, panel history, forces, or wake shedding, and never injects
      velocity into particle trajectories. Its strengths are produced solely
      by :meth:`refresh_coupled_solution`, which an external coupler calls
      before each boundary trace and after each particle replacement. That
      refresh calls :meth:`solve` only: no surface pressure or force is
      computed for this scope. It exists purely to supply the harmonic/body
      correction a coupler traces onto an external solver.
    - ``"normal"`` / ``"pressure"``: :meth:`advance` runs every step and,
      for static bodies, computes forces and surface pressure exactly like
      ``"full"``, but the result never enters particle trajectories and is
      never refreshed by a coupler. One-way, postprocessing-only aerodynamic
      force evaluation; the two names are not currently distinguished from
      each other.

    Examples
    --------
    .. code-block:: python

        # Two-way coupling: panel deflects every particle at every RK stage.
        panel = PanelSolver(coupling_scope="full")

        # One-way: panel only supplies a boundary condition to another solver.
        panel = PanelSolver(coupling_scope="vpm_boundary_condition")
    """

    def __init__(
        self,
        max_n_panels: int = 10000,
        float_dtype: str = "f64",
        linear_solver: Literal["SCIPY", "BICGSTAB_GPU"] = "SCIPY",
        force_config: ForceConfig | None = None,
        boundary_condition_type: Literal["DIRICHLET", "NEUMANN"] = "NEUMANN",
        density: float = 1.225,
        freestream_velocity: np.ndarray | None = None,
        logging_interval_steps: int = 1,
        coupling_scope: Literal["full", "vpm_boundary_condition", "normal", "pressure"] = "full",
        raise_on_non_convergence: bool = True,
        memory_budget_bytes: int = 4 * 1024**3,
        diagnostic_interval_steps: int = 0,
        diagnostic_sample_size: int = 4096,
        residual_tolerance: float | None = None,
        far_field_acceptance: float = 5.0,
        far_field_min_panels: int = 256,
        reuse_constrained_factorization: bool = True,
        collect_timing: bool = False,
    ) -> None:
        self.max_n_panels = max_n_panels
        self.float_dtype = float_dtype
        self.linear_solver_name = linear_solver
        self.force_config = force_config or ForceConfig.bernoulli()
        self.boundary_condition_type = boundary_condition_type
        self.density = density
        self.freestream_velocity = (
            None if freestream_velocity is None else np.array(freestream_velocity, dtype=np.float64)
        )
        self.logging_interval_steps = max(1, int(logging_interval_steps))
        if coupling_scope not in ("full", "vpm_boundary_condition", "normal", "pressure"):
            raise ValueError(
                "coupling_scope must be 'full', 'vpm_boundary_condition', 'normal', or 'pressure'"
            )
        self.coupling_scope = coupling_scope
        self.raise_on_non_convergence = raise_on_non_convergence
        self.memory_budget_bytes = memory_budget_bytes
        # None resolves per working precision at solver construction.
        self.residual_tolerance = residual_tolerance
        if float_dtype not in ("f32", "f64"):
            raise ValueError("float_dtype must be 'f32' or 'f64'")
        if far_field_acceptance <= 0.0:
            raise ValueError("far_field_acceptance must be positive")
        if far_field_min_panels < 1:
            raise ValueError("far_field_min_panels must be at least one")
        self.far_field_acceptance = float(far_field_acceptance)
        self.far_field_min_panels = int(far_field_min_panels)
        self.reuse_constrained_factorization = bool(reuse_constrained_factorization)
        self.collect_timing = bool(collect_timing)
        self.diagnostic_interval_steps = max(0, int(diagnostic_interval_steps))
        self.diagnostic_sample_size = max(1, int(diagnostic_sample_size))
        self.step = 0
        self.refresh_count = 0
        self._current_time = 0.0
        self._solved = False
        self._last_forces: dict[str, float] = {}
        self._last_reference_velocity: np.ndarray | None = None
        self._far_field_bodies = []
        self._last_far_field_fraction = 0.0
        self._geometry_revision = 0
        self._matrix_geometry_revision = -1
        self._constrained_factorization: EqualityConstrainedLeastSquaresFactorization | None = None
        self._factorization_geometry_revision = -1
        self._last_aic_assembly_seconds = 0.0
        self._last_aic_rebuilt = False

        # Lazy initialization state
        self.lattice: PanelLattice | None = None
        self.solver_strategy = None

        # Fields (initialized lazily)
        self.aerodynamic_influence_coefficient = None
        self.right_hand_side = None
        self.panel_force = None
        self.surface_velocity_absolute = None
        self.surface_velocity_relative = None
        # Compatibility alias for callers that historically consumed the
        # absolute/inertial surface velocity.
        self.surface_velocity = None

        self.results = {
            "force_history": [],
            "moment_history": [],
            "time_history": [],
            "diagnostic_history": [],
        }
        self.is_initialized = False

    def _check_memory_budget(self) -> None:
        """Fail fast if the dense influence matrix would exceed the memory budget.

        The influence matrix is ``max_n_panels x max_n_panels`` dense; a body
        with a large panel count silently allocating this before failing
        (or thrashing) is the failure mode this guard replaces.
        """
        itemsize = 4 if self.float_dtype == "f32" else 8
        # A reusable constrained factorization retains null-space, Q, and R
        # arrays in addition to the dense AIC.  Include that steady-state peak
        # in the same fail-fast budget rather than allocating it by surprise.
        dense_matrix_count = (
            4
            if self.boundary_condition_type == "NEUMANN"
            and self.linear_solver_name == "SCIPY"
            and self.reuse_constrained_factorization
            else 1
        )
        required_bytes = itemsize * self.max_n_panels**2 * dense_matrix_count
        if required_bytes <= self.memory_budget_bytes:
            return
        required_gib = required_bytes / 1024**3
        budget_gib = self.memory_budget_bytes / 1024**3
        suggested_max_n_panels = int(
            (self.memory_budget_bytes / (itemsize * dense_matrix_count)) ** 0.5
        )
        raise RuntimeError(
            f"Panel dense solve storage would require {required_gib:.2f} GiB "
            f"(max_n_panels={self.max_n_panels}, dtype={self.float_dtype}), "
            f"exceeding the {budget_gib:.2f} GiB memory_budget_bytes. "
            f"Lower max_n_panels to at most {suggested_max_n_panels}, or raise "
            "memory_budget_bytes if this much memory is actually available."
        )

    def _ensure_initialized(self) -> None:
        """Lazy initialization of GPU fields and sub-solvers after Taichi init."""
        if self.lattice is not None:
            return

        self._check_memory_budget()

        # 1. Create Lattice
        self.lattice = PanelLattice(self.max_n_panels, self.float_dtype)
        ti_dtype = self.lattice.ti_dtype

        # 2. Create GPU fields
        self.aerodynamic_influence_coefficient = ti.field(
            ti_dtype, shape=(self.max_n_panels, self.max_n_panels)
        )
        self.right_hand_side = ti.field(ti_dtype, shape=self.max_n_panels)
        self.panel_force = ti.Vector.field(3, dtype=ti_dtype, shape=self.max_n_panels)
        self.surface_velocity_absolute = ti.Vector.field(3, dtype=ti_dtype, shape=self.max_n_panels)
        self.surface_velocity_relative = ti.Vector.field(3, dtype=ti_dtype, shape=self.max_n_panels)
        self.surface_velocity = self.surface_velocity_absolute

        # 3. Strategy pattern for linear solver
        tolerance = (
            default_residual_tolerance(
                self.float_dtype,
                constrained=self.boundary_condition_type == "NEUMANN",
            )
            if self.residual_tolerance is None
            else self.residual_tolerance
        )
        if self.linear_solver_name == "SCIPY":
            self.solver_strategy = PanelScipySolver(residual_tolerance=tolerance)
        else:
            self.solver_strategy = PanelBiCGSTABSolver(
                self.max_n_panels, ti_dtype, residual_tolerance=tolerance
            )

        print(
            log_style.record(
                "vpm",
                "panel solver initialized",
                ("panels, max", f"{self.max_n_panels:,}"),
                ("precision", str(self.float_dtype)),
                ("linear solver", str(self.linear_solver_name)),
                stamped=True,
            )
        )

    def add_surface(
        self,
        uid: str,
        stl_path: str,
        kinematics: kin_module.PanelKinematics | None = None,
        group_id: int = 0,
        validate: bool = True,
        translation: tuple[float, float, float] | None = None,
        rotation_degrees: tuple[float, float, float] | None = None,
        rotation_centre: tuple[float, float, float] | None = None,
        reference_area: float | None = None,
    ) -> int:
        """Add one closed body from an STL file.

        The STL is loaded, audited, and oriented *before* any Taichi lattice
        or dense influence matrix is allocated, so invalid geometry cannot
        consume GPU memory on its way to being rejected.

        Exactly one connected component is accepted. A multi-component STL is
        rejected rather than uploaded, because this method maps one file to
        one :class:`~.lattice.PanelBody` with one uid and one kinematics
        object: separate shells silently merged into that single body could
        not be moved or identified independently, and nested shells would
        need a cavity-orientation setting that does not exist yet. Add each
        body from its own STL instead.
        """
        if any(body.uid == uid for body in getattr(self.lattice, "bodies", ())):
            raise ValueError(f"Duplicate panel body uid: {uid}")
        vertex_position = load_and_audit_body_stl(
            stl_path,
            validate=validate,
            max_panels=self.max_n_panels,
            expected_components=1,
        )
        vertex_position = self._apply_initial_placement(
            vertex_position, translation, rotation_degrees, rotation_centre
        )
        self._ensure_initialized()
        count = upload_body_to_lattice(
            self.lattice,
            uid,
            vertex_position,
            kinematics=kinematics,
            group_id=group_id,
            reference_area=reference_area,
        )
        # Appending a body changes the dense operator dimensions and requires
        # one complete influence-matrix rebuild before the next solve.
        self._geometry_revision += 1
        self.is_initialized = False
        self._solved = False
        self._far_field_bodies = []
        self._constrained_factorization = None
        self._factorization_geometry_revision = -1
        return count

    @staticmethod
    def _build_rotation_matrix(rotation_degrees) -> np.ndarray:
        """Build the XYZ placement rotation used by VLM scene metadata."""
        angles = (
            np.zeros(3, dtype=np.float64)
            if rotation_degrees is None
            else np.asarray(rotation_degrees, dtype=np.float64)
        )
        if angles.shape != (3,):
            raise ValueError("rotation_degrees must contain three coordinates")
        rotation = np.eye(3, dtype=np.float64)
        for axis, angle in enumerate(np.radians(angles)):
            if abs(angle) <= 1.0e-15:
                continue
            c, s = np.cos(angle), np.sin(angle)
            if axis == 0:
                elementary = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            elif axis == 1:
                elementary = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
            else:
                elementary = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            rotation = elementary @ rotation
        return rotation

    @classmethod
    def _apply_initial_placement(
        cls,
        vertex_position: np.ndarray,
        translation: tuple[float, float, float] | None,
        rotation_degrees: tuple[float, float, float] | None,
        rotation_centre: tuple[float, float, float] | None,
    ) -> np.ndarray:
        """Bake a declarative scene placement into the body's local geometry."""
        if translation is None and rotation_degrees is None:
            return vertex_position
        translation_vec = (
            np.zeros(3, dtype=np.float64)
            if translation is None
            else np.asarray(translation, dtype=np.float64)
        )
        centre = (
            np.zeros(3, dtype=np.float64)
            if rotation_centre is None
            else np.asarray(rotation_centre, dtype=np.float64)
        )
        if translation_vec.shape != (3,) or centre.shape != (3,):
            raise ValueError("translation and rotation_centre must contain three coordinates")
        rotation = cls._build_rotation_matrix(rotation_degrees)
        return (vertex_position - centre) @ rotation.T + centre + translation_vec

    def load_scene(self, layout_file: str) -> None:
        """
        Load a complete scene (assembly of bodies) from a JSON layout file.

        The layout file should define a list of bodies with their properties::

            {
                "bodies": [
                    {
                        "uid": "wing",
                        "stl_path": "path/to/wing.stl",
                        "translation": [x, y, z],
                        "rotation_degrees": [rx, ry, rz],
                        "rotation_centre": [cx, cy, cz],
                        "group_id": 0,
                        "kinematics": {
                            "type": "TranslatingPanel",
                            "velocity": [10.0, 0.0, 0.0]
                        }
                    }
                ]
            }

        Kinematics type strings match the class names in ``panels.coupling.kinematics``:
        ``StaticPanel``, ``TranslatingPanel``, ``RotatingPanel``, ``PitchingPanel``,
        ``HeavingPanel``, ``ManeuverPanel``, ``CompositePanel``, and
        ``RampedRotatingPanel``.
        """
        self._ensure_initialized()
        with open(layout_file) as f:
            data = json.load(f)

        base_dir = os.path.dirname(layout_file)

        for body_data in data.get("bodies", []):
            uid = body_data.get("uid", "body")
            stl_file = body_data.get("stl_path", body_data.get("file", ""))
            if not os.path.isabs(stl_file):
                stl_file = os.path.join(base_dir, stl_file)

            group_id = body_data.get("group_id", 0)

            kin_data = body_data.get("kinematics", {"type": "StaticPanel"})
            kin_type = kin_data.get("type")
            try:
                kin_cls = getattr(kin_module, kin_type)
            except AttributeError:
                print(f"Warning: Unknown kinematics type '{kin_type}', defaulting to StaticPanel")
                kin_cls = kin_module.StaticPanel

            kin_kwargs = {k: v for k, v in kin_data.items() if k != "type"}
            for k, v in kin_kwargs.items():
                if isinstance(v, list):
                    kin_kwargs[k] = np.array(v)

            kinematics = kin_cls(**kin_kwargs) if kin_kwargs else kin_cls()

            self.add_surface(
                uid=uid,
                stl_path=stl_file,
                kinematics=kinematics,
                group_id=group_id,
                translation=body_data.get("translation"),
                rotation_degrees=body_data.get("rotation_degrees"),
                rotation_centre=body_data.get("rotation_centre"),
                reference_area=body_data.get("reference_area"),
            )

    def save_results(self, filename: str, time: float = 0.0) -> None:
        """
        Save panel results to VTK file.

        Args:
            filename: Output filename (without extension)
            time: Physical simulation time stored in the output field.
        """
        self._ensure_initialized()
        n = self.lattice.n_panels
        if n == 0:
            return

        vertex_position = self.lattice.vertex_position.to_numpy()[:n]
        panel_centre = self.lattice.panel_centre.to_numpy()[:n]
        normal = self.lattice.normal.to_numpy()[:n]
        doublet_strength = self.lattice.doublet_strength.to_numpy()[:n]
        area = self.lattice.area.to_numpy()[:n]
        pressure_coefficient = self.lattice.pressure_coefficient.to_numpy()[:n]
        group_id = self.lattice.group_id.to_numpy()[:n]
        panel_force = (
            self.panel_force.to_numpy()[:n] if self.panel_force is not None else np.zeros((n, 3))
        )

        panel_mesh_to_vtp(
            vertex_position=vertex_position,
            panel_centre=panel_centre,
            normal=normal,
            doublet_strength=doublet_strength,
            area=area,
            pressure_coefficient=pressure_coefficient,
            panel_force=panel_force,
            group_id=group_id,
            time=time,
            filepath=f"{filename}.vtp",
        )

    def initialize(self, force: bool = False) -> None:
        """Assemble or refresh the active aerodynamic influence matrix."""
        self._ensure_initialized()
        self._last_aic_assembly_seconds = 0.0
        self._last_aic_rebuilt = False
        # Fallback to ensuring mesh is generated if we have bodies but no panels (if applicable)
        if (
            force
            or not self.is_initialized
            or self._matrix_geometry_revision != self._geometry_revision
        ):
            n = self.lattice.n_panels
            if n > 0:
                if self.collect_timing:
                    ti.sync()
                    assembly_started = perf_counter()
                if self.boundary_condition_type == "DIRICHLET":
                    build_dirichlet_aerodynamic_influence_coefficient_matrix(
                        self.lattice.vertex_position,
                        self.lattice.panel_centre,
                        self.lattice.normal,
                        self.aerodynamic_influence_coefficient,
                        n,
                    )
                else:
                    build_source_aerodynamic_influence_coefficient_matrix(
                        self.lattice.vertex_position,
                        self.lattice.panel_centre,
                        self.lattice.normal,
                        self.aerodynamic_influence_coefficient,
                        n,
                    )
                if self.collect_timing:
                    ti.sync()
                    self._last_aic_assembly_seconds = perf_counter() - assembly_started
                self._last_aic_rebuilt = True
                self.is_initialized = True
                self._matrix_geometry_revision = self._geometry_revision
                self._constrained_factorization = None
                self._factorization_geometry_revision = -1

    def _resolve_wake_field(self) -> ti.Vector.field:
        n = self.lattice.n_panels
        if hasattr(self, "_zero_wake") and self._zero_wake.shape[0] == n:
            return self._zero_wake
        self._zero_wake = ti.Vector.field(3, dtype=self.lattice.ti_dtype, shape=n)
        self._zero_wake.fill(0.0)
        return self._zero_wake

    def _refresh_far_field_expansions(self) -> None:
        """Refresh source moments after a solve or geometry change."""
        if self.lattice is None or self.lattice.n_panels == 0:
            self._far_field_bodies = []
            return
        if self.boundary_condition_type != "NEUMANN":
            self._far_field_bodies = []
            self._last_far_field_fraction = 0.0
            return
        n = self.lattice.n_panels
        self._far_field_bodies = build_far_field_bodies(
            self.lattice.panel_centre.to_numpy()[:n],
            self.lattice.area.to_numpy()[:n],
            self.lattice.source_strength.to_numpy()[:n],
            self.lattice.bodies,
        )

    def _far_field_arrays(self):
        """Return compact, dtype-matched arrays for Taichi moment kernels."""
        dtype = np.float32 if self.float_dtype == "f32" else np.float64
        bodies = self._far_field_bodies
        return (
            np.asarray([body.start_idx for body in bodies], dtype=np.int32),
            np.asarray([body.count for body in bodies], dtype=np.int32),
            np.asarray([body.centre for body in bodies], dtype=dtype).reshape(-1, 3),
            np.asarray([body.radius for body in bodies], dtype=dtype),
            np.asarray([body.monopole for body in bodies], dtype=dtype),
            np.asarray([body.dipole for body in bodies], dtype=dtype).reshape(-1, 3),
        )

    def _record_far_field_fraction(self, fraction: float) -> None:
        self._last_far_field_fraction = float(fraction)
        if self.results["diagnostic_history"]:
            self.results["diagnostic_history"][-1]["far_field_target_fraction"] = float(fraction)

    def _neumann_constraints(self, area: np.ndarray, n: int) -> np.ndarray:
        """Build the per-closed-body source-flux constraints ``C sigma = 0``."""
        constraints = np.zeros((len(self.lattice.bodies), n), dtype=area.dtype)
        for row, body in enumerate(self.lattice.bodies):
            body_slice = slice(body.start_idx, body.start_idx + body.count)
            constraints[row, body_slice] = area[body_slice]
        return constraints

    @staticmethod
    def _wake_velocity_numpy(wake_velocity: Any, n: int) -> np.ndarray:
        """Return the incident velocity at the active collocation points."""
        values = wake_velocity.to_numpy() if hasattr(wake_velocity, "to_numpy") else wake_velocity
        velocity = np.asarray(values, dtype=np.float64)
        if velocity.shape == (3,):
            return np.broadcast_to(velocity, (n, 3))
        if velocity.ndim != 2 or velocity.shape[1] != 3 or velocity.shape[0] < n:
            raise ValueError("wake_velocity must provide at least n panel velocity vectors")
        return velocity[:n]

    def analyze_neumann_residual(
        self, *, condition_max_panels: int = 2048
    ) -> dict[str, float | str | None]:
        """Compare the active constrained solution with an unconstrained reference.

        This is an explicit qualification diagnostic, not part of the normal
        solve path: the direct unconstrained reference and a two-norm condition
        number both have cubic cost.  It distinguishes a constrained-system
        compatibility floor from a failed linear solve while preserving the
        constrained CPU solve as the current reference implementation.

        ``condition_number_2`` is omitted above ``condition_max_panels`` so a
        diagnostic cannot unexpectedly dominate a large panel step.
        """
        self._ensure_initialized()
        if self.boundary_condition_type != "NEUMANN":
            raise ValueError("Neumann residual analysis is only defined for NEUMANN solves")
        if (
            not self.results["diagnostic_history"]
            or not self.is_initialized
            or self._matrix_geometry_revision != self._geometry_revision
        ):
            raise RuntimeError("Solve the Neumann system before analysing its residual")
        if condition_max_panels < 1:
            raise ValueError("condition_max_panels must be positive")

        n = self.lattice.n_panels
        matrix = self.aerodynamic_influence_coefficient.to_numpy()[:n, :n]
        rhs = self.right_hand_side.to_numpy()[:n]
        values = self.lattice.source_strength.to_numpy()[:n]
        constraints = self._neumann_constraints(self.lattice.area.to_numpy()[:n], n)

        constrained_metrics = constrained_least_squares_metrics(matrix, rhs, constraints, values)

        # ``solve`` uses this as the reference no-constraint system.  A direct
        # solve is preferred because the matrix is square; least squares keeps
        # the diagnostic defined if a malformed geometry makes it singular.
        import scipy.linalg as la

        try:
            unconstrained_values = la.solve(matrix, rhs, assume_a="gen", check_finite=False)
            unconstrained_method = "scipy.linalg.solve"
        except la.LinAlgError:
            unconstrained_values, _, _, _ = la.lstsq(
                matrix, rhs, lapack_driver="gelsy", check_finite=False
            )
            unconstrained_method = "scipy.linalg.lstsq(gelsy)"
        unconstrained_metrics = constrained_least_squares_metrics(
            matrix, rhs, constraints, unconstrained_values
        )

        condition_number: float | None = None
        condition_method: str | None = None
        if n <= condition_max_panels:
            condition_number = float(np.linalg.cond(matrix))
            condition_method = "numpy.linalg.cond_2"

        analysis: dict[str, float | str | None] = {
            "algebraic_residual": constrained_metrics["discrete_equation_residual"],
            "algebraic_residual_absolute": constrained_metrics[
                "discrete_equation_residual_absolute"
            ],
            "right_hand_side_norm": constrained_metrics["right_hand_side_norm"],
            "constraint_residual": constrained_metrics["constraint_residual"],
            "relative_constraint_residual": constrained_metrics["relative_constraint_residual"],
            "projected_optimality_residual": constrained_metrics["projected_optimality_residual"],
            "projected_optimality_residual_absolute": constrained_metrics[
                "projected_optimality_residual_absolute"
            ],
            "unconstrained_algebraic_residual": unconstrained_metrics["discrete_equation_residual"],
            "unconstrained_algebraic_residual_absolute": unconstrained_metrics[
                "discrete_equation_residual_absolute"
            ],
            "unconstrained_constraint_residual": unconstrained_metrics["constraint_residual"],
            "unconstrained_relative_constraint_residual": unconstrained_metrics[
                "relative_constraint_residual"
            ],
            "condition_number_2": condition_number,
            "condition_number_method": condition_method,
            "unconstrained_reference_method": unconstrained_method,
        }
        if self.results["diagnostic_history"]:
            latest = self.results["diagnostic_history"][-1]
            analysis.update(
                {
                    "no_penetration_residual": latest["no_penetration_residual"],
                    "no_penetration_max_residual": latest["no_penetration_max_residual"],
                    "no_penetration_reference_speed": latest["no_penetration_reference_speed"],
                    "relative_no_penetration_residual": latest["relative_no_penetration_residual"],
                }
            )
            latest.update(analysis)
        return analysis

    def solve(
        self,
        freestream_velocity: np.ndarray,
        wake_velocity: object | None,
        time: float,
    ) -> None:
        """Solve panel strengths for the declared freestream and wake state."""
        self._ensure_initialized()
        n = self.lattice.n_panels
        timings = {
            "aic_assembly": 0.0,
            "rhs_assembly": 0.0,
            "device_to_host": 0.0,
            "constraint_factorization": 0.0,
            "constrained_rhs_solve": 0.0,
            "constrained_diagnostics": 0.0,
            "host_to_device": 0.0,
            "surface_evaluation": 0.0,
            "far_field_refresh": 0.0,
            "total_solve": 0.0,
        }
        if self.collect_timing:
            ti.sync()
            total_started = perf_counter()
        self.initialize()
        timings["aic_assembly"] = self._last_aic_assembly_seconds

        if wake_velocity is None:
            wake_velocity = self._resolve_wake_field()

        scalar_dtype = ti.f32 if self.float_dtype == "f32" else ti.f64
        numpy_dtype = np.float32 if self.float_dtype == "f32" else np.float64
        ti_v_inf = ti.Vector(
            np.asarray(freestream_velocity, dtype=numpy_dtype).tolist(),
            dt=scalar_dtype,
        )

        if self.collect_timing:
            ti.sync()
            stage_started = perf_counter()
        if self.boundary_condition_type == "DIRICHLET":
            compute_dirichlet_right_hand_side_with_sources(
                self.lattice.vertex_position,
                self.lattice.panel_centre,
                self.lattice.normal,
                ti_v_inf,
                wake_velocity,
                self.right_hand_side,
                n,
            )
        else:
            # Assemble every Neumann right-hand side through the same f64
            # arithmetic. Taichi's template specialization otherwise changes
            # the final rounding depending on the incident-field
            # representation; an ill-conditioned influence matrix can amplify
            # that irrelevant difference between the CPU and GPU solves.
            incident = self._wake_velocity_numpy(wake_velocity, n)
            body_velocity = self.lattice.body_velocity.to_numpy()[:n]
            normal = self.lattice.normal.to_numpy()[:n]
            relative_incident = (
                np.asarray(freestream_velocity, dtype=np.float64) + incident - body_velocity
            )
            right_hand_side = -np.einsum("ij,ij->i", relative_incident, normal)
            right_hand_side_full = np.zeros(self.max_n_panels, dtype=numpy_dtype)
            right_hand_side_full[:n] = right_hand_side
            self.right_hand_side.from_numpy(right_hand_side_full)
        if self.collect_timing:
            ti.sync()
            timings["rhs_assembly"] = perf_counter() - stage_started

        doublet_strength = (
            self.lattice.doublet_strength
            if self.boundary_condition_type == "DIRICHLET"
            else self.lattice.source_strength
        )
        constraint_residual = 0.0
        relative_constraint_residual = 0.0
        projected_optimality_residual = float("nan")
        projected_optimality_residual_absolute = float("nan")
        net_source_flux: dict[str, float] = {}
        tolerance = float(getattr(self.solver_strategy, "residual_tolerance", np.nan))
        algebraic_residual_absolute = float("nan")
        right_hand_side_norm = float("nan")
        factorization = None
        factorization_reused = False
        if self.boundary_condition_type == "NEUMANN":
            if self.collect_timing:
                ti.sync()
                stage_started = perf_counter()
            area = self.lattice.area.to_numpy()[:n]
            if self.collect_timing:
                timings["device_to_host"] = perf_counter() - stage_started
            constraints = self._neumann_constraints(area, n)
            if self.linear_solver_name == "BICGSTAB_GPU":
                if self.collect_timing:
                    ti.sync()
                    stage_started = perf_counter()
                iterative_success = self.solver_strategy.solve_constrained_least_squares(
                    self.aerodynamic_influence_coefficient,
                    self.right_hand_side,
                    doublet_strength,
                    n,
                    constraints,
                    relative_tolerance=(1.0e-6 if self.float_dtype == "f32" else 1.0e-12),
                )
                if self.collect_timing:
                    ti.sync()
                    timings["constrained_rhs_solve"] = perf_counter() - stage_started
                    stage_started = perf_counter()
                metrics = self.solver_strategy.constrained_metrics(
                    self.aerodynamic_influence_coefficient,
                    self.right_hand_side,
                    doublet_strength,
                    n,
                    constraints,
                )
                values = doublet_strength.to_numpy()[:n]
                if self.collect_timing:
                    ti.sync()
                    timings["constrained_diagnostics"] = perf_counter() - stage_started
            else:
                if self.collect_timing:
                    stage_started = perf_counter()
                matrix = self.aerodynamic_influence_coefficient.to_numpy()[:n, :n]
                rhs = self.right_hand_side.to_numpy()[:n]
                if self.collect_timing:
                    timings["device_to_host"] += perf_counter() - stage_started
                factorization_reused = (
                    self.reuse_constrained_factorization
                    and self._constrained_factorization is not None
                    and self._factorization_geometry_revision == self._geometry_revision
                )
                if factorization_reused:
                    factorization = self._constrained_factorization
                else:
                    if self.collect_timing:
                        stage_started = perf_counter()
                    factorization = EqualityConstrainedLeastSquaresFactorization.factorize(
                        matrix, constraints
                    )
                    if self.collect_timing:
                        timings["constraint_factorization"] = perf_counter() - stage_started
                    if self.reuse_constrained_factorization:
                        self._constrained_factorization = factorization
                        self._factorization_geometry_revision = self._geometry_revision
                if self.collect_timing:
                    stage_started = perf_counter()
                values = factorization.solve(rhs)
                if self.collect_timing:
                    timings["constrained_rhs_solve"] = perf_counter() - stage_started
                full_values = np.zeros(self.lattice.max_n_panels, dtype=matrix.dtype)
                full_values[:n] = values
                if self.collect_timing:
                    stage_started = perf_counter()
                doublet_strength.from_numpy(full_values)
                if self.collect_timing:
                    ti.sync()
                    timings["host_to_device"] = perf_counter() - stage_started
                metrics = constrained_least_squares_metrics(matrix, rhs, constraints, values)
                iterative_success = True
            for body in self.lattice.bodies:
                body_slice = slice(body.start_idx, body.start_idx + body.count)
                net_source_flux[body.uid] = float(np.dot(values[body_slice], area[body_slice]))
            residual = metrics["discrete_equation_residual"]
            algebraic_residual_absolute = metrics["discrete_equation_residual_absolute"]
            right_hand_side_norm = metrics["right_hand_side_norm"]
            constraint_residual = metrics["constraint_residual"]
            relative_constraint_residual = metrics["relative_constraint_residual"]
            projected_optimality_residual = metrics["projected_optimality_residual"]
            projected_optimality_residual_absolute = metrics[
                "projected_optimality_residual_absolute"
            ]
            convergence_residual = max(
                relative_constraint_residual,
                projected_optimality_residual,
            )
            success = iterative_success and convergence_residual <= tolerance
            iterations = (
                self.solver_strategy.last_iterations
                if self.linear_solver_name == "BICGSTAB_GPU"
                else None
            )
            self.solver_strategy.last_residual = convergence_residual
            self.solver_strategy.last_iterations = None
        else:
            success = self.solver_strategy.solve(
                self.aerodynamic_influence_coefficient, self.right_hand_side, doublet_strength, n
            )
            iterations = getattr(self.solver_strategy, "last_iterations", None)
            if n > 0:
                matrix = self.aerodynamic_influence_coefficient.to_numpy()[:n, :n]
                values = doublet_strength.to_numpy()[:n]
                rhs = self.right_hand_side.to_numpy()[:n]
                algebraic_error = matrix @ values - rhs
                algebraic_residual_absolute = float(np.linalg.norm(algebraic_error))
                right_hand_side_norm = float(np.linalg.norm(rhs))
                residual = (
                    algebraic_residual_absolute / right_hand_side_norm
                    if right_hand_side_norm > 0.0
                    else algebraic_residual_absolute
                )
            else:
                residual = 0.0
                algebraic_residual_absolute = 0.0
                right_hand_side_norm = 0.0
            convergence_residual = residual
        if not success:
            if self.boundary_condition_type == "NEUMANN":
                message = (
                    f"Constrained panel solve failed KKT convergence (n_panels={n}, "
                    f"projected optimality={projected_optimality_residual:.3e}, "
                    f"relative flux={relative_constraint_residual:.3e}, "
                    f"tolerance={tolerance:.3e})."
                )
            else:
                message = (
                    f"Panel linear solver failed to converge (n_panels={n}, "
                    f"relative residual {residual:.3e} above the {tolerance:.3e} tolerance)."
                )
            if self.raise_on_non_convergence:
                raise RuntimeError(message)
            logger.error(message)
        self._solved = success
        no_penetration_residual = float("nan")
        max_no_penetration_residual = float("nan")
        no_penetration_reference_speed = float("nan")
        relative_no_penetration_residual = float("nan")
        if n > 0:
            if self.collect_timing:
                ti.sync()
                stage_started = perf_counter()
            self._update_surface_velocity(freestream_velocity, wake_velocity)
            normal_velocity = np.einsum(
                "ij,ij->i",
                self.surface_velocity_relative.to_numpy()[:n],
                self.lattice.normal.to_numpy()[:n],
            )
            no_penetration_residual = float(np.sqrt(np.mean(normal_velocity**2)))
            max_no_penetration_residual = float(np.max(np.abs(normal_velocity)))
            incident_relative = (
                np.asarray(freestream_velocity, dtype=np.float64)
                + self._wake_velocity_numpy(wake_velocity, n)
                - self.lattice.body_velocity.to_numpy()[:n]
            )
            no_penetration_reference_speed = float(
                np.sqrt(np.mean(np.einsum("ij,ij->i", incident_relative, incident_relative)))
            )
            if no_penetration_reference_speed > 0.0:
                relative_no_penetration_residual = (
                    no_penetration_residual / no_penetration_reference_speed
                )
            if self.collect_timing:
                ti.sync()
                timings["surface_evaluation"] = perf_counter() - stage_started
        if self.collect_timing:
            stage_started = perf_counter()
        self._refresh_far_field_expansions()
        if self.collect_timing:
            timings["far_field_refresh"] = perf_counter() - stage_started
            ti.sync()
            timings["total_solve"] = perf_counter() - total_started
        itemsize = 4 if self.float_dtype == "f32" else 8
        self.results["diagnostic_history"].append(
            {
                "step": int(self.step),
                "time": float(time),
                "n_panels": int(n),
                "linear_solver": (
                    (
                        "ProjectedCGLS(Taichi)"
                        if self.linear_solver_name == "BICGSTAB_GPU"
                        else "ConstrainedLeastSquares(SciPy)"
                    )
                    if self.boundary_condition_type == "NEUMANN"
                    else type(self.solver_strategy).__name__
                ),
                "requested_linear_solver": self.linear_solver_name,
                "factorization_reused": (
                    factorization_reused if self.boundary_condition_type == "NEUMANN" else False
                ),
                "factorization_cache_bytes": (
                    factorization.memory_bytes if factorization is not None else 0
                ),
                "active_aic_bytes": n * n * itemsize,
                "allocated_aic_bytes": self.max_n_panels * self.max_n_panels * itemsize,
                "aic_rebuilt": self._last_aic_rebuilt,
                "timings_seconds": timings if self.collect_timing else None,
                "linear_solver_success": bool(success),
                "residual": None if residual is None else float(residual),
                "algebraic_residual": None if residual is None else float(residual),
                "discrete_equation_residual": None if residual is None else float(residual),
                "algebraic_residual_absolute": algebraic_residual_absolute,
                "right_hand_side_norm": right_hand_side_norm,
                "iterations": None if iterations is None else int(iterations),
                "refresh_count": int(self.refresh_count),
                "force_method": self.force_config.method,
                "boundary_condition_type": self.boundary_condition_type,
                "precision": self.float_dtype,
                "residual_tolerance": tolerance,
                "relative_residual_vs_tolerance": (
                    float(residual / tolerance)
                    if self.boundary_condition_type != "NEUMANN" and tolerance > 0.0
                    else None
                ),
                "convergence_metric": (
                    "projected_kkt_optimality_and_relative_flux"
                    if self.boundary_condition_type == "NEUMANN"
                    else "relative_equation_residual"
                ),
                "convergence_residual": convergence_residual,
                "convergence_residual_vs_tolerance": (
                    float(convergence_residual / tolerance) if tolerance > 0.0 else float("nan")
                ),
                "no_penetration_residual": no_penetration_residual,
                "no_penetration_max_residual": max_no_penetration_residual,
                "no_penetration_reference_speed": no_penetration_reference_speed,
                "relative_no_penetration_residual": relative_no_penetration_residual,
                "net_source_flux": net_source_flux,
                "constraint_residual": constraint_residual,
                "relative_constraint_residual": relative_constraint_residual,
                "projected_optimality_residual": projected_optimality_residual,
                "projected_optimality_residual_absolute": projected_optimality_residual_absolute,
                "far_field_target_fraction": 0.0,
            }
        )
        if self.step % self.logging_interval_steps == 0:
            logger.info(
                "[Panel] panels=%d precision=%s equation_residual=%.3e "
                "convergence/tolerance=%.3e iterations=%s no_penetration=%.3e "
                "net_source_flux=%s constraint=%.3e projected_optimality=%.3e "
                "far_field_fraction=%.3f",
                n,
                self.float_dtype,
                residual,
                convergence_residual / tolerance if tolerance > 0.0 else float("nan"),
                iterations,
                no_penetration_residual,
                net_source_flux,
                constraint_residual,
                projected_optimality_residual,
                self._last_far_field_fraction,
            )

    def ensure_mesh_generated(self) -> None:
        """Require at least one loaded panel before a solve or advance."""
        self._ensure_initialized()
        if self.lattice.n_panels <= 0:
            raise RuntimeError(
                "PanelSolver has no panels loaded. Add a surface before solve/advance."
            )

    def _body_for_range(self, body_range):
        """Return the body exactly identified by ``(start_idx, end_idx)``."""
        start_idx, end_idx = body_range
        for body in self.lattice.bodies:
            if body.start_idx == start_idx and body.start_idx + body.count == end_idx:
                return body
        raise ValueError(f"No panel body has range [{start_idx}, {end_idx})")

    def get_body_pose(self, body_range: tuple[int, int]) -> BodyPose:
        """Return a copy of the current complete pose for one body."""
        self._ensure_initialized()
        pose = self._body_for_range(body_range).pose
        return BodyPose() if pose is None else pose.copy()

    def apply_body_pose(self, pose: BodyPose, body_range: tuple[int, int]) -> None:
        """Apply one authoritative geometry/body-velocity update to a body."""
        self._ensure_initialized()
        body = self._body_for_range(body_range)
        previous = BodyPose() if body.pose is None else body.pose
        self.lattice.apply_body_pose(body, pose)
        body.pose = pose.copy()

        geometry_changed = not all(
            np.array_equal(getattr(previous, name), getattr(pose, name))
            for name in ("rotation", "translation", "rotation_centre")
        )
        if geometry_changed:
            body.geometry_revision += 1
            self._geometry_revision += 1
            self.is_initialized = False
            self._solved = False
            self._far_field_bodies = []
            self._constrained_factorization = None
            self._factorization_geometry_revision = -1

    def apply_translation_update(
        self,
        displacement: np.ndarray,
        linear_velocity: np.ndarray,
        body_range: tuple[int, int],
    ) -> None:
        """Compatibility wrapper that updates only the translational pose terms."""
        self._ensure_initialized()
        pose = self.get_body_pose(body_range)
        pose.translation = np.asarray(displacement, dtype=np.float64)
        pose.linear_velocity = np.asarray(linear_velocity, dtype=np.float64)
        self.apply_body_pose(pose, body_range)

    def apply_rotation_update(
        self,
        rotation_matrix: np.ndarray,
        angular_velocity: np.ndarray,
        rotation_centre: np.ndarray,
        body_range: tuple[int, int],
    ) -> None:
        """Compatibility wrapper that updates only rotational pose terms."""
        self._ensure_initialized()
        pose = self.get_body_pose(body_range)
        pose.rotation = np.asarray(rotation_matrix, dtype=np.float64)
        pose.angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
        pose.rotation_centre = np.asarray(rotation_centre, dtype=np.float64)
        self.apply_body_pose(pose, body_range)

    def _update_surface_velocity(
        self, freestream_velocity: np.ndarray, wake_velocity: Any = None
    ) -> None:
        """Evaluate absolute and body-relative surface velocities.

        ``surface_velocity_absolute`` is the inertial fluid velocity from the
        incident and panel fields.  ``surface_velocity_relative`` subtracts
        the separately stored rigid-body velocity and is the only field used
        to evaluate impermeability.
        """
        n = self.lattice.n_panels
        if n == 0:
            return
        if wake_velocity is None:
            wake_velocity = self._resolve_wake_field()

        if self.boundary_condition_type == "DIRICHLET":
            normals_np = self.lattice.normal.to_numpy()[:n]
            ti_dtype = np.float32 if self.float_dtype == "f32" else np.float64
            source_full = np.zeros(self.lattice.max_n_panels, dtype=ti_dtype)
            source_full[:n] = (-np.dot(normals_np, freestream_velocity)).astype(ti_dtype)
            self.lattice.source_strength.from_numpy(source_full)

        scalar_dtype = ti.f32 if self.float_dtype == "f32" else ti.f64
        numpy_dtype = np.float32 if self.float_dtype == "f32" else np.float64
        # Keep a single field representation for the total incident flow.
        # A Taichi vector argument and an otherwise identical vector field
        # follow distinct compilation paths and used to produce f64 results
        # that differed by about one f32 ulp.  The surface solution must be
        # independent of whether uniform flow came from VPM or freestream.
        total_incident = self._wake_velocity_numpy(wake_velocity, n) + np.asarray(
            freestream_velocity, dtype=np.float64
        )
        if (
            not hasattr(self, "_total_incident_velocity")
            or self._total_incident_velocity.shape[0] != n
        ):
            self._total_incident_velocity = ti.Vector.field(
                3,
                dtype=self.lattice.ti_dtype,
                shape=n,
            )
        self._total_incident_velocity.from_numpy(total_incident.astype(numpy_dtype))
        compute_surface_velocity_with_sources(
            self.lattice.vertex_position,
            self.lattice.panel_centre,
            self.lattice.normal,
            self.lattice.doublet_strength,
            self.lattice.source_strength,
            ti.Vector([0.0, 0.0, 0.0], dt=scalar_dtype),
            self._total_incident_velocity,
            self.surface_velocity_absolute,
            n,
        )
        compute_relative_surface_velocity(
            self.surface_velocity_absolute,
            self.lattice.body_velocity,
            self.surface_velocity_relative,
            n,
        )

    def _has_body_motion(self) -> bool:
        """Whether any loaded body has non-zero rigid-body velocity."""
        if self.lattice is None or self.lattice.n_panels == 0:
            return False
        velocity = self.lattice.body_velocity.to_numpy()[: self.lattice.n_panels]
        return bool(np.any(np.abs(velocity) > 1.0e-14))

    def compute_forces(
        self,
        freestream_velocity: np.ndarray,
        wake_velocity: object | None,
        time_step_size: float,
        density: float,
    ) -> dict[int, np.ndarray]:
        """
        Compute integrated force vector per body group using Bernoulli or Impulse.
        """
        n = self.lattice.n_panels
        if n == 0:
            return {}
        if self._has_body_motion():
            raise NotImplementedError(
                "Steady Bernoulli panel forces are unsupported for moving bodies; "
                "the unsteady dphi/dt term has not been implemented."
            )

        v_inf_mag = np.linalg.norm(freestream_velocity)
        if v_inf_mag < 1e-10:
            v_inf_mag = 1.0

        if wake_velocity is None:
            wake_velocity = self._resolve_wake_field()

        # Surface velocity comes from the shared source-doublet evaluation.  The
        # doublet-only kernel used here previously omitted the source panels
        # entirely, which under NEUMANN — where the solve fills source_strength
        # and leaves doublet_strength at zero — dropped the whole body contribution.
        self._update_surface_velocity(freestream_velocity, wake_velocity)

        if self.force_config.method == "BERNOULLI":
            compute_pressure_bernoulli(
                self.surface_velocity, float(v_inf_mag), self.lattice.pressure_coefficient, n
            )
            compute_forces_bernoulli(
                self.surface_velocity,
                float(v_inf_mag),
                self.lattice.area,
                self.lattice.normal,
                density,
                self.panel_force,
                n,
            )
        elif self.force_config.method == "KUTTA_JOUKOWSKI":
            compute_forces_kutta_joukowski(
                self.lattice.doublet_strength,
                self.surface_velocity,
                self.lattice.vertex_position,
                self.lattice.area,
                self.lattice.normal,
                density,
                self.panel_force,
                n,
            )
        else:
            raise ValueError(f"Unknown panel force method: {self.force_config.method}")

        # Summarize forces by group_id
        panel_force = self.panel_force.to_numpy()[:n]
        total_forces: dict[int, np.ndarray] = {}

        for body in self.lattice.bodies:
            gid = body.group_id
            body_force = np.sum(panel_force[body.start_idx : body.start_idx + body.count], axis=0)
            if gid not in total_forces:
                total_forces[gid] = np.zeros(3)
            total_forces[gid] += body_force

        self.results["force_history"].append(total_forces)

        for gid, f in total_forces.items():
            logger.info(f"Step {self.step}: Group {gid} Force = {f}")
        return total_forces

    def compute_loads(
        self,
        freestream_velocity: np.ndarray,
        wake_velocity: object | None,
        time_step_size: float,
        density: float,
    ) -> dict[int, np.ndarray]:
        """Compute the configured integrated force for each panel group."""
        return self.compute_forces(freestream_velocity, wake_velocity, time_step_size, density)

    def compute_induced_velocity(self, points: np.ndarray) -> np.ndarray:
        """Evaluate panel-induced velocity at an ``(N, 3)`` point cloud."""
        self._ensure_initialized()
        dtype = np.float32 if self.float_dtype == "f32" else np.float64
        points = np.asarray(points, dtype=dtype)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        n_panels = self.lattice.n_panels
        if n_panels == 0:
            return np.zeros_like(points)

        self._refresh_far_field_expansions()

        vertex_position = self.lattice.vertex_position.to_numpy()[:n_panels].astype(
            dtype, copy=False
        )
        velocity = np.zeros_like(points)

        if self.boundary_condition_type == "NEUMANN":
            normal = self.lattice.normal.to_numpy()[:n_panels].astype(dtype, copy=False)
            doublet_strength = self.lattice.source_strength.to_numpy()[:n_panels].astype(
                dtype, copy=False
            )
            eligible = any(
                body.count >= self.far_field_min_panels and body.radius > 0.0
                for body in self._far_field_bodies
            )
            if not eligible:
                # Keep the production cube path exactly on the original kernel:
                # 108 panels is below the far-field threshold by design.
                compute_source_induced_velocity_kernel(
                    vertex_position, normal, doublet_strength, points, velocity
                )
                self._record_far_field_fraction(0.0)
            else:
                (
                    body_start,
                    body_count,
                    body_centre,
                    body_radius,
                    body_monopole,
                    body_dipole,
                ) = self._far_field_arrays()
                compute_source_panel_velocity_with_far_field(
                    vertex_position,
                    normal,
                    doublet_strength,
                    points,
                    velocity,
                    body_start,
                    body_count,
                    body_centre,
                    body_radius,
                    body_monopole,
                    body_dipole,
                    len(self._far_field_bodies),
                    self.far_field_acceptance,
                    self.far_field_min_panels,
                )
                self._record_far_field_fraction(
                    far_field_interaction_fraction(
                        points,
                        self._far_field_bodies,
                        self.far_field_acceptance,
                        self.far_field_min_panels,
                    )
                )
            return velocity

        doublet_strength = self.lattice.doublet_strength.to_numpy()[:n_panels].astype(
            dtype, copy=False
        )
        if n_panels >= 1000 or n_panels * len(points) >= 100_000:
            compute_induced_velocity_kernel(vertex_position, doublet_strength, points, velocity)
        else:

            def _segment_velocity(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
                r1 = p - a
                r2 = p - b
                r1xr2 = np.cross(r1, r2)
                d1 = np.linalg.norm(r1)
                d2 = np.linalg.norm(r2)
                denom = d1 * d2 + np.dot(r1, r2) + 1e-12
                coeff = (
                    (1.0 / (4.0 * np.pi))
                    * (1.0 / (d1 + 1e-12) + 1.0 / (d2 + 1e-12))
                    * (1.0 / denom)
                )
                return -coeff * r1xr2

            for j in range(n_panels):
                v0, v1, v2 = vertex_position[j, 0], vertex_position[j, 1], vertex_position[j, 2]
                vortex_strength = doublet_strength[j]
                for i in range(points.shape[0]):
                    p = points[i]
                    velocity[i] += vortex_strength * (
                        _segment_velocity(p, v0, v1)
                        + _segment_velocity(p, v1, v2)
                        + _segment_velocity(p, v2, v0)
                    )

        return velocity

    def advance(
        self,
        particles: "Particles | None" = None,
        physics: "PhysicsEngine | None" = None,
        config: "Numerics | None" = None,
        freestream_velocity: np.ndarray | None = None,
        wake_velocity: object | None = None,
        time_step_size: float | None = None,
        time: float | None = None,
        step: int | None = None,
        logging_interval_steps: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """
        Advance panel simulation by one time step.

        Args:
            particles: Live VPM particle container (coupled mode).
            physics: VPM physics/evaluation backend used for induced velocity.
            config: VPM setup providing the freestream velocity when not given.
            freestream_velocity: Freestream velocity vector; defaults to the
                configured solver value.
            wake_velocity: Prescribed wake-induced velocity (standalone mode).
            time_step_size: Time step size for this advance.
            time: Current physical time.
            step: Current step index.
            logging_interval_steps: Steps between force log reports.

        Returns newly shed wake particles to be added to the VPM system.
        """
        if freestream_velocity is None and config is not None:
            freestream_velocity = np.array(getattr(config, "freestream_velocity", [1.0, 0.0, 0.0]))
        if freestream_velocity is None:
            freestream_velocity = (
                self.freestream_velocity
                if self.freestream_velocity is not None
                else np.array([1.0, 0.0, 0.0])
            )
        if time_step_size is None:
            time_step_size = 0.01
        if time is None:
            time = self._current_time
        if step is not None:
            self.step = step
        if logging_interval_steps is not None:
            self.logging_interval_steps = logging_interval_steps

        if (
            self.boundary_condition_type == "DIRICHLET"
            and particles is not None
            and physics is not None
        ):
            raise NotImplementedError(
                "DIRICHLET panel coupling is unsupported for VPM velocity fields. "
                "Use NEUMANN, which only requires incident normal velocity."
            )

        self.ensure_mesh_generated()

        # 1. Save history for BDF2
        self.lattice.save_old_doublet_strength()

        # 2. Update geometry via kinematics (if any)
        for body in self.lattice.bodies:
            kinematics = getattr(body, "kinematics", None)
            if kinematics is not None:
                kinematics.update(
                    self,
                    time,
                    time_step_size,
                    (body.start_idx, body.start_idx + body.count),
                )

        # 3. Compute VPM-induced velocity at collocation points (if coupled)
        if particles is not None and physics is not None:
            wake_velocity = self._set_coupled_wake_velocity(particles, physics)

        # 4. Solve potential flow
        self.solve(freestream_velocity, wake_velocity, time)

        # cube_flow's vpm_boundary_condition-only panel supplies the irrotational boundary
        # correction, while the FVM owns force/pressure_coefficient reporting.  Avoid a second
        # surface-velocity/force pass unless this panel is authoritative for
        # particle dynamics or post-processing.
        if self.coupling_scope != "vpm_boundary_condition" and not self._has_body_motion():
            self.compute_postprocess(
                freestream_velocity,
                freestream_velocity,
                self.density,
                time_step_size=time_step_size,
                coupled=(particles is not None),
            )
            self.compute_loads(freestream_velocity, wake_velocity, time_step_size, self.density)
            log_freq = self.logging_interval_steps
            if log_freq > 0 and self.step % log_freq == 0:
                try:
                    self.log_forces_table(self.density, freestream_velocity)
                except Exception as e:
                    print(f"   (Warning) Could not compute panel forces: {e}")
        elif self.coupling_scope != "vpm_boundary_condition":
            logger.info(
                "[Panel] Steady Bernoulli force post-processing skipped for moving body motion; "
                "unsteady forces are unsupported."
            )

        self.results["time_history"].append(float(time))
        self._current_time = float(time)
        self._last_freestream_velocity = freestream_velocity
        self._last_wake_velocity = wake_velocity

        self.step += 1
        return None

    def _set_coupled_wake_velocity(self, particles: Any, physics: Any) -> Any:
        """Update panel collocation velocity from the current particle state."""
        if self.boundary_condition_type == "DIRICHLET":
            raise NotImplementedError(
                "DIRICHLET panel coupling is unsupported for VPM velocity fields. "
                "Use NEUMANN, which only requires incident normal velocity."
            )
        # NumPy targets follow PhysicsBase's configured TREECODE route. Passing
        # the Taichi field directly would bypass that branch and launch a direct
        # M-by-N target kernel at every panel centre.
        lattice = self.lattice
        if lattice is None:
            raise RuntimeError("panel lattice is not initialized")
        n_panels = lattice.n_panels
        centres = lattice.panel_centre.to_numpy()[:n_panels]
        induced = physics.compute_target_velocity(
            particles,
            centres,
            include_freestream=False,
        )
        incident_velocity = lattice.incident_velocity.to_numpy()
        incident_velocity[:n_panels] = np.asarray(induced, dtype=incident_velocity.dtype)
        lattice.incident_velocity.from_numpy(incident_velocity)
        return lattice.incident_velocity

    def refresh_coupled_solution(
        self,
        *,
        particles: "Particles",
        physics: "PhysicsEngine",
        freestream_velocity: np.ndarray,
        time: float,
    ) -> None:
        """Resolve the body potential for a replaced particle state at fixed time.

        This is a state refresh, not a time advance: panel history, kinematics,
        force history, step counters, and wake shedding are left untouched.
        """
        if getattr(self, "boundary_condition_type", "NEUMANN") == "DIRICHLET":
            raise NotImplementedError(
                "DIRICHLET panel coupling is unsupported for VPM velocity fields. "
                "Use NEUMANN, which only requires incident normal velocity."
            )
        self.ensure_mesh_generated()
        wake_velocity = self._set_coupled_wake_velocity(particles, physics)
        self.refresh_count += 1
        self.solve(freestream_velocity, wake_velocity, time)
        self._record_particle_velocity_diagnostic(particles)

    def induced_velocity_diagnostic_is_due(self) -> bool:
        """Whether this refresh should pay for a panel-induced-velocity sample."""
        return (
            self.diagnostic_interval_steps > 0
            and self.refresh_count % self.diagnostic_interval_steps == 0
        )

    def _record_particle_velocity_diagnostic(self, particles: Any) -> None:
        """Sample panel-induced velocity at particles, when scheduled.

        Evaluating every panel at every particle is an
        ``n_panels * n_particles`` direct calculation — for a coupled cube
        run, tens of millions of interactions per refresh. It is therefore
        off unless ``diagnostic_interval_steps`` is set, and even then reads
        a fixed-stride subsample so its cost is bounded by
        ``diagnostic_sample_size`` rather than by the particle count.
        """
        if not self.induced_velocity_diagnostic_is_due():
            return
        n_particles_total = getattr(particles, "n_particles_total", 0)
        if n_particles_total <= 0 or not self.results["diagnostic_history"]:
            return

        np_dtype = getattr(particles, "_np_float_dtype", np.float32)
        position = particles.position.to_numpy()[:n_particles_total].astype(np_dtype)
        # A fixed stride keeps the sample deterministic across repeat runs and
        # restarts, which a random subsample would not.
        stride = max(1, -(-n_particles_total // self.diagnostic_sample_size))
        sample = position[::stride]
        induced_norm = np.linalg.norm(self.compute_induced_velocity(sample), axis=1)
        self.results["diagnostic_history"][-1].update(
            {
                "max_induced_velocity_at_particles": float(np.max(induced_norm)),
                "rms_induced_velocity_at_particles": float(np.sqrt(np.mean(induced_norm**2))),
                "induced_velocity_sample_size": int(sample.shape[0]),
                "induced_velocity_sample_stride": int(stride),
            }
        )

    def advance_time(self, time_step_size: float, current_time: float) -> None:
        """Advance kinematics state and geometry for all surfaces (VLM-compatible API)."""
        self.ensure_mesh_generated()
        self._current_time = float(current_time)

        freestream_velocity = (
            self._last_freestream_velocity
            if hasattr(self, "_last_freestream_velocity")
            else np.array([1.0, 0.0, 0.0])
        )
        wake_velocity = self._last_wake_velocity if hasattr(self, "_last_wake_velocity") else None

        self.lattice.save_old_doublet_strength()
        for body in self.lattice.bodies:
            kinematics = getattr(body, "kinematics", None)
            if kinematics is not None:
                kinematics.update(
                    self,
                    current_time,
                    time_step_size,
                    (body.start_idx, body.start_idx + body.count),
                )
        self.solve(freestream_velocity, wake_velocity, current_time)

    def compute_induced_velocity_direct(self, particles: "Particles") -> None:
        """Add panel-induced velocity to every active particle, on device.

        Called by the core VPM solver to couple the panel method to the
        particle method. Every particle in ``[0, n_particles_total)`` is
        treated identically; the panel solver draws no distinction between
        particles injected by a coupler and particles it already carried.

        The accumulation runs entirely in Taichi against the lattice and
        particle fields. Round-tripping particle position and velocity
        through numpy here would move several arrays of length
        ``n_particles_total`` across the host boundary at every Runge-Kutta
        stage of every step, which dominates the cost of a coupled run.
        """
        if self.lattice is None or self.lattice.n_panels == 0:
            return
        if self._has_body_motion():
            raise NotImplementedError(
                "Steady Bernoulli post-processing is unsupported for moving bodies; "
                "the unsteady dphi/dt term has not been implemented."
            )

        n_particles_total = particles.n_particles_total
        if n_particles_total == 0:
            return

        # The self-induced velocity kernel may still be in flight.
        ti.sync()
        self.accumulate_induced_velocity_on_field(
            particles.position, particles.velocity, n_particles_total
        )

    def accumulate_induced_velocity_on_field(
        self,
        target_position: object,
        target_velocity: object,
        n_targets: int,
    ) -> None:
        """Add panel-induced velocity into a Taichi vec3 field, in place."""
        if self.lattice is None or n_targets <= 0:
            return
        n_panels = self.lattice.n_panels
        if n_panels == 0:
            return

        if self.boundary_condition_type == "NEUMANN":
            eligible = any(
                body.count >= self.far_field_min_panels and body.radius > 0.0
                for body in self._far_field_bodies
            )
            if not eligible:
                accumulate_source_panel_velocity_on_field(
                    self.lattice.vertex_position,
                    self.lattice.normal,
                    self.lattice.source_strength,
                    target_position,
                    target_velocity,
                    n_panels,
                    n_targets,
                )
                self._record_far_field_fraction(0.0)
            else:
                (
                    body_start,
                    body_count,
                    body_centre,
                    body_radius,
                    body_monopole,
                    body_dipole,
                ) = self._far_field_arrays()
                far_interactions = accumulate_source_panel_velocity_with_far_field_on_field(
                    self.lattice.vertex_position,
                    self.lattice.normal,
                    self.lattice.source_strength,
                    target_position,
                    target_velocity,
                    body_start,
                    body_count,
                    body_centre,
                    body_radius,
                    body_monopole,
                    body_dipole,
                    len(self._far_field_bodies),
                    n_targets,
                    self.far_field_acceptance,
                    self.far_field_min_panels,
                )
                n_eligible = sum(
                    body.count >= self.far_field_min_panels and body.radius > 0.0
                    for body in self._far_field_bodies
                )
                self._record_far_field_fraction(
                    float(far_interactions) / float(max(1, n_targets * n_eligible))
                )
        else:
            accumulate_doublet_panel_velocity_on_field(
                self.lattice.vertex_position,
                self.lattice.doublet_strength,
                target_position,
                target_velocity,
                n_panels,
                n_targets,
            )

    def absorb_particles(self, particles: "Particles", tolerance: float = 0.05) -> int:
        """
        Detect and remove particles colliding with the panel surface.

        For thick bodies (sphere, cube, etc.), particles that pass through
        the surface can cause numerical issues. This method detects particles
        whose perpendicular distance to any panel is less than tolerance and
        whose projection lies inside the panel triangle, and removes them.

        Args:
            particles: VPM particles object
            tolerance: Collision distance threshold [m]

        Returns:
            Number of particles removed
        """
        if self.lattice is None or self.lattice.n_panels == 0:
            return 0

        n_particles_total = particles.n_particles_total
        n_panels = self.lattice.n_panels

        if n_particles_total == 0 or n_panels == 0:
            return 0

        # Extract data
        position = particles.position.to_numpy()[:n_particles_total]
        panel_centre = self.lattice.panel_centre.to_numpy()[:n_panels]
        normal = self.lattice.normal.to_numpy()[:n_panels]
        vertex_position = self.lattice.vertex_position.to_numpy()[:n_panels]

        # Simple collision detection: check if particle is inside the body
        # by checking if it's on the negative side of all panel normal
        # (assuming outward-pointing normal for a closed body)
        keep_mask = np.ones(n_particles_total, dtype=bool)

        for i in range(n_particles_total):
            p = position[i]
            # Check distance to each panel center
            for j in range(n_panels):
                # Vector from panel center to particle
                r = p - panel_centre[j]
                # Perpendicular distance to panel plane
                d_perp = abs(np.dot(r, normal[j]))
                if d_perp < tolerance:
                    # Check if projection is inside triangle (simplified check)
                    # Project onto panel plane
                    p_proj = p - d_perp * normal[j] * np.sign(np.dot(r, normal[j]))
                    # Check if inside triangle using barycentric coordinates
                    v0 = vertex_position[j, 1] - vertex_position[j, 0]
                    v1 = vertex_position[j, 2] - vertex_position[j, 0]
                    v2 = p_proj - vertex_position[j, 0]
                    d00 = np.dot(v0, v0)
                    d01 = np.dot(v0, v1)
                    d11 = np.dot(v1, v1)
                    d20 = np.dot(v2, v0)
                    d21 = np.dot(v2, v1)
                    denom = d00 * d11 - d01 * d01
                    if abs(denom) > 1e-12:
                        v = (d11 * d20 - d01 * d21) / denom
                        w = (d00 * d21 - d01 * d20) / denom
                        u = 1.0 - v - w
                        if u >= -0.1 and v >= -0.1 and w >= -0.1:
                            keep_mask[i] = False
                            break

        n_removed = int(np.sum(~keep_mask))
        if n_removed == 0:
            return 0

        # Compact particles
        n_keep = int(np.sum(keep_mask))
        if n_keep == 0:
            particles.n_particles_total = 0
            return n_removed

        # Extract kept particles
        new_position = position[keep_mask]
        new_velocity = particles.velocity.to_numpy()[:n_particles_total][keep_mask]
        new_vortex_strength = particles.vortex_strength.to_numpy()[:n_particles_total][keep_mask]
        new_vorticity = particles.vorticity.to_numpy()[:n_particles_total][keep_mask]
        new_core_radius = particles.core_radius.to_numpy()[:n_particles_total][keep_mask]
        new_particle_volume = particles.particle_volume.to_numpy()[:n_particles_total][keep_mask]
        new_kinematic_viscosity = particles.kinematic_viscosity.to_numpy()[:n_particles_total][
            keep_mask
        ]
        new_eddy_viscosity = particles.eddy_viscosity.to_numpy()[:n_particles_total][keep_mask]
        new_effective_viscosity = particles.effective_viscosity.to_numpy()[:n_particles_total][
            keep_mask
        ]
        new_group_id = particles.group_id.to_numpy()[:n_particles_total][keep_mask]
        new_velocity_gradient = particles.velocity_gradient.to_numpy()[:n_particles_total][
            keep_mask
        ]
        new_strain_rate = particles.strain_rate.to_numpy()[:n_particles_total][keep_mask]

        # Write back to particles
        particles._copy_to_taichi_vectors(new_position, particles.position, 0, n_keep)
        particles._copy_to_taichi_vectors(new_velocity, particles.velocity, 0, n_keep)
        particles._copy_to_taichi_vectors(new_vortex_strength, particles.vortex_strength, 0, n_keep)
        particles._copy_to_taichi_vectors(new_vorticity, particles.vorticity, 0, n_keep)
        particles._copy_to_taichi_scalars(new_core_radius, particles.core_radius, 0, n_keep)
        particles._copy_to_taichi_scalars(new_particle_volume, particles.particle_volume, 0, n_keep)
        particles._copy_to_taichi_scalars(
            new_kinematic_viscosity, particles.kinematic_viscosity, 0, n_keep
        )
        particles._copy_to_taichi_scalars(new_eddy_viscosity, particles.eddy_viscosity, 0, n_keep)
        particles._copy_to_taichi_scalars(
            new_effective_viscosity, particles.effective_viscosity, 0, n_keep
        )
        particles._copy_to_taichi_scalars(new_group_id, particles.group_id, 0, n_keep)
        particles._copy_to_taichi_matrices(
            new_velocity_gradient, particles.velocity_gradient, 0, n_keep
        )
        particles._copy_to_taichi_matrices(new_strain_rate, particles.strain_rate, 0, n_keep)

        particles.n_particles_total = n_keep
        particles.sync_device_counter()

        print(f"   (Panel) Absorbed {n_removed} particles impinging on surface.")
        return n_removed

    def compute_postprocess(
        self,
        external_velocity_np: np.ndarray,
        reference_velocity: np.ndarray,
        density: float,
        time_step_size: float | None = None,
        coupled: bool = False,
    ) -> None:
        """
        Compute derived quantities (velocity, pressures, forces) after solve.

        Args:
            external_velocity_np: External velocity (N, 3) or (3,)
            reference_velocity: Reference velocity vector [ux, uy, uz] (m/s)
            density: Fluid density
            time_step_size: Time step size
            coupled: Whether in coupled mode (unused for panel method)
        """
        if self.lattice is None or self.lattice.n_panels == 0:
            return
        if self._has_body_motion():
            raise NotImplementedError(
                "Steady Bernoulli post-processing is unsupported for moving bodies; "
                "the unsteady dphi/dt term has not been implemented."
            )

        n = self.lattice.n_panels
        reference_velocity_mag = np.linalg.norm(reference_velocity)
        if reference_velocity_mag < 1e-10:
            reference_velocity_mag = 1.0

        # Resolve freestream_velocity
        if external_velocity_np.ndim == 1 and external_velocity_np.shape[0] == 3:
            freestream_velocity = external_velocity_np
        else:
            freestream_velocity = np.mean(external_velocity_np, axis=0)

        self._update_surface_velocity(freestream_velocity)

        # Compute pressure coefficients
        compute_pressure_bernoulli(
            self.surface_velocity,
            float(reference_velocity_mag),
            self.lattice.pressure_coefficient,
            n,
        )

        # Compute forces
        compute_forces_bernoulli(
            self.surface_velocity,
            float(reference_velocity_mag),
            self.lattice.area,
            self.lattice.normal,
            density,
            self.panel_force,
            n,
        )

    def compute_forces_coefficients(
        self,
        density: float,
        reference_velocity: np.ndarray | None = None,
        reference_area: float | None = None,
        reference_chord: float | None = None,
        reference_span: float | None = None,
    ) -> dict[str, float]:
        """
        Compute integrated aerodynamic forces and moments as coefficients.

        Returns a dictionary with canonical force, moment, coefficient, and
        reference-quantity names.

        Args:
            density: Fluid density (kg/m³)
            reference_velocity: Reference velocity vector [ux, uy, uz] (for coefficients and L/D axes).
                   If None, uses self.freestream_velocity or auto-computed.
            reference_area: Reference area (m²). If None, uses sum of panel area.
            reference_chord: Reference chord (m). If None, uses sqrt(reference_area).
            reference_span: Reference span (m). If None, uses sqrt(reference_area).

        Returns:
            Dictionary with force components and coefficients
        """
        if self.lattice is None or self.lattice.n_panels == 0:
            return {}

        if self._has_body_motion():
            raise NotImplementedError(
                "Steady Bernoulli force coefficients are unsupported for moving bodies."
            )

        if not self._solved:
            raise RuntimeError("Must solve system before computing forces")

        # Resolve reference values
        if reference_velocity is None:
            reference_velocity = (
                self.freestream_velocity
                if self.freestream_velocity is not None
                else np.array([1.0, 0.0, 0.0])
            )
        reference_velocity_mag = np.linalg.norm(reference_velocity)
        if reference_velocity_mag < 1e-10:
            reference_velocity_mag = 1.0

        n = self.lattice.n_panels
        panel_force = self.panel_force.to_numpy()[:n]
        total_force = np.sum(panel_force, axis=0)
        force_x, force_y, force_z = total_force

        # Decompose into lift, drag, side-force
        reference_direction = reference_velocity / reference_velocity_mag
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

        lift = float(np.dot(total_force, lift_direction))
        drag = float(np.dot(total_force, reference_direction))
        side_force = float(np.dot(total_force, side_force_direction))

        # Compute moments about reference center
        panel_centre = self.lattice.panel_centre.to_numpy()[:n]
        reference_point = np.mean(panel_centre, axis=0)
        total_moment = np.sum(np.cross(panel_centre - reference_point, panel_force), axis=0)
        moment_x, moment_y, moment_z = total_moment

        # Reference values
        if reference_area is None:
            panel_area = self.lattice.area.to_numpy()[:n]
            reference_area = float(np.sum(panel_area))
        if reference_chord is None:
            reference_chord = float(np.sqrt(reference_area))
        if reference_span is None:
            reference_span = float(np.sqrt(reference_area))

        # Non-dimensionalize
        dynamic_pressure = 0.5 * density * reference_velocity_mag**2
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
        else:
            lift_coefficient = drag_coefficient = side_force_coefficient = force_coefficient_x = (
                force_coefficient_y
            ) = force_coefficient_z = 0.0
            rolling_moment_coefficient = 0.0
            pitching_moment_coefficient = 0.0
            yawing_moment_coefficient = 0.0

        result = {
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
            "dynamic_pressure": dynamic_pressure,
            "reference_area": reference_area,
            "reference_chord": reference_chord,
            "reference_span": reference_span,
            "reference_point": reference_point,
        }

        self._last_forces = result
        self._last_reference_velocity = reference_velocity
        return result

    def log_forces_table(
        self, density: float, reference_velocity: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Log panel forces in a formatted table matching VLM diagnostics style.

        Prints per-surface forces and total forces with descriptions.

        Args:
            density: Fluid density (kg/m^3)
            reference_velocity: Reference velocity vector

        Returns:
            Dictionary of force coefficients
        """
        print("\n" + "-" * 60)
        print("PANEL AERODYNAMIC FORCES")
        print("-" * 60)
        method_name = self.force_config.method
        print(f"  Force computation method: {method_name}")
        print("    lift = force perpendicular to freestream")
        print("    drag = force parallel to freestream")
        print("    lift_coefficient, drag_coefficient = normalized force coefficients")
        print()

        # Get total forces
        total_forces = self.compute_forces_coefficients(density, reference_velocity)
        surface_forces = self.compute_per_surface_forces(density, reference_velocity)

        if len(surface_forces) > 1:
            print(
                f"  {'Body':<15} {'lift [N]':>12} {'drag [N]':>12} "
                f"{'lift_coefficient':>18} {'drag_coefficient':>18} {'Panels':>8}"
            )
            print(f"  {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 18} {'-' * 18} {'-' * 8}")
            for uid, forces in surface_forces.items():
                print(
                    f"  {uid:<15} {forces['lift']:>12.3f} {forces['drag']:>12.3f} "
                    f"{forces['lift_coefficient']:>18.3f} {forces['drag_coefficient']:>18.3f} "
                    f"{forces['panel_count']:>8}"
                )
            print(f"  {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 18} {'-' * 18} {'-' * 8}")

        # Print totals
        L = total_forces.get("lift", 0.0)
        D = total_forces.get("drag", 0.0)
        lift_coefficient = total_forces.get("lift_coefficient", 0.0)
        drag_coefficient = total_forces.get("drag_coefficient", 0.0)
        side_force_coefficient = total_forces.get("side_force_coefficient", 0.0)
        L_D = L / D if abs(D) > 1e-10 else float("inf")

        print(
            f"  {'TOTAL':<15} {L:>12.3f} {D:>12.3f} {lift_coefficient:>10.3f} {drag_coefficient:>10.3f}"
        )
        print()
        print(f"  Lift/Drag Ratio (L/D)    : {L_D:.2f}")
        print(f"  Side-force coefficient   : {side_force_coefficient:.3f}")

        # Moments
        rolling_moment_coefficient = total_forces.get("rolling_moment_coefficient", 0.0)
        pitching_moment_coefficient = total_forces.get("pitching_moment_coefficient", 0.0)
        yawing_moment_coefficient = total_forces.get("yawing_moment_coefficient", 0.0)

        print()
        print("  Moment Coefficients:")
        print(f"    rolling_moment_coefficient : {rolling_moment_coefficient:>12.3f}")
        print(f"    pitching_moment_coefficient: {pitching_moment_coefficient:>12.3f}")
        print(f"    yawing_moment_coefficient  : {yawing_moment_coefficient:>12.3f}")

        if "reference_point" in total_forces:
            r = total_forces["reference_point"]
            print(f"    Ref Center   : [{r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f}]")

        print("-" * 60, flush=True)

        return total_forces

    def compute_per_surface_forces(
        self,
        density: float,
        reference_velocity: np.ndarray | None = None,
        reference_area: float | None = None,
        reference_chord: float | None = None,
        reference_span: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute forces for each individual surface (body group).

        Args:
            density: Fluid density (kg/m³)
            reference_velocity: Reference velocity vector [ux, uy, uz]
            reference_area: Reference area (m²). If None, uses sum of panel area.
            reference_chord: Reference chord (m). If None, uses sqrt(reference_area).
            reference_span: Reference span (m). If None, uses sqrt(reference_area).

        Returns:
            Dictionary mapping surface name to force dictionary
        """
        if self.lattice is None or self.lattice.n_panels == 0:
            return {}

        if self._has_body_motion():
            raise NotImplementedError(
                "Steady Bernoulli force coefficients are unsupported for moving bodies."
            )

        if not self._solved:
            return {}

        # Resolve reference values
        if reference_velocity is None:
            reference_velocity = (
                self.freestream_velocity
                if self.freestream_velocity is not None
                else np.array([1.0, 0.0, 0.0])
            )
        reference_velocity_mag = np.linalg.norm(reference_velocity)
        if reference_velocity_mag < 1e-10:
            reference_velocity_mag = 1.0

        n = self.lattice.n_panels
        panel_force = self.panel_force.to_numpy()[:n]
        panel_area = self.lattice.area.to_numpy()[:n]
        panel_centre = self.lattice.panel_centre.to_numpy()[:n]

        dynamic_pressure = 0.5 * density * reference_velocity_mag**2
        reference_direction = (
            reference_velocity / reference_velocity_mag
            if reference_velocity_mag > 1e-10
            else np.array([1.0, 0.0, 0.0])
        )
        vertical_direction = np.array([0.0, 0.0, 1.0])
        lift_direction = (
            vertical_direction
            - np.dot(vertical_direction, reference_direction) * reference_direction
        )
        lift_norm = np.linalg.norm(lift_direction)
        lift_direction = lift_direction / lift_norm if lift_norm > 1.0e-10 else vertical_direction
        side_force_direction = np.cross(reference_direction, lift_direction)

        result = {}
        for body in self.lattice.bodies:
            uid = body.uid
            body_slice = slice(body.start_idx, body.start_idx + body.count)
            body_force = panel_force[body_slice]
            body_area = panel_area[body_slice]
            surface_force = np.sum(body_force, axis=0)
            force_x, force_y, force_z = surface_force
            drag = np.dot(surface_force, reference_direction)
            lift = np.dot(surface_force, lift_direction)
            side_force = np.dot(surface_force, side_force_direction)
            body_reference_area = (
                float(reference_area)
                if reference_area is not None
                else (
                    float(body.reference_area)
                    if body.reference_area is not None
                    else float(np.sum(body_area))
                )
            )
            body_reference_chord = (
                float(reference_chord)
                if reference_chord is not None
                else float(np.sqrt(body_reference_area))
            )
            body_reference_span = (
                float(reference_span)
                if reference_span is not None
                else float(np.sqrt(body_reference_area))
            )
            body_force_normalization = dynamic_pressure * body_reference_area
            body_centres = panel_centre[body_slice]
            if np.sum(body_area) > 0.0:
                body_reference_point = np.average(body_centres, axis=0, weights=body_area)
            else:
                body_reference_point = np.mean(body_centres, axis=0)
            body_moment = np.sum(np.cross(body_centres - body_reference_point, body_force), axis=0)
            if body_force_normalization > 1e-10:
                body_coefficients = {
                    "lift_coefficient": float(lift / body_force_normalization),
                    "drag_coefficient": float(drag / body_force_normalization),
                    "side_force_coefficient": float(side_force / body_force_normalization),
                    "force_coefficient_x": float(force_x / body_force_normalization),
                    "force_coefficient_y": float(force_y / body_force_normalization),
                    "force_coefficient_z": float(force_z / body_force_normalization),
                    "rolling_moment_coefficient": float(
                        body_moment[0] / (body_force_normalization * body_reference_span)
                    ),
                    "pitching_moment_coefficient": float(
                        body_moment[1] / (body_force_normalization * body_reference_chord)
                    ),
                    "yawing_moment_coefficient": float(
                        body_moment[2] / (body_force_normalization * body_reference_span)
                    ),
                }
            else:
                body_coefficients = dict.fromkeys(
                    (
                        "lift_coefficient",
                        "drag_coefficient",
                        "side_force_coefficient",
                        "force_coefficient_x",
                        "force_coefficient_y",
                        "force_coefficient_z",
                        "rolling_moment_coefficient",
                        "pitching_moment_coefficient",
                        "yawing_moment_coefficient",
                    ),
                    0.0,
                )
            result[uid] = {
                "lift": float(lift),
                "drag": float(drag),
                "side_force": float(side_force),
                "force_x": float(force_x),
                "force_y": float(force_y),
                "force_z": float(force_z),
                "moment_x": float(body_moment[0]),
                "moment_y": float(body_moment[1]),
                "moment_z": float(body_moment[2]),
                "reference_point": body_reference_point,
                "reference_area": body_reference_area,
                **body_coefficients,
                "panel_count": body.count,
            }

        return result
