"""One FVM domain of a coupled run and the checks that keep several domains apart.

A coupled run holds one particle set and one or more FVM domains, each with its
own mesh, mesh motion, outer boundary and transfer region. A
:class:`CoupledDomain` carries what belongs to one domain: its FVM solver, the
frame of its mesh, its vorticity transfer, the boundary data of its outer
boundary and their history. The shared particle set, the coupling clock and the
freestream belong to the :class:`~openonda.coupler.solver.CoupledSolver`, which
the domain reads through the properties below.
"""

from __future__ import annotations

from itertools import combinations
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from openonda.core.parallel import collective_phase

from .boundary import (
    advance_fvm,
    evaluate_vpm_boundary,
    reference_speed,
    update_boundary_history_after_replacement,
)
from .config import CouplerConfig
from .frame import MeshFrame

if TYPE_CHECKING:
    from openonda.fvm import FVMCase, FVMSolver

    from .solver import CoupledSolver

logger = logging.getLogger("openonda.coupler")

INTERVAL_PHASES = ("boundary", "fvm", "transfer")
"""Phases of one domain whose wall time each coupling interval records."""

M4_SUPPORT_SPACINGS = 2.0
"""Particle spacings beyond its remeshing region that a transfer may change: the M4' support."""


class CoupledDomain:
    """One FVM domain of a coupled run: its solver, frame, transfer and boundary data.

    Parameters
    ----------
    coupler : CoupledSolver
        The coupled solver that holds the particle set and the clock.
    fvm_case : FVMCase
        Mesh, mesh motion, flow settings and output of this domain.
    config : CouplerConfig
        Coupling settings; those of :data:`~openonda.coupler.config.DOMAIN_SETTINGS`
        belong to this domain, the others equal those of every domain.
    fvm_solver : FVMSolver
        The constructed FVM solver of ``fvm_case``.
    follows_mesh : bool
        Whether the particle frame translates with this mesh (one domain).

    Attributes
    ----------
    name : str
        The FVM case name, which names the domain in diagnostics and backups.
    frame : MeshFrame
        Translation and rotation of the mesh seen from the particle frame.
    freestream_velocity : numpy.ndarray
        Freestream in m/s relative to this mesh.
    fvm_time_step_size : float
        Current FVM substep in s.
    n_fvm_substeps : int
        FVM substeps of the last coupling interval.
    """

    def __init__(
        self,
        coupler: CoupledSolver,
        fvm_case: FVMCase,
        config: CouplerConfig,
        fvm_solver: FVMSolver,
        *,
        follows_mesh: bool,
    ) -> None:
        self.coupler = coupler
        self.name = fvm_case.name
        self.fvm_case = fvm_case
        self.config = config
        self.fvm_solver = fvm_solver
        self.frame = MeshFrame(fvm_solver, follows_mesh=follows_mesh)
        self.freestream_velocity = config.freestream_velocity_vector
        self.boundary_data_correction: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.previous_boundary_data_correction: tuple[np.ndarray, np.ndarray, np.ndarray] | None = (
            None
        )
        # Lengths in s of the coupling intervals of the two corrections above.
        self.boundary_data_correction_interval = 0.0
        self.previous_boundary_data_correction_interval = 0.0
        self._velocity_boundary_condition_old: np.ndarray | None = None
        self._normal_velocity_boundary_condition_old: np.ndarray | None = None
        self._normal_velocity_boundary_condition: np.ndarray | None = None
        self._normal_gradient_boundary_condition_old: np.ndarray | None = None
        self._normal_gradient_boundary_condition: np.ndarray | None = None
        self._gathered_velocity: np.ndarray | None = None
        self._gathered_velocity_gradient: np.ndarray | None = None
        self._last_vpm_boundary_condition_flux_diagnostics = {
            "raw_mismatch": 0.0,
            "raw_relative": 0.0,
            "acceptance_limit": 0.0,
            "applied_correction": 0.0,
            "corrected_mismatch": 0.0,
        }
        self._last_fvm_boundary_data_diagnostics = {
            "mean_velocity_mismatch": 0.0,
            "maximum_velocity_mismatch": 0.0,
            "maximum_normal_velocity_mismatch": 0.0,
            "mean_outflow_velocity_mismatch": 0.0,
            "maximum_outflow_velocity_mismatch": 0.0,
        }
        self._last_transfer_result = None
        self._step_transfer_stats: dict[str, float | int] | None = None
        self._interval_pose = None
        # Wall time in s of this domain's phases in the current coupling interval.
        self._interval_seconds = dict.fromkeys(INTERVAL_PHASES, 0.0)
        self.fvm_time_step_size = float(fvm_case.numerics.time_step_size)
        self.n_fvm_substeps = 1
        control = fvm_case.numerics.time_step_control
        self.maximum_fvm_time_step_size = (
            float(control.maximum_time_step_size)
            if control is not None and control.maximum_time_step_size is not None
            else self.fvm_time_step_size
            if control is None
            else None
        )

    # Shared state of the coupled solver

    @property
    def vpm_solver(self):
        """The particle solver on the root rank, ``None`` elsewhere."""
        return self.coupler.vpm_solver

    @property
    def vpm_case(self):
        return self.coupler.vpm_case

    @property
    def _comm(self):
        return self.coupler._comm

    @property
    def _is_root(self) -> bool:
        return self.coupler._is_root

    @property
    def step(self) -> int:
        """Number of completed coupling steps."""
        return self.coupler.step

    @property
    def time(self) -> float:
        """Accepted physical time in s."""
        return self.fvm_solver.time

    @property
    def end_time(self) -> float:
        return self.coupler.end_time

    @property
    def vpm_time_step_size(self) -> float:
        return self.coupler.vpm_time_step_size

    @property
    def maximum_coupling_time_step_size(self) -> float:
        return self.coupler.maximum_coupling_time_step_size

    @property
    def vpm_particle_spacing(self) -> float:
        return self.coupler.vpm_particle_spacing

    @property
    def vpm_core_radius_ratio(self) -> float:
        return self.coupler.vpm_core_radius_ratio

    @property
    def kinematic_viscosity(self) -> float:
        return self.coupler.kinematic_viscosity

    @property
    def absolute_freestream_velocity(self) -> np.ndarray:
        """Inertial-frame freestream in m/s."""
        return self.coupler.absolute_freestream_velocity

    def coupling_time(self, step: int) -> float:
        return self.coupler.coupling_time(step)

    def _coupling_event_schedules(self) -> tuple:
        return self.coupler._coupling_event_schedules()

    def inertial_boundary_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Convert a particle velocity, with the particle freestream, to the inertial frame."""
        return self.coupler.inertial_boundary_velocity(velocity)

    # Geometry of the outer boundary

    def read_outer_boundary(self) -> None:
        """Read the outer-boundary faces in body coordinates (collective)."""
        fvm = self.fvm_solver
        patch = self.config.outer_boundary_patch
        frame = self.frame
        self.face_centre_body = frame.to_body(
            frame.to_frame(
                np.asarray(fvm.gather_boundary_face_centre_coordinates(patch)).reshape(-1, 3)
            )
        )
        self._face_normal_body = frame.vectors_to_body(
            np.asarray(fvm.gather_boundary_face_normal(patch)).reshape(-1, 3)
        )
        self._face_area = np.asarray(fvm.gather_boundary_face_area(patch)).ravel()
        self._face_geometry = self._face_geometry_at(None)

    def _face_geometry_at(self, pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Frame centres, unit normals and areas of the outer-boundary faces at a mesh pose."""
        return (
            self.frame.from_body(self.face_centre_body, pose),
            self.frame.vectors_from_body(self._face_normal_body, pose),
            self._face_area,
        )

    def place_for_interval(self, time_end: float) -> None:
        """Move the outer-boundary faces seen by the particles to the pose at ``time_end``."""
        if self.frame.moves:
            self._interval_pose = self.frame.pose_after(np.asarray([time_end]))
            self._face_geometry = self._face_geometry_at(self._interval_pose)

    def remeshing_region_in_frame(self, pose=None) -> tuple[float, ...] | None:
        """Frame bounds of the remeshing region, translated with the body; ``None`` without one.

        The region is kept in body axes: a rotating mesh is not turned, as for
        the strain limit of the particles it excludes.
        """
        lattice = self.vorticity_transfer._transfer_lattice
        if lattice is None:
            return None
        box = np.asarray(lattice.remeshing_region_bounds, dtype=np.float64)
        if self.frame.translates:
            box = box + np.repeat(self.frame.offset(pose), 2)
        return tuple(float(value) for value in box)

    # Vorticity transfer

    def transfer_vorticity_to_vpm(self):
        """Replace the particles of this domain's remeshing region by its FVM vorticity."""
        started = time.perf_counter()
        velocity = self._gather_velocity_field()
        gradient = self._gather_velocity_gradient_field()
        frame = self.frame
        if frame.moving:
            # The particles see the flow relative to their frame; the transfer
            # reads it in body axes, G[i, j] = d(u_j)/d(x_i).
            velocity = frame.vectors_to_body(velocity - frame.velocity(self.fvm_solver.time))
            if frame.rotates:
                rotation = frame.rotation()
                gradient = rotation.T @ gradient @ rotation
        gather_seconds = time.perf_counter() - started
        transfer_result = None
        with collective_phase(self._comm, "vorticity transfer"):
            if self._is_root:
                # The first transfer at the start precedes any boundary data.
                boundary_velocity = self._velocity_boundary_condition_old
                self.vorticity_transfer.reference_speed = reference_speed(
                    self,
                    np.empty((0, 3)) if boundary_velocity is None else boundary_velocity,
                )
                transfer_result = self.vorticity_transfer.transfer(
                    self.vpm_solver, velocity=velocity, velocity_gradient=gradient
                )
                self._step_transfer_stats = {
                    "n_before": transfer_result.n_particles_before,
                    "n_after": transfer_result.n_particles_after,
                    "sum_before": transfer_result.input_vortex_strength_l1,
                    "sum_after": transfer_result.particle_set_vortex_strength_l1,
                    "face_count": len(self._face_geometry[0]),
                    "fvm_gather_seconds": gather_seconds,
                }
        self._last_transfer_result = transfer_result
        seconds = time.perf_counter() - started
        self._interval_seconds["transfer"] += seconds
        return transfer_result, seconds

    def _gather_velocity_field(self) -> np.ndarray:
        if self._gathered_velocity is None:
            self._gathered_velocity = np.ascontiguousarray(
                self.fvm_solver.gather_velocity_field(), dtype=np.float64
            ).reshape(-1, 3)
        else:
            self.fvm_solver.gather_velocity_field_into(self._gathered_velocity)
        return self._gathered_velocity

    def _gather_velocity_gradient_field(self) -> np.ndarray:
        if self._gathered_velocity_gradient is None:
            self._gathered_velocity_gradient = np.ascontiguousarray(
                self.fvm_solver.gather_velocity_gradient_field(), dtype=np.float64
            ).reshape(-1, 3, 3)
        else:
            self.fvm_solver.gather_velocity_gradient_field_into(self._gathered_velocity_gradient)
        return self._gathered_velocity_gradient

    def advance_fvm(self, velocity: np.ndarray) -> float:
        """Advance the FVM substeps of the interval with the end-time face velocity; return seconds."""
        seconds = advance_fvm(self, *self._face_geometry, velocity)
        self._interval_seconds["fvm"] += seconds
        return seconds

    def refresh_boundary_data(self) -> float:
        """Evaluate the start-of-interval boundary data from the particles after the transfers."""
        started = time.perf_counter()
        update_boundary_history_after_replacement(self, *self._face_geometry)
        seconds = time.perf_counter() - started
        self._interval_seconds["boundary"] += seconds
        return seconds

    def predictor_boundary_data(self) -> tuple[np.ndarray, float]:
        """Evaluate the end-of-interval face velocity from the predictor particles; return it and seconds."""
        _, velocity, seconds = evaluate_vpm_boundary(self, *self._face_geometry)
        self._interval_seconds["boundary"] += seconds
        return velocity, seconds

    # Output

    def fvm_frame_written(self, frame_step: int | None) -> bool:
        """Whether the FVM frame of the current coupling step already exists."""
        if frame_step == self.step:
            return True
        schedule = self.fvm_case.output.schedule
        fvm = self.fvm_solver
        return (
            self.step > 0
            and schedule is not None
            and schedule.is_due(fvm.step, fvm.time, fvm._accepted_time_step_size)
        )


def _separated(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two convex point sets, eight box corners each, have a separating plane.

    The candidate normals are the face normals of both boxes and their edge
    cross products (the separating-axis test of two oriented boxes).
    """
    edges = []
    for corners in (first, second):
        # Corners from meshgrid(ij): neighbours along the three box axes.
        edges.append([corners[4] - corners[0], corners[2] - corners[0], corners[1] - corners[0]])
    axes = [*edges[0], *edges[1]]
    axes.extend(np.cross(a, b) for a in edges[0] for b in edges[1])
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12 * max(1.0, float(np.max(np.abs(first)))):
            continue
        a = first @ (axis / norm)
        b = second @ (axis / norm)
        if a.max() < b.min() or b.max() < a.min():
            return True
    return False


def _expanded(box: np.ndarray, margin: float) -> np.ndarray:
    return np.asarray(box, dtype=np.float64) + margin * np.array([-1.0, 1.0] * 3)


def remeshing_regions_overlap(first: CoupledDomain, second: CoupledDomain, poses) -> bool:
    """Whether the remeshing regions of two domains, with their M4' supports, meet at ``poses``.

    ``poses`` is the pair of mesh poses of the two domains at one time.
    """
    margin = M4_SUPPORT_SPACINGS * first.vpm_particle_spacing
    corners = [
        domain.frame.box_corners(
            _expanded(domain.vorticity_transfer._transfer_lattice.remeshing_region_bounds, margin),
            pose,
        )
        for domain, pose in zip((first, second), poses, strict=True)
    ]
    return not _separated(*corners)


def check_remeshing_regions_apart(
    domains, start_time: float, end_time: float, time_step_size: float
) -> None:
    """Stop before the first step when the remeshing regions of two domains meet during the run.

    A transfer replaces the particles of its remeshing region and may change
    those of its M4' support, two particle spacings beyond it; the supports
    of two domains must therefore stay apart at every time, so that the
    transfers of the domains act on distinct particles and their order does
    not matter. The poses are those of the prescribed motions, four per FVM
    step; a flow-driven motion is checked again every coupling interval
    (:func:`check_interval_regions_apart`).
    """
    for first, second in combinations(domains, 2):
        if first.vorticity_transfer._transfer_lattice is None:
            continue
        if second.vorticity_transfer._transfer_lattice is None:
            continue
        times, first_poses = first.frame.sample_poses(start_time, end_time, time_step_size)
        _, second_poses = second.frame.sample_poses(start_time, end_time, time_step_size)
        for when, poses in zip(times, zip(first_poses, second_poses, strict=True), strict=True):
            if remeshing_regions_overlap(first, second, poses):
                raise ValueError(_overlap_message(first, second, float(when)))
            if not (first.frame.moves or second.frame.moves):
                break


def check_interval_regions_apart(domains, time_end: float) -> None:
    """Stop when the remeshing regions of a flow-driven domain and another meet at ``time_end``."""
    for first, second in combinations(domains, 2):
        if not (first.frame.flow_driven or second.frame.flow_driven):
            continue
        poses = (first._interval_pose, second._interval_pose)
        if remeshing_regions_overlap(first, second, poses):
            raise ValueError(_overlap_message(first, second, time_end))


def _overlap_message(first: CoupledDomain, second: CoupledDomain, when: float) -> str:
    spacings = 2.0 * M4_SUPPORT_SPACINGS
    return (
        f"The remeshing regions of the FVM domains {first.name!r} and {second.name!r} come "
        f"closer than {spacings:g} particle spacings at t = {when:.6g} s: their transfers would "
        "change the same particles. Move the bodies apart or shrink the transfer regions."
    )


def lattice_phase(anchors: list[np.ndarray]) -> np.ndarray:
    """The particle-lattice anchor shared by all domains: the smallest natural anchor.

    The smallest in the order (x, y, z) does not depend on the order in which
    the domains are listed.
    """
    return np.asarray(min(anchors, key=lambda anchor: tuple(anchor.tolist())), dtype=np.float64)


__all__ = [
    "CoupledDomain",
    "check_interval_regions_apart",
    "check_remeshing_regions_apart",
    "lattice_phase",
    "remeshing_regions_overlap",
]
