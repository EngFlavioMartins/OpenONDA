"""Coupling iterations: fixed-point (Picard) iteration of the boundary data in one coupling interval."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import TypeAlias, TypedDict

import numpy as np

from openonda.core.schedules import EverySteps, EveryTime
from openonda.core.simulation.parallel import collective_phase
from openonda.fvm.io.backup import capture_restart_state, restore_restart_state

from .boundary import advance_fvm, update_boundary_history_after_replacement
from .vorticity_transfer import _particle_state_snapshot, _restore_particle_state

logger = logging.getLogger("openonda.coupler")
_BOUNDARY_DATA_FIELDS = (
    "velocity_boundary_condition",
    "normal_velocity_boundary_condition",
    "normal_gradient_boundary_condition",
)

BoundaryData: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]
"""Face velocity in m/s, normal velocity in m/s and normal gradient of the tangential velocity in 1/s."""


class CouplingIterationRow(TypedDict):
    """Area-weighted RMS change of the boundary data over one coupling iteration."""

    iteration: int
    normal_velocity_residual: float
    normal_gradient_residual: float
    scaled_residual: float
    converged: bool


def validate_output_schedules(
    fvm_case, n_fvm_substeps: int, coupling_time_step_size: float
) -> None:
    """Reject FVM frames and samples that would fall inside a coupling interval.

    With coupling iterations the FVM output waits for the accepted end of each
    coupling interval, so every FVM output schedule must select coupling times only.
    """
    schedules = [fvm_case.output.schedule]
    schedules.extend(sampler.schedule for sampler in fvm_case.samplers.samples)
    for schedule in schedules:
        if isinstance(schedule, EverySteps):
            aligned = schedule.interval % n_fvm_substeps == 0 and (
                schedule.first_step is None or schedule.first_step % n_fvm_substeps == 0
            )
        elif isinstance(schedule, EveryTime):
            aligned = all(
                np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-9)
                for ratio in (
                    schedule.interval / coupling_time_step_size,
                    schedule.start_time / coupling_time_step_size,
                )
            )
        else:
            aligned = True
        if not aligned:
            raise ValueError(
                "Coupling iterations require FVM output and sampling schedules on coupling times"
            )


def _read_boundary_data(coupler, *, old=False, velocity=None) -> BoundaryData:
    suffix = "_old" if old else ""
    if old:
        velocity = coupler._velocity_boundary_condition_old
    return (
        np.asarray(velocity).copy(),
        np.asarray(getattr(coupler, "_" + _BOUNDARY_DATA_FIELDS[1] + suffix)).copy(),
        np.asarray(getattr(coupler, "_" + _BOUNDARY_DATA_FIELDS[2] + suffix)).copy(),
    )


def _write_boundary_data(coupler, values: BoundaryData, *, old=False) -> None:
    suffix = "_old" if old else ""
    for name, value in zip(_BOUNDARY_DATA_FIELDS, values, strict=True):
        if old or name != _BOUNDARY_DATA_FIELDS[0]:
            setattr(coupler, "_" + name + suffix, value.copy())


def _first_estimate(coupler, predictor: BoundaryData, face_normal: np.ndarray) -> BoundaryData:
    """Add the previous interval's boundary-data correction to the VPM predictor data.

    The vorticity transfer and the FVM response change the boundary data by nearly
    the same amount in consecutive coupling intervals, so this estimate usually
    lies within the tolerances after one coupling iteration. The normal velocity is
    recomputed from the corrected face velocity.
    """
    correction = coupler.boundary_data_correction
    if correction is None:
        return predictor
    velocity = predictor[0] + correction[0]
    return velocity, np.einsum("ij,ij->i", velocity, face_normal), predictor[2] + correction[2]


def advance_coupling_iterations(coupler, geometry, next_velocity):
    """Repeat the FVM interval and the vorticity transfer until the boundary data converge.

    The VPM advances once per coupling interval, to the predictor. Every coupling
    iteration restarts the FVM and the advected particles from the same state,
    sub-cycles the FVM with the end-of-interval estimate ``b^(m)`` of the boundary
    data, transfers the FVM vorticity and evaluates the boundary data of the
    corrected particles. Those data are the next estimate. The first estimate is :func:`_first_estimate`. Only the accepted
    iteration writes FVM output. MPI ranks share the root's residuals and
    stopping decision.

    Returns
    -------
    tuple
        Accepted transfer result, FVM seconds including the accepted output, and
        transfer seconds including state capture and restoration.
    """
    fvm, vpm, transfer = coupler.fvm_solver, coupler.vpm_solver, coupler.vorticity_transfer
    comm = getattr(fvm.parallel, "comm", None)
    face_area = geometry[2]
    started = perf_counter()
    with collective_phase(comm, "coupling-iteration state capture"):
        fvm_start = capture_restart_state(fvm)
        particles_start = (
            _particle_state_snapshot(vpm, slot="coupler-predictor") if coupler._is_root else None
        )
        old = _read_boundary_data(coupler, old=True)
        predictor = _read_boundary_data(coupler, velocity=next_velocity)
        estimate = _first_estimate(coupler, predictor, geometry[1])
    first_estimate = "vpm_predictor" if estimate is predictor else "previous_correction"
    transfer_step = transfer.step
    phase_seconds = {
        "state_capture": perf_counter() - started,
        "state_restore": 0.0,
        "boundary_refresh": 0.0,
        "accepted_output": 0.0,
    }
    fvm_seconds = transfer_seconds = 0.0
    rows: list[CouplingIterationRow] = []
    tolerance = (
        coupler.config.normal_velocity_tolerance,
        coupler.config.normal_gradient_tolerance,
    )
    for iteration in range(1, coupler.config.max_coupling_iterations + 1):
        started = perf_counter()
        if iteration > 1:
            restore_restart_state(fvm, fvm_start)
            if coupler._is_root:
                _restore_particle_state(vpm, particles_start)
            transfer.step = transfer_step
            _write_boundary_data(coupler, old, old=True)
        _write_boundary_data(coupler, estimate)
        phase_seconds["state_restore"] += perf_counter() - started
        fvm_seconds += advance_fvm(coupler, *geometry, old[0], estimate[0])
        started = perf_counter()
        result, _ = coupler._transfer_vorticity_to_vpm(*geometry)
        transfer_seconds += perf_counter() - started
        started = perf_counter()
        update_boundary_history_after_replacement(coupler, *geometry)
        phase_seconds["boundary_refresh"] += perf_counter() - started
        row: CouplingIterationRow | None = None
        if coupler._is_root:
            updated = _read_boundary_data(coupler, old=True)
            normal = float(np.sqrt(np.average((updated[1] - estimate[1]) ** 2, weights=face_area)))
            gradient = float(
                np.sqrt(
                    np.average(np.sum((updated[2] - estimate[2]) ** 2, axis=1), weights=face_area)
                )
            )
            row = {
                "iteration": iteration,
                "normal_velocity_residual": normal,
                "normal_gradient_residual": gradient,
                "scaled_residual": max(normal / tolerance[0], gradient / tolerance[1]),
                "converged": bool(normal <= tolerance[0] and gradient <= tolerance[1]),
            }
            estimate = updated
        if fvm.parallel.is_parallel:
            row = fvm.parallel.comm.bcast(row, root=0)
        assert row is not None
        rows.append(row)
        if row["converged"]:
            break
    converged = rows[-1]["converged"]
    # The accepted end-of-interval data are those evaluated in the last iteration.
    coupler.boundary_data_correction = (
        tuple(accepted - first for accepted, first in zip(estimate, predictor, strict=True))
        if converged
        else None
    )
    started = perf_counter()
    fvm.write_step_output()
    phase_seconds["accepted_output"] = perf_counter() - started
    fvm_seconds += phase_seconds["accepted_output"]
    transfer_seconds += phase_seconds["state_capture"] + phase_seconds["state_restore"]
    coupler._last_coupling_iteration_diagnostics = {
        "iterations": len(rows),
        "converged": converged,
        "first_estimate": first_estimate,
        "residuals": rows,
        "phase_seconds": phase_seconds,
    }
    if coupler._is_root:
        logger.log(
            logging.INFO if converged else logging.WARNING,
            "coupler  coupling iterations | iterations=%d | converged=%s | "
            "normal velocity=%.3e | normal gradient=%.3e",
            len(rows),
            converged,
            rows[-1]["normal_velocity_residual"],
            rows[-1]["normal_gradient_residual"],
        )
    return result, fvm_seconds, transfer_seconds
