"""Picard iteration of the FVM--VPM interface trace within one exchange."""

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
_TRACE_FIELDS = (
    "velocity_boundary_condition",
    "normal_velocity_boundary_condition",
    "tangential_gradient_boundary_condition",
)

BoundaryTrace: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]
"""Face velocity in m/s, normal velocity in m/s and tangential du/dn in 1/s."""


class IterationRow(TypedDict):
    """Area-weighted RMS change of the interface trace over one sweep."""

    sweep: int
    normal_residual_rms: float
    gradient_residual_rms: float
    scaled_residual: float
    converged: bool


def validate_output_schedules(
    fvm_case, n_fvm_substeps: int, exchange_time_step_size: float
) -> None:
    """Reject FVM frames and samples that would fall inside a provisional interval.

    Iterated exchanges defer the FVM output to the converged end of each
    exchange, so every FVM output schedule must select exchange times only.
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
                    schedule.interval / exchange_time_step_size,
                    schedule.start_time / exchange_time_step_size,
                )
            )
        else:
            aligned = True
        if not aligned:
            raise ValueError(
                "Iterated coupling requires FVM output and sampling schedules on exchange times"
            )


def _read_trace(coupler, *, old=False, velocity=None) -> BoundaryTrace:
    suffix = "_old" if old else ""
    if old:
        velocity = coupler._velocity_boundary_condition_old
    return (
        np.asarray(velocity).copy(),
        np.asarray(getattr(coupler, "_" + _TRACE_FIELDS[1] + suffix)).copy(),
        np.asarray(getattr(coupler, "_" + _TRACE_FIELDS[2] + suffix)).copy(),
    )


def _write_trace(coupler, values: BoundaryTrace, *, old=False) -> None:
    suffix = "_old" if old else ""
    for name, value in zip(_TRACE_FIELDS, values, strict=True):
        if old or name != _TRACE_FIELDS[0]:
            setattr(coupler, "_" + name + suffix, value.copy())


def _initial_guess(coupler, predictor: BoundaryTrace, face_normal: np.ndarray) -> BoundaryTrace:
    """Add the previous exchange's converged trace correction to the VPM predictor.

    Renewal and the FVM response change the interface trace by nearly the same
    amount in consecutive exchanges, so this guess usually lies within the
    tolerance after one sweep. The normal velocity is recomputed from the
    corrected face velocity.
    """
    correction = coupler.interface_trace_correction
    if correction is None:
        return predictor
    velocity = predictor[0] + correction[0]
    return velocity, np.einsum("ij,ij->i", velocity, face_normal), predictor[2] + correction[2]


def advance_iterated_interface(coupler, geometry, next_velocity):
    """Repeat the FVM interval and renewal until the interface trace converges.

    The VPM is advanced once per exchange. Every sweep restarts the FVM and the
    advected particles from the same state, sub-cycles the FVM with the
    candidate trace at ``t_{n+1}``, renews the overlap and re-evaluates the VPM
    trace. That re-evaluated trace is the next candidate (Picard iteration).
    The first candidate is :func:`_initial_guess`. Only the final sweep writes
    FVM output. MPI ranks share the master's residuals and stop decision.

    Returns
    -------
    tuple
        Final transfer result, FVM seconds including accepted output, and
        transfer seconds including state capture and restoration.
    """
    fvm, vpm, transfer = coupler.fvm_solver, coupler.vpm_solver, coupler.vorticity_transfer
    comm = getattr(fvm.parallel, "comm", None)
    face_area = geometry[2]
    started = perf_counter()
    with collective_phase(comm, "interface initial state capture"):
        fvm_start = capture_restart_state(fvm)
        particles_start = (
            _particle_state_snapshot(vpm, slot="coupler-predictor") if coupler._is_root else None
        )
        old = _read_trace(coupler, old=True)
        predictor = _read_trace(coupler, velocity=next_velocity)
        candidate = _initial_guess(coupler, predictor, geometry[1])
    initial_guess = "vpm_trace" if candidate is predictor else "previous_correction"
    transfer_step = transfer.step
    phase_seconds = {
        "state_capture": perf_counter() - started,
        "state_restore": 0.0,
        "boundary_refresh": 0.0,
        "accepted_output": 0.0,
    }
    fvm_seconds = transfer_seconds = 0.0
    rows: list[IterationRow] = []
    tolerance = (
        coupler.config.interface_normal_tolerance,
        coupler.config.interface_gradient_tolerance,
    )
    for sweep in range(1, coupler.config.interface_iterations + 1):
        started = perf_counter()
        if sweep > 1:
            restore_restart_state(fvm, fvm_start)
            if coupler._is_root:
                _restore_particle_state(vpm, particles_start)
            transfer.step = transfer_step
            _write_trace(coupler, old, old=True)
        _write_trace(coupler, candidate)
        phase_seconds["state_restore"] += perf_counter() - started
        fvm_seconds += advance_fvm(coupler, *geometry, old[0], candidate[0])
        started = perf_counter()
        result, _ = coupler._transfer_vorticity_to_vpm(*geometry)
        transfer_seconds += perf_counter() - started
        started = perf_counter()
        update_boundary_history_after_replacement(coupler, *geometry)
        phase_seconds["boundary_refresh"] += perf_counter() - started
        row: IterationRow | None = None
        if coupler._is_root:
            updated = _read_trace(coupler, old=True)
            normal = float(np.sqrt(np.average((updated[1] - candidate[1]) ** 2, weights=face_area)))
            gradient = float(
                np.sqrt(
                    np.average(np.sum((updated[2] - candidate[2]) ** 2, axis=1), weights=face_area)
                )
            )
            row = {
                "sweep": sweep,
                "normal_residual_rms": normal,
                "gradient_residual_rms": gradient,
                "scaled_residual": max(normal / tolerance[0], gradient / tolerance[1]),
                "converged": bool(normal <= tolerance[0] and gradient <= tolerance[1]),
            }
            candidate = updated
        if fvm.parallel.is_parallel:
            row = fvm.parallel.comm.bcast(row, root=0)
        assert row is not None
        rows.append(row)
        if row["converged"]:
            break
    converged = rows[-1]["converged"]
    # The accepted endpoint is the re-evaluated trace of the final sweep.
    coupler.interface_trace_correction = (
        tuple(final - first for final, first in zip(candidate, predictor, strict=True))
        if converged
        else None
    )
    started = perf_counter()
    fvm.write_step_output()
    phase_seconds["accepted_output"] = perf_counter() - started
    fvm_seconds += phase_seconds["accepted_output"]
    transfer_seconds += phase_seconds["state_capture"] + phase_seconds["state_restore"]
    coupler._last_interface_iteration_diagnostics = {
        "sweeps": len(rows),
        "converged": converged,
        "initial_guess": initial_guess,
        "residuals": rows,
        "phase_seconds": phase_seconds,
    }
    if coupler._is_root:
        logger.log(
            logging.INFO if converged else logging.WARNING,
            "coupler  interface iteration | sweeps=%d | converged=%s | normal=%.3e | gradient=%.3e",
            len(rows),
            converged,
            rows[-1]["normal_residual_rms"],
            rows[-1]["gradient_residual_rms"],
        )
    return result, fvm_seconds, transfer_seconds
