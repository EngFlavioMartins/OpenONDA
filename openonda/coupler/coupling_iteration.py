"""Fixed-point (Picard) iterations of boundary data within one VPM interval."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import TypeAlias, TypedDict

import numpy as np

from openonda.core.simulation.parallel import collective_phase
from openonda.fvm.io.backup import capture_restart_state, restore_restart_state

from .boundary import advance_fvm, reference_speed, update_boundary_history_after_replacement
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
    """Add the expected boundary-data correction of this interval to the VPM predictor data.

    The vorticity transfer and the FVM response change the boundary data at a
    rate ``r = c / Δt`` that varies slowly from interval to interval, while the
    intervals themselves differ (output times shorten some). The estimate for the
    next interval ``Δt`` is ``Δt r_k`` with ``boundary_data_extrapolation="constant"``,
    and with ``"linear"`` and two accepted corrections the rate extrapolated
    linearly in time to the middle of the next interval,
    ``Δt [r_k + (r_k − r_(k−1)) (Δt + Δt_k) / (Δt_k + Δt_(k−1))]``, which is
    ``2 c_k − c_(k−1)`` for equal intervals. The normal velocity is recomputed from
    the corrected face velocity.
    """
    correction = coupler.boundary_data_correction
    if correction is None:
        return predictor
    interval = float(coupler.vpm_time_step_size)
    rate = tuple(c / coupler.boundary_data_correction_interval for c in correction)
    previous = coupler.previous_boundary_data_correction
    if coupler.config.boundary_data_extrapolation == "linear" and previous is not None:
        previous_interval = coupler.previous_boundary_data_correction_interval
        previous_rate = tuple(c / previous_interval for c in previous)
        reach = (interval + coupler.boundary_data_correction_interval) / (
            coupler.boundary_data_correction_interval + previous_interval
        )
        rate = tuple(r + (r - p) * reach for r, p in zip(rate, previous_rate, strict=True))
    correction = tuple(interval * r for r in rate)
    velocity = predictor[0] + correction[0]
    return velocity, np.einsum("ij,ij->i", velocity, face_normal), predictor[2] + correction[2]


def coupling_tolerances(coupler, face_velocity: np.ndarray) -> tuple[float, float]:
    """Return the absolute tolerances of the normal-velocity (m/s) and normal-gradient (1/s) residuals.

    The configured tolerances are relative: to the reference speed ``U`` of the
    coupled boundary (:func:`reference_speed`) and to ``U / h`` with ``h`` the
    particle spacing, so one value fits every case.
    """
    speed = reference_speed(coupler, face_velocity)
    spacing = float(coupler.vpm_particle_spacing)
    return (
        coupler.config.normal_velocity_tolerance * speed,
        coupler.config.normal_gradient_tolerance * speed / spacing,
    )


def advance_coupling_iterations(coupler, geometry, next_velocity):
    """Repeat the FVM interval and the vorticity transfer until the boundary data converge.

    The VPM advances once to the shared step's end-time predictor. Each coupling
    iteration restarts the FVM and advected particles from the same state, solves
    the FVM substeps with the end-time estimate ``b^(m)`` of the boundary
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
    tolerance = coupling_tolerances(coupler, predictor[0])
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
        fvm_seconds += advance_fvm(coupler, *geometry, estimate[0])
        started = perf_counter()
        result, _ = coupler._transfer_vorticity_to_vpm(geometry[0])
        transfer_seconds += perf_counter() - started
        started = perf_counter()
        update_boundary_history_after_replacement(coupler, *geometry)
        phase_seconds["boundary_refresh"] += perf_counter() - started
        row: CouplingIterationRow | None = None
        imposed = estimate
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
    if not converged:
        if coupler.config.stop_on_unconverged_coupling:
            raise RuntimeError(
                f"the coupling iterations did not converge in {len(rows)} iterations at "
                f"t = {fvm.time:.6g} s: normal-velocity residual "
                f"{rows[-1]['normal_velocity_residual']:.3e} m/s (tolerance {tolerance[0]:.3e}), "
                f"normal-gradient residual {rows[-1]['normal_gradient_residual']:.3e} 1/s "
                f"(tolerance {tolerance[1]:.3e})"
            )
        if coupler._is_root:
            # The FVM ended the interval on the data it was given, so the next interval
            # starts from them rather than from the newer evaluation.
            _write_boundary_data(coupler, imposed, old=True)
    # The accepted end-of-interval data are those evaluated in the last iteration.
    coupler.previous_boundary_data_correction = (
        coupler.boundary_data_correction if converged else None
    )
    coupler.previous_boundary_data_correction_interval = coupler.boundary_data_correction_interval
    coupler.boundary_data_correction_interval = float(coupler.vpm_time_step_size)
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
