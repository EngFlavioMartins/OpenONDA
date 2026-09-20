"""Fixed-predictor interface sweeps with in-memory state and accepted output."""

from __future__ import annotations

import hashlib
import logging
from time import perf_counter

import numpy as np

from source.solvers.fvm.io.backup import capture_restart_payload, publish_restart_payload

from .boundary import advance_fvm, update_boundary_history_after_replacement
from .vorticity_transfer import _particle_state_snapshot, _restore_particle_state

logger = logging.getLogger("coupler")
_TRACE_FIELDS = (
    "velocity_boundary_condition",
    "normal_velocity_boundary_condition",
    "tangential_gradient_boundary_condition",
)


def validate_output_schedules(coupler) -> None:
    """Reject output events that would fall inside a provisional interval."""
    fvm = coupler.fvm_solver
    schedules = [fvm._output_schedule, fvm._backup_config.schedule]
    schedules.extend(fvm._sampler_schedules.values())
    if coupler.n_fvm_substeps > 1 and any(
        getattr(sampler, "schedule", None) is None for sampler in getattr(fvm, "_samplers", ())
    ):
        raise ValueError("Iterated coupling requires explicit aligned FVM sampler schedules")
    for schedule in schedules:
        if schedule is None or schedule.final_only:
            continue
        if schedule.every_n_steps is not None:
            aligned = schedule.every_n_steps % coupler.n_fvm_substeps == 0
        else:
            ratio = schedule.every_time / coupler.vpm_time_step_size
            aligned = round(ratio) >= 1 and np.isclose(ratio, round(ratio), rtol=0, atol=1e-10)
        if not aligned:
            raise ValueError(
                "Iterated coupling requires FVM output and sampling schedules aligned with VPM exchange times"
            )


def _read_trace(coupler, *, old=False, velocity=None):
    suffix = "_old" if old else ""
    if old:
        velocity = coupler._velocity_boundary_condition_old
    return (
        np.asarray(velocity).copy(),
        *(np.asarray(getattr(coupler, "_" + name + suffix)).copy() for name in _TRACE_FIELDS[1:]),
    )


def _write_trace(coupler, values, *, old=False):
    suffix = "_old" if old else ""
    for name, value in zip(_TRACE_FIELDS, values, strict=True):
        if old or name != _TRACE_FIELDS[0]:
            setattr(coupler, "_" + name + suffix, value.copy())


def advance_iterated_interface(coupler, geometry, next_velocity):
    """Advance only FVM/renewal repeatedly; the VPM predictor stays fixed.

    Each sweep begins from the same accepted FVM state and advected particles.
    Only the converged/final sweep publishes samples. No reference field,
    experiment replay, compressed backup, or per-sweep archive is involved.
    MPI ranks restore their local FVM state and share the master's stop decision.
    """
    fvm, vpm, transfer = coupler.fvm_solver, coupler.vpm_solver, coupler.vorticity_transfer
    fvm_start = capture_restart_payload(fvm)
    predictor = _particle_state_snapshot(vpm) if coupler._is_master else None
    transfer_step = transfer.step
    old = _read_trace(coupler, old=True)
    candidate = _read_trace(coupler, velocity=next_velocity)
    fvm_seconds = transfer_seconds = 0.0
    records = []
    previous_candidate = None
    for sweep in range(1, coupler.setup.interface_iterations + 1):
        if sweep > 1:
            started = perf_counter()
            publish_restart_payload(fvm, fvm_start)
            if coupler._is_master:
                _restore_particle_state(vpm, predictor)
            transfer.step = transfer_step
            _write_trace(coupler, old, old=True)
            _write_trace(coupler, candidate)
            transfer_seconds += perf_counter() - started
        fvm_seconds += advance_fvm(coupler, *geometry, old[0], candidate[0])
        started = perf_counter()
        result, _ = coupler._transfer_vorticity_to_vpm(*geometry)
        update_boundary_history_after_replacement(coupler, *geometry)
        transfer_seconds += perf_counter() - started
        row = None
        if coupler._is_master:
            post = _read_trace(coupler, old=True)
            normal = float(np.sqrt(np.average((post[1] - candidate[1]) ** 2, weights=geometry[2])))
            gradient = float(
                np.sqrt(
                    np.average(np.sum((post[2] - candidate[2]) ** 2, axis=1), weights=geometry[2])
                )
            )
            row = {
                "sweep": sweep,
                "normal_residual_rms": normal,
                "gradient_residual_rms": gradient,
                "converged": bool(
                    normal <= coupler.setup.interface_normal_tolerance
                    and gradient <= coupler.setup.interface_gradient_tolerance
                ),
            }
            if getattr(getattr(vpm, "induction", None), "planar_span", None) is not None:
                position = np.ascontiguousarray(vpm.particles.position_cpu())
                row["particle_count"] = int(vpm.particles.n_particles_total)
                row["particle_support_digest"] = hashlib.sha256(position.tobytes()).hexdigest()[:16]
                if previous_candidate is not None:
                    row["two_sweep_normal_residual_rms"] = float(
                        np.sqrt(
                            np.average((post[1] - previous_candidate[1]) ** 2, weights=geometry[2])
                        )
                    )
                    row["two_sweep_gradient_residual_rms"] = float(
                        np.sqrt(
                            np.average(
                                np.sum((post[2] - previous_candidate[2]) ** 2, axis=1),
                                weights=geometry[2],
                            )
                        )
                    )
                previous_candidate = candidate
            candidate = post
        if fvm.parallel.is_parallel:
            row = fvm.parallel.comm.bcast(row, root=0)
        records.append(row)
        if row["converged"]:
            break
    coupler._last_interface_iteration_diagnostics = {
        "sweeps": len(records),
        "converged": records[-1]["converged"],
        "residuals": records,
    }
    if coupler._is_master:
        final = records[-1]
        logger.log(
            logging.INFO if final["converged"] else logging.WARNING,
            "coupler  interface iteration | sweeps=%d | converged=%s | normal=%.3e | gradient=%.3e",
            len(records),
            final["converged"],
            final["normal_residual_rms"],
            final["gradient_residual_rms"],
        )
    started = perf_counter()
    fvm.write_accepted_step_output()
    fvm_seconds += perf_counter() - started
    return result, fvm_seconds, transfer_seconds
