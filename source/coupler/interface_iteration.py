"""Fixed-predictor interface sweeps with in-memory state and accepted output."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import logging
from time import perf_counter
from typing import Any, NotRequired, TypeAlias, TypedDict

import numpy as np

from source.solvers.fvm.io.backup import (
    RestartState,
    capture_restart_state,
    restore_restart_state,
)

from .boundary import advance_fvm, update_boundary_history_after_replacement
from .interface_prediction import discard_interface_prediction_on_failure
from .parallel import collective_phase
from .vorticity_transfer import _particle_state_snapshot, _restore_particle_state

logger = logging.getLogger("coupler")
_TRACE_FIELDS = (
    "velocity_boundary_condition",
    "normal_velocity_boundary_condition",
    "tangential_gradient_boundary_condition",
)
_ROLLBACK_ATTRIBUTES = (
    "_last_vpm_boundary_condition_flux_diagnostics",
    "_last_fvm_boundary_trace_diagnostics",
    "_step_transfer_stats",
)
_TRANSFER_ROLLBACK_ATTRIBUTES = (
    "last_interface_flow",
    "last_vortex_line_closure",
    "last_spanwise_metrics",
)
_FVM_ROLLBACK_ATTRIBUTES = ("_last_residuals", "last_diagnostics")
_VPM_PHYSICS_ROLLBACK_ATTRIBUTES = (
    "last_solid_projection",
    # These diagnostics are exposed through read-only, copy-returning
    # properties. Roll back the owned state, not those reporting views.
    "_last_gbd_wall_transfer",
    "_last_gbd_moment_recovery",
)

BoundaryTrace: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]


class IterationRow(TypedDict):
    sweep: int
    normal_residual_rms: float
    gradient_residual_rms: float
    scaled_residual: float
    converged: bool
    accepted: bool
    prediction_probe: NotRequired[bool]
    prediction_rejected: NotRequired[bool]
    picard_sweep: NotRequired[int]
    particle_count: NotRequired[int]
    particle_support_digest: NotRequired[str]
    two_sweep_normal_residual_rms: NotRequired[float]
    two_sweep_gradient_residual_rms: NotRequired[float]


class TrialFallback(TypedDict):
    fvm: RestartState
    fvm_attributes: dict[str, Any]
    fvm_patch: dict[str, Any] | None
    particles: Any
    physics_attributes: dict[str, Any]
    induction_last_tail: Any
    transfer_step: int
    transfer_attributes: dict[str, Any]
    coupler_attributes: dict[str, Any]
    input_trace: BoundaryTrace
    output_trace: BoundaryTrace


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
        np.asarray(getattr(coupler, "_" + _TRACE_FIELDS[1] + suffix)).copy(),
        np.asarray(getattr(coupler, "_" + _TRACE_FIELDS[2] + suffix)).copy(),
    )


def _write_trace(coupler, values, *, old=False):
    suffix = "_old" if old else ""
    for name, value in zip(_TRACE_FIELDS, values, strict=True):
        if old or name != _TRACE_FIELDS[0]:
            setattr(coupler, "_" + name + suffix, value.copy())


def _capture_trial_fallback(
    coupler, input_trace: BoundaryTrace, output_trace: BoundaryTrace
) -> TrialFallback:
    """Save the coherent endpoint before probing a predicted trace."""
    transfer = coupler.vorticity_transfer
    fvm = coupler.fvm_solver
    vpm = coupler.vpm_solver if coupler._is_master else None
    physics = getattr(vpm, "physics", None)
    induction = getattr(physics, "induction", None)
    patch = (
        next(
            (
                boundary
                for boundary in getattr(fvm, "boundaries", ())
                if boundary["name"] == coupler.setup.coupling_patch
            ),
            None,
        )
        if hasattr(coupler.setup, "coupling_patch")
        else None
    )
    return {
        "fvm": capture_restart_state(fvm),
        "fvm_attributes": {
            name: deepcopy(getattr(fvm, name))
            for name in _FVM_ROLLBACK_ATTRIBUTES
            if hasattr(fvm, name)
        },
        "fvm_patch": deepcopy(patch) if patch is not None else None,
        "particles": _particle_state_snapshot(vpm, slot="coupler-trial")
        if vpm is not None
        else None,
        "physics_attributes": {
            name: deepcopy(getattr(physics, name))
            for name in _VPM_PHYSICS_ROLLBACK_ATTRIBUTES
            if physics is not None and hasattr(physics, name)
        },
        "induction_last_tail": (
            deepcopy(induction.last_tail)
            if induction is not None and hasattr(induction, "last_tail")
            else None
        ),
        "transfer_step": transfer.step,
        "transfer_attributes": {
            name: deepcopy(getattr(transfer, name))
            for name in _TRANSFER_ROLLBACK_ATTRIBUTES
            if hasattr(transfer, name)
        },
        "coupler_attributes": {
            name: deepcopy(getattr(coupler, name))
            for name in _ROLLBACK_ATTRIBUTES
            if hasattr(coupler, name)
        },
        "input_trace": input_trace,
        "output_trace": output_trace,
    }


def _restore_trial_fallback(coupler, fallback: TrialFallback):
    fvm = coupler.fvm_solver
    restore_restart_state(fvm, fallback["fvm"])
    for name, value in fallback["fvm_attributes"].items():
        setattr(fvm, name, value)
    if fallback["fvm_patch"] is not None:
        patch = next(
            boundary
            for boundary in fvm.boundaries
            if boundary["name"] == fallback["fvm_patch"]["name"]
        )
        patch.clear()
        patch.update(fallback["fvm_patch"])
    if coupler._is_master:
        particles = fallback["particles"]
        assert particles is not None
        _restore_particle_state(coupler.vpm_solver, particles)
        physics = getattr(coupler.vpm_solver, "physics", None)
        for name, value in fallback["physics_attributes"].items():
            setattr(physics, name, value)
        induction = getattr(physics, "induction", None)
        if induction is not None and hasattr(induction, "last_tail"):
            induction.last_tail = fallback["induction_last_tail"]
    transfer = coupler.vorticity_transfer
    transfer.step = fallback["transfer_step"]
    for name, value in fallback["transfer_attributes"].items():
        setattr(transfer, name, value)
    for name, value in fallback["coupler_attributes"].items():
        setattr(coupler, name, value)
    _write_trace(coupler, fallback["input_trace"])
    _write_trace(coupler, fallback["output_trace"], old=True)


@discard_interface_prediction_on_failure
def advance_iterated_interface(coupler, geometry, next_velocity):
    """Advance only FVM/renewal repeatedly; the VPM predictor stays fixed.

    Each sweep begins from the same accepted FVM state and advected particles.
    Only the converged/final sweep publishes samples. No reference field,
    experiment replay, compressed backup, or per-sweep archive is involved.
    MPI ranks restore their local FVM state and share the master's stop decision.
    """
    fvm, vpm, transfer = coupler.fvm_solver, coupler.vpm_solver, coupler.vorticity_transfer
    comm = getattr(fvm.parallel, "comm", None)
    snapshot_started = perf_counter()
    with collective_phase(comm, "interface initial state capture"):
        fvm_start = capture_restart_state(fvm)
        predictor = (
            _particle_state_snapshot(vpm, slot="coupler-predictor") if coupler._is_master else None
        )
        old = _read_trace(coupler, old=True)
        candidate: BoundaryTrace = _read_trace(coupler, velocity=next_velocity)
    phase_seconds = {
        "state_capture": perf_counter() - snapshot_started,
        "state_restore": 0.0,
        "boundary_refresh": 0.0,
        "accepted_output": 0.0,
    }
    transfer_step = transfer.step
    raw_predictor = candidate
    history = getattr(coupler, "interface_predictor", None)
    prediction = {"enabled": False, "attempted": False, "reason": "unavailable"}
    seed_fallback = None
    if history is not None:
        started = perf_counter()
        seed, prediction = history.prepare(coupler, geometry, old, raw_predictor)
        prediction["input_check_seconds"] = perf_counter() - started
        prediction["snapshot_seconds"] = 0.0
        if seed is not None:
            snapshot_started = perf_counter()
            with collective_phase(comm, "interface seed snapshot and installation"):
                seed_fallback = _capture_trial_fallback(coupler, raw_predictor, old)
                candidate = seed
                # Unlike later sweeps, a seed must install its matching
                # normal/gradient fields before the first FVM call.
                _write_trace(coupler, candidate)
            prediction["snapshot_seconds"] = perf_counter() - snapshot_started
        phase_seconds["state_capture"] += perf_counter() - started
    probing = bool(prediction["attempted"])
    prediction.update(accepted=False, fallback=False)
    fvm_seconds = transfer_seconds = 0.0
    records = []
    previous_candidate = None
    accepted_row = None
    accepted_sweep = 0
    for sweep in range(1, coupler.setup.interface_iterations + 1 + int(probing)):
        prediction_probe = probing and sweep == 1
        picard_sweep = sweep - int(probing)
        if picard_sweep > 1:
            started = perf_counter()
            restore_restart_state(fvm, fvm_start)
            if coupler._is_master:
                _restore_particle_state(vpm, predictor)
            transfer.step = transfer_step
            _write_trace(coupler, old, old=True)
            _write_trace(coupler, candidate)
            elapsed = perf_counter() - started
            phase_seconds["state_restore"] += elapsed
            transfer_seconds += elapsed
        fvm_seconds += advance_fvm(coupler, *geometry, old[0], candidate[0])
        started = perf_counter()
        result, _ = coupler._transfer_vorticity_to_vpm(*geometry)
        boundary_started = perf_counter()
        update_boundary_history_after_replacement(coupler, *geometry)
        refresh_seconds = perf_counter() - boundary_started
        phase_seconds["boundary_refresh"] += refresh_seconds
        transfer_seconds += perf_counter() - started - refresh_seconds
        row: IterationRow | None = None
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
                "scaled_residual": max(
                    normal / coupler.setup.interface_normal_tolerance,
                    gradient / coupler.setup.interface_gradient_tolerance,
                ),
                "converged": bool(
                    normal <= coupler.setup.interface_normal_tolerance
                    and gradient <= coupler.setup.interface_gradient_tolerance
                ),
                "accepted": True,
            }
            if history is not None:
                row.update(
                    prediction_probe=prediction_probe,
                    prediction_rejected=prediction_probe and not row["converged"],
                    picard_sweep=picard_sweep,
                )
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
            if prediction_probe and not row["converged"]:
                # This speculative solve is outside the original allowance.
                # A rejected probe never supplies the next Picard input.
                row["accepted"] = False
            else:
                candidate = post
        if fvm.parallel.is_parallel:
            row = fvm.parallel.comm.bcast(row, root=0)
        assert row is not None
        records.append(row)
        if prediction_probe and not row["converged"]:
            started = perf_counter()
            assert seed_fallback is not None
            with collective_phase(comm, "interface seed rejection restore"):
                _restore_trial_fallback(coupler, seed_fallback)
            candidate = raw_predictor
            previous_candidate = None
            elapsed = perf_counter() - started
            phase_seconds["state_restore"] += elapsed
            transfer_seconds += elapsed
            prediction["fallback"] = True
            seed_fallback = None
            continue
        if prediction_probe:
            prediction["accepted"] = True
        accepted_row = row
        accepted_sweep = sweep
        if accepted_row is not None and accepted_row["converged"]:
            break
    assert accepted_row is not None
    coupler._last_interface_iteration_diagnostics = {
        "sweeps": len(records),
        "accepted_sweep": accepted_sweep,
        "converged": accepted_row["converged"],
        "residuals": records,
        "phase_seconds": phase_seconds,
        "prediction": prediction,
        "picard_sweeps": len(records) - int(probing),
    }
    if coupler._is_master:
        final = accepted_row
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
    phase_seconds["accepted_output"] = perf_counter() - started
    fvm_seconds += phase_seconds["accepted_output"]
    if history is not None:
        started = perf_counter()
        with collective_phase(getattr(fvm.parallel, "comm", None), "interface history staging"):
            history.stage(
                coupler,
                geometry,
                raw_predictor,
                _read_trace(coupler, old=True),
                converged=accepted_row["converged"],
            )
        prediction["history_stage_seconds"] = perf_counter() - started
        phase_seconds["state_capture"] += prediction["history_stage_seconds"]
    # Capture cost used to fall outside the four reported phase totals.
    transfer_seconds += phase_seconds["state_capture"]
    return result, fvm_seconds, transfer_seconds
