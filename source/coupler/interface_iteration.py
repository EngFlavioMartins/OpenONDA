"""Fixed-predictor interface sweeps with in-memory state and accepted output."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import logging
from time import perf_counter
from typing import Any, NotRequired, TypeAlias, TypedDict

import numpy as np

from source.solvers.fvm.io.backup import (
    RestartPayload,
    capture_restart_payload,
    publish_restart_payload,
)

from .boundary import advance_fvm, update_boundary_history_after_replacement
from .vorticity_transfer import _particle_state_snapshot, _restore_particle_state

logger = logging.getLogger("coupler")
_TRACE_FIELDS = (
    "velocity_boundary_condition",
    "normal_velocity_boundary_condition",
    "tangential_gradient_boundary_condition",
)
_ACCELERATION_MIN = 0.5
_ACCELERATION_MAX = 1.5
_ACCELERATION_DENOMINATOR_FLOOR = 1.0e-12
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
    "last_gbd_wall_transfer",
    "last_gbd_moment_recovery",
)

BoundaryTrace: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]


class IterationRow(TypedDict):
    sweep: int
    normal_residual_rms: float
    gradient_residual_rms: float
    scaled_residual: float
    converged: bool
    acceleration_alpha: float | None
    acceleration_rejected: bool
    next_acceleration_alpha: float | None
    accepted: bool
    particle_count: NotRequired[int]
    particle_support_digest: NotRequired[str]
    two_sweep_normal_residual_rms: NotRequired[float]
    two_sweep_gradient_residual_rms: NotRequired[float]


class TrialFallback(TypedDict):
    fvm: RestartPayload
    fvm_attributes: dict[str, Any]
    fvm_patch: dict[str, Any] | None
    particles: dict[str, np.ndarray] | None
    physics_attributes: dict[str, Any]
    induction_last_tail: Any
    transfer_step: int
    transfer_attributes: dict[str, Any]
    coupler_attributes: dict[str, Any]
    input_trace: BoundaryTrace
    output_trace: BoundaryTrace
    result: Any


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


def _scaled_residual_norm(residual, areas, normal_tolerance, gradient_tolerance):
    """Area-weighted, dimensionless norm of velocity and gradient increments."""
    weights = np.asarray(areas, dtype=np.float64)
    if weights.ndim != 1 or not len(weights) or not np.all(np.isfinite(weights)):
        return float("nan")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        return float("nan")
    velocity, _, gradient = residual
    values = (
        np.sum(np.asarray(velocity) ** 2, axis=1) / normal_tolerance**2
        + np.sum(np.asarray(gradient) ** 2, axis=1) / gradient_tolerance**2
    )
    return float(np.sqrt(np.average(values, weights=weights)))


def _trace_difference(left, right):
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _aitken_candidate(
    previous_input,
    previous_output,
    current_input,
    current_output,
    previous_alpha,
    face_normal,
    face_area,
    normal_tolerance,
    gradient_tolerance,
):
    """Bound a scalar Aitken step on the complete physical boundary trace.

    One coefficient for velocity and tangential gradient keeps the two FVM
    inputs synchronized. Reconstructing normal velocity from the mixed
    velocity makes its discrete flux exactly consistent with that field.
    Each source velocity has already passed the flux check and correction in
    ``evaluate_vpm_velocity``; their affine combination has the same net flux.
    """
    previous_residual = _trace_difference(previous_output, previous_input)
    current_residual = _trace_difference(current_output, current_input)
    delta = _trace_difference(current_residual, previous_residual)
    previous_norm = _scaled_residual_norm(
        previous_residual, face_area, normal_tolerance, gradient_tolerance
    )
    delta_norm = _scaled_residual_norm(delta, face_area, normal_tolerance, gradient_tolerance)
    if (
        not np.isfinite(previous_norm)
        or not np.isfinite(delta_norm)
        or delta_norm <= _ACCELERATION_DENOMINATOR_FLOOR * max(previous_norm, 1.0)
    ):
        return None, None
    weights = np.asarray(face_area, dtype=np.float64)
    previous_velocity, _, previous_gradient = previous_residual
    delta_velocity, _, delta_gradient = delta
    product = np.sum(previous_velocity * delta_velocity, axis=1) / normal_tolerance**2
    product += np.sum(previous_gradient * delta_gradient, axis=1) / gradient_tolerance**2
    numerator = float(np.average(product, weights=weights))
    alpha = -previous_alpha * numerator / delta_norm**2
    if not np.isfinite(alpha) or alpha <= 0.0:
        return None, None
    alpha = float(np.clip(alpha, _ACCELERATION_MIN, _ACCELERATION_MAX))
    velocity = current_input[0] + alpha * current_residual[0]
    normal = np.einsum("ij,ij->i", velocity, face_normal)
    gradient = current_input[2] + alpha * current_residual[2]
    if not all(np.all(np.isfinite(value)) for value in (velocity, normal, gradient)):
        return None, None
    return (velocity, normal, gradient), alpha


def _capture_trial_fallback(
    coupler, result, input_trace: BoundaryTrace, output_trace: BoundaryTrace
) -> TrialFallback:
    """Save the last coherent endpoint before risking an accelerated sweep."""
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
        "fvm": capture_restart_payload(fvm),
        "fvm_attributes": {
            name: deepcopy(getattr(fvm, name))
            for name in _FVM_ROLLBACK_ATTRIBUTES
            if hasattr(fvm, name)
        },
        "fvm_patch": deepcopy(patch) if patch is not None else None,
        "particles": _particle_state_snapshot(vpm) if vpm is not None else None,
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
        "result": result,
    }


def _restore_trial_fallback(coupler, fallback: TrialFallback):
    fvm = coupler.fvm_solver
    publish_restart_payload(fvm, fallback["fvm"])
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
    return fallback["result"]


def advance_iterated_interface(coupler, geometry, next_velocity):
    """Advance only FVM/renewal repeatedly; the VPM predictor stays fixed.

    Each sweep begins from the same accepted FVM state and advected particles.
    Only the converged/final sweep publishes samples. No reference field,
    experiment replay, compressed backup, or per-sweep archive is involved.
    MPI ranks restore their local FVM state and share the master's stop decision.
    """
    fvm, vpm, transfer = coupler.fvm_solver, coupler.vpm_solver, coupler.vorticity_transfer
    snapshot_started = perf_counter()
    fvm_start = capture_restart_payload(fvm)
    predictor = _particle_state_snapshot(vpm) if coupler._is_master else None
    phase_seconds = {
        "state_capture": perf_counter() - snapshot_started,
        "state_restore": 0.0,
        "boundary_refresh": 0.0,
        "accepted_output": 0.0,
    }
    transfer_step = transfer.step
    old = _read_trace(coupler, old=True)
    candidate: BoundaryTrace = _read_trace(coupler, velocity=next_velocity)
    fvm_seconds = transfer_seconds = 0.0
    records = []
    previous_candidate = None
    acceleration_enabled = getattr(coupler.setup, "interface_acceleration", "none") == "aitken"
    acceleration_disabled = False
    previous_pair = None
    applied_alpha = 1.0
    trial_accelerated = False
    fallback: TrialFallback | None = None
    accepted_row = None
    accepted_sweep = 0
    for sweep in range(1, coupler.setup.interface_iterations + 1):
        input_trace = candidate
        if sweep > 1:
            started = perf_counter()
            publish_restart_payload(fvm, fvm_start)
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
        phase_seconds["boundary_refresh"] += perf_counter() - boundary_started
        transfer_seconds += perf_counter() - started
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
                "acceleration_alpha": applied_alpha if trial_accelerated else None,
                "acceleration_rejected": False,
                "next_acceleration_alpha": None,
                "accepted": True,
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
            if (
                trial_accelerated
                and accepted_row is not None
                and (
                    not np.isfinite(row["normal_residual_rms"])
                    or not np.isfinite(row["gradient_residual_rms"])
                    or row["normal_residual_rms"]
                    > accepted_row["normal_residual_rms"] * (1.0 + 1.0e-12)
                    or row["gradient_residual_rms"]
                    > accepted_row["gradient_residual_rms"] * (1.0 + 1.0e-12)
                )
            ):
                row["acceleration_rejected"] = True
                row["accepted"] = False
                acceleration_disabled = True
                # A failed trial never supplies the accepted endpoint or the
                # input to another Aitken estimate.
                assert fallback is not None
                candidate = fallback["output_trace"]
                previous_pair = None
                applied_alpha = 1.0
            else:
                row["accepted"] = True
                if (
                    acceleration_enabled
                    and not acceleration_disabled
                    and not row["converged"]
                    and sweep < coupler.setup.interface_iterations
                    and previous_pair is not None
                ):
                    accelerated, alpha = _aitken_candidate(
                        previous_pair[0],
                        previous_pair[1],
                        input_trace,
                        post,
                        applied_alpha,
                        geometry[1],
                        geometry[2],
                        coupler.setup.interface_normal_tolerance,
                        coupler.setup.interface_gradient_tolerance,
                    )
                    if accelerated is not None:
                        candidate = accelerated
                        row["next_acceleration_alpha"] = alpha
                        applied_alpha = alpha
                    else:
                        candidate = post
                        applied_alpha = 1.0
                else:
                    candidate = post
                    applied_alpha = 1.0
                previous_pair = (input_trace, post)
        if fvm.parallel.is_parallel:
            row = fvm.parallel.comm.bcast(row, root=0)
        assert row is not None
        records.append(row)
        if row["acceleration_rejected"]:
            started = perf_counter()
            assert fallback is not None
            result = _restore_trial_fallback(coupler, fallback)
            elapsed = perf_counter() - started
            phase_seconds["state_restore"] += elapsed
            transfer_seconds += elapsed
            fallback = None
            if not coupler._is_master:
                candidate = _read_trace(coupler, old=True)
        else:
            accepted_row = row
            accepted_sweep = sweep
        trial_accelerated = row["next_acceleration_alpha"] is not None
        if trial_accelerated:
            started = perf_counter()
            fallback = _capture_trial_fallback(
                coupler,
                result,
                input_trace,
                _read_trace(coupler, old=True),
            )
            elapsed = perf_counter() - started
            phase_seconds["state_capture"] += elapsed
        if accepted_row is not None and accepted_row["converged"]:
            break
    assert accepted_row is not None
    coupler._last_interface_iteration_diagnostics = {
        "sweeps": len(records),
        "accepted_sweep": accepted_sweep,
        "converged": accepted_row["converged"],
        "residuals": records,
        "phase_seconds": phase_seconds,
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
    # Capture cost used to fall outside the four reported phase totals.
    transfer_seconds += phase_seconds["state_capture"]
    return result, fvm_seconds, transfer_seconds
