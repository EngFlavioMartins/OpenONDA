"""Logging, metadata, and per-step diagnostics for coupled runs."""

from collections.abc import Mapping
from dataclasses import fields
import logging
from pathlib import Path
import time

import numpy as np

from openonda.core import log_style
from openonda.core.file_output import append_jsonl
from openonda.core.simulation.parallel import collective_phase


def _diagnostic_json_value(value, path="diagnostic", *, _active=None):
    """Copy structured evidence into JSON types, rejecting every nonfinite leaf.

    Counts, flags, optional values, ordered lists and nested mappings retain
    their meaning. NumPy scalar/array values are supported without coercing
    lists to scalars, dropping details, stringifying objects, or mutating the
    producer. Non-string mapping keys and cycles are rejected explicitly.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        result = float(value)
        if not np.isfinite(result):
            raise FloatingPointError(f"non-finite diagnostic at {path}")
        return result
    if not isinstance(value, Mapping | list | tuple | np.ndarray):
        raise TypeError(f"unsupported diagnostic value {type(value).__name__} at {path}")
    active = set() if _active is None else _active
    object_id = id(value)
    if object_id in active:
        raise ValueError(f"cyclic diagnostic structure at {path}")
    active.add(object_id)
    try:
        if isinstance(value, np.ndarray):
            return _diagnostic_json_value(value.tolist(), path, _active=active)
        if isinstance(value, Mapping):
            result = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"diagnostic mapping key must be a string at {path}: {key!r}")
                result[key] = _diagnostic_json_value(child, f"{path}[{key!r}]", _active=active)
            return result
        return [
            _diagnostic_json_value(child, f"{path}[{index}]", _active=active)
            for index, child in enumerate(value)
        ]
    finally:
        active.remove(object_id)


def configure_coupler_logging(path: Path | None, *, console: bool) -> logging.Logger:
    """Send ``openonda.coupler`` to ``solution/coupler.log``, one report per accepted coupling step."""
    logger = log_style.configure_logging("openonda.coupler", path, console=console)
    for progress_filter in list(logger.filters):
        if isinstance(progress_filter, _CouplingProgress):
            logger.removeFilter(progress_filter)
    logger.addFilter(_CouplingProgress())
    return logger


def close_coupler_logging() -> None:
    """Close the handlers and the held coupling-step records of ``openonda.coupler``."""
    logger = logging.getLogger("openonda.coupler")
    for progress_filter in list(logger.filters):
        if isinstance(progress_filter, _CouplingProgress):
            logger.removeFilter(progress_filter)
    log_style.close_logging("openonda.coupler")


def flush_log(logger: logging.Logger) -> None:
    """Flush all handlers attached to ``logger``."""
    for handler in logger.handlers:
        handler.flush()


class _CouplingProgress(logging.Filter):
    """Keep only this coupling interval's latest module records until acceptance.

    Holding the log rows does not depend on the particle or mesh size. Skipped
    reports are not formatted and keep no history across coupling steps.
    Warnings are written at once; the JSONL diagnostics do not pass through here.
    """

    def __init__(self) -> None:
        super().__init__()
        self.started = time.perf_counter()
        self.last_report: float | None = None
        self.context: tuple[int, int | None, float] | None = None
        self.records: dict[str, log_style.Event] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            self.context is not None
            and record.levelno < logging.WARNING
            and isinstance(record.msg, log_style.Event)
        ):
            self.records[record.msg.topic] = record.msg
            return False
        return True

    def begin(self, step: int, total_steps: int | None, flow_time: float) -> None:
        """Start a tentative interval without publishing a completed-step banner."""
        self.context = (step, total_steps, flow_time)
        self.records.clear()

    def finish(self, logger: logging.Logger, *, final: bool = False) -> None:
        """Publish one accepted block when due, retaining immediate warnings."""
        context, self.context = self.context, None
        if context is None:
            return
        now = time.perf_counter()
        step, total_steps, flow_time = context
        if (
            self.last_report is None
            or now - self.last_report >= 30.0
            or final
            or (total_steps is not None and step == total_steps)
        ):
            report: list[str] = [
                log_style.step_header(
                    step, flow_time, now - self.started, scope="COUPLER", total_steps=total_steps
                )
            ]
            report.extend(
                str(event)
                for event in sorted(
                    self.records.values(),
                    key=lambda event: log_style.step_section_order(event.topic),
                )
            )
            logger.info(log_style.FormattedText("\n".join(report)))
            self.last_report = now
        self.records.clear()


def begin_coupling_step(
    logger: logging.Logger, step: int, total_steps: int | None, time_end: float
) -> None:
    """Collect one shared step without claiming its acceptance."""
    for progress_filter in logger.filters:
        if isinstance(progress_filter, _CouplingProgress):
            progress_filter.begin(step, total_steps, time_end)


def finish_coupling_step(logger: logging.Logger, *, final: bool = False) -> None:
    """Close accepted console reporting independently of native diagnostic output."""
    for progress_filter in logger.filters:
        if isinstance(progress_filter, _CouplingProgress):
            progress_filter.finish(logger, final=final)


def format_coupler_log(topic: str, *rows: log_style.Row) -> log_style.Event:
    """Collect already-computed host rows; formatting is deferred until emission."""
    return log_style.Event(topic, rows)


def boundary_induction_summary(vpm_solver, *, target_count: int, wall_seconds: float) -> dict:
    """Capture scalar query evidence before another operation replaces it.

    Backend timings are nested parts of the measured query wall time. Image
    descriptors, source arrays and allocator histories are deliberately omitted.
    """
    induction = getattr(getattr(vpm_solver, "physics", None), "induction", None)
    tail = getattr(induction, "last_tail", None) or {}
    mesh = tail.get("mesh") or {}
    finite = mesh.get("finite_evaluation") or {}
    summary = {
        "target_count": int(target_count),
        "wall_seconds": float(wall_seconds),
        "operator": type(induction).__name__ if induction is not None else None,
        "execution_backend": finite.get("execution_backend"),
        "memory_fallback": finite.get("memory_fallback"),
    }
    if isinstance(summary["memory_fallback"], str):
        summary["memory_fallback"] = summary["memory_fallback"][:512]
    if "seconds" in tail:
        summary["induction_seconds"] = tail["seconds"]
    for name in (
        "source_snapshot_seconds",
        "tail_bound_seconds",
        "field_build_seconds",
        "field_prepare_seconds",
        "field_close_seconds",
        "field_reused",
        "finite_images",
    ):
        if name in mesh:
            summary[name] = mesh[name]
    for name in ("query_seconds", "gather_seconds", "query_batches", "source_count"):
        if name in finite:
            summary[name] = finite[name]
    return _diagnostic_json_value(summary, "boundary induction")


def _transfer_record(result) -> dict[str, object]:
    """Flatten a transfer result; 3-vectors become ``_x``, ``_y``, ``_z`` scalars.

    ``None`` (no transfer yet) gives zero counts and errors.
    """
    from .vorticity_transfer import TransferResult

    if result is None:
        result = TransferResult(0, 0, 0, 0, 0, 0, 0.0)
    record: dict[str, object] = {}
    for item in fields(result):
        value = getattr(result, item.name)
        if isinstance(value, np.ndarray):
            for axis, component in zip("xyz", np.asarray(value, dtype=np.float64), strict=True):
                record[f"{item.name}_{axis}"] = float(component)
        else:
            record[item.name] = value
    if not all(np.isfinite(value) for value in record.values() if isinstance(value, int | float)):
        raise FloatingPointError("non-finite transfer diagnostic")
    return record


def compute_diagnostics(coupler, transfer_result=None) -> dict:
    """Flatten the latest coupling state into JSON-safe diagnostic values.

    Parameters
    ----------
    coupler : CoupledSolver
        Coupled solver with the latest boundary data, transfer and solver state.
    transfer_result : TransferResult or None, optional
        Result for the current replacement. ``None`` uses the coupler cache;
        before the first transfer, zero/``None`` placeholders are returned.

    Returns
    -------
    dict[str, object]
        Nested boundary-data, transfer, transfer-surface and moment-correction records. Vector
        quantities are split into scalar components for JSONL serialization;
        strengths use m³/s, velocities m/s, and normalized errors are
        dimensionless.

    Notes
    -----
    The function is read-only. :func:`record_step` appends its result to the
    in-memory history and ``coupler_diagnostics.jsonl``.
    """
    result = transfer_result if transfer_result is not None else coupler._last_transfer_result
    transfer = _transfer_record(result)
    particle_count = transfer["n_particles_after"]
    boundary_flux = {
        name: float(coupler._last_vpm_boundary_condition_flux_diagnostics[name])
        for name in (
            "raw_mismatch",
            "raw_relative",
            "acceptance_limit",
            "applied_correction",
            "corrected_mismatch",
        )
    }
    if not all(np.isfinite(value) for value in boundary_flux.values()):
        raise FloatingPointError("non-finite VPM boundary-flux diagnostic")
    boundary_data = {
        str(name): float(value)
        for name, value in getattr(coupler, "_last_fvm_boundary_data_diagnostics", {}).items()
    }
    if not all(np.isfinite(value) for value in boundary_data.values()):
        raise FloatingPointError("non-finite FVM boundary-data diagnostic")

    surface_flow = {}
    surface_vorticity = {}
    if coupler.vorticity_transfer is not None:
        surface_flow = {
            str(name): float(value)
            for name, value in coupler.vorticity_transfer.last_transfer_surface_flow.items()
        }
        surface_vorticity = {
            str(name): float(value)
            for name, value in coupler.vorticity_transfer.last_transfer_surface_vorticity.items()
        }
    gbd_moment_correction = {
        "applied": False,
        "nonzero_node_count": 0,
        "retained_node_count": 0,
        "removed_node_count": 0,
        "support_augmented_node_count": 0,
        "correction_fraction": 0.0,
        "normalized_vortex_strength_residual": 0.0,
        "normalized_linear_impulse_residual": 0.0,
        "normalized_angular_impulse_residual": 0.0,
    }
    vpm_solver = getattr(coupler, "vpm_solver", None)
    if vpm_solver is not None:
        correction = getattr(vpm_solver.physics, "last_gbd_moment_correction", None)
        if correction is not None:
            gbd_moment_correction = {
                "applied": bool(correction["applied"]),
                "nonzero_node_count": int(correction["nonzero_node_count"]),
                "retained_node_count": int(correction["retained_node_count"]),
                "removed_node_count": int(correction["removed_node_count"]),
                "support_augmented_node_count": int(
                    correction.get("support_augmented_node_count", 0)
                ),
                "correction_fraction": float(correction["correction_fraction"]),
                "normalized_vortex_strength_residual": float(
                    correction["normalized_vortex_strength_residual"]
                ),
                "normalized_linear_impulse_residual": float(
                    correction["normalized_linear_impulse_residual"]
                ),
                "normalized_angular_impulse_residual": float(
                    correction["normalized_angular_impulse_residual"]
                ),
            }
            numeric_correction = [
                value for name, value in gbd_moment_correction.items() if name != "applied"
            ]
            if not all(np.isfinite(value) for value in numeric_correction):
                raise FloatingPointError("non-finite GBD moment-correction diagnostic")
    diagnostics = {
        "vpm_boundary_condition_flux": boundary_flux,
        "fvm_boundary_data": boundary_data,
        "transfer": transfer,
        "transfer_surface_normal_velocity": surface_flow,
        "transfer_surface_normal_vorticity": surface_vorticity,
        "gbd_moment_correction": gbd_moment_correction,
        "n_transfer_particles": int(particle_count),
        # VPM positions are relative to the FVM mesh; add this to obtain
        # inertial-frame positions.
        "mesh_translation": coupler.frame.translation().tolist(),
    }
    if vpm_solver is not None:
        physics = vpm_solver.physics
        projection = getattr(physics, "last_solid_projection", None)
        if projection is not None:
            projection_record = {
                "stage_count": int(projection["stage_count"]),
                "accepted_count": int(projection["accepted_count"]),
                "stage_displacement_l1": float(projection["stage_displacement_l1"]),
                "accepted_displacement_l1": float(projection["accepted_displacement_l1"]),
                "stage_impulse_change": np.asarray(
                    projection["stage_impulse_change"], dtype=np.float64
                ).tolist(),
                "accepted_impulse_change": np.asarray(
                    projection["accepted_impulse_change"], dtype=np.float64
                ).tolist(),
            }
            if not all(
                np.isfinite(value)
                for value in (
                    projection_record["stage_displacement_l1"],
                    projection_record["accepted_displacement_l1"],
                    *projection_record["stage_impulse_change"],
                    *projection_record["accepted_impulse_change"],
                )
            ):
                raise FloatingPointError("non-finite solid-projection budget")
            diagnostics["solid_projection"] = projection_record
        wall_transfer = getattr(physics, "last_gbd_wall_transfer", None)
        if wall_transfer is not None:
            wall_record = {
                "wall_adjacent_particles": int(wall_transfer["wall_adjacent_particles"]),
                "excluded_signed_weight_l1": float(wall_transfer["excluded_signed_weight_l1"]),
                "fluid_correction_l1": float(wall_transfer["fluid_correction_l1"]),
            }
            if not all(np.isfinite(value) for value in wall_record.values()):
                raise FloatingPointError("non-finite GBD wall-transfer budget")
            diagnostics["gbd_wall_transfer"] = wall_record
        induction = getattr(vpm_solver.physics, "induction", None)
        tail = getattr(induction, "last_tail", None)
        if tail is not None:
            # This describes only the most recent image query, not the total
            # induction work across the coupling step. Its bound and mesh
            # records are structured evidence and must stay intact.
            diagnostics["last_induction_image_call"] = _diagnostic_json_value(
                tail, "slip-slab image diagnostic"
            )
        # Cumulative counts: a reuse means a request at an unchanged accepted
        # particle state published the stored induction instead of a new sum.
        stage_rhs = getattr(vpm_solver, "stage_rhs", None)
        reuse = getattr(stage_rhs, "induction_reuse_statistics", None)
        if reuse is not None:
            diagnostics["induction_reuse"] = {
                "evaluations": int(reuse.evaluations),
                "reuses": int(reuse.reuses),
            }
        boundary_induction = getattr(coupler, "_last_boundary_induction_diagnostics", None)
        if boundary_induction is not None:
            diagnostics["boundary_induction"] = _diagnostic_json_value(
                boundary_induction, "boundary induction diagnostic"
            )
    return diagnostics


def record_step(
    coupler,
    step: int,
    time_end: float,
    timing: tuple[float, float, float, float],
    transfer_result,
    *,
    logger: logging.Logger,
    interval_started: float,
    output_seconds: float,
) -> None:
    """Report one accepted coupling step, write its due backup and its diagnostics record.

    Boundary time includes the predictor boundary data and every evaluation after a
    vorticity transfer. Transfer time includes the vorticity transfers and the state
    capture and restoration of the coupling iterations. The four
    evolution timers exclude output and backup; the total includes them,
    reporting, logging and collective waits. A failed backup does not undo the
    accepted coupling step: its diagnostics record is written and the backup error
    is raised on every rank.
    """
    comm = coupler._comm
    reporting_started = time.perf_counter()
    with collective_phase(comm, "coupled step reporting"):
        t_vpm, t_vpm_boundary_condition, t_fvm, t_transfer = timing
        diagnostics = compute_diagnostics(coupler, transfer_result)
        coupling_iteration = getattr(coupler, "_last_coupling_iteration_diagnostics", None)
        if coupling_iteration is not None:
            diagnostics["coupling_iteration"] = coupling_iteration
        diagnostics = _diagnostic_json_value(diagnostics, "coupled step diagnostic")
        timing_data = {
            "vpm": float(t_vpm),
            "vpm_boundary_condition": float(t_vpm_boundary_condition),
            "fvm": float(t_fvm),
            "transfer": float(t_transfer),
            "evolution_total": float(sum(timing)),
            "state_checks_and_samplers": float(output_seconds),
        }
        stats = coupler._step_transfer_stats or {}
        timing_data["last_transfer_fvm_gather"] = float(stats.get("fvm_gather_seconds", 0.0))
        if coupler._is_root:
            logger.info(
                format_coupler_log(
                    "FVM",
                    ("accepted FVM step", coupler.fvm_solver.step),
                    ("physical time", coupler.fvm_solver.time, "s"),
                    ("Courant, max", coupler.fvm_solver.max_courant_number),
                )
            )
            logger.info(
                format_coupler_log(
                    "VPM",
                    ("particles, before", log_style.count(stats.get("n_before", 0))),
                    ("particles, after", log_style.count(stats.get("n_after", 0))),
                    (
                        "vortex strength, before",
                        f"{float(stats.get('sum_before', 0.0)):.4e}",
                        "m^3/s",
                    ),
                    (
                        "vortex strength, after",
                        f"{float(stats.get('sum_after', 0.0)):.4e}",
                        "m^3/s",
                    ),
                )
            )
            finish_coupling_step(
                logger,
                final=time_end >= coupler.end_time - max(1.0e-14, abs(coupler.end_time) * 1.0e-12),
            )
            flush_log(logger)

    reporting_time = time.perf_counter() - reporting_started
    backup_started = time.perf_counter()
    backup_due = coupler._backup_due(step)
    backup_failure = None
    if backup_due:
        try:
            coupler.save_backup()
        except BaseException as error:
            backup_failure = error
    backup_time = time.perf_counter() - backup_started
    measured_end = time.perf_counter()
    phases = float(sum(timing) + output_seconds + reporting_time + backup_time)
    total = measured_end - interval_started
    timing_data.update(
        {
            "reporting": float(reporting_time),
            "backup": float(backup_time),
            "coupling_control_and_wait": float(total - phases),
            "total": float(total),
        }
    )
    try:
        with collective_phase(comm, "accepted coupling-step diagnostics output"):
            if coupler._is_root:
                diagnostics.update(
                    {
                        "step": int(step),
                        "time": float(time_end),
                        "timing_seconds": timing_data,
                        "backup_phase": {
                            "scheduled": backup_due,
                            "status": (
                                "failed"
                                if backup_failure is not None
                                else "completed"
                                if backup_due
                                else "not_scheduled"
                            ),
                            "error": (
                                None
                                if backup_failure is None
                                else f"{type(backup_failure).__name__}: {backup_failure}"
                            ),
                        },
                    }
                )
                diagnostics = _diagnostic_json_value(
                    diagnostics, "accepted coupling-step diagnostic"
                )
                coupler.coupling_diagnostics.append(diagnostics)
                append_jsonl(coupler.solution_dir / "coupler_diagnostics.jsonl", diagnostics)
                logger.info(
                    format_coupler_log(
                        "timing",
                        ("wall time", f"{timing_data['total']:.3f}", "s"),
                        ("  vpm", f"{timing_data['vpm']:.3f}", "s"),
                        ("  boundary", f"{timing_data['vpm_boundary_condition']:.3f}", "s"),
                        ("  fvm", f"{timing_data['fvm']:.3f}", "s"),
                        ("  transfer", f"{timing_data['transfer']:.3f}", "s"),
                        ("  state checks and samplers", f"{output_seconds:.3f}", "s"),
                        ("  reporting", f"{reporting_time:.3f}", "s"),
                        ("  backup", f"{backup_time:.3f}", "s"),
                        ("  control and wait", f"{total - phases:.3f}", "s"),
                    )
                )
                flush_log(logger)
    except BaseException as output_failure:
        if backup_failure is None:
            raise
        backup_failure.add_note(
            "Accepted-step diagnostic output also failed: "
            f"{type(output_failure).__name__}: {output_failure}"
        )
    if backup_failure is not None:
        raise backup_failure.with_traceback(backup_failure.__traceback__)


__all__ = [
    "begin_coupling_step",
    "boundary_induction_summary",
    "close_coupler_logging",
    "compute_diagnostics",
    "configure_coupler_logging",
    "finish_coupling_step",
    "flush_log",
    "format_coupler_log",
    "record_step",
]
