"""Logging, metadata, and per-step diagnostics for coupled runs."""

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

from source.coupler.checkpoint import CHECKPOINT_DIRECTORY

_REAL_STDOUT = sys.stdout


class OutputRedirector:
    """Capture Python and native output by temporarily replacing file descriptors."""

    def __init__(self, logfile=None, append=True):
        self.logfile = logfile
        self.append = append
        self._log_fd = None
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None
        self.log_file = None

    def __enter__(self):
        if not self.logfile:
            return self

        mode = "a" if self.append else "w"
        self.log_file = open(self.logfile, mode)
        self._log_fd = self.log_file.fileno()

        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(self._log_fd, 1)
        os.dup2(self._log_fd, 2)
        return self

    def __exit__(self, _exc_type, _exc_value, traceback):
        if not self.logfile:
            return

        sys.stdout.flush()
        sys.stderr.flush()

        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)
            os.close(self._saved_stdout_fd)

        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)

        if self.log_file:
            self.log_file.close()


class _CaseFileHandler(logging.FileHandler):
    """File handler used for the per-case coupler log."""


def configure_logging(solution_dir: Path, logger: logging.Logger) -> None:
    """Attach the case log while preserving caller-installed handlers."""
    log_path = (solution_dir / "coupler.log").resolve()
    for handler in list(logger.handlers):
        if isinstance(handler, _CaseFileHandler):
            logger.removeHandler(handler)
            handler.close()

    has_external_handlers = bool(logger.handlers)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = _CaseFileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(file_handler)
    if not has_external_handlers:
        console_handler = logging.StreamHandler(_REAL_STDOUT)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)


def flush_log(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def format_coupler_log(section: str, headline: str = "", *rows: str) -> str:
    """Return one searchable coupler record with indented detail rows."""
    header = f"[Coupler][{section}]"
    if headline:
        header = f"{header} {headline}"
    return "\n".join((header, *(f"  {row}" for row in rows)))


def _domain_dict(box: np.ndarray) -> dict[str, float]:
    return dict(
        zip(
            ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
            (float(value) for value in box),
            strict=True,
        )
    )


def write_run_metadata(coupler) -> None:
    """Write the resolved solver and coupling state used by post-processing."""
    assert coupler.fvm_box is not None
    metadata = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "case_dir": str(coupler.case_dir),
        "physics": {
            "freestream_velocity": coupler.setup.freestream_velocity,
            "kinematic_viscosity": coupler.kinematic_viscosity,
            "density": coupler.density,
            "fvm_time_step_size": coupler.fvm_time_step_size,
            "end_time": coupler.end_time,
            "checkpoint_interval_steps": coupler.setup.checkpoint_interval_steps,
        },
        "fvm_solver": {
            "coupling_patch": coupler.setup.coupling_patch,
            "boundary_condition_mode": coupler.setup.boundary_condition_mode,
            "fvm_domain": _domain_dict(coupler.fvm_box),
        },
        "vpm_solver": {
            "vpm_particle_spacing": coupler.setup.vpm_particle_spacing,
            "authority_ramp_width": coupler.setup.authority_ramp_width,
            "vpm_only_width": coupler.setup.vpm_only_width,
        },
        **coupler.setup.to_dict(),
        "vpm_time_step_size": coupler.vpm_time_step_size,
        "n_fvm_substeps": coupler.n_fvm_substeps,
    }
    (coupler.solution_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def compute_diagnostics(coupler, transfer_result=None) -> dict:
    """Return finite transfer and boundary-flux diagnostics."""
    result = transfer_result
    if result is None:
        result = coupler._last_transfer_result

    if result is None:
        transfer = {
            "diagnostics_evaluated": False,
            "n_existing_particles": 0,
            "n_updated_particles": 0,
            "n_added_particles": 0,
            "n_support_nodes": 0,
            "correction_vortex_strength_l1": 0.0,
            "correction_vortex_strength_net_x": 0.0,
            "correction_vortex_strength_net_y": 0.0,
            "correction_vortex_strength_net_z": 0.0,
            "divergence_correction_l2": None,
            "divergence_correction_linf": None,
        }
        particle_count = 0
    else:
        correction_net = np.asarray(result.correction_vortex_strength_net, dtype=np.float64)
        transfer = {
            "diagnostics_evaluated": bool(result.diagnostics_evaluated),
            "n_existing_particles": int(result.n_existing_particles),
            "n_updated_particles": int(result.n_updated_particles),
            "n_added_particles": int(result.n_added_particles),
            "n_support_nodes": int(result.n_support_nodes),
            "correction_vortex_strength_l1": float(result.correction_vortex_strength_l1),
            "correction_vortex_strength_net_x": float(correction_net[0]),
            "correction_vortex_strength_net_y": float(correction_net[1]),
            "correction_vortex_strength_net_z": float(correction_net[2]),
            "divergence_correction_l2": (
                float(result.divergence_correction_l2) if result.diagnostics_evaluated else None
            ),
            "divergence_correction_linf": (
                float(result.divergence_correction_linf) if result.diagnostics_evaluated else None
            ),
        }
        particle_count = result.n_total_particles
    if not all(
        np.isfinite(value)
        for value in transfer.values()
        if value is not None and not isinstance(value, bool)
    ):
        raise FloatingPointError("non-finite transfer diagnostic")
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

    interface = {}
    closure = {}
    if coupler.vorticity_transfer is not None:
        interface = {
            str(name): float(value)
            for name, value in coupler.vorticity_transfer.last_interface_flow.items()
        }
        closure = {
            str(name): float(value)
            for name, value in coupler.vorticity_transfer.last_vortex_line_closure.items()
        }
    pressure_shift = (
        0.0 if coupler.pressure_reference is None else coupler.pressure_reference.last_shift
    )
    return {
        "vpm_boundary_condition_flux": boundary_flux,
        "transfer": transfer,
        "boundary_normal_velocity": interface,
        "vortex_line_closure": closure,
        "pressure_reference_shift": float(pressure_shift),
        "n_fvm_substeps": int(coupler.n_fvm_substeps),
        "n_transfer_particles": int(particle_count),
    }


def record_step(
    coupler,
    step: int,
    time_end: float,
    timing: tuple[float, float, float, float],
    transfer_result,
    *,
    logger: logging.Logger,
    comm=None,
) -> None:
    """Persist diagnostics and synchronize a completed coupling step."""
    t_vpm, t_vpm_boundary_condition, t_fvm, t_transfer = timing
    diagnostics = compute_diagnostics(coupler, transfer_result)
    timing_data = {
        "vpm": float(t_vpm),
        "vpm_boundary_condition": float(t_vpm_boundary_condition),
        "fvm": float(t_fvm),
        "transfer": float(t_transfer),
        "total": float(sum(timing)),
    }
    if coupler._is_master:
        diagnostics.update(
            {"step": int(step), "time": float(time_end), "timing_seconds": timing_data}
        )
        coupler.coupling_diagnostics.append(diagnostics)
        with (coupler.solution_dir / "coupler_diagnostics.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(diagnostics, separators=(",", ":")) + "\n")

        stats = coupler._step_transfer_stats or {}
        logger.info(
            format_coupler_log(
                "VPMState",
                f"step {step:,}",
                "particles        "
                f"{int(stats.get('n_before', 0)):,} -> "
                f"{int(stats.get('n_after', 0)):,}",
                "vortex strength  sum of magnitudes "
                f"{float(stats.get('sum_before', 0.0)):.4e} -> "
                f"{float(stats.get('sum_after', 0.0)):.4e} m^3/s",
            )
        )
        logger.info(
            format_coupler_log(
                "Timing",
                f"step {step:,} | total {timing_data['total']:.3f} s",
                f"stages  VPM {timing_data['vpm']:.3f} s"
                f" | boundary {timing_data['vpm_boundary_condition']:.3f} s"
                f" | FVM {timing_data['fvm']:.3f} s"
                f" | transfer {timing_data['transfer']:.3f} s",
            )
        )
        flush_log(logger)

    if comm is not None and comm.Get_size() > 1:
        comm.Barrier()
    checkpoint_due = (
        coupler.setup.checkpoint_interval_steps > 0
        and step % coupler.setup.checkpoint_interval_steps == 0
    )
    if checkpoint_due:
        coupler.save_state(coupler.solution_dir / CHECKPOINT_DIRECTORY, coupling_step=step)


__all__ = [
    "OutputRedirector",
    "compute_diagnostics",
    "configure_logging",
    "flush_log",
    "format_coupler_log",
    "record_step",
    "write_run_metadata",
]
