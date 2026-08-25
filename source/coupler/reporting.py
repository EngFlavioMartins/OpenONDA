"""Logging, metadata, and per-step diagnostics for coupled runs."""

from dataclasses import asdict
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

from source import log_style
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


def format_coupler_log(topic: str, *rows: log_style.Row) -> str:
    """Return one coupler record: a topic header over indented detail rows."""
    return log_style.record("coupler", topic, *rows)


def format_coupler_step(step: int, n_steps: int, time_end: float) -> str:
    """Return the banner that opens a coupling step."""
    return log_style.banner(
        f"coupling step {step:,} of {n_steps:,}",
        f"physical time {time_end:.6f} s",
    )


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
    assert coupler.vpm_solver is not None
    viscous_config = asdict(coupler.vpm_solver.setup.viscous)
    metadata = {
        "schema_version": 2,
        "coupling_method": "absolute_common_m4_lattice_blend",
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
            "vpm_particle_spacing": coupler.vpm_particle_spacing,
            "vpm_core_radius_ratio": coupler.vpm_core_radius_ratio,
            "eta_blend_width": coupler.setup.eta_blend_width,
            "viscous_scheme": coupler.vpm_solver.setup.viscous.scheme,
            "viscous_config": viscous_config,
            "panel_coupling_scope": (
                None
                if coupler.vpm_solver.panel_solver is None
                else coupler.vpm_solver.panel_solver.coupling_scope
            ),
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
            "transfer_method": "none",
            "eta_blending_enabled": False,
            "n_particles_before": 0,
            "n_particles_retained": 0,
            "n_particles_removed": 0,
            "n_particles_blended": 0,
            "n_particles_injected": 0,
            "n_particles_after": 0,
            "injected_vortex_strength_l1": 0.0,
            "replaced_vortex_strength_l1": 0.0,
            "injected_vortex_strength_net_x": 0.0,
            "injected_vortex_strength_net_y": 0.0,
            "injected_vortex_strength_net_z": 0.0,
            "replaced_vortex_strength_net_x": 0.0,
            "replaced_vortex_strength_net_y": 0.0,
            "replaced_vortex_strength_net_z": 0.0,
            "state_change_vortex_strength_net_x": 0.0,
            "state_change_vortex_strength_net_y": 0.0,
            "state_change_vortex_strength_net_z": 0.0,
            "mapped_target_nodes": 0,
            "excluded_solid_target_nodes": 0,
            "blend_cross_divergence_l2_before": 0.0,
            "blend_cross_divergence_l2_after": 0.0,
            "projection_vorticity_relative_error": 0.0,
            "projection_velocity_relative_error": None,
            "projection_condition_number": 0.0,
            "selective_support_births": 0,
            "renewal_guard_width": 0.0,
            "renewal_diffusion_substeps": 0,
        }
        particle_count = 0
    else:
        injected_net = np.asarray(result.injected_vortex_strength_net, dtype=np.float64)
        replaced_net = np.asarray(result.replaced_vortex_strength_net, dtype=np.float64)
        state_change_net = np.asarray(result.state_change_vortex_strength_net, dtype=np.float64)
        transfer = {
            "transfer_method": str(result.transfer_method),
            "eta_blending_enabled": bool(result.eta_blending_enabled),
            "n_particles_before": int(result.n_particles_before),
            "n_particles_retained": int(result.n_particles_retained),
            "n_particles_removed": int(result.n_particles_removed),
            "n_particles_blended": int(result.n_particles_blended),
            "n_particles_injected": int(result.n_particles_injected),
            "n_particles_after": int(result.n_particles_after),
            "injected_vortex_strength_l1": float(result.injected_vortex_strength_l1),
            "replaced_vortex_strength_l1": float(result.replaced_vortex_strength_l1),
            "injected_vortex_strength_net_x": float(injected_net[0]),
            "injected_vortex_strength_net_y": float(injected_net[1]),
            "injected_vortex_strength_net_z": float(injected_net[2]),
            "replaced_vortex_strength_net_x": float(replaced_net[0]),
            "replaced_vortex_strength_net_y": float(replaced_net[1]),
            "replaced_vortex_strength_net_z": float(replaced_net[2]),
            "state_change_vortex_strength_net_x": float(state_change_net[0]),
            "state_change_vortex_strength_net_y": float(state_change_net[1]),
            "state_change_vortex_strength_net_z": float(state_change_net[2]),
            "mapped_target_nodes": int(result.mapped_target_nodes),
            "excluded_solid_target_nodes": int(result.excluded_solid_target_nodes),
            "blend_cross_divergence_l2_before": float(result.blend_cross_divergence_l2_before),
            "blend_cross_divergence_l2_after": float(result.blend_cross_divergence_l2_after),
            "projection_vorticity_relative_error": float(
                result.projection_vorticity_relative_error
            ),
            "projection_velocity_relative_error": (
                None
                if result.projection_velocity_relative_error is None
                else float(result.projection_velocity_relative_error)
            ),
            "projection_condition_number": float(result.projection_condition_number),
            "selective_support_births": int(result.selective_support_births),
            "renewal_guard_width": float(result.renewal_guard_width),
            "renewal_diffusion_substeps": int(result.renewal_diffusion_substeps),
        }
        particle_count = result.n_particles_after
    if not all(
        np.isfinite(value)
        for value in transfer.values()
        if value is not None and isinstance(value, int | float | np.number)
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
    return {
        "vpm_boundary_condition_flux": boundary_flux,
        "transfer": transfer,
        "boundary_normal_velocity": interface,
        "vortex_line_closure": closure,
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
                "vpm state",
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
        logger.info(
            format_coupler_log(
                f"step {step:,} complete",
                ("wall time", f"{timing_data['total']:.3f}", "s"),
                ("  vpm", f"{timing_data['vpm']:.3f}", "s"),
                ("  boundary", f"{timing_data['vpm_boundary_condition']:.3f}", "s"),
                ("  fvm", f"{timing_data['fvm']:.3f}", "s"),
                ("  transfer", f"{timing_data['transfer']:.3f}", "s"),
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
    "format_coupler_step",
    "record_step",
    "write_run_metadata",
]
