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


def write_run_metadata(
    coupler,
    *,
    start_step: int = 0,
    stop_step: int | None = None,
) -> None:
    """Write the resolved solver and coupling state used by post-processing."""
    assert coupler.fvm_box is not None
    assert coupler.vpm_solver is not None
    viscous_config = asdict(coupler.vpm_solver.setup.viscous)
    configured_end_step = coupler._derive_coupling_step_count(
        coupler.end_time,
        coupler.vpm_time_step_size,
    )
    resolved_stop_step = configured_end_step if stop_step is None else int(stop_step)
    metadata = {
        "schema_version": 3,
        "coupling_method": coupler.setup.transfer_method,
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
        "execution": {
            "start_coupling_step": int(start_step),
            "stop_coupling_step": resolved_stop_step,
            "configured_end_coupling_step": configured_end_step,
            "start_time": float(start_step * coupler.vpm_time_step_size),
            "stop_time": float(resolved_stop_step * coupler.vpm_time_step_size),
            "is_limited": resolved_stop_step < configured_end_step,
        },
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
            "excluded_solid_active_nodes": 0,
            "excluded_solid_vortex_strength_l1": 0.0,
            "excluded_solid_vortex_strength_net_x": 0.0,
            "excluded_solid_vortex_strength_net_y": 0.0,
            "excluded_solid_vortex_strength_net_z": 0.0,
            "excluded_solid_first_moment_norm": 0.0,
            "mapped_target_vortex_strength_l1": 0.0,
            "mapped_target_vortex_strength_net_x": 0.0,
            "mapped_target_vortex_strength_net_y": 0.0,
            "mapped_target_vortex_strength_net_z": 0.0,
            "maximum_mapped_vortex_strength": 0.0,
            "fvm_mapping_vortex_strength_error": 0.0,
            "fvm_mapping_first_moment_error": 0.0,
            "blend_cross_divergence_l2_before": 0.0,
            "blend_cross_divergence_l2_after": 0.0,
            "blend_cross_divergence_relative": 0.0,
            "mapped_vorticity_divergence_error": None,
            "mapped_vortex_strength_misalignment_degrees": None,
            "mapped_mean_overlap_ratio": None,
            "projection_vorticity_relative_error": 0.0,
            "projection_velocity_relative_error": None,
            "projection_condition_number": 0.0,
            "selective_support_births": 0,
            "renewal_guard_width": 0.0,
            "renewal_diffusion_substeps": 0,
            "renewed_input_particles": 0,
            "renewed_output_particles": 0,
            "preserved_outer_particles": 0,
            "coalesced_outer_particles": 0,
            "pruned_lattice_nodes": 0,
            "pruned_vortex_strength_l1": 0.0,
            "pruned_vortex_strength_fraction": 0.0,
            "population_pruned_particles": 0,
            "population_pruned_vortex_strength_fraction": 0.0,
            "population_pruned_velocity_bound": 0.0,
            "renewal_cfl": 0.0,
            "renewal_raw_vortex_strength_error": 0.0,
            "renewal_applied_vortex_strength_correction": 0.0,
            "renewal_conservation_error": 0.0,
            "renewal_vortex_strength_tolerance": 0.0,
            "renewal_raw_linear_impulse_error": 0.0,
            "renewal_applied_linear_impulse_correction": 0.0,
            "renewal_linear_impulse_error": 0.0,
            "renewal_linear_impulse_tolerance": 0.0,
            "renewal_raw_angular_impulse_error": 0.0,
            "renewal_applied_angular_impulse_correction": 0.0,
            "renewal_angular_impulse_error": 0.0,
            "renewal_applied_particle_strength_fraction": 0.0,
            "population_renewal_raw_vortex_strength_error": 0.0,
            "population_renewal_applied_vortex_strength_correction": 0.0,
            "population_renewal_conservation_error": 0.0,
            "population_renewal_raw_linear_impulse_error": 0.0,
            "population_renewal_applied_linear_impulse_correction": 0.0,
            "population_renewal_linear_impulse_error": 0.0,
            "population_renewal_applied_particle_strength_fraction": 0.0,
            "representation_residual_before_prune": None,
            "representation_residual_after_prune": None,
            "maximum_transfer_amplification": 0.0,
        }
        particle_count = 0
    else:
        injected_net = np.asarray(result.injected_vortex_strength_net, dtype=np.float64)
        replaced_net = np.asarray(result.replaced_vortex_strength_net, dtype=np.float64)
        state_change_net = np.asarray(result.state_change_vortex_strength_net, dtype=np.float64)
        mapped_target_net = np.asarray(result.mapped_target_vortex_strength_net, dtype=np.float64)
        excluded_solid_net = np.asarray(result.excluded_solid_vortex_strength_net, dtype=np.float64)
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
            "excluded_solid_active_nodes": int(result.excluded_solid_active_nodes),
            "excluded_solid_vortex_strength_l1": float(result.excluded_solid_vortex_strength_l1),
            "excluded_solid_vortex_strength_net_x": float(excluded_solid_net[0]),
            "excluded_solid_vortex_strength_net_y": float(excluded_solid_net[1]),
            "excluded_solid_vortex_strength_net_z": float(excluded_solid_net[2]),
            "excluded_solid_first_moment_norm": float(
                np.linalg.norm(result.excluded_solid_first_moment)
            ),
            "mapped_target_vortex_strength_l1": float(result.mapped_target_vortex_strength_l1),
            "mapped_target_vortex_strength_net_x": float(mapped_target_net[0]),
            "mapped_target_vortex_strength_net_y": float(mapped_target_net[1]),
            "mapped_target_vortex_strength_net_z": float(mapped_target_net[2]),
            "maximum_mapped_vortex_strength": float(result.maximum_mapped_vortex_strength),
            "fvm_mapping_vortex_strength_error": (
                None
                if result.transfer_method == "buffered_m4_renewal"
                else float(
                    np.linalg.norm(
                        result.fvm_mapped_vortex_strength_net - result.fvm_donor_vortex_strength_net
                    )
                )
            ),
            "fvm_mapping_first_moment_error": (
                None
                if result.transfer_method == "buffered_m4_renewal"
                else float(
                    np.linalg.norm(result.fvm_mapped_first_moment - result.fvm_donor_first_moment)
                )
            ),
            "blend_cross_divergence_l2_before": float(result.blend_cross_divergence_l2_before),
            "blend_cross_divergence_l2_after": float(result.blend_cross_divergence_l2_after),
            "blend_cross_divergence_relative": float(result.blend_cross_divergence_relative),
            "mapped_vorticity_divergence_error": (
                None
                if result.mapped_vorticity_divergence_error is None
                else float(result.mapped_vorticity_divergence_error)
            ),
            "mapped_vortex_strength_misalignment_degrees": (
                None
                if result.mapped_vortex_strength_misalignment_degrees is None
                else float(result.mapped_vortex_strength_misalignment_degrees)
            ),
            "mapped_mean_overlap_ratio": (
                None
                if result.mapped_mean_overlap_ratio is None
                else float(result.mapped_mean_overlap_ratio)
            ),
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
            "renewed_input_particles": int(result.renewed_input_particles),
            "renewed_output_particles": int(result.renewed_output_particles),
            "preserved_outer_particles": int(result.preserved_outer_particles),
            "coalesced_outer_particles": int(result.coalesced_outer_particles),
            "pruned_lattice_nodes": int(result.pruned_lattice_nodes),
            "pruned_vortex_strength_l1": float(result.pruned_vortex_strength_l1),
            "pruned_vortex_strength_fraction": float(result.pruned_vortex_strength_fraction),
            "population_pruned_particles": int(result.population_pruned_particles),
            "population_pruned_vortex_strength_fraction": float(
                result.population_pruned_vortex_strength_fraction
            ),
            "population_pruned_velocity_bound": float(result.population_pruned_velocity_bound),
            "renewal_cfl": float(result.renewal_cfl),
            "renewal_raw_vortex_strength_error": float(result.renewal_raw_vortex_strength_error),
            "renewal_applied_vortex_strength_correction": float(
                result.renewal_applied_vortex_strength_correction
            ),
            "renewal_conservation_error": float(result.renewal_conservation_error),
            "renewal_vortex_strength_tolerance": float(result.renewal_vortex_strength_tolerance),
            "renewal_raw_linear_impulse_error": float(result.renewal_raw_linear_impulse_error),
            "renewal_applied_linear_impulse_correction": float(
                result.renewal_applied_linear_impulse_correction
            ),
            "renewal_linear_impulse_error": float(result.renewal_linear_impulse_error),
            "renewal_linear_impulse_tolerance": float(result.renewal_linear_impulse_tolerance),
            "renewal_raw_angular_impulse_error": float(result.renewal_raw_angular_impulse_error),
            "renewal_applied_angular_impulse_correction": float(
                result.renewal_applied_angular_impulse_correction
            ),
            "renewal_angular_impulse_error": float(result.renewal_angular_impulse_error),
            "renewal_applied_particle_strength_fraction": float(
                result.renewal_applied_particle_strength_fraction
            ),
            "population_renewal_raw_vortex_strength_error": float(
                result.population_renewal_raw_vortex_strength_error
            ),
            "population_renewal_applied_vortex_strength_correction": float(
                result.population_renewal_applied_vortex_strength_correction
            ),
            "population_renewal_conservation_error": float(
                result.population_renewal_conservation_error
            ),
            "population_renewal_raw_linear_impulse_error": float(
                result.population_renewal_raw_linear_impulse_error
            ),
            "population_renewal_applied_linear_impulse_correction": float(
                result.population_renewal_applied_linear_impulse_correction
            ),
            "population_renewal_linear_impulse_error": float(
                result.population_renewal_linear_impulse_error
            ),
            "population_renewal_applied_particle_strength_fraction": float(
                result.population_renewal_applied_particle_strength_fraction
            ),
            "representation_residual_before_prune": (
                None
                if result.representation_residual_before_prune is None
                else float(result.representation_residual_before_prune)
            ),
            "representation_residual_after_prune": (
                None
                if result.representation_residual_after_prune is None
                else float(result.representation_residual_after_prune)
            ),
            "maximum_transfer_amplification": float(result.maximum_transfer_amplification),
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
    gbd_moment_recovery = {
        "applied": False,
        "nonzero_node_count": 0,
        "retained_node_count": 0,
        "pruned_node_count": 0,
        "support_augmented_node_count": 0,
        "correction_fraction": 0.0,
        "normalized_vortex_strength_residual": 0.0,
        "normalized_linear_impulse_residual": 0.0,
        "normalized_angular_impulse_residual": 0.0,
    }
    vpm_solver = getattr(coupler, "vpm_solver", None)
    if vpm_solver is not None:
        recovery = getattr(vpm_solver.physics, "last_gbd_moment_recovery", None)
        if recovery is not None:
            gbd_moment_recovery = {
                "applied": bool(recovery["applied"]),
                "nonzero_node_count": int(recovery["nonzero_node_count"]),
                "retained_node_count": int(recovery["retained_node_count"]),
                "pruned_node_count": int(recovery["pruned_node_count"]),
                "support_augmented_node_count": int(
                    recovery.get("support_augmented_node_count", 0)
                ),
                "correction_fraction": float(recovery["correction_fraction"]),
                "normalized_vortex_strength_residual": float(
                    recovery["normalized_vortex_strength_residual"]
                ),
                "normalized_linear_impulse_residual": float(
                    recovery["normalized_linear_impulse_residual"]
                ),
                "normalized_angular_impulse_residual": float(
                    recovery["normalized_angular_impulse_residual"]
                ),
            }
            numeric_recovery = [
                value for name, value in gbd_moment_recovery.items() if name != "applied"
            ]
            if not all(np.isfinite(value) for value in numeric_recovery):
                raise FloatingPointError("non-finite GBD moment-recovery diagnostic")
    return {
        "vpm_boundary_condition_flux": boundary_flux,
        "transfer": transfer,
        "boundary_normal_velocity": interface,
        "vortex_line_closure": closure,
        "gbd_moment_recovery": gbd_moment_recovery,
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
