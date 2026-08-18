"""Logging, metadata, and per-step diagnostics for coupled runs."""

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

from source.coupler.checkpoint import CHECKPOINT_DIRECTORY


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
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)


def flush_log(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


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
            "u_inf": coupler.config.u_inf,
            "nu": coupler.nu,
            "rho": coupler.rho,
            "dt": coupler.dt_fvm,
            "t_end": coupler.t_end,
            "backup_period": coupler.config.backup_period,
        },
        "fvm_solver": {
            "patch_name": coupler.config.patch_name,
            "vpm_bc_mode": coupler.config.vpm_bc_mode,
            "fvm_domain": _domain_dict(coupler.fvm_box),
        },
        "vpm_solver": {
            "particle_spacing": coupler.config.h,
            "buffer_thickness": coupler.config.buffer_thickness,
            "dead_zone_h": coupler.config.dead_zone_h,
        },
        **coupler.config.to_dict(),
        "dt_fvm": coupler.dt_fvm,
        "dt_vpm": coupler.dt_vpm,
        "n_fvm_substeps": coupler.n_fvm_substeps,
    }
    (coupler.solution_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def compute_diagnostics(coupler, handoff_result=None) -> dict:
    """Return finite transfer, boundary-flux, and conservation diagnostics."""
    result = handoff_result
    if result is None:
        result = coupler._last_handoff_result

    zero_invariants = {
        "circulation": 0.0,
        "linear_impulse": 0.0,
        "angular_impulse": 0.0,
    }

    def finite_invariants(values) -> dict[str, float]:
        output = dict(zero_invariants)
        if values:
            output.update({name: float(value) for name, value in values.items()})
        if not all(np.isfinite(value) for value in output.values()):
            raise FloatingPointError("non-finite conservation diagnostic")
        return output

    if result is None:
        conservation = {
            name: dict(zero_invariants)
            for name in ("raw_mismatch", "applied_correction", "corrected_mismatch")
        }
        transfer = {
            "cfl": 0.0,
            "n_remesh_in": 0,
            "n_remesh_out": 0,
            "n_free": 0,
            "n_excluded": 0,
            "n_pruned": 0,
            "pruned_circulation_fraction": 0.0,
            "n_population_pruned": 0,
            "population_pruned_circulation_fraction": 0.0,
            "population_pruned_velocity_bound": 0.0,
            "flux_ratio": 0.0,
            "transfer_in_band_residual": 0.0,
            "transfer_pre_prune_residual": 0.0,
            "transfer_out_of_band_fraction": 0.0,
            "transfer_max_amplification": 0.0,
        }
        spectrum = {}
        particle_count = 0
    else:
        conservation = {
            "raw_mismatch": finite_invariants(result.conservation_raw_mismatch),
            "applied_correction": finite_invariants(result.conservation_applied_correction),
            "corrected_mismatch": finite_invariants(result.conservation_corrected_mismatch),
        }
        transfer = {
            "cfl": float(result.cfl),
            "n_remesh_in": int(result.n_remesh_in),
            "n_remesh_out": int(result.n_remesh_out),
            "n_free": int(result.n_free),
            "n_excluded": int(result.n_excluded),
            "n_pruned": int(result.n_pruned),
            "pruned_circulation_fraction": float(result.pruned_circulation_fraction),
            "n_population_pruned": int(result.n_population_pruned),
            "population_pruned_circulation_fraction": float(
                result.population_pruned_circulation_fraction
            ),
            "population_pruned_velocity_bound": float(result.population_pruned_velocity_bound),
            "flux_ratio": float(result.flux_ratio),
            "transfer_in_band_residual": float(result.transfer_in_band_residual),
            "transfer_pre_prune_residual": float(result.transfer_pre_prune_residual),
            "transfer_out_of_band_fraction": float(result.transfer_out_of_band_fraction),
            "transfer_max_amplification": float(result.transfer_max_amplification),
        }
        spectrum = {str(name): float(value) for name, value in result.spectral_band_ratio.items()}
        particle_count = result.n_total
    if not all(np.isfinite(value) for value in transfer.values()):
        raise FloatingPointError("non-finite handoff diagnostic")
    if not all(np.isfinite(value) for value in spectrum.values()):
        raise FloatingPointError("non-finite spectral handoff diagnostic")

    boundary_flux = {
        name: float(coupler._last_vpm_bc_flux_diagnostics[name])
        for name in ("raw_mismatch", "applied_correction", "corrected_mismatch")
    }
    if not all(np.isfinite(value) for value in boundary_flux.values()):
        raise FloatingPointError("non-finite VPM boundary-flux diagnostic")

    interface = {}
    closure = {}
    if coupler.handoff is not None:
        interface = {
            str(name): float(value) for name, value in coupler.handoff.last_interface_flow.items()
        }
        closure = {
            str(name): float(value)
            for name, value in coupler.handoff.last_vortex_line_closure.items()
        }
    pressure_shift = (
        0.0 if coupler.pressure_reference is None else coupler.pressure_reference.last_shift
    )
    return {
        "conservation": conservation,
        "vpm_bc_flux": boundary_flux,
        "handoff": transfer,
        "spectral_band_ratio": spectrum,
        "interface_normal_velocity": interface,
        "vortex_line_closure": closure,
        "pressure_datum_shift": float(pressure_shift),
        "n_fvm_substeps": int(coupler.n_fvm_substeps),
        "handoff_particle_count": int(particle_count),
    }


def record_step(
    coupler,
    step: int,
    time_end: float,
    timing: tuple[float, float, float, float, float],
    handoff_result,
    *,
    logger: logging.Logger,
    comm=None,
) -> None:
    """Persist diagnostics and synchronize a completed coupling step."""
    t_vpm, t_blending, t_vpm_bc, t_fvm, t_handoff = timing
    diagnostics = compute_diagnostics(coupler, handoff_result)
    timing_data = {
        "vpm": float(t_vpm),
        "vpm_bc": float(t_vpm_bc),
        "blending": float(t_blending),
        "fvm": float(t_fvm),
        "handoff": float(t_handoff),
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
            "     [Handoff] N_before=%d  N_after=%d  |Γ|_before=%.4e  |Γ|_after=%.4e",
            int(stats.get("n_before", 0)),
            int(stats.get("n_after", 0)),
            float(stats.get("sum_before", 0.0)),
            float(stats.get("sum_after", 0.0)),
        )
        logger.info(
            "[Timing step=%d] VPM=%.3fs vpm_bc=%.3fs blending=%.3fs "
            "FVM=%.3fs handoff=%.3fs total=%.3fs",
            step,
            timing_data["vpm"],
            timing_data["vpm_bc"],
            timing_data["blending"],
            timing_data["fvm"],
            timing_data["handoff"],
            timing_data["total"],
        )
        print()
        print(f"[Step {step:4d}] t={time_end:.3f}s | Particles: {int(stats.get('n_after', 0))}")
        print(
            f"     Timing: VPM={t_vpm:.2f}s | BC={t_vpm_bc:.2f}s | "
            f"Blending={t_blending:.2f}s | FVM={t_fvm:.2f}s | Handoff={t_handoff:.2f}s"
        )
        sys.stdout.flush()
        flush_log(logger)

    if comm is not None and comm.Get_size() > 1:
        comm.Barrier()
    if coupler.config.backup_period > 0 and step % coupler.config.backup_period == 0:
        coupler.save_state(coupler.solution_dir / CHECKPOINT_DIRECTORY, coupling_step=step)


__all__ = [
    "OutputRedirector",
    "compute_diagnostics",
    "configure_logging",
    "flush_log",
    "record_step",
    "write_run_metadata",
]
