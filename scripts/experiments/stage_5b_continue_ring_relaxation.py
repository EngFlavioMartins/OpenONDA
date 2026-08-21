#!/usr/bin/env python3
"""Continue a restartable Gaussian-ring relaxation to an absolute final time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.solvers.VPM import VPMSolver


def continue_run(
    *,
    checkpoint: Path,
    target_time_star: float,
    extension_label: str,
    time_step: float | None = None,
    output_directory: Path | None = None,
) -> None:
    checkpoint = checkpoint.resolve()
    # The public restart API expects the checkpoint basename, while users and
    # manifests naturally point at the HDF5 file itself.
    if checkpoint.suffix == ".h5":
        checkpoint = checkpoint.with_suffix("")
    solver = VPMSolver.continue_from_checkpoint(str(checkpoint))
    if solver is None:
        raise RuntimeError(f"could not restore {checkpoint}")
    if target_time_star <= solver.time:
        raise ValueError("target time must be later than the restart state")
    if time_step is not None:
        if time_step <= 0.0:
            raise ValueError("time step must be positive")
        solver.time_step_size = time_step

    alternate_output = output_directory is not None
    output_directory = (
        output_directory.resolve()
        if output_directory is not None
        else Path(solver.setup.checkpoint_directory).resolve()
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    if alternate_output:
        # Do not mix scheduled snapshots or diagnostic rows from a convergence
        # branch with the authoritative trajectory.
        solver.checkpoint_interval_steps = 0
        solver.logging_interval_steps = 0
    manifest_path = output_directory / f"extension_manifest_{extension_label}.json"
    final_base = output_directory / f"vpm_{extension_label}_final"
    if manifest_path.exists() or final_base.with_suffix(".h5").exists():
        raise FileExistsError(f"refusing to overwrite extension {extension_label}")

    remaining_steps = int(np.ceil((target_time_star - solver.time) / solver.time_step_size))
    initial_strength = float(np.abs(solver.particles.vortex_strength_cpu()).max())
    initial_energy = solver.total_kinetic_energy
    initial_dissipation = float(solver._flow_integrals["vorticity_dissipation_rate"])
    manifest = {
        "status": "running",
        "claim_scope": "continuation of unperturbed axisymmetric relaxation only",
        "restart": str(checkpoint.with_suffix(".h5")),
        "start_time_star": solver.time,
        "target_time_star": target_time_star,
        "requested_additional_steps": remaining_steps,
        "particle_spacing": solver.setup.viscous.particle_spacing,
        "time_step_size": solver.time_step_size,
        "velocity_method": solver.setup.velocity.method,
        "molecular_diffusion": solver.setup.viscous.scheme,
        "sgs_model": "none",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    termination_reason = None
    health_cadence = max(1, round(0.25 / solver.time_step_size))
    for _ in range(remaining_steps):
        solver.advance()
        if np.abs(solver.particles.vortex_strength_cpu()).max() > 50.0 * initial_strength:
            termination_reason = "peak particle strength exceeded 50 times restart value"
            break
        if solver.step % health_cadence:
            continue
        health = solver._discretization_health
        if float(health["vorticity_divergence_error"]) > 0.12:
            termination_reason = "normalized particle divergence exceeded 0.12"
            break
        if float(health["strength_misalignment_deg"]) > 45.0:
            termination_reason = "particle-strength misalignment exceeded 45 degrees"
            break

    if alternate_output:
        solver.stepper._update_velocity_and_gradients()
        solver._update_all_flow_integrals()
    else:
        solver.record_diagnostics(refresh_fields=True)
    final_energy = solver.total_kinetic_energy
    final_dissipation = float(solver._flow_integrals["vorticity_dissipation_rate"])
    solver.save_state(str(final_base))
    duration = solver.time - float(manifest["start_time_star"])
    predicted_energy_change = 0.5 * duration * (initial_dissipation + final_dissipation)
    measured_energy_change = final_energy - initial_energy
    manifest.update(
        status="resolution_lost" if termination_reason else "completed",
        completed_steps=solver.step,
        completed_time_star=solver.time,
        final_particles=len(solver.particles),
        initial_kinetic_energy=initial_energy,
        final_kinetic_energy=final_energy,
        measured_energy_change=measured_energy_change,
        trapezoidal_molecular_energy_change=predicted_energy_change,
        energy_balance_relative_residual=abs(
            (measured_energy_change - predicted_energy_change) / predicted_energy_change
        ),
    )
    if termination_reason:
        manifest["termination_reason"] = termination_reason
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-time-star", type=float, required=True)
    parser.add_argument("--extension-label", required=True)
    parser.add_argument("--time-step-size", type=float)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    continue_run(
        checkpoint=args.checkpoint,
        target_time_star=args.target_time_star,
        extension_label=args.extension_label,
        time_step=args.time_step_size,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()
