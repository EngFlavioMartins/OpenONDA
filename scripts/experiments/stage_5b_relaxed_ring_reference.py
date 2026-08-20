#!/usr/bin/env python3
"""Generate the unperturbed Gaussian-ring base state used by the Widnall gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RING_ASSETS = ROOT / "tutorials" / "VPM" / "vortexRing" / "assets"
sys.path.insert(0, str(RING_ASSETS))

from ring_diagnostics import RingDiagnosticsSampler, RingModeDiagnosticsSampler  # noqa: E402

from openonda.vpm import (  # noqa: E402
    AdvectionConfig,
    ParticleDistributor,
    StabilizationConfig,
    StretchingConfig,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
    VortexRingVPM,
    VPMSetup,
    VPMSolver,
)

RING_RADIUS = 1.0
RING_CIRCULATION = 1.0
CORE_RADIUS = 0.4131
REYNOLDS_NUMBER = 3000.0
DEFAULT_TAIL_FRACTION = 2.0e-3


def cadence(period: float, time_step: float) -> int:
    return max(1, round(period / time_step))


def viscous_config(
    scheme: str, spacing: float, viscosity: float, core_radius_ratio: float
) -> ViscousConfig:
    if scheme == "CS":
        return ViscousConfig.cs(
            kinematic_viscosity=viscosity,
            particle_spacing=spacing,
        )
    return ViscousConfig.gbd(
        particle_spacing=spacing,
        padding=5.0,
        threshold=1.0e-5,
        threshold_mode="budget",
        kinematic_viscosity=viscosity,
        max_nodes=150_000,
        cap_absolute_fraction=0.999,
        core_radius_ratio=core_radius_ratio,
    )


def run(
    *,
    output_directory: Path,
    label: str,
    spacing: float,
    time_step: float,
    final_time_star: float,
    viscous_scheme: str,
    velocity_method: str,
    compute_device: str,
    tail_fraction: float,
    backup_period: float,
    axisymmetric: bool,
    conserve_inviscid_invariants: bool,
    core_radius_ratio: float,
) -> None:
    if spacing <= 0.0 or time_step <= 0.0 or final_time_star <= 0.0 or backup_period <= 0.0:
        raise ValueError("spacing, time step, final time, and backup period must be positive")
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail fraction must lie strictly between zero and one")
    if core_radius_ratio <= 0.0:
        raise ValueError("regeneration radius ratio must be positive")
    particle_radius = core_radius_ratio * spacing if viscous_scheme == "GBD" else 2.0 * spacing
    represented_core_sq = CORE_RADIUS**2 - particle_radius**2
    if represented_core_sq <= 0.0:
        raise ValueError("particle radius must be smaller than the Gaussian core radius")

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / f"run_manifest_{label}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")

    tube_radius = np.sqrt(represented_core_sq) * np.sqrt(-np.log(tail_fraction))
    distributed = ParticleDistributor.toroidal_distribution(
        RING_RADIUS,
        tube_radius,
        spacing,
        epsilon_w=0.0,
        return_orbit_ids=axisymmetric,
    )
    if axisymmetric:
        position, volume, radius, orbit_id = distributed
    else:
        position, volume, radius = distributed
        orbit_id = None
    radius.fill(particle_radius)
    viscosity = RING_CIRCULATION / REYNOLDS_NUMBER
    velocity, particle_viscosity, circulation = VortexRingVPM(
        viscosity=viscosity,
        ring_center=[0.0, 0.0, 0.0],
        ring_radius=RING_RADIUS,
        ring_strength=RING_CIRCULATION,
        ring_thickness=CORE_RADIUS,
        avg_particle_radius=particle_radius,
        positions=position,
        volumes=volume,
        epsilon_W=0.0,
        max_modes=1,
        anti_diffuse_flag=True,
        normalize_circulation=True,
    )

    mode_sampler = RingModeDiagnosticsSampler(
        maximum_mode=16,
        azimuthal_bins=96,
        reference_radius=RING_RADIUS,
        transverse_origin=(0.0, 0.0),
    )
    solver = VPMSolver(
        setup=VPMSetup(
            time_step_size=time_step,
            compute_device=compute_device,
            time_integration="COUPLED",
            axisymmetric_no_swirl_axis="x" if axisymmetric else None,
            advection=AdvectionConfig(scheme="RK3"),
            turbulence=TurbulenceConfig.dns(),
            stretching=StretchingConfig.transposed(
                scheme="RK3",
                conserve_moments=conserve_inviscid_invariants,
                conserve_energy=conserve_inviscid_invariants,
            ),
            stabilization=StabilizationConfig.disabled(),
            velocity=(
                VelocityConfig.direct()
                if velocity_method == "DIRECT"
                else VelocityConfig.treecode(
                    theta=0.1,
                    sort_particle_targets=True,
                    traversal_block_dim=128,
                )
            ),
            viscous=viscous_config(viscous_scheme, spacing, viscosity, core_radius_ratio),
            logging_interval_steps=cadence(0.25, time_step),
            checkpoint_interval_steps=cadence(backup_period, time_step),
            checkpoint_name=label,
            checkpoint_directory=str(output_directory),
            sample_subdirectory=None,
            samplers=(RingDiagnosticsSampler(), mode_sampler),
            max_particles=150_000,
        )
    )
    solver.add_vortex_particles(
        position=position,
        velocity=velocity,
        circulation=circulation,
        radius=radius,
        volume=volume,
        viscosity=particle_viscosity,
        group_id=0,
        zone_id=orbit_id,
    )

    requested_steps = int(np.ceil(final_time_star / time_step))
    initial_strength = float(np.abs(solver.particles.vortex_strength_cpu()).max())
    manifest = {
        "status": "running",
        "source": "Verzicco and Shariff (1994), CTR proceedings, pp. 221-228",
        "claim_scope": "unperturbed axisymmetric relaxation only; no SGS model",
        "output_label": label,
        "ring_radius": RING_RADIUS,
        "ring_circulation": RING_CIRCULATION,
        "gaussian_core_radius": CORE_RADIUS,
        "gaussian_tail_fraction_at_cloud_boundary": tail_fraction,
        "circulation_reynolds_number": REYNOLDS_NUMBER,
        "particle_spacing": spacing,
        "particle_radius": particle_radius,
        "core_radius_ratio": core_radius_ratio if viscous_scheme == "GBD" else None,
        "initial_particles": len(position),
        "time_step": time_step,
        "requested_final_time_star": final_time_star,
        "requested_steps": requested_steps,
        "backup_period_time_star": backup_period,
        "time_integration": "COUPLED_RK3",
        "axisymmetric_no_swirl_axis": "x" if axisymmetric else None,
        "conserve_inviscid_moments_and_energy": conserve_inviscid_invariants,
        "stretching": "TRANSPOSED",
        "velocity_method": velocity_method,
        "treecode_theta": 0.1 if velocity_method == "TREECODE" else None,
        "molecular_diffusion": viscous_scheme,
        "sgs_model": "none",
        "processing_unit": compute_device,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    solver.record_diagnostics(refresh_fields=True)
    solver.backup_solution(str(output_directory / f"vpm_{label}"))
    termination_reason = None
    for _ in range(requested_steps):
        solver.advance()
        if np.abs(solver.particles.vortex_strength_cpu()).max() > 50.0 * initial_strength:
            termination_reason = "peak particle strength exceeded 50 times its initial value"
            break
        if solver.time_step % cadence(0.25, time_step):
            continue
        health = solver._discretization_health
        if float(health["vorticity_divergence_error"]) > 0.12:
            termination_reason = "normalized particle divergence exceeded 0.12"
            break
        if float(health["strength_misalignment_deg"]) > 45.0:
            termination_reason = "particle-strength misalignment exceeded 45 degrees"
            break

    # The final state must be present in sampled histories even when it falls
    # between regular output times.
    solver.record_diagnostics(refresh_fields=True)
    solver.save_state(str(output_directory / f"vpm_{label}_final"))
    manifest.update(
        status="resolution_lost" if termination_reason else "completed",
        completed_steps=solver.time_step,
        completed_time_star=solver.time,
        final_particles=len(solver.particles),
    )
    if termination_reason:
        manifest["termination_reason"] = termination_reason
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--time-step", type=float, required=True)
    parser.add_argument("--final-time-star", type=float, required=True)
    parser.add_argument("--tail-fraction", type=float, default=DEFAULT_TAIL_FRACTION)
    parser.add_argument("--backup-period", type=float, default=1.0)
    parser.add_argument("--axisymmetric", action="store_true")
    parser.add_argument("--conserve-inviscid-invariants", action="store_true")
    parser.add_argument("--viscous-scheme", choices=("CS", "GBD"), default="GBD")
    parser.add_argument("--regen-radius-ratio", type=float, default=2.5)
    parser.add_argument("--velocity-method", choices=("DIRECT", "TREECODE"), default="DIRECT")
    parser.add_argument(
        "--processing-unit",
        choices=("AUTO", "CPU", "METAL", "VULKAN", "CUDA"),
        default="CPU",
    )
    args = parser.parse_args()
    run(
        output_directory=args.output_directory,
        label=args.label,
        spacing=args.spacing,
        time_step=args.time_step,
        final_time_star=args.final_time_star,
        viscous_scheme=args.viscous_scheme,
        velocity_method=args.velocity_method,
        compute_device=args.compute_device,
        tail_fraction=args.tail_fraction,
        backup_period=args.backup_period,
        axisymmetric=args.axisymmetric,
        conserve_inviscid_invariants=args.conserve_inviscid_invariants,
        core_radius_ratio=args.core_radius_ratio,
    )


if __name__ == "__main__":
    main()
