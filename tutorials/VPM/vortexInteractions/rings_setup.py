#!/usr/bin/env python3
"""
Run coaxial vortex-ring interactions: leapfrogging or head-on collision.
======================================================================
Two vortex rings are initialised coaxially.  Their relative circulation
signs determine the interaction type:
  gamma1 * gamma2 > 0   →  leapfrogging (same sign)
  gamma1 * gamma2 < 0   →  head-on collision (opposite sign)

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: May 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import numpy as np
from pathlib import Path
import subprocess

from source.solvers.VPM import Solver, SolverConfig, VelocityConfig, ParticleDistributor
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    TurbulenceConfig,
    ViscousConfig,
    StretchingConfig,
)
from source.solvers.VPM.utils import VortexRingVPM
from source.solvers.VPM.utils.field_samplers import SurfaceSampler


def main():
    parser = argparse.ArgumentParser(
        description="Run coaxial vortex-ring interaction: leapfrog or collision."
    )
    parser.add_argument(
        "--gamma1", type=float, default=np.pi, help="Circulation of ring 1 [m²/s] (default: π)."
    )
    parser.add_argument(
        "--gamma2",
        type=float,
        default=np.pi,
        help="Circulation of ring 2 [m²/s] (default: π). "
        "Same sign → leapfrog; opposite sign → head-on collision.",
    )
    parser.add_argument(
        "--mode",
        choices=["dns", "les"],
        default="les",
        help="Simulation mode: dns | les",
    )
    parser.add_argument(
        "--name",
        default="leapfrog_LES",
        help="Output sub-directory / file-name prefix (default: leapfrog_LES)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.103,
        help="Time step size [s]. DVH pins dt to Δt_d = β·R_d²/(4nu) "
        "(≈ 0.103 s here) — the diffusion operator fires once per step. "
        "A user dt differing from Δt_d is overridden to Δt_d.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1200,
        help="Number of time steps (default: 1200)",
    )
    parser.add_argument(
        "--particle-spacing",
        type=float,
        default=0.025,
        help="Initial particle spacing h [m] (default: 0.025).",
    )
    parser.add_argument(
        "--output-root",
        default="solution",
        help="Parent directory; this case is written to <output-root>/<name>.",
    )
    parser.add_argument("--backup-frequency", type=int, default=2)
    parser.add_argument("--logging-frequency", type=int, default=2)
    parser.add_argument("--max-particles", type=int, default=500_000)
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=1,
        help="Minimum complete VPM physics microsteps per output step.",
    )
    parser.add_argument(
        "--deformation-cfl",
        type=float,
        default=None,
        help="Optional adaptive limit max(||S||_F)*dt_sub.",
    )
    parser.add_argument("--max-physics-substeps", type=int, default=64)
    parser.add_argument(
        "--isr",
        type=float,
        default=0.0,
        help="(legacy) enable blend relaxation with strain-gate constant C "
        "(0 = disabled). Equivalent to --relaxation blend --isr-C <value>.",
    )
    parser.add_argument(
        "--relaxation",
        choices=["none", "blend", "pedrizzetti"],
        default="none",
        help="Strength relaxation stabilizer: blend (conservative ADM residual "
        "filter) or pedrizzetti (|Γ|-preserving realignment).",
    )
    parser.add_argument(
        "--isr-C",
        type=float,
        default=1.0,
        help="Strain-gate rate constant C for the relaxation (default 1.0).",
    )
    parser.add_argument(
        "--deconv",
        type=int,
        default=1,
        help="Van Cittert deconvolution iterations for the relaxation target "
        "(0 = raw mollified field; default 1).",
    )
    parser.add_argument(
        "--stretching",
        choices=["transposed", "gradu", "rvpm"],
        default="transposed",
        help="Stretching scheme: transposed (direct O(N²)), gradu (local O(N)), "
        "or rvpm (Alvarez & Ning reformulation, local O(N), conserves σ²|Γ|).",
    )
    parser.add_argument(
        "--viscous",
        choices=["dvh", "cs"],
        default="dvh",
        help="Viscous scheme: dvh (default) or cs (Core Spreading).",
    )
    parser.add_argument(
        "--dvh-threshold-mode",
        choices=["budget", "absolute", "relative_max"],
        default="budget",
        help="DVH regen-node survival criterion. 'budget' (default) bounds the "
        "circulation destroyed per firing to --dvh-threshold × Σ|Γ| — an "
        "absolute threshold (the old default, 3e-5) was measured to destroy "
        "~1.2%% of Σ|Γ| PER FIRING on the rings (total evaporation by step "
        "~450).",
    )
    parser.add_argument(
        "--dvh-threshold",
        type=float,
        default=2e-4,
        help="Threshold value: budget → fractional Σ|Γ| loss allowed per "
        "firing (default 2e-4); absolute → node |Γ| floor in m³/s.",
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "vulkan", "cpu"],
        default="gpu",
        help="Compute backend (default: gpu = CUDA). Use 'vulkan' for the "
        "LES+relaxation cases: Taichi 1.7.4's CUDA backend corrupts its "
        "docs/vpm_stabilization_audit.md.",
    )
    args = parser.parse_args()

    gamma1 = args.gamma1
    gamma2 = args.gamma2

    case_label = "leapfrog" if gamma1 * gamma2 >= 0 else "collide"

    # ================================================
    # 1. Physical Parameters  (Alvarez et al. 2024)
    # ================================================
    ring_radius = 1.0  # Major radius [m]
    reference_gamma = np.pi  # Reference circulation for Re [m²/s]
    kinematic_viscosity = reference_gamma / 3000.0  # Re = Γ/nu = 3000
    core_radius = 0.1  # Vortex core radius [m]

    # ================================================
    # 2. Numerical Parameters
    # ================================================
    particle_spacing = args.particle_spacing  # Particle spacing h [m]
    if particle_spacing <= 0.0:
        parser.error("--particle-spacing must be positive")
    time_step = args.dt  # [s]
    num_steps = args.num_steps  # Simulation steps

    # Relaxation: --relaxation flag, with --isr <C> as a legacy alias for blend
    relaxation_mode = args.relaxation
    isr_C = args.isr_C
    if args.isr > 0.0 and relaxation_mode == "none":
        relaxation_mode = "blend"
        isr_C = args.isr
    isr_enabled = relaxation_mode != "none"

    # ================================================
    # 3. Create Initial Particle Distribution
    # ================================================
    domain_bounds = [-0.15, 0.15, -1.5, 1.5, -1.5, 1.5]
    positions, volumes, radii = ParticleDistributor.hexagonal_distribution(
        domain_bounds, particle_spacing
    )

    # ================================================
    # 4. Configure Solver
    # ================================================
    if args.mode == "dns":
        turbulence = TurbulenceConfig.dns()
    else:
        turbulence = TurbulenceConfig.les_smagorinsky(cs=0.16, ce=1.048)

    if args.stretching == "rvpm":
        stretching = StretchingConfig.rvpm()
    elif args.stretching == "gradu":
        stretching = StretchingConfig.gradu()
    else:
        stretching = StretchingConfig.transposed()

    output_dir = Path(args.output_root) / args.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty result directory: {output_dir}. "
            "Choose a new --name or --output-root."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # XZ-plane field sampler for diagnostic analysis
    if case_label == "collide":
        xz_sampler = SurfaceSampler(
            point=[0, 0, 0],
            normal=[0, 1, 0],
            bounds=[-7, 7, -4, 4],
            spacing=particle_spacing,
            file_name="xz_slice",
            output_dir=str(output_dir / "samples"),
        )
    else:
        xz_sampler = SurfaceSampler(
            point=[0, 0, 0],
            normal=[0, 1, 0],
            bounds=[-0.5, 11.5, -2.0, 2.0],
            spacing=particle_spacing,
            file_name="xz_slice",
            output_dir=str(output_dir / "samples"),
        )

    if args.viscous == "cs":
        viscous_cfg = ViscousConfig.cs(
            viscosity=kinematic_viscosity,
            characteristic_distance=particle_spacing,
        )
    else:
        viscous_cfg = ViscousConfig.dvh(
            h=particle_spacing,
            dvh_rd_ratio=3,
            viscosity=kinematic_viscosity,
            threshold_mode=args.dvh_threshold_mode,
            threshold=args.dvh_threshold,
            max_nodes=250_000,
        )

    advection_cfg = AdvectionConfig(scheme="RK2")

    solver_config = SolverConfig(
        time_step_size=time_step,
        processing_unit={"cpu": "CPU", "vulkan": "GPU_VULKAN", "gpu": "GPU"}[args.device],
        turbulence=turbulence,
        stretching=stretching,
        velocity=VelocityConfig.treecode(theta=0.3),
        viscous=viscous_cfg,
        advection=advection_cfg,
        backup_frequency=args.backup_frequency,
        logging_frequency=args.logging_frequency,
        backup_file_name=args.name,
        solution_name=str(output_dir),
        backup_directory=str(output_dir),
        max_particles=args.max_particles,
        samplers=[(xz_sampler, "xz_slice")],
        isr_enabled=isr_enabled,
        isr_mode=relaxation_mode if isr_enabled else "blend",
        isr_C=isr_C,
        isr_deconv=args.deconv,
        isr_conserve=True,
        isr_cfl=0.2,
        physics_substeps=args.physics_substeps,
        deformation_cfl=args.deformation_cfl,
        max_physics_substeps=args.max_physics_substeps,
    )

    def _jsonable(value):
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        revision = ""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision or None,
        "arguments": vars(args),
        "case_label": case_label,
        "output_directory": str(output_dir),
        "solver_config": _jsonable(solver_config.to_dict()),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    vpm = Solver(config=solver_config)

    # ================================================
    # 5. Initialize Two Vortex Rings
    # ================================================
    if gamma1 * gamma2 >= 0:
        # Leapfrogging: rings half a major-radius apart, same axis.
        ring_centers = [[-0.5, 0, 0], [0.5, 0, 0]]
        ring_strengths = [gamma1, gamma2]
    else:
        # Head-on collision: rings 4 diameters apart
        ring_separation = 4.0 * (2.0 * ring_radius)
        ring_centers = [[ring_separation / 2, 0, 0], [-ring_separation / 2, 0, 0]]
        ring_strengths = [gamma1, gamma2]

    for group_index, (center, strength) in enumerate(zip(ring_centers, ring_strengths)):
        vel, visc, circ = VortexRingVPM(
            viscosity=kinematic_viscosity,
            ring_center=np.zeros(3),
            ring_radius=ring_radius,
            ring_strength=strength,
            ring_thickness=core_radius,
            avg_particle_radius=radii.mean(),
            positions=positions,
            volumes=volumes,
            epsilon_W=0.05,
            anti_diffuse_flag=True,
        )

        vpm.add_vortex_particles(
            position=positions - np.array(center),
            velocity=vel,
            circulation=circ,
            radius=radii,
            volume=volumes,
            viscosity=visc,
            group_id=np.full(len(positions), group_index, dtype=np.int32),
        )
        # Prune each ring before allocating the next full background lattice.
        # At h=0.020 the unfiltered two-ring lattice exceeds 8e5 particles even
        # though only a narrow tubular core carries meaningful circulation.
        vpm.remove_weak_particles(percent=0.1, per_group=True)

    vpm.info()

    # ================================================
    # 6. Run simulation
    # ================================================
    initial_max_norm = float(np.linalg.norm(vpm.particles_circulation, axis=1).max())
    blowup_threshold = max(50.0 * initial_max_norm, 0.1)

    metric_fields = [
        "step",
        "time",
        "n_particles",
        "max_gamma",
        "p999_gamma",
        "sum_gamma_magnitude",
        "max_strain_frobenius",
        "max_parallel_strain",
        "radius_min",
        "radius_max",
        "physics_substeps",
        "physics_dt_sub",
    ]

    def stability_metrics(step):
        """Return inexpensive indicators of the stretching feedback loop."""
        circulation = vpm.particles_circulation
        n_particles = len(circulation)
        gamma_norm = np.linalg.norm(circulation, axis=1)
        strain = vpm.particles.strain_rate_cpu(use_cache=False)[:n_particles]
        strain_norm = np.linalg.norm(strain.reshape(n_particles, 9), axis=1)

        parallel_strain = np.zeros(n_particles)
        nonzero = gamma_norm > np.finfo(float).eps
        if np.any(nonzero):
            gamma_hat = circulation[nonzero] / gamma_norm[nonzero, None]
            parallel_strain[nonzero] = np.einsum(
                "ni,nij,nj->n", gamma_hat, strain[nonzero], gamma_hat
            )

        radii = vpm.particles_radii
        microstep_info = getattr(vpm, "last_physics_substeps", {})
        return {
            "step": step,
            "time": vpm.flow_time,
            "n_particles": n_particles,
            "max_gamma": float(gamma_norm.max()),
            "p999_gamma": float(np.quantile(gamma_norm, 0.999)),
            "sum_gamma_magnitude": float(gamma_norm.sum()),
            "max_strain_frobenius": float(strain_norm.max()),
            "max_parallel_strain": float(parallel_strain.max()),
            "radius_min": float(radii.min()),
            "radius_max": float(radii.max()),
            "physics_substeps": int(microstep_info.get("count", 1)),
            "physics_dt_sub": float(microstep_info.get("dt_sub", time_step)),
        }

    metrics_path = output_dir / "stability_metrics.csv"
    with metrics_path.open("w", newline="", buffering=1) as metrics_file:
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
        metrics_writer.writeheader()
        metrics_writer.writerow(stability_metrics(0))

        for step in range(num_steps):
            # The complete-physics microstep path is fail-fast by design: it
            # raises when the adaptive substep count exceeds its safety ceiling
            # (acceptance criterion 5) or when the strain used to size the
            # microsteps is non-finite (criterion 1).  Both are *defined* failure
            # outcomes of the stability challenge, not script bugs — record them
            # like a blow-up so the partial trajectory is still recoverable for
            # post-processing instead of crashing the whole ladder.
            try:
                vpm.update_state()
            except (RuntimeError, FloatingPointError) as exc:
                print(
                    f"\n*** INSTABILITY ABORT at step {step + 1} "
                    f"(t={vpm.flow_time:.2f} s) — {exc} ***"
                )
                vpm.save_state(str(output_dir / "pre_blowup"))
                break
            metrics = stability_metrics(step + 1)
            metrics_writer.writerow(metrics)
            max_norm = metrics["max_gamma"]
            if not np.isfinite(max_norm) or max_norm > blowup_threshold:
                print(
                    f"\n*** BLOWUP DETECTED at step {step + 1} (t={vpm.flow_time:.2f} s) "
                    f"— max|Γ|={max_norm:.4f} > {blowup_threshold:.4f} ***"
                )
                vpm.save_state(str(output_dir / "pre_blowup"))
                break
        else:
            print(f"Simulation completed {num_steps} steps without blowup.")


if __name__ == "__main__":
    main()
