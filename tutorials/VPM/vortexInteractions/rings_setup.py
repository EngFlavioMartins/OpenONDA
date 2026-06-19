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
import numpy as np
from pathlib import Path

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
    particle_spacing = 0.025  # Particle spacing h [m]
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

    output_dir = Path("solution") / args.name

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
        backup_frequency=2,
        logging_frequency=2,
        backup_file_name=args.name,
        solution_name=str(output_dir),
        backup_directory=str(output_dir),
        samplers=[(xz_sampler, "xz_slice")],
        isr_enabled=isr_enabled,
        isr_mode=relaxation_mode if isr_enabled else "blend",
        isr_C=isr_C,
        isr_deconv=args.deconv,
        isr_conserve=True,
        isr_cfl=0.2,
    )

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

    vpm.remove_weak_particles(percent=0.1, per_group=True)
    vpm.info()

    # ================================================
    # 6. Run simulation
    # ================================================
    initial_max_norm = float(np.linalg.norm(vpm.particles_circulation, axis=1).max())
    blowup_threshold = max(50.0 * initial_max_norm, 0.1)

    for step in range(num_steps):
        vpm.update_state()
        max_norm = float(np.linalg.norm(vpm.particles_circulation, axis=1).max())
        if max_norm > blowup_threshold:
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
