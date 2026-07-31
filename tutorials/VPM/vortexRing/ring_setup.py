#!/usr/bin/env python3
"""
Run a single vortex ring for DNS/LES validation.
=================================================
This script simulates a single vortex ring and tracks its self-induced
velocity decay against the Saffman analytical model.

Objective: test the different stretching schemes

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: April 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import argparse
import numpy as np
from pathlib import Path

from openonda.vpm import (
    ParticleDistributor,
    Solver,
    VPMSetup,
    VelocityConfig,
)
from openonda.vpm import (
    AdvectionConfig,
    StabilizationConfig,
    TurbulenceConfig,
    ViscousConfig,
    StretchingConfig,
)
from openonda.vpm import VortexRingVPM


def main():
    parser = argparse.ArgumentParser(description="Run leapfrogging rings simulation.")
    parser.add_argument(
        "--mode", choices=["dns", "les"], default="les", help="Simulation mode (dns or les)"
    )
    parser.add_argument("--name", default="LES", help="Output file name prefix")
    parser.add_argument(
        "--stretching",
        choices=["direct", "transposed", "mixed"],
        default="transposed",
        help="Vortex stretching formulation.",
    )
    parser.add_argument(
        "--stretching-scheme",
        type=str.upper,
        choices=["EULER", "RK2", "RK3", "RK4"],
        default="RK3",
        help="Time integrator for the stretching substep.",
    )
    parser.add_argument(
        "--solution-dir",
        default="solution",
        help="Root directory for solution output (default: solution/)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=600,
        help="Number of time steps (default: 600)",
    )
    parser.add_argument(
        "--particle-spacing",
        type=float,
        default=0.035,
        help="Initial particle spacing [m] (about 2.9 points per core radius).",
    )
    parser.add_argument(
        "--processing-unit",
        default="CUDA",
        choices=["AUTO", "CPU", "VULKAN", "CUDA", "METAL"],
        help="Compute backend. Default CUDA keeps the tutorial on the tested NVIDIA path.",
    )
    parser.add_argument(
        "--device-memory-fraction",
        type=float,
        default=0.5,
        help="Fraction of GPU memory reserved by Taichi.",
    )
    parser.add_argument(
        "--backup-frequency",
        type=int,
        default=6,
        help="Backup interval in steps (default: 6, giving 100 snapshots for 600 steps).",
    )
    parser.add_argument(
        "--logging-frequency",
        type=int,
        default=None,
        help="Flow-diagnostic logging interval in steps (default: backup frequency).",
    )

    args = parser.parse_args()

    # ================================================
    # 1. Physical Parameters
    # ================================================
    ring_radius = 1.0  # Major radius of the vortex ring [m]
    ring_strength = np.pi  # Ring circulation [m²/s]
    kinematic_viscosity = ring_strength / 3000.0  # Re = Gamma/nu = 3000
    core_radius = 0.1  # Vortex core radius [m]

    # ================================================
    # 2. Numerical Parameters
    # ================================================
    particle_spacing = args.particle_spacing  # Grid spacing [m]
    time_step = 0.02  # [s]
    num_steps = args.num_steps
    logging_frequency = (
        args.backup_frequency if args.logging_frequency is None else args.logging_frequency
    )

    # ================================================
    # 3. Create Initial Particle Distribution
    # ================================================
    domain_bounds = [-0.15, 0.15, -1.5, 1.5, -1.5, 1.5]
    positions, volumes, radii = ParticleDistributor.hexagonal_distribution(
        domain_bounds, particle_spacing
    )

    # ================================================
    # 4. Configure Solver for simulation mode
    # ================================================
    turbulence = (
        TurbulenceConfig.dns() if args.mode == "dns" else TurbulenceConfig.les_smagorinsky()
    )

    stretching_scheme = args.stretching_scheme
    _stretching_map = {
        "direct": StretchingConfig.direct(scheme=stretching_scheme),
        "transposed": StretchingConfig.transposed(scheme=stretching_scheme),
        "mixed": StretchingConfig.mixed(scheme=stretching_scheme),
    }
    stretching = _stretching_map[args.stretching]
    stabilization = StabilizationConfig.disabled()

    output_dir = Path(args.solution_dir) / args.name

    solver_config = VPMSetup(
        time_step_size=time_step,
        advection=AdvectionConfig(scheme="RK3"),
        turbulence=turbulence,
        stretching=stretching,
        stabilization=stabilization,
        velocity=VelocityConfig.treecode(
            theta=0.35,
            sort_particle_targets=True,
            traversal_block_dim=128,
        ),
        viscous=ViscousConfig.cs(),
        processing_unit=args.processing_unit,
        device_memory_fraction=args.device_memory_fraction,
        backup_frequency=args.backup_frequency,
        logging_frequency=logging_frequency,
        timing_frequency=max(1, 6 * logging_frequency) if logging_frequency > 0 else 0,
        backup_file_name=args.name,
        backup_directory=str(output_dir),
        max_particles=30_000,
    )

    vpm = Solver(setup=solver_config)

    # ================================================
    # 5. Initialize One Vortex Ring (centered at origin)
    # ================================================
    vel, visc, circ = VortexRingVPM(
        viscosity=kinematic_viscosity,
        ring_center=[0, 0, 0],
        ring_radius=ring_radius,
        ring_strength=ring_strength,
        ring_thickness=core_radius,
        avg_particle_radius=radii.mean(),
        positions=positions,
        volumes=volumes,
        epsilon_W=0.025,
        anti_diffuse_flag=True,
    )

    vpm.add_vortex_particles(
        position=positions,
        velocity=vel,
        circulation=circ,
        radius=radii,
        volume=volumes,
        viscosity=visc,
        group_id=0,
    )

    vpm.remove_weak_particles(percent=0.1, per_group=True)

    # ================================================
    # 6. Run Simulation
    # ================================================
    max_circ0 = np.abs(vpm.particles.circulation_cpu()).max()
    for _ in range(num_steps):
        vpm.update_state()
        if np.abs(vpm.particles.circulation_cpu()).max() > 50 * max_circ0:
            print("Solution blew up — stopping.")
            break


if __name__ == "__main__":
    main()
