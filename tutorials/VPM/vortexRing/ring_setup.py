#!/usr/bin/env python3
"""
Run a single vortex ring for DNS/LES validation.
=================================================
This script simulates a single vortex ring and tracks its self-induced
velocity decay against the Saffman analytical model.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: April 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import argparse
import numpy as np
from pathlib import Path

from source.solvers.VPM import Solver, SolverConfig, VelocityConfig, ParticleDistributor
from source.solvers.VPM.config.types import (
    TurbulenceConfig,
    ViscousConfig,
    AdaptationConfig,
    StretchingConfig,
)
from source.solvers.VPM.utils import VortexRingVPM


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
        help="Vortex stretching scheme for LES (ignored for DNS)",
    )
    parser.add_argument(
        "--solution-dir",
        default="solution",
        help="Root directory for solution output (default: solution/)",
    )
    parser.add_argument(
        "--direction-alpha",
        type=float,
        default=1.0,
        help="Pedrizzetti direction-relaxation coefficient (default: 1.0 = disabled, "
        "range [0.85, 0.99] recommended for stabilization)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=600,
        help="Number of time steps (default: 600)",
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
    particle_spacing = 0.03  # Grid spacing [m]
    time_step = 0.02  # [s]
    num_steps = args.num_steps

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
        TurbulenceConfig.dns()
        if args.mode == "dns"
        else TurbulenceConfig.les_smagorinsky(cs=0.16, ce=1.048)
    )

    _stretching_map = {
        "direct": StretchingConfig.classical(),
        "transposed": StretchingConfig.transposed(),
        "mixed": StretchingConfig.mixed(),
    }
    stretching = _stretching_map[args.stretching]

    output_dir = Path(args.solution_dir) / args.name

    # Map legacy direction_alpha to the ISR API:
    #   alpha == 1.0  → disabled (ISR off, default)
    #   alpha  < 1.0  → Pedrizzetti constant-relaxation mode (range [0.85, 0.99])
    isr_kwargs: dict = {}
    if args.direction_alpha < 1.0:
        isr_kwargs = dict(
            isr_enabled=True,
            isr_mode="pedrizzetti",
            isr_gate="constant",
            isr_rlx=args.direction_alpha,
            isr_conserve=True,
        )

    solver_config = SolverConfig(
        time_step_size=time_step,
        turbulence=turbulence,
        stretching=stretching,
        velocity=VelocityConfig.treecode(theta=0.5),
        viscous=ViscousConfig.cs(),
        backup_frequency=6,
        logging_frequency=6,
        backup_file_name=args.name,
        solution_name=str(output_dir),
        backup_directory=str(output_dir),
        **isr_kwargs,
    )

    vpm = Solver(config=solver_config)

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
        epsilon_W=0.05,
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

    vpm.remove_weak_particles(percent=1.0, per_group=True)

    # ================================================
    # 6. Run Simulation
    # ================================================
    for step in range(num_steps):
        vpm.update_state()


if __name__ == "__main__":
    main()
