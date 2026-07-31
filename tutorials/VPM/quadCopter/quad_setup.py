"""
QuadCopter VPM Tutorial
=======================
Simulates the wake of a toy quadcopter in hover.

Physical setup:
  - Vehicle moves upward (+Z), equivalent to a uniform inflow of -Uz
    in the vehicle frame.  This pushes shed wake particles downward,
    clearing the rotor disk plane.
  - 4× rotors at R≈0.15 m, alternating CW/CCW about the Z axis.
  - Flat-plate blades with 12°→6° linear twist (root→tip), tapered
    25 mm → 15 mm chord.

Numerical setup:
  - WINCKELMANS particle kernel (algebraic, stronger regularisation)
  - Stretching DISABLED (prevents helical-wake vorticity amplification)
  - Core-spreading viscous diffusion (DNS)
  - σ_factor = 2.5 for sufficient particle overlap
  - Background velocity [0, 0, -U_climb] pushes particles away from disk

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026
"""

import argparse
import numpy as np
from pathlib import Path

from assets.generate_blade import create_rotor_blade, save_blade
from openonda.vpm import Solver
from openonda.vpm import (
    VPMSetup,
    StabilizationConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
    TurbulenceConfig,
)
from openonda.vpm import VLMSurfaceSetup, VLMSetup
from openonda.vpm import RotatingVLM
from openonda.vpm import SurfaceSampler


def main():
    parser = argparse.ArgumentParser(description="QuadCopter VPM tutorial.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override total step count (default: 6 revolutions).",
    )
    parser.add_argument(
        "--processing-unit",
        default="CUDA",
        choices=["AUTO", "CPU", "VULKAN", "CUDA", "METAL"],
        help="Compute backend. Default CUDA keeps the tutorial on the tested NVIDIA path.",
    )
    args = parser.parse_args()

    # =========================================================================
    # 1. Physical Parameters
    # =========================================================================

    RPM = 200.0  # Rotor speed [rev/min]
    omega = RPM * 2 * np.pi / 60.0  # ≈ 62.83 rad/s
    R_tip = 0.15  # Blade tip radius [m]
    R_hub = 0.03  # Hub cutout radius [m]
    V_tip = omega * R_tip  # ≈ 9.42 m/s

    rho = 1.225  # Air density [kg/m³]
    nu = 1.5e-5  # Kinematic viscosity [m²/s]

    n_blades = 2  # Number of blades per rotor
    U_climb = 0.8  # [m/s] upward climb speed
    background_velocity = [0.0, 0.0, -U_climb]

    # =========================================================================
    # 2. Time Stepping
    # =========================================================================

    deg_per_step = 7.5
    dt = (deg_per_step * np.pi / 180.0) / omega
    steps_per_rev = int(360.0 / deg_per_step)
    n_revolutions = 6
    n_steps = args.num_steps if args.num_steps is not None else n_revolutions * steps_per_rev

    # =========================================================================
    # 3. Generate Blade Geometry
    # =========================================================================
    blade_ccw_json = "assets/blade_ccw.json"
    blade_cw_json = "assets/blade_cw.json"

    blade_params = dict(
        R_hub=R_hub,
        R_tip=R_tip,
        chord_root=0.025,  # 25 mm
        chord_tip=0.015,  # 15 mm
        pitch_root_deg=12.0,  # deg — high pitch at hub
        pitch_tip_deg=6.0,  # deg — low pitch at tip (washout)
        n_chord=4,
        n_span=12,
    )

    blade_ccw = create_rotor_blade(**blade_params, clockwise=False)
    save_blade(blade_ccw, blade_ccw_json)

    blade_cw = create_rotor_blade(**blade_params, clockwise=True)
    save_blade(blade_cw, blade_cw_json)

    # =========================================================================
    # 4. VLM System Setup
    # =========================================================================

    # Rotor layout — alternating CW / CCW for torque balance
    arm_length = 0.16  # [m] hub centre distance from vehicle CG
    rotors = [
        ("rotor_0", [arm_length, arm_length, 0.0], 1.0),  # CCW
        ("rotor_1", [-arm_length, arm_length, 0.0], -1.0),  # CW
        ("rotor_2", [-arm_length, -arm_length, 0.0], 1.0),  # CCW
        ("rotor_3", [arm_length, -arm_length, 0.0], -1.0),  # CW
    ]

    vlm_setup = VLMSetup(
        surfaces=tuple(
            VLMSurfaceSetup(
                blade_ccw_json if direction > 0 else blade_cw_json,
                name=f"{name}_blade_{blade_index}",
                kinematics=RotatingVLM(
                    omega=omega * direction,
                    axis=[0.0, 0.0, 1.0],
                    center=position,
                ),
                translation=tuple(position),
                rotation_deg=(0.0, 0.0, 360.0 / n_blades * blade_index),
                rotation_center=(0.0, 0.0, 0.0),
                group_id=rotor_index + 1,
            )
            for rotor_index, (name, position_values, direction) in enumerate(rotors)
            for position in (np.array(position_values),)
            for blade_index in range(n_blades)
        ),
        viscosity=nu,
        density=rho,
        sigma_factor=2.5,
    )

    # =========================================================================
    # 5. Field Samplers
    # =========================================================================

    # Plane at z = -1.2 (4 rotor diameters downstream)
    # Covers 50% more area than the quad footprint: [0.76 x 0.76] m
    output_dir = "solution/quadcopter"
    sampler_z15 = SurfaceSampler(
        point=[0.0, 0.0, -1.2],
        normal=[0.0, 0.0, 1.0],
        bounds=[-0.4, 0.4, -0.4, 0.4],
        spacing=0.0075,
        file_name="sampled_zplane",
        output_dir=f"{output_dir}/samples",
    )
    samplers = [sampler_z15]

    # =========================================================================
    # 6. VPM Solver Configuration
    # =========================================================================

    # Spatial bounds for particle removal (rotors at ±0.25, wake goes -Z)
    wake_bounds = [-1.5, 1.5, -1.5, 1.5, -3.0, 1.0]

    config = VPMSetup(
        time_step_size=dt,
        vlm=vlm_setup,
        backup_frequency=6,
        logging_frequency=6,
        timing_frequency=40,
        stretching=StretchingConfig.disabled(),  # Required for hover stability
        viscous=ViscousConfig.cs(),  # Core-spreading diffusion
        velocity=VelocityConfig.treecode(
            theta=0.35,
            sort_particle_targets=True,
            traversal_block_dim=128,
        ),
        turbulence=TurbulenceConfig.dns(),
        particles_kernel="WINCKELMANS",
        background_velocity=background_velocity,
        backup_file_name="quadcopter",
        backup_directory=output_dir,
        stabilization=StabilizationConfig(
            remove_particles_by_bounds=wake_bounds,
        ),
        samplers=samplers,
        processing_unit=args.processing_unit,
    )

    vpm = Solver(setup=config)

    # =========================================================================
    # 6. Simulation Loop
    # =========================================================================
    for step in range(1, n_steps + 1):
        vpm.update_state()


if __name__ == "__main__":
    main()
