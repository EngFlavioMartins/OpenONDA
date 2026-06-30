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
import sys

# OpenONDA imports
from source.solvers.VPM.core.solver import Solver
from source.solvers.VPM.config.types import (
    SolverConfig,
    StabilizationConfig,
    StretchingConfig,
    ViscousConfig,
    TurbulenceConfig,
)
from source.solvers.VPM.boundary_elements.vlm.solver.vlm_solver import VLMSolver
from source.solvers.VPM.boundary_elements.vlm.coupling.kinematics import RotatingVLM
from source.solvers.VPM.utils.field_samplers import SurfaceSampler

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "assets"))

# Local helper for blade generation
from generate_blade import create_rotor_blade, save_blade


def main():
    parser = argparse.ArgumentParser(description="QuadCopter VPM tutorial.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override total step count (default: 8 revolutions).",
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

    deg_per_step = 5.0
    dt = (deg_per_step * np.pi / 180.0) / omega
    steps_per_rev = int(360.0 / deg_per_step)
    n_revolutions = 8
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

    vlm = VLMSolver(
        viscosity=nu,
        density=rho,
        max_panels=10000,
        linear_solver="SCIPY",
        sigma_factor=2.5,  # Overlap ratio for stable rotor wakes
    )

    # Rotor layout — alternating CW / CCW for torque balance
    arm_length = 0.16  # [m] hub centre distance from vehicle CG
    rotors = [
        ("rotor_0", [arm_length, arm_length, 0.0], 1.0),  # CCW
        ("rotor_1", [-arm_length, arm_length, 0.0], -1.0),  # CW
        ("rotor_2", [-arm_length, -arm_length, 0.0], 1.0),  # CCW
        ("rotor_3", [arm_length, -arm_length, 0.0], -1.0),  # CW
    ]

    for i, (name, pos, direction) in enumerate(rotors):
        pos = np.array(pos)

        # Use the correct blade geometry for rotation direction
        blade_file = blade_ccw_json if direction > 0 else blade_cw_json

        # Add multiple blades symmetrically around the hub center
        for b in range(n_blades):
            blade_name = f"{name}_blade_{b}"
            offset_deg = 360.0 / n_blades * b

            # Create a separate kinematics instance for each blade to track its state
            kinematics = RotatingVLM(
                omega=omega * direction,
                axis=[0.0, 0.0, 1.0],
                center=pos,
            )

            vlm.add_surface(
                blade_file,
                surface_name=blade_name,
                kinematics=kinematics,
                translation=pos,
                rotation_deg=[0.0, 0.0, offset_deg],
                rotation_center=[0.0, 0.0, 0.0],
                group_id=i + 1,
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
        spacing=0.005,
        file_name="sampled_zplane",
        output_dir=f"{output_dir}/samples",
    )
    samplers = [sampler_z15]

    # =========================================================================
    # 6. VPM Solver Configuration
    # =========================================================================

    # Spatial bounds for particle removal (rotors at ±0.25, wake goes -Z)
    wake_bounds = [-1.5, 1.5, -1.5, 1.5, -3.0, 1.0]

    config = SolverConfig(
        time_step_size=dt,
        vlm_solver=vlm,
        backup_frequency=3,  # Save every 6 steps (90 deg) to see rotation
        logging_frequency=3,
        timing_frequency=40,
        stretching=StretchingConfig.disabled(),  # Required for hover stability
        viscous=ViscousConfig.cs(),  # Core-spreading diffusion
        turbulence=TurbulenceConfig.dns(),
        particles_kernel="WINCKELMANS",
        background_velocity=background_velocity,
        backup_file_name="quadcopter",
        backup_directory=output_dir,
        solution_name=output_dir,
        stabilization=StabilizationConfig(
            remove_particles_by_bounds=wake_bounds,
        ),
        samplers=samplers,
    )

    vpm = Solver(config=config)

    # =========================================================================
    # 6. Simulation Loop
    # =========================================================================
    for step in range(1, n_steps + 1):
        vpm.update_state()


if __name__ == "__main__":
    main()
