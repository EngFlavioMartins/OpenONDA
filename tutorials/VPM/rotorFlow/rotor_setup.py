#!/usr/bin/env python3
"""
Rotor VLM-VPM Setup Runner
===========================
Parametric runner for the rotorFlow tutorial.  Builds a three-bladed rotor with
design-optimal twist (Betz optimum, TSR = 7) and a flat-plate VLM, then runs
the VPM wake simulation.  Post-processing is handled by allplot.sh.

Usage::

    python rotor_setup.py --num-steps 3500

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import sys
import argparse
from pathlib import Path

import numpy as np

# =========================================================
# Ensure assets/ is importable when run from the tutorial root
# =========================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "assets"))

from source.solvers.VPM import Solver, SolverConfig, StabilizationConfig
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    TurbulenceConfig,
    StretchingConfig,
    ViscousConfig
)
from source.solvers.VPM.boundary_elements.vlm import VLMSolver
from source.solvers.VPM.boundary_elements.vlm.coupling.kinematics import ManeuverVLM
from source.solvers.VPM.utils.field_samplers import SurfaceSampler
from generate_openvsp_blade import RotorBladeDesign, generate_rotorflow_openvsp_blade


def main():
    parser = argparse.ArgumentParser(description="RotorFlow VLM-VPM simulation")

    parser.add_argument("--num-steps", type=int, default=3500, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=0.005, help="Time-step size [s].")
    parser.add_argument(
        "--ramp-rotations",
        type=float,
        default=1.0,
        help="Smooth sin-squared spin-up duration [rotor rotations].",
    )
    parser.add_argument("--solution-dir", default="solution/rotor", help="Output directory.")
    args = parser.parse_args()

    # ================================================
    # 1. Physical Parameters
    # ================================================
    freestream_velocity = 7.0  # Inflow velocity [m/s]
    tip_speed_ratio = 7.0  # TSR = ωR/U
    rotor_radius = 6.0  # Blade tip radius [m]
    hub_radius = 1.0  # Hub radius [m]
    kinematic_viscosity = 1.5e-5  # [m²/s]
    rho = 1.225  # [kg/m³] (air at ~20°C)
    angular_velocity = tip_speed_ratio * freestream_velocity / rotor_radius

    # ================================================
    # 2. Numerical Parameters
    # ================================================
    time_step = args.dt
    num_steps = args.num_steps

    # ================================================
    # 3. Create VLM Blade Geometry From OpenVSP
    # ================================================
    blade_file = "./assets/blade.json"

    blade_design = RotorBladeDesign(
        radius=rotor_radius,
        hub_radius=hub_radius,
        root_chord=0.6,
        tip_chord=0.35,
        freestream_velocity=freestream_velocity,
        tip_speed_ratio=tip_speed_ratio,
        axial_induction_design=1.0 / 3.0,
        alpha_design_deg=5.0,
        n_stations=23,
        chord_stations=7,
    )
    generate_rotorflow_openvsp_blade(
        output_dir="./assets/openvsp",
        json_path=blade_file,
        design=blade_design,
    )

    # ================================================
    # 4. Configure VLM Solver
    # ================================================
    vlm = VLMSolver(
        max_panels=512,
        viscosity=kinematic_viscosity,
        density=rho,
        linear_solver="SCIPY",
        sample_surface_forces=True,
        logging_frequency=10,
    )

    rotation_period = 2.0 * np.pi / angular_velocity
    ramp_time = max(0.0, args.ramp_rotations * rotation_period)

    def rotor_angular_velocity(t: float) -> np.ndarray:
        if ramp_time > 0.0 and t < ramp_time:
            factor = np.sin(0.5 * np.pi * max(t, 0.0) / ramp_time) ** 2
        else:
            factor = 1.0
        return np.array([-angular_velocity * factor, 0.0, 0.0])

    rotation_kinematics = ManeuverVLM(
        angular_velocity_fn=rotor_angular_velocity,
        rotation_center=np.zeros(3),
    )

    # Add three blades at 120° azimuthal spacing
    blade_azimuths = [0, 120, 240]
    for blade_index, azimuth in enumerate(blade_azimuths):
        vlm.add_surface(
            blade_file,
            surface_name=f"blade_{blade_index}",
            kinematics=rotation_kinematics,
            rotation_deg=[azimuth, 0, 0],
            mesh_refinement_type="geometric",
            mesh_refinement_ratio=3.0,
            mesh_refinement_region="both",
        )

    # ================================================
    # 5. Configure VPM Solver
    # ================================================
    backup_dir = args.solution_dir

    # Downstream YZ cross-plane samplers for wake / induction validation.
    off_wake = rotor_radius * 1.2
    wake_spacing = rotor_radius / 36  
    plane_samplers = [
        SurfaceSampler(
            point=[x_loc, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-off_wake, off_wake, -off_wake, off_wake],
            spacing=wake_spacing,
            file_name=f"slice_x{int(round(x_loc))}m",
            output_dir=backup_dir + "/samples",
        )
        for x_loc in [1.5 * rotor_radius, 3.0 * rotor_radius, 4.5 * rotor_radius]
    ]

    advection=AdvectionConfig(scheme="RK3")

    turbulence=TurbulenceConfig.les_smagorinsky()

    stretching=StretchingConfig.transposed()

    stabilization=StabilizationConfig(
            parallel_strain_enabled=True,
            parallel_strain_f=0.0,
            parallel_strain_g=1.0 / 3.0,
            # ISR blend relaxation is the second net — it drains runaway |Γ|.
            relaxation_enabled=True,
            relaxation_mode='blend', # try also 'pedrizzetti'
            remove_particles_by_bounds=[
                -2.0 * rotor_radius,
                20.0 * rotor_radius,
                -2.0 * rotor_radius,
                 2.0 * rotor_radius,
                -2.0 * rotor_radius,
                 2.0 * rotor_radius,
            ],
        )

    viscous=ViscousConfig(scheme='CS')

    solver_config = SolverConfig(
        time_step_size=time_step,
        advection=advection,
        vlm_solver=vlm,
        background_velocity=[freestream_velocity, 0, 0],
        turbulence=turbulence,
        stretching=stretching,
        viscous=viscous,
        samplers=plane_samplers,
        backup_file_name="rotor",
        backup_directory=backup_dir,
        solution_name=backup_dir,
        backup_frequency=10,
        logging_frequency=10,
        timing_frequency=10,
        # Platform-best GPU (CUDA on NVIDIA, Vulkan otherwise).  Forcing Vulkan
        # risks the Taichi 1.7.x per-shape staging-buffer leak on long runs.
        processing_unit="GPU",
    )

    vpm = Solver(config=solver_config)
    vpm.info()

    # ================================================
    # 6. Run Simulation
    # ================================================
    for _ in range(num_steps):
        vpm.update_state()

if __name__ == "__main__":
    main()
