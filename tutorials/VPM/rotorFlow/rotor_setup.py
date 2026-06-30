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
)
from source.solvers.VPM.boundary_elements.vlm import VLMSolver
from source.solvers.VPM.boundary_elements.vlm.coupling.kinematics import ManeuverVLM
from source.solvers.VPM.utils.field_samplers import SurfaceSampler
from generate_openvsp_blade import RotorBladeDesign, generate_rotorflow_openvsp_blade


def main():
    parser = argparse.ArgumentParser(description="RotorFlow VLM-VPM simulation")

    parser.add_argument("--num-steps", type=int, default=3500, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=0.005, help="Time-step size [s].")
    # --- Stabiliser knobs (in-house LES + stretching stabiliser) -------------
    parser.add_argument(
        "--cs", type=float, default=0.16, help="Smagorinsky constant (in-house LES)."
    )
    parser.add_argument(
        "--rvpm-g",
        type=float,
        default=1.0 / 3.0,
        help="rVPM g: 0.2 (partial) … 1/3 (default; full parallel-growth suppression).",
    )
    parser.add_argument("--rvpm-f", type=float, default=0.0, help="rVPM f parameter.")
    parser.add_argument(
        "--relaxation",
        choices=["pedrizzetti", "blend"],
        default="blend",
        help="Strength stabiliser: 'pedrizzetti' (|Γ|-preserving realign) or "
        "'blend' (dissipative ADM-residual filter — can DRAIN runaway |Γ|).",
    )

    parser.add_argument(
        "--ramp-rotations",
        type=float,
        default=1.0,
        help="Smooth sin-squared spin-up duration [rotor rotations].",
    )
    parser.add_argument(
        "--max-strength",
        type=float,
        default=1.0e3,
        help="Abort cleanly if any particle strength exceeds this value.",
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
        n_stations=21,
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
        logging_frequency=25,
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
    rotor_diameter = 1.2 * rotor_radius
    wake_half = rotor_radius * 2.0
    wake_spacing = rotor_radius / 36  
    plane_samplers = [
        SurfaceSampler(
            point=[x_loc, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-wake_half, wake_half, -wake_half, wake_half],
            spacing=wake_spacing,
            file_name=f"slice_x{int(round(x_loc))}m",
            output_dir=backup_dir + "/samples",
        )
        for x_loc in [1 * rotor_diameter, 2 * rotor_diameter, 3 * rotor_diameter]
    ]

    advection=AdvectionConfig(scheme="RK2")

    turbulence=TurbulenceConfig.les_smagorinsky(cs=args.cs)

    #stretching=StretchingConfig.rvpm(f=args.rvpm_f, g=args.rvpm_g),
    stretching=StretchingConfig.transposed(scheme='Euler')

    stabilization=StabilizationConfig(
            remove_particles_by_bounds=[
                -rotor_diameter,
                20 * rotor_radius,
                -2.0 * rotor_radius,
                2.0 * rotor_radius,
                -2.0 * rotor_radius,
                2.0 * rotor_radius,
            ],
        )

    solver_config = SolverConfig(
        time_step_size=time_step,
        advection=advection,
        vlm_solver=vlm,
        background_velocity=[freestream_velocity, 0, 0],
        turbulence=turbulence,
        stretching=stretching,
        samplers=plane_samplers,
        stabilization=stabilization,
        backup_file_name="rotor",
        backup_directory=backup_dir,
        solution_name=backup_dir,
        backup_frequency=15,
        logging_frequency=15,
        timing_frequency=15,
        processing_unit="GPU_VULKAN",
    )

    vpm = Solver(config=solver_config)
    vpm.info()

    # ================================================
    # 6. Run Simulation
    # ================================================
    for step in range(num_steps):
        vpm.update_state()

if __name__ == "__main__":
    main()
