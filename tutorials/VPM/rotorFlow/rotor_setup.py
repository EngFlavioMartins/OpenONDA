#!/usr/bin/env python3
"""
Rotor VLM-VPM Setup Runner
===========================
Parametric runner for the rotorFlow tutorial.  Builds a three-bladed rotor with
design-optimal twist (Betz optimum, TSR = 7) and a flat-plate VLM, then runs
the VPM wake simulation.  Post-processing is handled by allplot.sh.

Usage::

    python rotor_setup.py --num-steps 1232

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import sys
import argparse
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Ensure assets/ is importable when run from the tutorial root
# ------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "assets"))

from source.solvers.VPM import Solver, SolverConfig
from source.solvers.VPM.config.types import AdaptationConfig, TurbulenceConfig, StretchingConfig
from source.solvers.VPM.boundary_elements.vlm import VLMSolver, RotatingVLM
from source.solvers.VPM.utils.field_samplers import SurfaceSampler
from generate_blade import create_blade, save_surface


def main():
    parser = argparse.ArgumentParser(description="RotorFlow VLM-VPM simulation")
    parser.add_argument("--num-steps", type=int, default=1232, help="Number of time steps")
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
    time_step = 0.005
    num_steps = args.num_steps

    # Wake cutoff: remove particles beyond this x-coordinate:
    wake_cutoff_distance = 20 * rotor_radius

    # ================================================
    # 3. Create VLM Blade Geometry
    # ================================================
    blade_file = "./assets/blade.json"
    # Compute design-optimal twist: theta(r) = phi(r) - alpha_des,
    # where phi = atan(U*(1-a_Betz) / (omega*r)) and beta = 90 - theta.
    # Using a multi-segment blade so the nonlinear profile is captured exactly.
    _a_betz = 1.0 / 3.0
    _alpha_des = np.radians(5.0)
    _n_sched = 11  # 11 breakpoints → 10 segments, 2 span panels each
    _r_sched = np.linspace(hub_radius, rotor_radius, _n_sched)
    _phi_sched = np.arctan2(freestream_velocity * (1.0 - _a_betz), angular_velocity * _r_sched)
    _beta_sched = 90.0 - np.degrees(_phi_sched - _alpha_des)
    blade_surface = create_blade(
        radius=rotor_radius,
        hub_radius=hub_radius,
        root_chord=0.6,
        tip_chord=0.35,
        n_span=20,
        n_chord=6,
        r_schedule=_r_sched,
        beta_schedule=_beta_sched,
        pitch_schedule=np.zeros(_n_sched),
    )
    save_surface(blade_surface, blade_file)

    # ================================================
    # 4. Configure VLM Solver
    # ================================================
    vlm = VLMSolver(
        viscosity=kinematic_viscosity,
        density=rho,
        linear_solver="SCIPY",
        sample_surface_forces=True,
    )
    rotation_kinematics = RotatingVLM(omega=angular_velocity, axis=[1, 0, 0])

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
    backup_dir = "solution"

    # Downstream YZ cross-plane samplers for wake / induction validation
    # Planes at 1R, 2R, 4R downstream (x-axis is axial / freestream direction)
    wake_half = rotor_radius * 1.4  # bounds: ±1.4R in y and z
    wake_spacing = rotor_radius / 12  # ~0.5 m for R=6
    plane_samplers = [
        SurfaceSampler(
            point=[x_loc, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-wake_half, wake_half, -wake_half, wake_half],
            spacing=wake_spacing,
            file_name=f"slice_x{int(round(x_loc))}m",
            output_dir=backup_dir + "/samples",
        )
        for x_loc in [rotor_radius, 2 * rotor_radius, 4 * rotor_radius]
    ]

    solver_config = SolverConfig(
        time_step_size=time_step,
        vlm_solver=vlm,
        background_velocity=[freestream_velocity, 0, 0],
        turbulence=TurbulenceConfig.les_smagorinsky(cs=0.3),
        stretching=StretchingConfig.transposed(),
        backup_file_name="rotor",
        backup_directory=backup_dir,
        backup_frequency=3,
        logging_frequency=3,
        adaptation=AdaptationConfig(
            max_core_radius=2.0,
        ),
        processing_unit="GPU_VULKAN",
        samplers=plane_samplers,
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
