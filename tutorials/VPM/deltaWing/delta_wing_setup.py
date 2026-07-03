#!/usr/bin/env python3
"""
Two delta wings crossing each other's wake — VLM-VPM tutorial.
==============================================================
Two identical delta wings are held at fixed x-stations in a uniform −x
free-stream (no body translation) and forced in PLUNGE + PITCH.  The two wings
plunge OUT OF PHASE (π apart): the downstream wing repeatedly rises through the
wake shed by the upstream wing, so the case shows the aerodynamic effect of
crossing up and down through another wing's wake.

Geometry / kinematics
---------------------
* Free-stream  U∞ = 5 m/s in −x  (background velocity; wings do not translate).
* Wings yawed 180° so the leading edge faces the −x wind (lift stays +z).
* Upstream ("front") wing at x = +separation; downstream ("rear") wing at x = 0,
  so the front wake convects in −x onto the rear wing.
* Plunge h(t)=A(1−cos(ωt)); pitch holds the mean AoA against the plunge-induced
  inflow angle.  Rear wing uses a π phase shift.
* Run covers ≥10 plunge periods.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from source.solvers.VPM import Solver, SolverConfig, StabilizationConfig
from source.solvers.VPM.config.types import TurbulenceConfig, VelocityConfig
from source.solvers.VPM.boundary_elements.vlm import VLMSolver
from source.solvers.VPM.boundary_elements.vlm.coupling.kinematics import ManeuverVLM
from source.solvers.VPM.utils.field_samplers import SurfaceSampler

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "assets"))
from generate_surface import create_delta_wing, save_surface  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Two-wing wake-crossing delta-wing tutorial.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2200,  # ~8.8 plunge periods at f=1 Hz, dt=0.004 s
        help="Number of time steps (default: 2200 ≈ 8.8 plunge cycles).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.004,
        help="Time-step size [s].",
    )
    parser.add_argument(
        "--processing-unit",
        default="CUDA",
        choices=["CPU", "GPU", "GPU_VULKAN", "VULKAN", "CUDA", "GPU_METAL", "METAL"],
        help="Compute backend. Default CUDA keeps the tutorial on the tested NVIDIA path.",
    )
    args = parser.parse_args()

    # ================================================
    # 1. Physical parameters
    # ================================================
    freestream_velocity = 5.0  # [m/s]  (applied as a −x background flow)
    kinematic_viscosity = 1.0e-3  # [m²/s]
    root_chord = 0.5  # [m]
    tip_chord = 0.1  # [m]
    half_span = 0.5  # [m]   (full span = 1.0 m)
    angle_of_attack = 15.0  # [deg]  mean AoA (geometry)
    rho = 1.225  # [kg/m³]

    # Two-wing layout: 5 half-spans apart in x; front upstream (+x), rear at 0.
    separation = 5.0 * half_span  # = 2.5 m

    # ================================================
    # 2. Numerical parameters
    # ================================================
    time_step = args.dt  # [s]
    num_steps = args.num_steps

    # Motion parameters
    heave_amplitude = 0.2  # [m]
    heave_frequency = 1.0  # [Hz]
    pivot_x = root_chord / 3.0  # pitch pivot at 1/3 chord (in body x)
    omega = 2.0 * np.pi * heave_frequency
    A = heave_amplitude

    # ================================================
    # 3. VLM surface geometry (shared by both wings)
    # ================================================
    surface_file = str(_SCRIPT_DIR / "delta_wing_surface.json")
    surface = create_delta_wing(
        root_chord=root_chord,
        tip_chord=tip_chord,
        half_span=half_span,
        alpha=angle_of_attack,
        n_chord=8,
        n_span=18,
    )
    save_surface(surface, surface_file)

    # ================================================
    # 4. VLM solver with two phase-shifted wings
    # ================================================
    vlm = VLMSolver(
        viscosity=kinematic_viscosity,
        density=rho,
        sample_surface_forces=True,  # per-wing force history → samples/vlm_forces.csv
    )

    def make_heave(phase):
        # h(t) = A(1 − cos(ωt+φ)) → vz = A ω sin(ωt+φ)
        def vfn(t):
            return np.array([0.0, 0.0, A * omega * np.sin(omega * t + phase)])
        return vfn

    def make_pitch(phase):
        # Hold mean AoA against the plunge-induced inflow angle.
        def wfn(t):
            vz = A * omega * np.sin(omega * t + phase)
            dvz = A * omega * omega * np.cos(omega * t + phase)
            u = vz / freestream_velocity
            dtheta = (dvz / freestream_velocity) / (1.0 + u * u)
            return np.array([0.0, -1.0, 0.0]) * dtheta
        return wfn

    wings = [
        ("front_wing", separation, 0.0),       # upstream, phase 0
        ("rear_wing", 0.0, np.pi),             # downstream, phase π (out of phase)
    ]
    for name, x0, phase in wings:
        kin = ManeuverVLM(
            velocity_fn=make_heave(phase),
            angular_velocity_fn=make_pitch(phase),
            rotation_center=[x0 + pivot_x, 0.0, 0.0],
        )
        vlm.add_surface(
            surface_file,
            surface_name=name,
            kinematics=kin,
            translation=np.array([x0, 0.0, 0.0]),
            rotation_deg=np.array([0.0, 0.0, 180.0]),  # face the −x free-stream
            rotation_center=np.array([x0 + pivot_x, 0.0, 0.0]),
            mesh_refinement_type="geometric",
            mesh_refinement_ratio=3.0,
            mesh_refinement_region="end",
        )

    # ================================================
    # 5. Wake crossflow samplers: 1, 5, 10 half-spans downstream of rear wing.
    #    Free-stream is −x, so "downstream" of the rear wing (x=0) is −x.
    # ================================================
    backup_dir = "solution/delta_wing"
    sampler_planes = []
    for n_sp in (1, 5, 10):
        x_loc = -n_sp * half_span
        sampler_planes.append(
            SurfaceSampler(
                point=[x_loc, 0.0, 0.0],
                normal=[1, 0, 0],
                bounds=[-1.5, 1.5, -1.0, 1.0],  # y, z
                spacing=0.04,
                file_name=f"wake_{n_sp}span",
                output_dir=backup_dir + "/samples",
            )
        )

    # ================================================
    # 6. VPM solver
    # ================================================
    solver_config = SolverConfig(
        time_step_size=time_step,
        turbulence=TurbulenceConfig.les_smagorinsky(cs=0.3),
        vlm_solver=vlm,
        velocity=VelocityConfig.treecode(
            theta=0.35,
            sort_particle_targets=True,
            traversal_block_dim=128,
        ),
        background_velocity=[-freestream_velocity, 0, 0],
        backup_file_name="wing",
        backup_directory=backup_dir,
        backup_frequency=20,
        logging_frequency=20,
        timing_frequency=40,
        processing_unit=args.processing_unit,
        # Keep the wake bounded (≥10 cycles would otherwise grow without limit).
        stabilization=StabilizationConfig(
            remove_particles_by_bounds=[-8.0, separation + 1.0, -2.0, 2.0, -1.5, 1.5],
        ),
        samplers=sampler_planes,
    )

    vpm = Solver(config=solver_config)

    # Save motion parameters so the post-processing can reconstruct the wing
    # plunge trajectories z(t) without re-deriving them.
    meta = {
        "A": A, "omega": omega, "dt": time_step, "num_steps": num_steps,
        "separation": separation, "half_span": half_span,
        "wings": {"front_wing": 0.0, "rear_wing": np.pi},
    }
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    (Path(backup_dir) / "motion_params.json").write_text(json.dumps(meta, indent=2))

    # ================================================
    # 7. Run
    # ================================================
    for _step in range(num_steps):
        vpm.update_state()


if __name__ == "__main__":
    main()
