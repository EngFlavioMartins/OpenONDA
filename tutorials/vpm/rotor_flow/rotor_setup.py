#!/usr/bin/env python3
"""Rotor in forward flight with a fully-coupled VLM--VPM wake (LES).

A three-bladed rotor flies at a tip-speed ratio of 7.0. The wake is resolved
with vortex particles convected by a coupled (implicit) advection integrator;
the blade loading and the downstream wake planes are sampled for the
``plot_all.sh`` figures.

Usage:
    python rotor_setup.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import openonda.vpm as vpm

TUTORIAL_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"
CASE_NAME = "rotor"

FREESTREAM_SPEED = 7.0
TIP_SPEED_RATIO = 7.0
ROTOR_RADIUS = 6.0
HUB_RADIUS = 1.0
KINEMATIC_VISCOSITY = 1.5e-5
AIR_DENSITY = 1.225
N_RADIAL_STATIONS = 23
ANGULAR_VELOCITY = TIP_SPEED_RATIO * FREESTREAM_SPEED / ROTOR_RADIUS

COUPLED_MAX_STRAIN_INCREMENT = 0.08
COUPLED_MAX_ADVECTION_FRACTION = 0.25
COUPLED_MAX_SUBSTEPS = 128
TREECODE_THETA = 0.20
TIME_STEP_SIZE = 0.006
N_STEPS = 2400
RAMP_ROTATIONS = 1.0
GUARD_INTERVAL_STEPS = 20
MAX_PARTICLE_STRENGTH = 10.0
SAMPLE_INTERVAL_TIME = 0.12  # write a snapshot every this many seconds
CHECKPOINT_INTERVAL_TIME = 0.03  # about 26 animation frames per rotor revolution


def nominal_wake_spacing(time_step_size: float) -> float:
    """Return the resolved wake length used by displacement subcycling.

    A VLM step creates one streamwise row of particles.  The limiting nominal
    spacing is therefore the smaller of the radial panel spacing and the
    fully-spun-up tip travel per macro step.
    """
    radial_spacing = (ROTOR_RADIUS - HUB_RADIUS) / (N_RADIAL_STATIONS - 1)
    tip_streamwise_spacing = ANGULAR_VELOCITY * ROTOR_RADIUS * time_step_size
    return min(radial_spacing, tip_streamwise_spacing)


def cadence_steps(period: float, time_step_size: float) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / time_step_size))


def build_solver_config(
    sample_interval_time: float,
    checkpoint_interval_time: float,
    *,
    vlm_setup: vpm.VLMSetup | None = None,
    samplers: tuple[vpm.SurfaceSampler, ...] | list[vpm.SurfaceSampler] = (),
) -> vpm.VPMSetup:
    """Build the rotor VPM configuration."""
    wake_spacing = nominal_wake_spacing(TIME_STEP_SIZE)
    return vpm.VPMSetup(
        time_step_size=TIME_STEP_SIZE,
        compute_device="AUTO",
        time_integration="COUPLED",
        coupled_max_strain_increment=COUPLED_MAX_STRAIN_INCREMENT,
        coupled_max_advection_fraction=COUPLED_MAX_ADVECTION_FRACTION,
        coupled_max_substeps=COUPLED_MAX_SUBSTEPS,
        advection=vpm.AdvectionConfig(scheme="RK2"),
        vlm=vlm_setup,
        freestream_velocity=[FREESTREAM_SPEED, 0.0, 0.0],
        turbulence=vpm.TurbulenceConfig.les_smagorinsky(),
        stretching=vpm.StretchingConfig.transposed(
            scheme="RK2",
            use_treecode=True,
            treecode_theta=TREECODE_THETA,
        ),
        stabilization=vpm.StabilizationConfig.bounded_domain(
            bounds=[
                -2.0 * ROTOR_RADIUS,
                20.0 * ROTOR_RADIUS,
                -2.0 * ROTOR_RADIUS,
                2.0 * ROTOR_RADIUS,
                -2.0 * ROTOR_RADIUS,
                2.0 * ROTOR_RADIUS,
            ]
        ),
        viscous=vpm.ViscousConfig.cs(
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            particle_spacing=wake_spacing,
        ),
        velocity=vpm.VelocityConfig.treecode(
            theta=TREECODE_THETA,
            sort_particle_targets=True,
            traversal_block_dim=128,
        ),
        particle_kernel="WINCKELMANS",
        samplers=list(samplers),
        checkpoint_name=CASE_NAME,
        checkpoint_directory=str(SOLUTION_DIR),
        sample_subdirectory=CASE_NAME,
        checkpoint_interval_steps=cadence_steps(checkpoint_interval_time, TIME_STEP_SIZE),
        logging_interval_steps=cadence_steps(sample_interval_time, TIME_STEP_SIZE),
        export_flow_integrals=True,
    )


def enforce_wake_admissibility(solver: vpm.VPMSolver, max_particle_strength: float) -> None:
    """Stop a divergent wake without altering vortex strength or core radius."""
    fields = {
        "position": solver.particle_position,
        "vortex_strength": solver.particle_vortex_strength,
        "core_radius": solver.particle_core_radius,
        "particle_volume": solver.particle_volume,
    }
    if not len(fields["core_radius"]):
        return

    failures = [name for name, values in fields.items() if not np.isfinite(values).all()]
    if np.any(fields["core_radius"] <= 0.0):
        failures.append("non-positive core radius")
    if np.any(fields["particle_volume"] <= 0.0):
        failures.append("non-positive particle_volume")

    vortex_strength_magnitude = np.linalg.norm(fields["vortex_strength"], axis=1)
    maximum = float(vortex_strength_magnitude.max())
    if not np.isfinite(maximum) or maximum > max_particle_strength:
        failures.append(f"max|vortex_strength|={maximum:.4g} > {max_particle_strength:.4g}")
    if failures:
        raise RuntimeError(
            "Rotor wake admissibility failed at "
            f"step={solver.step}, t={solver.time:.6e}: "
            + "; ".join(failures)
            + ". The run was stopped without modifying the particle field."
        )


def write_manifest(solver: vpm.VPMSolver) -> None:
    """Store the numerical settings beside the sampled results."""
    cfg = solver.setup
    output_dir = TUTORIAL_DIR / "samples" / CASE_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "case": "rotor_flow",
        "time_step_size": TIME_STEP_SIZE,
        "n_steps": N_STEPS,
        "sampling_interval_time": cfg.logging_interval_steps * TIME_STEP_SIZE,
        "checkpoint_interval_time": cfg.checkpoint_interval_steps * TIME_STEP_SIZE,
        "treecode_theta": cfg.velocity.theta,
        "kernel": cfg.particle_kernel,
        "kinematic_viscosity": cfg.viscous.kinematic_viscosity,
        "wake_particle_spacing": cfg.viscous.particle_spacing,
        "coupled_max_strain_increment": cfg.coupled_max_strain_increment,
        "coupled_max_advection_fraction": cfg.coupled_max_advection_fraction,
        "coupled_max_substeps": cfg.coupled_max_substeps,
        "retention_bounds": cfg.stabilization.remove_particles_by_bounds,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    from assets.generate_openvsp_blade import RotorBladeDesign, generate_rotorflow_openvsp_blade

    sample_interval_time = SAMPLE_INTERVAL_TIME
    checkpoint_interval_time = CHECKPOINT_INTERVAL_TIME

    blade_file = TUTORIAL_DIR / "assets/blade.json"

    blade_design = RotorBladeDesign(
        rotor_radius=ROTOR_RADIUS,
        hub_radius=HUB_RADIUS,
        root_chord=0.6,
        tip_chord=0.35,
        freestream_speed=FREESTREAM_SPEED,
        tip_speed_ratio=TIP_SPEED_RATIO,
        design_axial_induction_factor=1.0 / 3.0,
        design_angle_of_attack_degrees=5.0,
        n_radial_stations=N_RADIAL_STATIONS,
        n_chordwise_stations=7,
    )

    if Path(blade_file).exists():
        print(f"Using cached VLM blade surface: {blade_file} (skipping OpenVSP regeneration)")
    else:
        generate_rotorflow_openvsp_blade(
            output_dir=str(TUTORIAL_DIR / "assets/openvsp"),
            json_path=str(blade_file),
            design=blade_design,
        )

    rotation_period = 2.0 * np.pi / ANGULAR_VELOCITY
    ramp_time = RAMP_ROTATIONS * rotation_period

    def rotor_angular_velocity(t: float) -> np.ndarray:
        if ramp_time > 0.0 and t < ramp_time:
            factor = np.sin(0.5 * np.pi * max(t, 0.0) / ramp_time) ** 2
        else:
            factor = 1.0
        return np.array([-ANGULAR_VELOCITY * factor, 0.0, 0.0])

    rotation_kinematics = vpm.ManeuverVLM(
        angular_velocity_function=rotor_angular_velocity,
        rotation_centre=np.zeros(3),
    )

    vlm_setup = vpm.VLMSetup(
        surfaces=tuple(
            vpm.VLMSurfaceSetup(
                str(blade_file),
                name=f"blade_{blade_index}",
                kinematics=rotation_kinematics,
                rotation_degrees=(azimuth, 0.0, 0.0),
            )
            for blade_index, azimuth in enumerate((0.0, 120.0, 240.0))
        ),
        mesh=vpm.VLMMeshSetup.geometric(ratio=3.0),
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        sample_surface_forces=True,
        logging_interval_steps=cadence_steps(sample_interval_time, TIME_STEP_SIZE),
    )

    # Downstream planes at 1.5R, 3R, and 4.5R.
    off_wake = ROTOR_RADIUS * 1.2
    sample_spacing = ROTOR_RADIUS / 36
    plane_samplers = [
        vpm.SurfaceSampler(
            point=[x_loc, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-off_wake, off_wake, -off_wake, off_wake],
            spacing=sample_spacing,
            file_name=f"slice_x{int(round(x_loc))}m",
        )
        for x_loc in [1.5 * ROTOR_RADIUS, 3.0 * ROTOR_RADIUS, 4.5 * ROTOR_RADIUS]
    ]

    solver_config = build_solver_config(
        sample_interval_time,
        checkpoint_interval_time,
        vlm_setup=vlm_setup,
        samplers=plane_samplers,
    )
    solver = vpm.VPMSolver(setup=solver_config, case_dir=TUTORIAL_DIR)
    write_manifest(solver)
    solver.info()

    print("\n===== SIMULATION =====")
    try:
        for step in range(N_STEPS):
            solver.advance()
            if (step + 1) % GUARD_INTERVAL_STEPS == 0:
                enforce_wake_admissibility(solver, MAX_PARTICLE_STRENGTH)
    except RuntimeError:
        solver.save_state(str(SOLUTION_DIR / "rejected_state"))
        raise
    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./plot_all.sh to make the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
