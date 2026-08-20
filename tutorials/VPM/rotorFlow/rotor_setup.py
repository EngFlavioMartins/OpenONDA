#!/usr/bin/env python3
"""Rotor in forward flight with a fully-coupled VLM--VPM wake (LES).

A three-bladed rotor flies at a tip-speed ratio of 7.0. The wake is resolved
with vortex particles convected by a coupled (implicit) advection integrator;
the blade loading and the downstream wake planes are sampled for the
``allplot.sh`` figures.

Usage:
    python rotor_setup.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from openonda.vpm import (
    AdvectionConfig,
    ManeuverVLM,
    VPMSolver,
    StabilizationConfig,
    StretchingConfig,
    SurfaceSampler,
    TurbulenceConfig,
    VelocityConfig,
    ViscousConfig,
    VLMMeshSetup,
    VLMSurfaceSetup,
    VLMSetup,
    VPMSetup,
)
from source.solvers.VPM.io.sampling import resolve_samples_dir

TUTORIAL_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = TUTORIAL_DIR / "solution"
CASE_NAME = "rotor"

FREESTREAM_VELOCITY = 7.0
TIP_SPEED_RATIO = 7.0
ROTOR_RADIUS = 6.0
HUB_RADIUS = 1.0
KINEMATIC_VISCOSITY = 1.5e-5
AIR_DENSITY = 1.225
NUM_RADIAL_STATIONS = 23
ANGULAR_VELOCITY = TIP_SPEED_RATIO * FREESTREAM_VELOCITY / ROTOR_RADIUS

COUPLED_MAX_STRAIN_INCREMENT = 0.08
COUPLED_MAX_ADVECTION_FRACTION = 0.25
COUPLED_MAX_SUBSTEPS = 128
TREECODE_THETA = 0.20
TIME_STEP = 0.006
NUMBER_OF_STEPS = 2400
RAMP_ROTATIONS = 1.0
GUARD_FREQUENCY = 20
MAX_PARTICLE_STRENGTH = 10.0
SAMPLE_PERIOD = 0.12  # write a snapshot every this many seconds
BACKUP_PERIOD = 0.03  # about 26 animation frames per rotor revolution


def nominal_wake_spacing(time_step: float) -> float:
    """Return the resolved wake length used by displacement subcycling.

    A VLM step creates one streamwise row of particles.  The limiting nominal
    spacing is therefore the smaller of the radial panel spacing and the
    fully-spun-up tip travel per macro step.
    """
    radial_spacing = (ROTOR_RADIUS - HUB_RADIUS) / (NUM_RADIAL_STATIONS - 1)
    tip_streamwise_spacing = ANGULAR_VELOCITY * ROTOR_RADIUS * time_step
    return min(radial_spacing, tip_streamwise_spacing)


def cadence_steps(period: float, time_step: float) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / time_step))


def build_solver_config(
    sample_period: float,
    backup_period: float,
    *,
    vlm_setup: VLMSetup | None = None,
    samplers: tuple[SurfaceSampler, ...] | list[SurfaceSampler] = (),
) -> VPMSetup:
    """Build the rotor VPM configuration."""
    wake_spacing = nominal_wake_spacing(TIME_STEP)
    return VPMSetup(
        time_step_size=TIME_STEP,
        compute_device="AUTO",
        time_integration="COUPLED",
        coupled_max_strain_increment=COUPLED_MAX_STRAIN_INCREMENT,
        coupled_max_advection_fraction=COUPLED_MAX_ADVECTION_FRACTION,
        coupled_max_substeps=COUPLED_MAX_SUBSTEPS,
        advection=AdvectionConfig(scheme="RK2"),
        vlm=vlm_setup,
        freestream_velocity=[FREESTREAM_VELOCITY, 0.0, 0.0],
        turbulence=TurbulenceConfig.les_smagorinsky(),
        stretching=StretchingConfig.transposed(
            scheme="RK2",
            use_treecode=True,
            treecode_theta=TREECODE_THETA,
        ),
        stabilization=StabilizationConfig.bounded_domain(
            bounds=[
                -2.0 * ROTOR_RADIUS,
                20.0 * ROTOR_RADIUS,
                -2.0 * ROTOR_RADIUS,
                2.0 * ROTOR_RADIUS,
                -2.0 * ROTOR_RADIUS,
                2.0 * ROTOR_RADIUS,
            ]
        ),
        viscous=ViscousConfig.cs(
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            particle_spacing=wake_spacing,
        ),
        velocity=VelocityConfig.treecode(
            theta=TREECODE_THETA,
            sort_particle_targets=True,
            traversal_block_dim=128,
        ),
        particle_kernel="WINCKELMANS",
        samplers=list(samplers),
        checkpoint_name=CASE_NAME,
        checkpoint_directory=str(SOLUTION_DIR),
        sample_subdirectory=CASE_NAME,
        checkpoint_interval_steps=cadence_steps(backup_period, TIME_STEP),
        logging_interval_steps=cadence_steps(sample_period, TIME_STEP),
        export_flow_integrals=True,
    )


def enforce_wake_admissibility(solver: VPMSolver, max_particle_strength: float) -> None:
    """Stop a divergent wake without altering circulation or core size."""
    fields = {
        "position": solver.particles_positions,
        "circulation": solver.particles_circulation,
        "radius": solver.particles_radii,
        "volume": solver.particles_volumes,
    }
    if not len(fields["radius"]):
        return

    failures = [name for name, values in fields.items() if not np.isfinite(values).all()]
    if np.any(fields["radius"] <= 0.0):
        failures.append("non-positive radius")
    if np.any(fields["volume"] <= 0.0):
        failures.append("non-positive volume")

    strength = np.linalg.norm(fields["circulation"], axis=1)
    maximum = float(strength.max())
    if not np.isfinite(maximum) or maximum > max_particle_strength:
        failures.append(f"max|Gamma|={maximum:.4g} > {max_particle_strength:.4g}")
    if failures:
        raise RuntimeError(
            "Rotor wake admissibility failed at "
            f"step={solver.step}, t={solver.time:.6e}: "
            + "; ".join(failures)
            + ". The run was stopped without modifying the particle field."
        )


def write_manifest(solver: VPMSolver) -> None:
    """Store the numerical settings beside the sampled results."""
    cfg = solver.setup
    output_dir = resolve_samples_dir(SOLUTION_DIR, CASE_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "case": "rotorFlow",
        "dt": TIME_STEP,
        "num_steps": NUMBER_OF_STEPS,
        "sample_interval": cfg.logging_interval_steps * TIME_STEP,
        "raw_backup_interval": cfg.checkpoint_interval_steps * TIME_STEP,
        "treecode_theta": cfg.velocity.theta,
        "kernel": cfg.particle_kernel,
        "molecular_viscosity": cfg.viscous.viscosity,
        "wake_characteristic_distance": cfg.viscous.particle_spacing,
        "coupled_max_strain_increment": cfg.coupled_max_strain_increment,
        "coupled_max_advection_fraction": cfg.coupled_max_advection_fraction,
        "coupled_max_substeps": cfg.coupled_max_substeps,
        "retention_bounds": cfg.stabilization.remove_particles_by_bounds,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    from assets.generate_openvsp_blade import RotorBladeDesign, generate_rotorflow_openvsp_blade

    sample_period = SAMPLE_PERIOD
    backup_period = BACKUP_PERIOD

    blade_file = TUTORIAL_DIR / "assets/blade.json"

    blade_design = RotorBladeDesign(
        radius=ROTOR_RADIUS,
        hub_radius=HUB_RADIUS,
        root_chord=0.6,
        tip_chord=0.35,
        freestream_velocity=FREESTREAM_VELOCITY,
        tip_speed_ratio=TIP_SPEED_RATIO,
        axial_induction_design=1.0 / 3.0,
        alpha_design_deg=5.0,
        n_stations=NUM_RADIAL_STATIONS,
        chord_stations=7,
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

    rotation_kinematics = ManeuverVLM(
        angular_velocity_fn=rotor_angular_velocity,
        rotation_center=np.zeros(3),
    )

    vlm_setup = VLMSetup(
        surfaces=tuple(
            VLMSurfaceSetup(
                str(blade_file),
                name=f"blade_{blade_index}",
                kinematics=rotation_kinematics,
                rotation_deg=(azimuth, 0.0, 0.0),
            )
            for blade_index, azimuth in enumerate((0.0, 120.0, 240.0))
        ),
        mesh=VLMMeshSetup.geometric(ratio=3.0),
        viscosity=KINEMATIC_VISCOSITY,
        density=AIR_DENSITY,
        sample_surface_forces=True,
        logging_interval_steps=cadence_steps(sample_period, TIME_STEP),
    )

    # Downstream planes at 1.5R, 3R, and 4.5R.
    off_wake = ROTOR_RADIUS * 1.2
    sample_spacing = ROTOR_RADIUS / 36
    plane_samplers = [
        SurfaceSampler(
            point=[x_loc, 0.0, 0.0],
            normal=[1, 0, 0],
            bounds=[-off_wake, off_wake, -off_wake, off_wake],
            spacing=sample_spacing,
            file_name=f"slice_x{int(round(x_loc))}m",
        )
        for x_loc in [1.5 * ROTOR_RADIUS, 3.0 * ROTOR_RADIUS, 4.5 * ROTOR_RADIUS]
    ]

    solver_config = build_solver_config(
        sample_period,
        backup_period,
        vlm_setup=vlm_setup,
        samplers=plane_samplers,
    )
    vpm = VPMSolver(setup=solver_config)
    write_manifest(vpm)
    vpm.info()

    print("\n===== SIMULATION =====")
    try:
        for step in range(NUMBER_OF_STEPS):
            vpm.advance()
            if (step + 1) % GUARD_FREQUENCY == 0:
                enforce_wake_admissibility(vpm, MAX_PARTICLE_STRENGTH)
    except RuntimeError:
        vpm.save_state(str(SOLUTION_DIR / "rejected_state"))
        raise
    print("\n===== DONE =====")
    print("Simulation completed successfully. Run ./allplot.sh to make the figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
