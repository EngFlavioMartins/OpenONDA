#!/usr/bin/env python3
"""Physics-preserving rotor VLM-VPM setup.

The wake is advanced with common RK stages for position and vortex strength,
physical core spreading, and automatic strain/displacement subcycling.  This
is the scalable open-wake counterpart of the exact direct-pair methodology in
``vortexInteractions/rings_setup.py``:

* transposed stretching preserves vector circulation;
* the treecode tolerance controls (rather than hides) the approximation error;
* a finite wake box is only a declared outflow/retention policy;
* runtime guards stop an inadmissible field instead of clipping its strength.

The rotor is a forced, open system, so its global impulses are not constants.
``assets/validate_results.py`` therefore checks the physically relevant
blade-force/wake-impulse budget after the run.

Usage::

    python rotor_setup.py --num-steps 2400 --dt 0.006

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import argparse
import json
from pathlib import Path

import numpy as np

from openonda.vpm import Solver, VPMSetup, StabilizationConfig
from openonda.vpm import (
    AdvectionConfig,
    TurbulenceConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)
from openonda.vpm import VLMMeshSetup, VLMSurfaceSetup, VLMSetup
from openonda.vpm import ManeuverVLM
from openonda.vpm import SurfaceSampler

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RotorFlow VLM-VPM simulation")
    parser.add_argument("--num-steps", type=int, default=2400, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=0.006, help="Time-step size [s].")
    parser.add_argument(
        "--ramp-rotations",
        type=float,
        default=1.0,
        help="Smooth sin-squared spin-up duration [rotor rotations].",
    )
    parser.add_argument("--solution-dir", default="solution/rotor", help="Output directory.")
    parser.add_argument(
        "--processing-unit",
        default="AUTO",
        choices=["AUTO", "CPU", "VULKAN", "CUDA", "METAL"],
        help="Compute backend. GPU selects Metal on macOS and CUDA/Vulkan elsewhere.",
    )
    parser.add_argument(
        "--treecode-theta",
        type=float,
        default=TREECODE_THETA,
        help="Barnes-Hut opening angle; smaller is more accurate.",
    )
    parser.add_argument(
        "--coupled-max-strain-increment",
        type=float,
        default=COUPLED_MAX_STRAIN_INCREMENT,
        help="Maximum dt_sub*||S||_2 accepted by coupled subcycling.",
    )
    parser.add_argument(
        "--coupled-max-advection-fraction",
        type=float,
        default=COUPLED_MAX_ADVECTION_FRACTION,
        help="Maximum displacement per substep as a fraction of wake spacing.",
    )
    parser.add_argument(
        "--coupled-max-substeps",
        type=int,
        default=COUPLED_MAX_SUBSTEPS,
        help="Stop instead of filtering if a macro step needs more substeps.",
    )
    parser.add_argument(
        "--guard-frequency",
        type=int,
        default=20,
        help="Check particle-field admissibility every N accepted steps.",
    )
    parser.add_argument(
        "--max-particle-strength",
        type=float,
        default=10.0,
        help="Fail-fast upper bound for a single wake-particle |Gamma| [m^3/s].",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid controls before allocating the VPM/VLM solvers."""
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive.")
    if args.num_steps < 0:
        raise ValueError("--num-steps must be non-negative.")
    if args.ramp_rotations < 0.0:
        raise ValueError("--ramp-rotations must be non-negative.")
    if not 0.0 < args.treecode_theta < 2.0:
        raise ValueError("--treecode-theta must be in (0, 2).")
    if args.coupled_max_strain_increment <= 0.0:
        raise ValueError("--coupled-max-strain-increment must be positive.")
    if args.coupled_max_advection_fraction <= 0.0:
        raise ValueError("--coupled-max-advection-fraction must be positive.")
    if args.coupled_max_substeps < 1:
        raise ValueError("--coupled-max-substeps must be at least one.")
    if args.guard_frequency < 1:
        raise ValueError("--guard-frequency must be at least one.")
    if args.max_particle_strength <= 0.0:
        raise ValueError("--max-particle-strength must be positive.")


def nominal_wake_spacing(time_step: float) -> float:
    """Return the resolved wake length used by displacement subcycling.

    A VLM step creates one streamwise row of particles.  The limiting nominal
    spacing is therefore the smaller of the radial panel spacing and the
    fully-spun-up tip travel per macro step.
    """
    radial_spacing = (ROTOR_RADIUS - HUB_RADIUS) / (NUM_RADIAL_STATIONS - 1)
    tip_streamwise_spacing = ANGULAR_VELOCITY * ROTOR_RADIUS * time_step
    return min(radial_spacing, tip_streamwise_spacing)


def build_solver_config(
    args: argparse.Namespace,
    *,
    vlm_setup: VLMSetup | None = None,
    samplers: tuple[SurfaceSampler, ...] | list[SurfaceSampler] = (),
) -> VPMSetup:
    """Build the rotor's scalable physics-preserving VPM policy."""
    validate_arguments(args)
    wake_spacing = nominal_wake_spacing(args.dt)
    return VPMSetup(
        time_step_size=args.dt,
        time_integration="COUPLED",
        coupled_max_strain_increment=args.coupled_max_strain_increment,
        coupled_max_advection_fraction=args.coupled_max_advection_fraction,
        coupled_max_substeps=args.coupled_max_substeps,
        advection=AdvectionConfig(scheme="RK2"),
        vlm=vlm_setup,
        background_velocity=[FREESTREAM_VELOCITY, 0.0, 0.0],
        turbulence=TurbulenceConfig.les_smagorinsky(),
        stretching=StretchingConfig.transposed(
            scheme="RK2",
            use_treecode=True,
            treecode_theta=args.treecode_theta,
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
            viscosity=KINEMATIC_VISCOSITY,
            characteristic_distance=wake_spacing,
        ),
        velocity=VelocityConfig.treecode(
            theta=args.treecode_theta,
            sort_particle_targets=True,
            traversal_block_dim=128,
        ),
        particles_kernel="WINCKELMANS",
        samplers=list(samplers),
        backup_file_name="rotor",
        backup_directory=args.solution_dir,
        backup_frequency=20,
        logging_frequency=20,
        timing_frequency=40,
        processing_unit=args.processing_unit,
        export_flow_integrals=True,
    )


def enforce_wake_admissibility(solver: Solver, max_particle_strength: float) -> None:
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
            f"step={solver.time_step}, t={solver.flow_time:.6e}: "
            + "; ".join(failures)
            + ". The run was stopped without modifying the particle field."
        )


def write_manifest(args: argparse.Namespace, solver: Solver) -> None:
    """Record the numerical contract needed to reproduce the run."""
    cfg = solver.config
    output_dir = Path(args.solution_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "case": "rotorFlow",
        "system": "forced_open_wake",
        "time_integration": cfg.time_integration,
        "advection_scheme": cfg.advection.scheme,
        "stretching_mode": cfg.stretching.mode,
        "stretching_scheme": cfg.stretching.scheme,
        "stretching_treecode": cfg.stretching.use_treecode,
        "velocity_method": cfg.velocity.method,
        "treecode_theta": cfg.velocity.theta,
        "kernel": cfg.particles_kernel,
        "viscous_scheme": cfg.viscous.scheme,
        "molecular_viscosity": cfg.viscous.viscosity,
        "wake_characteristic_distance": cfg.viscous.characteristic_distance,
        "coupled_max_strain_increment": cfg.coupled_max_strain_increment,
        "coupled_max_advection_fraction": cfg.coupled_max_advection_fraction,
        "coupled_max_substeps": cfg.coupled_max_substeps,
        "field_modification": "none",
        "retention_bounds": cfg.stabilization.remove_particles_by_bounds,
        "physical_acceptance": "blade-force/wake-impulse budget",
        "processing_unit": solver.processing_unit,
        "processing_unit_requested": args.processing_unit,
        "dt": args.dt,
        "num_steps": args.num_steps,
        "guard_frequency": args.guard_frequency,
        "max_particle_strength": args.max_particle_strength,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    from assets.generate_openvsp_blade import RotorBladeDesign, generate_rotorflow_openvsp_blade

    args = build_arg_parser().parse_args()
    validate_arguments(args)

    # ================================================
    # 1. Runtime Controls
    # ================================================
    num_steps = args.num_steps

    # ================================================
    # 2. Create VLM Blade Geometry From OpenVSP
    # ================================================
    blade_file = _SCRIPT_DIR / "assets/blade.json"

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
            output_dir=str(_SCRIPT_DIR / "assets/openvsp"),
            json_path=str(blade_file),
            design=blade_design,
        )

    # ================================================
    # 3. Configure VLM Solver
    # ================================================
    rotation_period = 2.0 * np.pi / ANGULAR_VELOCITY
    ramp_time = max(0.0, args.ramp_rotations * rotation_period)

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
        logging_frequency=10,
    )

    # ================================================
    # 4. Configure VPM Solver
    # ================================================
    backup_dir = args.solution_dir

    # Downstream YZ cross-plane samplers for wake / induction validation.
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

    solver_config = build_solver_config(args, vlm_setup=vlm_setup, samplers=plane_samplers)
    vpm = Solver(setup=solver_config)
    write_manifest(args, vpm)
    vpm.info()

    # ================================================
    # 5. Run Simulation
    # ================================================
    try:
        for step in range(num_steps):
            vpm.update_state()
            if (step + 1) % args.guard_frequency == 0:
                enforce_wake_admissibility(vpm, args.max_particle_strength)
    except RuntimeError:
        vpm.save_state(str(Path(backup_dir) / "rejected_state"))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
