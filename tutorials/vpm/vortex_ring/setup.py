#!/usr/bin/env python3
"""Measure vortex-ring instability onset across stretching formulations.

All cases use the hierarchical treecode gradient. The three DNS cases use
the direct, transposed, and symmetric mixed vortex-stretching formulations;
``les_transposed`` adds the equilibrium Smagorinsky sub-grid model.
Every case uses the same Lagrangian-CFL endpoint; crossing it is the measured
outcome and returns normally so the remaining cases can run.

Examples (from this case directory)::

    python -m openonda.tutorial_runner . setup --variant dns_direct
    python -m openonda.tutorial_runner . setup --variant les_transposed --resume
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

if not __package__:
    from openonda.tutorial_runner import case_package
    from pathlib import Path as _CasePath

    __package__ = case_package(_CasePath(__file__).resolve().parents[0]) + ""

from .assets.ring_diagnostics import (
    RingDiagnosticsSampler,
    vortex_ring_mode_sampler,
)

TUTORIAL_DIR = Path(__file__).resolve().parent

# ---- Physics ---------------------------------------------------------------
RING_RADIUS = 1.0  # ring major radius [m]
RING_STRENGTH = np.pi  # circulation [m²/s]
REYNOLDS_NUMBER = 3000.0  # Re = Γ/ν — sets the vortex Reynolds number
CORE_RADIUS = 0.1  # initial Gaussian core radius [m]

# ---- Numerics --------------------------------------------------------------
PARTICLE_SPACING = 0.035  # in-plane particle spacing [m]
TIME_STEP_SIZE = 0.02  # Δt [s]
N_STEPS = 3000  # total number of time steps
SAMPLE_INTERVAL_TIME = 0.1  # write a sample every this many seconds
BACKUP_INTERVAL_TIME = 0.5  # keep an animation frame every this many seconds
WIDNALL_MODES = 24  # number of azimuthal bending modes
DEFAULT_WIDNALL_AMPLITUDE = 0.005  # resolved broadband centreline perturbation amplitude
TOROIDAL_TAIL_FRACTION = 0.05  # toroidal particle distribution tail fraction
MAX_N_PARTICLES = 100_000  # particle count guard
SMAGORINSKY_COEFFICIENT = 0.20  # Smagorinsky coefficient for LES
RANDOM_SEED = 42
ENERGY_DIAGNOSTIC_VERSION = 2
RUN_METADATA_SCHEMA_VERSION = 4
MAX_LAGRANGIAN_CFL = 1.0
MAX_VORTICITY_DIVERGENCE = 0.12
MAX_VORTEX_MISALIGNMENT = 25.0
EXPERIMENT = "stretching_instability_onset"

VARIANT_CONFIG = {
    "dns_direct": ("DNS", "DIRECT"),
    "dns_transposed": ("DNS", "TRANSPOSED"),
    "dns_mixed": ("DNS", "MIXED"),
    "les_transposed": ("LES_SMAGORINSKY", "TRANSPOSED"),
}
VARIANTS = tuple(VARIANT_CONFIG)

# -- Derived quantities ------------------------------------------------------
KINEMATIC_VISCOSITY = RING_STRENGTH / REYNOLDS_NUMBER  # ν = Γ/Re


def cadence_steps(period: float, time_step_size: float = TIME_STEP_SIZE) -> int:
    """Convert a physical output period to solver steps."""
    return max(1, round(period / time_step_size))


def turbulence_config(variant: str) -> vpm.TurbulenceConfig:
    """Return the explicit DNS or LES part of a ring configuration."""
    if variant not in VARIANT_CONFIG:
        raise ValueError(f"Unknown vortex-ring variant {variant!r}; expected one of {VARIANTS}")
    if VARIANT_CONFIG[variant][0] == "DNS":
        return vpm.TurbulenceConfig.dns()
    return vpm.TurbulenceConfig.les_smagorinsky(smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT)


def stretching_scheme(variant: str) -> str:
    """Return the vortex-stretching formulation for one case."""
    try:
        return VARIANT_CONFIG[variant][1]
    except KeyError as error:
        raise ValueError(
            f"Unknown vortex-ring variant {variant!r}; expected one of {VARIANTS}"
        ) from error


def result_exists(variant: str, n_steps: int) -> bool:
    """Return whether this instability test has already been run."""
    try:
        sample_directory = TUTORIAL_DIR / "samples" / variant
        metadata = json.loads((sample_directory / "run_metadata.json").read_text(encoding="utf-8"))
        expected = {
            "schema_version": RUN_METADATA_SCHEMA_VERSION,
            "variant": variant,
            "time_step_size": TIME_STEP_SIZE,
            "requested_steps": n_steps,
            "backup_interval_steps": cadence_steps(BACKUP_INTERVAL_TIME),
            "integrator": "SSPRK3",
            "induction_backend": "TREECODE",
            "strength_rate_mode": "HIERARCHICAL_GRADIENT",
            "stretching_scheme": stretching_scheme(variant),
            "turbulence_model": turbulence_config(variant).model,
            "viscous_scheme": "CS",
            "ring_radius": RING_RADIUS,
            "ring_circulation": RING_STRENGTH,
            "core_radius": CORE_RADIUS,
            "particle_spacing": PARTICLE_SPACING,
            "particle_core_radius": 2.0 * PARTICLE_SPACING,
            "reynolds_number": REYNOLDS_NUMBER,
            "smagorinsky_coefficient": (
                0.0 if VARIANT_CONFIG[variant][0] == "DNS" else SMAGORINSKY_COEFFICIENT
            ),
            "widnall_modes": WIDNALL_MODES,
            "widnall_amplitude": DEFAULT_WIDNALL_AMPLITUDE,
            "random_seed": RANDOM_SEED,
            "energy_diagnostic_version": ENERGY_DIAGNOSTIC_VERSION,
            "stabilization": "DISABLED",
            "health_limit_action": "STOP",
            "experiment": EXPERIMENT,
            "maximum_lagrangian_cfl": MAX_LAGRANGIAN_CFL,
            "maximum_vorticity_divergence_error": MAX_VORTICITY_DIVERGENCE,
            "maximum_vortex_misalignment_degrees": MAX_VORTEX_MISALIGNMENT,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        completed_steps = int(metadata["completed_steps"])
        status = metadata.get("status")
        if status == "horizon_reached":
            if not metadata.get("completed") or completed_steps != n_steps:
                return False
        elif status == "instability_detected":
            if metadata.get("completed") or not 0 <= completed_steps < n_steps:
                return False
            if not metadata.get("instability_reason"):
                return False
        else:
            return False
        expected_time = completed_steps * TIME_STEP_SIZE
        if not np.isclose(metadata.get("final_time", -1.0), expected_time, rtol=0.0, atol=1.0e-10):
            return False
        for name in ("flow_integrals.csv", "ring_diagnostics.csv", "ring_modes.csv"):
            with (sample_directory / name).open(newline="", encoding="utf-8") as stream:
                last_row = None
                for last_row in csv.DictReader(stream):
                    pass
            if last_row is None or int(last_row["step"]) != completed_steps:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def write_metadata(metadata: dict) -> None:
    """Write the case parameters and the measured instability time."""
    destination = TUTORIAL_DIR / "samples" / metadata["variant"] / "run_metadata.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def write_running_metadata(
    *,
    variant: str,
    n_steps: int,
    particle_core_radius: float,
    initial_n_particles_total: int,
    compute_device: str,
) -> None:
    """Make partial results available to the plotting scripts."""
    write_metadata(
        {
            "schema_version": RUN_METADATA_SCHEMA_VERSION,
            "status": "running",
            "completed": False,
            "variant": variant,
            "time_step_size": TIME_STEP_SIZE,
            "requested_steps": n_steps,
            "completed_steps": 0,
            "final_time": 0.0,
            "backup_interval_steps": cadence_steps(BACKUP_INTERVAL_TIME),
            "sample_interval_steps": cadence_steps(SAMPLE_INTERVAL_TIME),
            "integrator": "SSPRK3",
            "integrator_order": 3,
            "induction_backend": "TREECODE",
            "strength_rate_mode": "HIERARCHICAL_GRADIENT",
            "stretching_scheme": stretching_scheme(variant),
            "turbulence_model": turbulence_config(variant).model,
            "viscous_scheme": "CS",
            "compute_backend": compute_device,
            "write_precision": "f32",
            "ring_radius": RING_RADIUS,
            "ring_circulation": RING_STRENGTH,
            "core_radius": CORE_RADIUS,
            "particle_spacing": PARTICLE_SPACING,
            "particle_core_radius": particle_core_radius,
            "reynolds_number": REYNOLDS_NUMBER,
            "kinematic_viscosity": KINEMATIC_VISCOSITY,
            "smagorinsky_coefficient": (
                0.0 if VARIANT_CONFIG[variant][0] == "DNS" else SMAGORINSKY_COEFFICIENT
            ),
            "widnall_modes": WIDNALL_MODES,
            "widnall_amplitude": DEFAULT_WIDNALL_AMPLITUDE,
            "random_seed": RANDOM_SEED,
            "initial_n_particles_total": initial_n_particles_total,
            "final_n_particles_total": initial_n_particles_total,
            "energy_diagnostic_version": ENERGY_DIAGNOSTIC_VERSION,
            "stabilization": "DISABLED",
            "health_limit_action": "STOP",
            "experiment": EXPERIMENT,
            "maximum_lagrangian_cfl": MAX_LAGRANGIAN_CFL,
            "maximum_vorticity_divergence_error": MAX_VORTICITY_DIVERGENCE,
            "maximum_vortex_misalignment_degrees": MAX_VORTEX_MISALIGNMENT,
            "outcome": "running",
            "instability_step": None,
            "instability_time": None,
            "instability_reason": None,
            "termination_reason": None,
        }
    )


def write_run_metadata(
    *,
    variant: str,
    n_steps: int,
    particle_core_radius: float,
    initial_n_particles_total: int,
    solver,
) -> None:
    """Record when this stretching formulation became unstable."""
    status = str(solver.run_status)
    completed = status == "completed"
    instability_detected = status == "resolution_lost"
    public_status = "horizon_reached" if completed else "instability_detected"
    failure = solver.run_failure
    metadata = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "status": public_status,
        "completed": completed,
        "variant": variant,
        "time_step_size": TIME_STEP_SIZE,
        "requested_steps": n_steps,
        "completed_steps": int(solver.step),
        "final_time": float(solver.time),
        "backup_interval_steps": cadence_steps(BACKUP_INTERVAL_TIME),
        "sample_interval_steps": cadence_steps(SAMPLE_INTERVAL_TIME),
        "integrator": solver.integrator_tableau.name,
        "integrator_order": int(solver.integrator_tableau.order),
        "induction_backend": solver.induction.method,
        "strength_rate_mode": solver.induction.strength_rate_mode,
        "stretching_scheme": solver.induction.stretching_scheme,
        "turbulence_model": turbulence_config(variant).model,
        "viscous_scheme": solver.viscous_scheme,
        "compute_backend": solver.compute_device,
        "write_precision": "f32",
        "ring_radius": RING_RADIUS,
        "ring_circulation": RING_STRENGTH,
        "core_radius": CORE_RADIUS,
        "particle_spacing": PARTICLE_SPACING,
        "particle_core_radius": particle_core_radius,
        "reynolds_number": REYNOLDS_NUMBER,
        "kinematic_viscosity": KINEMATIC_VISCOSITY,
        "smagorinsky_coefficient": (
            0.0 if VARIANT_CONFIG[variant][0] == "DNS" else SMAGORINSKY_COEFFICIENT
        ),
        "widnall_modes": WIDNALL_MODES,
        "widnall_amplitude": DEFAULT_WIDNALL_AMPLITUDE,
        "random_seed": RANDOM_SEED,
        "initial_n_particles_total": initial_n_particles_total,
        "final_n_particles_total": int(getattr(solver.particles, "n_particles_total", 0)),
        "energy_diagnostic_version": ENERGY_DIAGNOSTIC_VERSION,
        "stabilization": "DISABLED",
        "health_limit_action": "STOP",
        "experiment": EXPERIMENT,
        "maximum_lagrangian_cfl": MAX_LAGRANGIAN_CFL,
        "maximum_vorticity_divergence_error": MAX_VORTICITY_DIVERGENCE,
        "maximum_vortex_misalignment_degrees": MAX_VORTEX_MISALIGNMENT,
        "outcome": public_status,
        "instability_step": int(solver.step) if instability_detected else None,
        "instability_time": float(solver.time) if instability_detected else None,
        "instability_reason": (
            None if not instability_detected else f"{type(failure).__name__}: {failure}"
        ),
        "termination_reason": (None if failure is None else f"{type(failure).__name__}: {failure}"),
    }
    write_metadata(metadata)


class RingFlowIntegralsSampler(vpm.FlowIntegralsSampler):
    """Use the current typed sampler interface for append-safe flow diagnostics."""

    initial = True

    def write(self, context) -> None:
        self.save_csv(
            context.solver,
            context.output_directory / f"{self.file_name}.csv",
            time=context.time,
        )


def run_case(
    variant: str,
    *,
    compute_device: str = "AUTO",
    n_steps: int = N_STEPS,
    resume: bool = False,
) -> None:
    """Run one current vortex-ring configuration and write solution/samples."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown vortex-ring variant {variant!r}; expected one of {VARIANTS}")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if resume and result_exists(variant, n_steps):
        print(f"[resume] {variant}: reusing existing result", flush=True)
        return
    previous = [TUTORIAL_DIR / kind / variant for kind in ("samples", "solution")]
    if any(path.exists() for path in previous):
        archive_root = TUTORIAL_DIR / "solution" / ".previous_runs"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = Path(tempfile.mkdtemp(prefix=f"{variant}-", dir=archive_root))
        for path in previous:
            if path.exists():
                path.rename(archive / path.parent.name)
        print(f"[resume] {variant}: moved old results to {archive}", flush=True)

    # -- Particle distribution ----------------------------------------------
    particle_core_radius = 2.0 * PARTICLE_SPACING
    represented_core_sq = CORE_RADIUS**2 - particle_core_radius**2
    tube_radius = np.sqrt(represented_core_sq) * np.sqrt(-np.log(TOROIDAL_TAIL_FRACTION))
    distribution = vpm.ToroidalDistribution(
        ring_radius=RING_RADIUS,
        tube_radius=tube_radius,
        spacing=PARTICLE_SPACING,
        core_radius_ratio=particle_core_radius / PARTICLE_SPACING,
    )

    initial_condition = vpm.VortexRing(
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        centre=(0.0, 0.0, 0.0),
        radius=RING_RADIUS,
        circulation=RING_STRENGTH,
        vortex_core_radius=CORE_RADIUS,
        disturbance=vpm.WidnallDisturbance.broadband(
            amplitude=DEFAULT_WIDNALL_AMPLITUDE,
            number_of_modes=WIDNALL_MODES,
            seed=RANDOM_SEED,
        ),
        core_compensation=vpm.ParticleCoreCompensation(),
        distribution=distribution,
        group_id=0,
    )
    initial_n_particles_total = len(initial_condition.build())
    write_running_metadata(
        variant=variant,
        n_steps=n_steps,
        particle_core_radius=particle_core_radius,
        initial_n_particles_total=initial_n_particles_total,
        compute_device=compute_device,
    )

    sample_steps = cadence_steps(SAMPLE_INTERVAL_TIME)
    backup_steps = cadence_steps(BACKUP_INTERVAL_TIME)
    mode_sampler = vortex_ring_mode_sampler(
        reference_radius=RING_RADIUS,
        schedule=vpm.EverySteps(sample_steps),
    )

    stabilization = vpm.StabilizationConfig.disabled()

    # -- Solver setup --------------------------------------------------------
    case = vpm.VPMCase(
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device=compute_device,
            integrator=vpm.SSPRK3(),
            turbulence=turbulence_config(variant),
            stabilization=stabilization,
            induction=vpm.TreecodeInduction(stretching_scheme=stretching_scheme(variant)),
            viscous=vpm.ViscousConfig.cs(),
            write_precision="f32",
            max_n_particles=MAX_N_PARTICLES,
            random_seed=RANDOM_SEED,
            health_limits=vpm.HealthLimits(
                lagrangian_cfl=vpm.LagrangianCFLLimit(maximum=MAX_LAGRANGIAN_CFL),
                divergence=vpm.DivergenceLimit(maximum=MAX_VORTICITY_DIVERGENCE),
                misalignment=vpm.MisalignmentLimit(maximum_degrees=MAX_VORTEX_MISALIGNMENT),
            ),
        ),
        initial_conditions=(initial_condition,),
        backup=Backup(
            interval_steps=backup_steps,
            directory=str(Path("solution") / variant),
            log_directory=str(Path("solution") / variant),
        ),
        samplers=Samplers(
            samples=(
                RingFlowIntegralsSampler(schedule=vpm.EverySteps(sample_steps)),
                RingDiagnosticsSampler(schedule=vpm.EverySteps(sample_steps)),
                mode_sampler,
            ),
            directory=variant,
        ),
        run=vpm.RunPlan(
            steps=n_steps,
            final_backup=False,
            health_limit_action="STOP",
        ),
        directory=TUTORIAL_DIR,
    )
    solver = vpm.VPMSolver(case)
    solver.run()
    write_run_metadata(
        variant=variant,
        n_steps=n_steps,
        particle_core_radius=particle_core_radius,
        initial_n_particles_total=initial_n_particles_total,
        solver=solver,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--steps", type=int, default=N_STEPS, help="accepted steps to run")
    parser.add_argument("--resume", action="store_true", help="reuse compatible completed outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_case(args.variant, n_steps=args.steps, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
