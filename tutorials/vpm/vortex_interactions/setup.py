#!/usr/bin/env python3
"""Compare VPM stabilization methods for two leapfrogging vortex rings.

Every case uses the same transposed LES formulation and SSPRK3 update. The
only changed quantity is the stabilization method selected by the case name.
Crossing a resolution limit ends that case normally so the comparison can
continue.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import cache
import json
from pathlib import Path
import tempfile

import numpy as np

import openonda.vpm as vpm
from openonda.vpm import Backup, Samplers

TUTORIAL_DIR = Path(__file__).resolve().parent

# ---- Physics ---------------------------------------------------------------
RING_RADIUS = 1.0  # ring major radius [m]
RING_CIRCULATION = np.pi  # circulation of each ring [m²/s]
REYNOLDS_NUMBER = 3000.0  # Re = Gamma/nu
CORE_RADIUS = 0.1 * RING_RADIUS  # physical Gaussian core radius [m]
RING_SEPARATION = 1.0 * RING_RADIUS  # initial axial separation [m]
KINEMATIC_VISCOSITY = RING_CIRCULATION / REYNOLDS_NUMBER
DISTURBANCE_AMPLITUDE = 0.05  # fraction of ring radius
DISTURBANCE_MODE = 8

# ---- Numerics --------------------------------------------------------------
PARTICLE_SPACING = 0.035 * RING_RADIUS
PARTICLE_CORE_RADIUS = 2.0 * PARTICLE_SPACING
# Circulation normalization amplifies a truncated Gaussian. A 5% tail cut
# produced a 6--7% peak excess on the study lattice; retain the physical tail.
TOROIDAL_TAIL_FRACTION = 1.0e-4
TIME_STEP_SIZE = 20.0 * PARTICLE_SPACING**2 / RING_CIRCULATION
N_STEPS = 1200
SAMPLE_INTERVAL_STEPS = 5
CORE_SECTION_INTERVAL = 1.5  # physical seconds, independent of the integration timestep
CORE_SECTION_SPACING = 0.02 * RING_RADIUS
BACKUP_INTERVAL_STEPS = 50
MAX_N_PARTICLES = 120_000
SMAGORINSKY_COEFFICIENT = 0.20
RANDOM_SEED = 42

MAX_LAGRANGIAN_CFL = 1.0
MAX_VORTICITY_DIVERGENCE = 0.12
MAX_VORTEX_MISALIGNMENT = 25.0

# ---- Stabilization ---------------------------------------------------------
STRETCHING_VISCOSITY_COEFFICIENT = 0.5
PEDRIZZETTI_FACTOR = 0.3
PEDRIZZETTI_INTERVAL_STEPS = 1
SPLITTING_STRENGTH_FACTOR = 2.0
SPLITTING_INTERVAL_STEPS = 1
SPLITTING_OFFSET_FRACTION = 0.25
DIVERGENCE_RELAXATION_INTERVAL_STEPS = 25
REMESH_CORE_RADIUS_FACTOR = 2.0
REMESH_CORE_RADIUS_TRIGGER = REMESH_CORE_RADIUS_FACTOR * PARTICLE_CORE_RADIUS
REMESH_INTERVAL_STEPS = round(
    3.0 * PARTICLE_CORE_RADIUS**2 / (4.0 * KINEMATIC_VISCOSITY * TIME_STEP_SIZE)
)
REMESH_TAIL_BUDGET = 1.0e-3

CASES = (
    "baseline",
    "stretching_viscosity",
    "pedrizzetti",
    "splitting",
    "divergence_relaxation",
    "remeshing",
)
CASE_LABELS = {
    "baseline": "Baseline",
    "stretching_viscosity": "Stretching viscosity",
    "pedrizzetti": "Pedrizzetti relaxation",
    "splitting": "Filament refinement",
    "divergence_relaxation": "Divergence relaxation",
    "remeshing": "Conservative regularization",
}
RUN_METADATA_SCHEMA_VERSION = 2


class FlowIntegralsSampler(vpm.FlowIntegralsSampler):
    """Write the initial state as well as the regular diagnostic cadence."""

    initial = True


class RingDiagnosticsSampler(vpm.RingDiagnosticsSampler):
    """Write the initial ring geometry as well as the regular cadence."""

    initial = True


class CoreSectionSampler(vpm.SurfaceSampler):
    """Record the initial meridional section as well as the regular cadence."""

    initial = True


def core_section_samplers(
    *, interval: float = CORE_SECTION_INTERVAL
) -> tuple[vpm.SurfaceSampler, ...]:
    """Sample curl(u) on z=0, y>=0; here omega_theta equals omega_z.

    The fixed grid covers both rings' travel. Postprocessing crops the saved
    plane around the cores without recomputing or azimuthally averaging fields.
    A final sampler also records runs that end between cadence boundaries.
    """
    options = dict(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[-2.0 * RING_RADIUS, 14.0 * RING_RADIUS, 0.0, 1.8 * RING_RADIUS],
        spacing=CORE_SECTION_SPACING,
        file_name="core_section",
        include_derivatives=False,
    )
    return (
        CoreSectionSampler(**options, schedule=vpm.EveryTime(interval)),
        vpm.SurfaceSampler(**options, schedule=vpm.FinalOnly()),
    )


def create_ring(centre_x: float, group_id: int) -> vpm.VortexRing:
    """Build one disturbed Gaussian vortex ring."""
    represented_core_sq = CORE_RADIUS**2 - PARTICLE_CORE_RADIUS**2
    tube_radius = np.sqrt(represented_core_sq) * np.sqrt(-np.log(TOROIDAL_TAIL_FRACTION))
    centre = (centre_x, 0.0, 0.0)
    distribution = vpm.ToroidalDistribution(
        ring_radius=RING_RADIUS,
        tube_radius=tube_radius,
        spacing=PARTICLE_SPACING,
        core_radius_ratio=PARTICLE_CORE_RADIUS / PARTICLE_SPACING,
        centre=centre,
    )
    return vpm.VortexRing(
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        centre=centre,
        radius=RING_RADIUS,
        circulation=RING_CIRCULATION,
        vortex_core_radius=CORE_RADIUS,
        disturbance=vpm.WidnallDisturbance.single_mode(
            amplitude=DISTURBANCE_AMPLITUDE,
            mode=DISTURBANCE_MODE,
        ),
        core_compensation=vpm.ParticleCoreCompensation(),
        distribution=distribution,
        group_id=group_id,
    )


@cache
def initial_peak_strength() -> float:
    """Return the largest initial particle strength."""
    particles = create_ring(-0.5 * RING_SEPARATION, 0).build()
    return float(np.linalg.norm(particles.vortex_strength, axis=1).max())


def stabilization(case_name: str) -> vpm.StabilizationConfig:
    """Return the one stabilization method selected for this case."""
    if case_name == "baseline":
        return vpm.StabilizationConfig.disabled()
    if case_name == "stretching_viscosity":
        return vpm.StabilizationConfig.stretching_viscosity(
            coefficient=STRETCHING_VISCOSITY_COEFFICIENT
        )
    if case_name == "pedrizzetti":
        return vpm.StabilizationConfig.pedrizzetti_relaxation(
            factor=PEDRIZZETTI_FACTOR,
            interval_steps=PEDRIZZETTI_INTERVAL_STEPS,
        )
    if case_name == "splitting":
        return vpm.StabilizationConfig(
            filament_refinement=vpm.FilamentRefinementConfig.adaptive(
                interval_steps=SPLITTING_INTERVAL_STEPS,
                max_vortex_strength_factor=np.inf,
                max_absolute_vortex_strength=(SPLITTING_STRENGTH_FACTOR * initial_peak_strength()),
                offset_fraction=SPLITTING_OFFSET_FRACTION,
                max_n_particles=MAX_N_PARTICLES,
            )
        )
    if case_name == "divergence_relaxation":
        return vpm.StabilizationConfig(
            divergence_relaxation=vpm.DivergenceRelaxationConfig.constrained(
                interval_steps=DIVERGENCE_RELAXATION_INTERVAL_STEPS,
                start_step=DIVERGENCE_RELAXATION_INTERVAL_STEPS,
                grid_spacing=PARTICLE_SPACING,
            )
        )
    if case_name == "remeshing":
        return vpm.StabilizationConfig(
            regularization_interval_steps=REMESH_INTERVAL_STEPS,
            regularization_start_step=REMESH_INTERVAL_STEPS,
            regularization_grid_spacing=PARTICLE_SPACING,
            regularization_tail_budget=REMESH_TAIL_BUDGET,
            regularization_max_particles=MAX_N_PARTICLES,
            regularization_divergence_trigger=None,
            regularization_misalignment_trigger=None,
            regularization_core_radius_trigger=REMESH_CORE_RADIUS_TRIGGER,
            regularization_core_radius=PARTICLE_CORE_RADIUS,
        )
    raise ValueError(f"Unknown case {case_name!r}; expected one of {CASES}")


def build_case(
    case_name: str,
    *,
    n_steps: int = N_STEPS,
    compute_device: str = "AUTO",
) -> vpm.VPMCase:
    """Build one LES and transposed-stretching comparison case."""
    if case_name not in CASES:
        raise ValueError(f"Unknown case {case_name!r}; expected one of {CASES}")
    initial_conditions = tuple(
        create_ring(centre_x, group_id)
        for group_id, centre_x in enumerate((-0.5 * RING_SEPARATION, 0.5 * RING_SEPARATION))
    )
    return vpm.VPMCase(
        numerics=vpm.Numerics(
            time_step_size=TIME_STEP_SIZE,
            compute_device=compute_device,
            integrator=vpm.SSPRK3(),
            induction=vpm.TreecodeInduction(stretching_scheme="TRANSPOSED"),
            viscous=vpm.ViscousConfig.cs(),
            turbulence=vpm.TurbulenceConfig.les_smagorinsky(
                smagorinsky_coefficient=SMAGORINSKY_COEFFICIENT
            ),
            stabilization=stabilization(case_name),
            particle_kernel="GAUSSIAN",
            write_precision="f32",
            max_n_particles=MAX_N_PARTICLES,
            random_seed=RANDOM_SEED,
            health_limits=vpm.HealthLimits(
                lagrangian_cfl=vpm.LagrangianCFLLimit(maximum=MAX_LAGRANGIAN_CFL),
                divergence=vpm.DivergenceLimit(maximum=MAX_VORTICITY_DIVERGENCE),
                misalignment=vpm.MisalignmentLimit(maximum_degrees=MAX_VORTEX_MISALIGNMENT),
            ),
        ),
        initial_conditions=initial_conditions,
        backup=Backup(
            interval_steps=BACKUP_INTERVAL_STEPS,
            directory=str(Path("solution") / case_name),
            log_directory=str(Path("solution") / case_name),
        ),
        samplers=Samplers(
            samples=(
                FlowIntegralsSampler(schedule=vpm.EverySteps(SAMPLE_INTERVAL_STEPS)),
                RingDiagnosticsSampler(schedule=vpm.EverySteps(SAMPLE_INTERVAL_STEPS)),
                *core_section_samplers(),
            ),
            directory=case_name,
        ),
        run=vpm.RunPlan(
            steps=n_steps,
            final_backup=False,
            health_limit_action="STOP",
        ),
        directory=TUTORIAL_DIR,
    )


def _metadata(case_name: str, n_steps: int, initial_particles: int) -> dict:
    """Return the fixed settings recorded with one result."""
    return {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "case": case_name,
        "label": CASE_LABELS[case_name],
        "status": "running",
        "requested_steps": n_steps,
        "completed_steps": 0,
        "final_time": 0.0,
        "time_step_size": TIME_STEP_SIZE,
        "integrator": "SSPRK3",
        "induction_backend": "TREECODE",
        "stretching_scheme": "TRANSPOSED",
        "turbulence_model": "LES_SMAGORINSKY",
        "smagorinsky_coefficient": SMAGORINSKY_COEFFICIENT,
        "viscous_scheme": "CS",
        "stabilization": case_name,
        "stabilization_config": asdict(stabilization(case_name)),
        "particle_spacing": PARTICLE_SPACING,
        "particle_core_radius": PARTICLE_CORE_RADIUS,
        "ring_radius": RING_RADIUS,
        "ring_circulation": RING_CIRCULATION,
        "core_radius": CORE_RADIUS,
        "ring_separation": RING_SEPARATION,
        "reynolds_number": REYNOLDS_NUMBER,
        "disturbance_amplitude": DISTURBANCE_AMPLITUDE,
        "disturbance_mode": DISTURBANCE_MODE,
        "initial_n_particles_total": initial_particles,
        "final_n_particles_total": initial_particles,
        "maximum_lagrangian_cfl": MAX_LAGRANGIAN_CFL,
        "maximum_vorticity_divergence_error": MAX_VORTICITY_DIVERGENCE,
        "maximum_vortex_misalignment_degrees": MAX_VORTEX_MISALIGNMENT,
        "health_limit_action": "STOP",
        "termination_reason": None,
    }


def _write_metadata(metadata: dict) -> None:
    path = TUTORIAL_DIR / "samples" / metadata["case"] / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def result_exists(case_name: str, n_steps: int) -> bool:
    """Return whether a compatible terminal result is already available."""
    try:
        path = TUTORIAL_DIR / "samples" / case_name / "run_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        expected = _metadata(
            case_name,
            n_steps,
            int(metadata["initial_n_particles_total"]),
        )
        for key in (
            "schema_version",
            "case",
            "requested_steps",
            "time_step_size",
            "integrator",
            "induction_backend",
            "stretching_scheme",
            "turbulence_model",
            "smagorinsky_coefficient",
            "viscous_scheme",
            "stabilization",
            "stabilization_config",
            "particle_spacing",
            "particle_core_radius",
            "ring_radius",
            "ring_circulation",
            "core_radius",
            "ring_separation",
            "reynolds_number",
            "disturbance_amplitude",
            "disturbance_mode",
            "maximum_lagrangian_cfl",
            "maximum_vorticity_divergence_error",
            "maximum_vortex_misalignment_degrees",
            "health_limit_action",
        ):
            if metadata.get(key) != expected[key]:
                return False
        completed_steps = int(metadata["completed_steps"])
        if metadata.get("status") == "horizon_reached":
            return completed_steps == n_steps
        return metadata.get("status") == "resolution_lost" and 0 <= completed_steps < n_steps
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_case(
    case_name: str,
    *,
    n_steps: int = N_STEPS,
    resume: bool = False,
    compute_device: str = "AUTO",
) -> None:
    """Run one stabilization case and retain all available diagnostics."""
    if case_name not in CASES:
        raise ValueError(f"Unknown case {case_name!r}; expected one of {CASES}")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if resume and result_exists(case_name, n_steps):
        print(f"[resume] {case_name}: reusing existing result", flush=True)
        return

    previous = [TUTORIAL_DIR / kind / case_name for kind in ("samples", "solution")]
    if any(path.exists() for path in previous):
        archive_root = TUTORIAL_DIR / "solution" / ".previous_runs"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = Path(tempfile.mkdtemp(prefix=f"{case_name}-", dir=archive_root))
        for path in previous:
            if path.exists():
                path.rename(archive / path.parent.name)
        print(f"[resume] {case_name}: moved old results to {archive}", flush=True)

    case = build_case(case_name, n_steps=n_steps, compute_device=compute_device)
    initial_particles = sum(len(condition.build()) for condition in case.initial_conditions)
    metadata = _metadata(case_name, n_steps, initial_particles)
    _write_metadata(metadata)

    solver = None
    error = None
    root_manifest = TUTORIAL_DIR / "run_manifest.json"
    root_manifest.unlink(missing_ok=True)
    try:
        solver = vpm.VPMSolver(case)
        solver.run()
    except BaseException as exc:
        error = exc
        raise
    finally:
        if root_manifest.is_file():
            destination = TUTORIAL_DIR / "solution" / case_name / "run_manifest.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            root_manifest.replace(destination)
        failure = error if solver is None else (solver.run_failure or error)
        status = "failed" if solver is None else solver.run_status
        metadata.update(
            status="horizon_reached" if status == "completed" else status,
            completed_steps=0 if solver is None else int(solver.step),
            final_time=0.0 if solver is None else float(solver.time),
            final_n_particles_total=(
                initial_particles if solver is None else int(solver.particles.n_particles_total)
            ),
            active_stabilization=(
                [] if solver is None else list(solver.stabilization.active_mechanisms())
            ),
            termination_reason=(
                None if failure is None else f"{type(failure).__name__}: {failure}"
            ),
        )
        _write_metadata(metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_case(args.case, n_steps=args.steps, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
