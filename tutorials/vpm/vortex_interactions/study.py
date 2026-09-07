"""Reproducible stabilization experiments for disturbed interacting rings.

Each run has its own directory, full configuration and source fingerprint.
Relaxation is specified as a frequency, so dt refinement keeps f fixed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import traceback
from source.solvers.vpm.config.fingerprint import numerical_configuration

import numpy as np

import openonda.vpm as vpm
from . import setup

METHODS = (
    "baseline",
    "stretching_viscosity",
    "pedrizzetti",
    "splitting",
    "divergence_relaxation",
    "remeshing",
    "solenoidal_remeshing",
    "p_relaxation",
    "p_moments",
    "p_split",
    "p_remesh",
    "p_split_remesh",
)
STUDY_DIR = setup.TUTORIAL_DIR / "study_results"


class ParticleSnapshots:
    """Small particle archives for independent field and instability checks."""

    initial = True

    def __init__(self, interval: int = 100):
        self.schedule = vpm.EverySteps(interval)

    def write(self, context):
        path = context.output_directory / "particles"
        path.mkdir(parents=True, exist_ok=True)
        solver = context.solver
        np.savez_compressed(
            path / f"{context.step:06d}.npz",
            time=context.time,
            step=context.step,
            position=solver.particle_position,
            vortex_strength=solver.particle_vortex_strength,
            core_radius=solver.particle_core_radius,
            particle_volume=solver.particle_volume,
            group_id=solver.particle_group_id,
        )


def source_fingerprint() -> str:
    root = setup.TUTORIAL_DIR.parents[2]
    files = sorted((root / "source" / "solvers" / "vpm").rglob("*.py"))
    files += [Path(__file__), Path(setup.__file__)]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_experiment(args, directory: Path):
    h = args.spacing
    sigma = args.core_ratio * h
    if not 0 < args.initial_tail < 1:
        raise ValueError("initial_tail must be between zero and one")
    if sigma >= setup.CORE_RADIUS:
        raise ValueError("spacing must resolve the physical core: core_ratio*h < a0")
    rings = []
    for group, centre in enumerate((-0.5 * setup.RING_SEPARATION, 0.5 * setup.RING_SEPARATION)):
        model = setup.create_ring(centre, group)
        tube = np.sqrt(setup.CORE_RADIUS**2 - sigma**2) * np.sqrt(-np.log(args.initial_tail))
        distribution = replace(
            model.distribution, spacing=h, tube_radius=tube, core_radius_ratio=args.core_ratio
        )
        if args.support == "disturbed":
            distribution = replace(
                distribution,
                disturbance=vpm.WidnallDisturbance.single_mode(
                    amplitude=args.amplitude,
                    mode=setup.DISTURBANCE_MODE,
                    direction=args.seed_direction,
                ),
            )
        circulation = (
            -setup.RING_CIRCULATION
            if args.scenario == "collision" and group == 1
            else setup.RING_CIRCULATION
        )
        rings.append(
            replace(
                model,
                distribution=distribution,
                circulation=circulation,
                disturbance=vpm.WidnallDisturbance.single_mode(
                    amplitude=args.amplitude,
                    mode=setup.DISTURBANCE_MODE,
                    direction=args.seed_direction,
                ),
            )
        )
    base = setup.build_case("baseline", n_steps=args.steps, compute_device=args.device)
    if args.diffusion == "GBD":
        viscous = vpm.ViscousConfig.gbd(
            particle_spacing=h,
            kinematic_viscosity=setup.KINEMATIC_VISCOSITY,
            core_radius_ratio=args.core_ratio,
            padding=5.0,
            threshold=args.diffusion_tail,
            max_nodes=args.capacity,
            remeshing_kernel=args.gbd_remeshing,
        )
    else:
        viscous = replace(
            base.numerics.viscous,
            scheme=args.diffusion,
            particle_spacing=h,
            core_radius_ratio=args.core_ratio,
            kinematic_viscosity=setup.KINEMATIC_VISCOSITY,
        )
    config = vpm.StabilizationConfig.disabled()
    if args.method in ("stretching_viscosity", "pedrizzetti", "divergence_relaxation"):
        config = setup.stabilization(args.method)
        if args.method == "divergence_relaxation":
            config = replace(
                config, divergence_relaxation=replace(config.divergence_relaxation, grid_spacing=h)
            )
    if args.method.startswith("p_"):
        config = replace(
            config,
            pedrizzetti_relaxation_factor=args.frequency * args.dt,
            pedrizzetti_relaxation_preserve_vortex_strength=False,
            pedrizzetti_relaxation_preserve_moments=args.method != "p_relaxation",
        )
    if args.method in ("splitting", "p_split", "p_split_remesh"):
        peak = max(
            float(np.linalg.norm(ring.build().vortex_strength, axis=1).max()) for ring in rings
        )
        config = replace(
            config,
            filament_refinement=vpm.FilamentRefinementConfig.adaptive(
                interval_steps=5,
                max_vortex_strength_factor=2.0,
                max_absolute_vortex_strength=args.split_factor * peak,
                offset_fraction=0.25,
                max_n_particles=args.capacity,
            ),
        )
    if args.method in ("remeshing", "solenoidal_remeshing", "p_remesh", "p_split_remesh"):
        config = replace(
            config,
            regularization_interval_steps=args.remesh_interval,
            regularization_start_step=args.remesh_start or args.remesh_interval,
            regularization_grid_spacing=args.remesh_spacing or h,
            regularization_core_radius=sigma,
            regularization_core_radius_trigger=1.35 * sigma,
            regularization_divergence_trigger=args.divergence_trigger,
            regularization_misalignment_trigger=None,
            regularization_tail_budget=args.tail_budget,
            regularization_solenoidal_remesh=args.method == "solenoidal_remeshing",
            regularization_max_particles=args.capacity,
            regularization_projection_trigger=args.projection_trigger,
            regularization_total_kinetic_energy_dissipation_limit=0.02,
            regularization_total_enstrophy_dissipation_limit=0.05,
        )
    return replace(
        base,
        directory=directory,
        initial_conditions=tuple(rings),
        numerics=replace(
            base.numerics,
            induction=vpm.TreecodeInduction._for_testing(
                theta=args.tree_theta,
                multipole_order=args.tree_order,
                stretching_scheme=args.stretching,
            ),
            time_step_size=args.dt,
            viscous=viscous,
            domain_bounds=(-2.0, 12.0, -3.0, 3.0, -3.0, 3.0)
            if args.diffusion == "GBD"
            else base.numerics.domain_bounds,
            stabilization=config,
            turbulence=vpm.TurbulenceConfig.dns()
            if args.smagorinsky == 0
            else replace(base.numerics.turbulence, smagorinsky_coefficient=args.smagorinsky),
            max_n_particles=args.capacity,
            verbose=False,
            diagnostics=replace(base.numerics.diagnostics, detailed_timing=args.timing),
        ),
        backup=vpm.Backup(interval_steps=0, directory="solution", log_directory="solution"),
        samplers=vpm.Samplers(
            samples=(
                setup.FlowIntegralsSampler(schedule=vpm.EverySteps(10)),
                setup.RingDiagnosticsSampler(schedule=vpm.EverySteps(10)),
                ParticleSnapshots(interval=args.snapshot_interval),
            ),
            directory="diagnostics",
        ),
    )


def run(args):
    name = args.tag or f"{args.scenario}_{args.method}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("tag must be a simple directory name")
    directory = STUDY_DIR / name
    signature = {**vars(args), "source_sha256": source_fingerprint()}
    signature.pop("resume", None)
    metadata_path = directory / "result.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if (
            args.resume
            and previous.get("signature") == signature
            and previous.get("status") in ("horizon_reached", "resolution_lost")
        ):
            print(f"[reuse] {name}", flush=True)
            return 0
        archive = directory.with_name(f"{name}.previous-{time.time_ns()}")
        directory.rename(archive)
    directory.mkdir(parents=True, exist_ok=True)
    case = build_experiment(args, directory)
    initial_count = sum(len(ring.build()) for ring in case.initial_conditions)
    metadata = {
        "signature": signature,
        "status": "running",
        "initial_particles": initial_count,
        "stabilization": asdict(case.numerics.stabilization),
        "numerics": numerical_configuration(case.numerics),
        "physics": {
            "Re_Gamma": setup.REYNOLDS_NUMBER,
            "R0": setup.RING_RADIUS,
            "Gamma0": setup.RING_CIRCULATION,
            "a0": setup.CORE_RADIUS,
            "disturbance_amplitude": args.amplitude,
            "disturbance_mode": setup.DISTURBANCE_MODE,
        },
    }

    def write():
        temporary = metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(metadata_path)

    write()
    start = time.perf_counter()
    solver = None
    error = None
    try:
        solver = vpm.VPMSolver(case)
        solver.run()
        metadata["status"] = (
            "horizon_reached" if solver.run_status == "completed" else solver.run_status
        )
        error = solver.run_failure
    except (Exception, KeyboardInterrupt) as exc:
        error = exc
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        traceback.print_exc()
    finally:
        metadata.update(
            wall_seconds=time.perf_counter() - start,
            completed_steps=0 if solver is None else int(solver.step),
            final_time=0.0 if solver is None else float(solver.time),
            termination_reason=None if error is None else f"{type(error).__name__}: {error}",
        )
        write()
    print(
        json.dumps(
            {
                "run": name,
                **{
                    key: metadata[key]
                    for key in (
                        "status",
                        "completed_steps",
                        "final_time",
                        "wall_seconds",
                        "termination_reason",
                    )
                },
            }
        ),
        flush=True,
    )
    return 1 if metadata["status"] in ("failed", "interrupted") else 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", choices=("leapfrog", "collision"), default="leapfrog")
    result.add_argument("--method", choices=METHODS, default="baseline")
    result.add_argument("--steps", type=int, default=1200)
    result.add_argument("--dt", type=float, default=setup.TIME_STEP_SIZE)
    result.add_argument("--spacing", type=float, default=setup.PARTICLE_SPACING)
    result.add_argument("--amplitude", type=float, default=setup.DISTURBANCE_AMPLITUDE)
    result.add_argument("--seed-direction", choices=("radial", "axial"), default="radial")
    result.add_argument(
        "--support",
        choices=("circular", "disturbed"),
        default="circular",
        help="circular reproduces the original tutorial; disturbed follows the seeded centreline",
    )
    result.add_argument("--frequency", type=float, default=0.03 / setup.TIME_STEP_SIZE)
    result.add_argument("--smagorinsky", type=float, default=setup.SMAGORINSKY_COEFFICIENT)
    result.add_argument("--diffusion", choices=("CS", "GBD", "RWM", "NONE"), default="CS")
    result.add_argument("--core-ratio", type=float, default=2.0)
    result.add_argument(
        "--initial-tail",
        type=float,
        default=setup.TOROIDAL_TAIL_FRACTION,
        help="omitted initial Gaussian tail before circulation normalization",
    )
    result.add_argument("--diffusion-tail", type=float, default=1e-5)
    result.add_argument("--gbd-remeshing", choices=("M4_PRIME", "LAGRANGE6"), default="M4_PRIME")
    result.add_argument("--timing", action="store_true")
    result.add_argument("--snapshot-interval", type=int, default=100)
    result.add_argument(
        "--tree-theta",
        type=float,
        default=0.1,
        help="qualification override; verify velocity AND gradients",
    )
    result.add_argument("--tree-order", type=int, choices=(1, 2, 3), default=1)
    result.add_argument(
        "--stretching", choices=("DIRECT", "TRANSPOSED", "MIXED"), default="TRANSPOSED"
    )
    result.add_argument("--remesh-interval", type=int, default=50)
    result.add_argument("--remesh-start", type=int)
    result.add_argument("--divergence-trigger", type=float, default=0.06)
    result.add_argument(
        "--remesh-spacing", type=float, help="defaults to the initial particle spacing"
    )
    result.add_argument("--tail-budget", type=float, default=1e-3)
    result.add_argument(
        "--projection-trigger",
        type=float,
        default=1.0,
        help="remeshed-field divergence above which to try constrained Helmholtz correction",
    )
    result.add_argument("--split-factor", type=float, default=2.0)
    result.add_argument("--capacity", type=int, default=120000)
    result.add_argument("--device", choices=("AUTO", "CPU", "METAL"), default="AUTO")
    result.add_argument("--tag")
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
