"""Run an isolated, explicitly planar cylinder drag-recovery experiment.

This diagnostic retains the tutorial's XY wall mesh, molecular viscosity,
time steps, force normalization and coupling controls. Infinite-span planar
induction and four extruded FVM layers make parameter comparisons affordable;
they do not qualify the production three-dimensional slip-slab solver.

Example (run from the repository root)::

    python -m tests.support.cylinder.run_drag_recovery --output-dir /tmp/cylinder-fixed

The freestream is (1, 0.1, 0) until the accepted 2 s endpoint, then (1, 0, 0).
Both stages use ordinary coupled solves and native checkpoints. Outputs are
never reused unless --resume is supplied with identical experiment settings.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

import openonda.coupler as coupling
import openonda.fvm as fvm
import openonda.vpm as vpm
from source.restart import select_backup

REPOSITORY = Path(__file__).resolve().parents[3]
TUTORIAL = REPOSITORY / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
SPAN = 0.96
H = 0.04
FVM_DT = 0.008
EXCHANGE_DT = 0.04


def _positive_integer(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def planar_initial_velocity(positions, *, transverse_speed, amplitude=1.0e-3):
    """A compact XY curl disturbance, exactly invariant in z with w=0."""
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    x, y = points[:, 0], points[:, 1]
    radius_squared = 0.5**2
    support = np.maximum(1 - ((x - 0.65) ** 2 + y**2) / radius_squared, 0)
    velocity = np.zeros_like(points)
    velocity[:, 0] = 1 - 8 * amplitude * y / radius_squared * support**3
    velocity[:, 1] = transverse_speed + 8 * amplitude * (x - 0.65) / radius_squared * support**3
    return velocity


def build_experiment(options):
    """Construct tutorial-based configurations without allocating a solver."""
    spec = importlib.util.spec_from_file_location(
        "cylinder_drag_recovery_case", TUTORIAL / "setup.py"
    )
    tutorial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tutorial)
    tutorial.FVM_BOX = (-1.6, options.downstream_x, -1.6, 1.6, -SPAN / 2, SPAN / 2)
    tutorial.TRANSFER_REGION_BOX = (
        -1.25,
        options.downstream_x - options.transfer_gap,
        -1.25,
        1.25,
        -SPAN / 2,
        SPAN / 2,
    )
    setup, particles, exchange, mesh = tutorial.build_case(
        end_time=options.end_time,
        overrides={
            "hxy": H,
            "span": SPAN,
            "dz": SPAN / 4,
            "compute_device": options.device,
            "cores": options.cores,
            "particle_limit": options.particle_limit,
        },
    )
    background = (1.0, options.trigger_velocity, 0.0)
    force = fvm.ForceSampler(
        patch_names=["cylinder"],
        reference_velocity=1.0,
        reference_area=SPAN,
        reference_length=1.0,
        file_name="forces_history",
        schedule=fvm.RunSchedule(every_n_steps=5),
    )
    setup = replace(
        setup,
        time=replace(setup.time, output_schedule=fvm.RunSchedule(final_only=True)),
        samplers=(force,),
        initial_velocity=list(background),
        boundaries=[
            replace(boundary, velocity_value=list(background))
            if boundary.name == "numericalBoundary"
            else boundary
            for boundary in setup.boundaries
        ],
    )
    # Preserve the physical vorticity floor for cubic particle volume h³.
    viscous = replace(particles.numerics.viscous, gbd_threshold=0.01 * H**3)
    particles = replace(
        particles,
        numerics=replace(
            particles.numerics,
            induction=vpm.SlipSlabInduction(
                vpm.FMMInduction(), z_min=-SPAN / 2, z_max=SPAN / 2
            ),
            viscous=viscous,
            freestream_velocity=background,
        ),
        samplers=vpm.Samplers(),
        directory=options.output_dir,
    )
    exchange = replace(
        exchange,
        freestream_velocity=list(background),
        backup_interval_steps=125,
    )
    return setup, particles, exchange, mesh


def disable_trigger(solver):
    """Change both solvers' accepted freestream settings at the segment boundary.

    The initial lattice retains the slightly larger buffer built for the
    maximum startup speed. Reconstructing every run with that same lattice
    keeps restart geometry identical on either side of the trigger endpoint.
    The next solve rebuilds boundary history with the constant background at
    the accepted switch time; subsequent exchanges interpolate that trace.
    """
    background = (1.0, 0.0, 0.0)
    solver.setup = replace(solver.setup, freestream_velocity=list(background))
    solver.freestream_velocity = np.asarray(background, dtype=np.float64)
    solver.vorticity_transfer.config = solver.setup
    if solver._is_master:
        particles = solver.vpm_solver
        particles.setup = replace(particles.setup, freestream_velocity=background)
        particles.case = replace(particles.case, numerics=particles.setup)
        particles._set_freestream_velocity(background)


def experiment_settings(options):
    """Physical configuration checked before validating a native resume."""
    settings = {
        "schema": "openonda-cylinder-drag-recovery/1",
        "scope": "diagnostic planar reduction; not three-dimensional slab qualification",
        "downstream_x": options.downstream_x,
        "transfer_gap": options.transfer_gap,
        "hxy": H,
        "span": SPAN,
        "fvm_layers": 4,
        "fvm_dt": FVM_DT,
        "exchange_dt": EXCHANGE_DT,
        "end_time": options.end_time,
        "device": options.device,
        "cores": options.cores,
        "particle_limit": options.particle_limit,
        "trigger_velocity": options.trigger_velocity,
        "trigger_duration": options.trigger_duration,
        "initial_disturbance": "compact XY curl; amplitude 0.001; no span variation",
        "force_reference_velocity": 1.0,
    }
    mesh = getattr(options, "mesh", None)
    if mesh is not None:
        settings["mesh"] = {
            "path": str(mesh),
            "sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
        }
    return settings


def run(options):
    configuration = experiment_settings(options)
    metadata = options.output_dir / "drag_recovery_experiment.json"
    if options.resume:
        if not metadata.is_file() or json.loads(metadata.read_text()) != configuration:
            raise ValueError("--resume requires an identical drag_recovery_experiment.json")
    elif metadata.exists():
        raise FileExistsError(
            f"Experiment already exists; select a new output directory: {metadata}"
        )
    checkpoint = select_backup("latest", directory=options.output_dir / "solution", kind="coupled")
    if options.resume and checkpoint is None:
        raise ValueError("--resume requires a committed native coupled checkpoint")
    saved = (
        None
        if checkpoint is None
        else json.loads((checkpoint / "checkpoint_info.json").read_text())
    )
    start_step = 0 if saved is None else int(saved["coupling_step"])
    trigger_step = round(options.trigger_duration / EXCHANGE_DT)
    end_step = round(options.end_time / EXCHANGE_DT)
    setup, particles, exchange, mesh = build_experiment(options)
    if getattr(options, "mesh", None) is not None:
        # Native factory/initialization checks still apply. The mesh is only
        # an input; require_empty_output below protects the new destination.
        mesh = options.mesh
    with coupling.create_coupler(
        setup,
        particles,
        exchange,
        mesh=mesh,
        case_dir=options.output_dir,
        require_empty_output=not options.resume,
    ) as solver:
        if solver._is_master:
            metadata.write_text(json.dumps(configuration, indent=2, sort_keys=True) + "\n")
        solver.initialize()
        if not options.resume:
            count = solver.fvm_solver.mesh_data["n_cells"]
            solver.fvm_solver.set_initial_velocity(
                planar_initial_velocity(
                    solver.fvm_solver.geo_data["cell_centre"][:count],
                    transverse_speed=options.trigger_velocity,
                )
            )
        if options.resume:
            # A checkpoint exactly at the switch may still record the first
            # segment's background. Validate it unchanged before changing settings.
            saved_velocity = saved["config"]["coupler"]["freestream_velocity"]
            if np.allclose(saved_velocity, [1.0, 0.0, 0.0], rtol=0, atol=0):
                disable_trigger(solver)
            start_step = solver.load_backup(checkpoint)
            if start_step >= trigger_step:
                disable_trigger(solver)
            if start_step >= end_step:
                return start_step
        remaining = options.max_coupling_steps
        first_limit = remaining
        if start_step < trigger_step:
            first_limit = min(trigger_step - start_step, end_step - start_step)
            if remaining is not None:
                first_limit = min(first_limit, remaining)
        if options.resume:
            accepted = solver.solve(
                start_step=start_step,
                max_coupling_steps=first_limit,
                backup_at_stop=True,
            )
        else:
            accepted = solver.run(
                start_from="initial",
                max_coupling_steps=first_limit,
                backup_at_stop=True,
            )
        if remaining is not None:
            remaining -= accepted - start_step
        if (
            accepted == trigger_step
            and accepted < end_step
            and (remaining is None or remaining > 0)
        ):
            disable_trigger(solver)
            accepted = solver.solve(
                start_step=accepted,
                max_coupling_steps=remaining,
                backup_at_stop=True,
            )
        return accepted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mesh",
        type=Path,
        help="Reuse an existing native FVM mesh instead of generating one; SHA-256 is recorded",
    )
    parser.add_argument("--downstream-x", type=float, default=1.6)
    parser.add_argument("--transfer-gap", type=float, default=0.35)
    parser.add_argument("--end-time", type=float, default=60.0)
    parser.add_argument("--trigger-velocity", type=float, default=0.1)
    parser.add_argument("--trigger-duration", type=float, default=2.0)
    parser.add_argument("--device", choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--cores", type=_positive_integer, default=1)
    parser.add_argument("--particle-limit", type=_positive_integer, default=200_000)
    parser.add_argument("--max-coupling-steps", type=_positive_integer)
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args(argv)
    options.output_dir = options.output_dir.resolve()
    if options.mesh is not None:
        options.mesh = options.mesh.resolve(strict=True)
    for name in ("end_time", "trigger_duration"):
        value = getattr(options, name)
        if (
            not np.isfinite(value)
            or value <= 0
            or not np.isclose(value / EXCHANGE_DT, round(value / EXCHANGE_DT))
        ):
            parser.error(f"--{name.replace('_', '-')} must be positive and align to 0.04 s")
    if not np.isfinite(options.trigger_velocity):
        parser.error("--trigger-velocity must be finite")
    if not np.isfinite(options.transfer_gap) or options.transfer_gap <= 0:
        parser.error("--transfer-gap must be positive")
    if (
        not np.isfinite(options.downstream_x)
        or options.downstream_x - options.transfer_gap <= 0.5 + 6 * H
    ):
        parser.error("The downstream transfer blending weight must enclose the cylinder wall")
    run(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
