"""Autonomous cylinder controls with native restart and force measurements.

Run physical changes in private temporary cases. A developed checkpoint can
be reused when the FVM mesh and clocks agree; every changed numerical value
is recorded and passed through the native exact restart validation.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
import json
from pathlib import Path
import time

from openonda import coupler, vpm
from openonda.tutorial_runner import load_case_module
from source.coupler.backup import _backup_config, checkpoint_path_hash
from source.solvers.vpm.config.restart_changes import _evidence, _exact_mismatches, _value_at

from .run_coupled_checkpoint_control import digest, quiet_setup

REPOSITORY = Path(__file__).resolve().parents[3]
CASE = REPOSITORY / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def write_report(path, values):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(options):
    directory = options.directory.resolve()
    if not directory.is_relative_to(Path("/tmp")):
        raise ValueError("Scientific test output belongs in a private /tmp case")
    directory.mkdir(parents=True, exist_ok=False)
    module = load_case_module(CASE)
    if (
        options.initial_state
        or options.initial_volume_time is not None
        or options.steady_freestream
    ):
        module.STARTUP_FREESTREAM_VELOCITY = module.FREESTREAM_VELOCITY
    module.VPM_DOMAIN = (-5.0, options.wake_x, -5.0, 5.0, -0.5, 0.5)
    module.FVM_BOX = (
        -options.half_width,
        options.downstream_x,
        -options.half_width,
        options.half_width,
        -0.5,
        0.5,
    )
    module.TRANSFER_REGION_BOX = (
        -options.half_width + 0.35,
        options.downstream_x - 0.35,
        -options.half_width + 0.35,
        options.half_width - 0.35,
        -0.5,
        0.5,
    )
    flow, particles, exchange, mesh = module.build_case(
        end_time=options.end_time,
        overrides={
            "particle_spacing_ratio": options.particle_ratio,
            "core_radius_ratio": options.core_ratio,
            "exchange_dt": options.exchange_dt,
            "eta_blend_width": module.BLEND_WIDTH_RATIO * module.CELL_SIZE,
            "vpm_only_width": module.RELEASE_WIDTH_RATIO * module.CELL_SIZE,
            "interface_normal_tolerance": options.interface_tolerance,
            "interface_gradient_tolerance": options.interface_tolerance,
        },
    )
    flow = quiet_setup(flow)
    if options.pressure_tolerance:
        flow = replace(
            flow, linear=replace(flow.linear, pressure_tolerance=options.pressure_tolerance)
        )
    if options.particle_precision:
        particles = replace(
            particles, numerics=replace(particles.numerics, precision=options.particle_precision)
        )
    if options.native_slip_channel:
        particles = replace(
            particles,
            numerics=replace(
                particles.numerics,
                induction=vpm.PlanarChannelInduction(half_width=options.slip_half_width),
            ),
        )
    particles = replace(particles, directory=directory, samplers=vpm.Samplers())
    exchange = replace(exchange, backup_interval_steps=options.backup_steps)
    if options.interface_iterations:
        exchange = replace(exchange, interface_iterations=options.interface_iterations)
    if options.mesh is not None:
        mesh = options.mesh.resolve()
    input_hashes = {str(CASE / "setup.py"): digest(CASE / "setup.py")}
    if options.mesh is not None:
        input_hashes[str(mesh)] = digest(mesh)
    checkpoint = None
    stored = None
    if options.snapshot:
        checkpoint = options.snapshot.resolve() / "checkpoint"
        stored = json.loads((checkpoint / "checkpoint_info.json").read_text())
        mesh = options.snapshot.resolve() / "coupled_mesh.npz"
        input_hashes[str(mesh)] = digest(mesh)
        input_hashes[str(checkpoint / "checkpoint_info.json")] = digest(
            checkpoint / "checkpoint_info.json"
        )
        for name, relative in stored["checkpoint_files"].items():
            if checkpoint_path_hash(checkpoint / relative) != stored["file_sha256"][name]:
                raise ValueError("Frozen checkpoint hash differs: " + name)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "initializing",
        "parameters": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(options).items()
        },
        "input_sha256": input_hashes,
        "restart_changes": [],
        "scope": "Autonomous native FVM/VPM evolution; no reference forcing or force scaling",
    }
    report_path = directory / "experiment.json"
    write_report(report_path, report)
    started = time.perf_counter()
    with coupler.create_coupler(flow, particles, exchange, mesh=mesh, case_dir=directory) as owner:
        owner.initialize()
        owner.fvm_solver.auto_write = False
        start = 0
        if stored:
            current = json.loads(json.dumps(_backup_config(owner)))
            paths = sorted(_exact_mismatches(current, stored["config"]))
            expected = {
                path: (_value_at(stored["config"], path), _value_at(current, path))
                for path in paths
            }
            intended = (
                "coupler.backup_interval_steps",
                "vpm.domain_bounds[1]",
                "vpm.stabilization.remove_particles_by_bounds[1]",
                "vpm.viscous.particle_spacing",
                "vpm.viscous.gbd_grid_spacing",
                "vpm.viscous.gbd_threshold",
                "vpm.viscous.core_radius_ratio",
                "vpm.induction.channel_half_width",
                "vpm.induction.method",
                "vpm.induction.type",
                "coupler.eta_blend_width",
                "coupler.vpm_only_width",
                "coupler.interface_normal_tolerance",
                "coupler.interface_gradient_tolerance",
            )
            unexpected = set(paths) - set(intended)
            if unexpected:
                raise ValueError("Unexpected experimental restart changes: " + str(unexpected))
            start = owner.load_backup(
                checkpoint, allowed_config_differences=paths, expected_config_differences=expected
            )
            report["restart_changes"] = [
                {
                    "path": path,
                    "stored": _evidence(expected[path][0]),
                    "current": _evidence(expected[path][1]),
                }
                for path in paths
            ]
            report["initial_time"] = owner.fvm_solver.time
        elif options.initial_state:
            from .developed_initial_conditions import initialize_developed_flow

            report["developed_initial_conditions"] = initialize_developed_flow(
                owner, options.initial_state.resolve()
            )
            report["initial_time"] = owner.fvm_solver.time
        elif options.initial_volume_time is not None:
            from .developed_initial_conditions import initialize_saved_volume

            report["saved_volume_initial_conditions"] = initialize_saved_volume(
                owner, CASE, options.initial_volume_time
            )
            report["initial_time"] = owner.fvm_solver.time
        else:
            velocity = partial(
                module.cylinder_initial_velocity,
                freestream_velocity=module.STARTUP_FREESTREAM_VELOCITY,
                **module.INITIAL_PERTURBATION,
            )
            from source.simulation.forcing import apply_initial_velocity

            apply_initial_velocity(owner.fvm_solver, velocity)
            report["initial_time"] = 0.0
        report.update(
            status="running",
            fvm_cells=owner.fvm_solver.mesh_data["n_cells"],
            compute_device=owner.vpm_solver.compute_device,
        )
        write_report(report_path, report)
        measurements = {}
        if options.wall_potential:
            from .wall_potential_control import impermeable_cylinder_induction

            correction = impermeable_cylinder_induction(owner, measurements)
        else:
            correction = nullcontext()
        particle_measurements = {}
        channel_measurements = {}
        interface_measurements = {}
        summation_measurements = {}
        stable_image_measurements = {}
        with ExitStack() as controls:
            if options.induction_accumulation64:
                from .planar_summation_control import accurate_planar_summation

                controls.enter_context(accurate_planar_summation(owner, summation_measurements))
            if options.stable_channel_images:
                from .stable_channel_image_control import stable_channel_images

                controls.enter_context(
                    stable_channel_images(
                        owner,
                        stable_image_measurements,
                        accurate_summation=options.induction_accumulation64,
                    )
                )
            if options.linear_interface_prediction:
                from .interface_prediction_control import linearly_predicted_interface

                controls.enter_context(linearly_predicted_interface(owner, interface_measurements))
            if options.slip_half_width and not options.native_slip_channel:
                from .slip_channel_control import slip_channel_induction

                controls.enter_context(
                    slip_channel_induction(
                        owner,
                        options.slip_half_width,
                        channel_measurements,
                        inlet_x=options.inlet_x,
                    )
                )
            controls.enter_context(correction)
            if options.advection_substeps > 1:
                from .particle_time_step_control import particle_advection_substeps

                controls.enter_context(
                    particle_advection_substeps(
                        owner, options.advection_substeps, particle_measurements
                    )
                )
            final = owner.solve(
                start_step=start, max_coupling_steps=options.steps, backup_at_stop=True
            )
        records = owner.coupling_diagnostics
        report.update(
            status="complete",
            accepted_time=owner.fvm_solver.time,
            final_coupling_step=final,
            accepted_exchanges=final - start,
            unconverged_exchanges=sum(
                not row["interface_iteration"]["converged"] for row in records
            ),
            elapsed_seconds=time.perf_counter() - started,
            force_sha256=digest(directory / "samples/forces_history.csv"),
            wall_potential=measurements,
            particle_advection=particle_measurements,
            slip_channel=channel_measurements,
            planar_remeshing={"kernel": owner.vpm_solver._viscous_config.gbd_remeshing_kernel},
            interface_prediction=interface_measurements,
            induction_summation=summation_measurements,
            stable_channel_images=stable_image_measurements,
        )
    for path, expected_hash in input_hashes.items():
        if digest(path) != expected_hash:
            raise RuntimeError("Original scientific input changed: " + path)
    write_report(report_path, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--initial-volume-time", type=float)
    parser.add_argument("--steady-freestream", action="store_true")
    parser.add_argument("--backup-steps", type=int, default=100)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--end-time", type=float, default=100.0)
    parser.add_argument("--wake-x", type=float, default=15.0)
    parser.add_argument("--particle-ratio", type=float, default=1.0)
    parser.add_argument("--core-ratio", type=float, default=1.0)
    parser.add_argument("--particle-precision", choices=("f32", "f64"))
    parser.add_argument("--induction-accumulation64", action="store_true")
    parser.add_argument("--stable-channel-images", action="store_true")
    parser.add_argument("--exchange-dt", type=float, default=0.04)
    parser.add_argument("--interface-tolerance", type=float, default=1e-5)
    parser.add_argument("--interface-iterations", type=int)
    parser.add_argument("--pressure-tolerance", type=float)
    parser.add_argument("--downstream-x", type=float, default=2.4)
    parser.add_argument("--half-width", type=float, default=1.6)
    parser.add_argument("--wall-potential", action="store_true")
    parser.add_argument("--advection-substeps", type=int, default=1)
    parser.add_argument("--slip-half-width", type=float)
    parser.add_argument("--native-slip-channel", action="store_true")
    parser.add_argument("--inlet-x", type=float)
    parser.add_argument("--linear-interface-prediction", action="store_true")
    options = parser.parse_args()
    if (
        sum(
            (
                bool(options.snapshot),
                bool(options.initial_state),
                options.initial_volume_time is not None,
            )
        )
        > 1
    ):
        parser.error("Choose one physical initial condition or native restart")
    if options.advection_substeps < 1:
        parser.error("Particle advection substeps must be positive")
    if options.inlet_x is not None and not options.slip_half_width:
        parser.error("The inlet image control requires a slip-channel half width")
    if options.native_slip_channel and (not options.slip_half_width or options.inlet_x is not None):
        parser.error("Native planar channel induction requires a half width and has no inlet image")
    try:
        run(options)
    except BaseException as error:
        path = options.directory / "experiment.json"
        if path.exists():
            report = json.loads(path.read_text())
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_report(path, report)
        raise


if __name__ == "__main__":
    main()
