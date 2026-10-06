"""Unequal native mixed-boundary Picard trials from one accepted cylinder state.

The controlled declared interval is always 100.00 to 100.04 s. Trial A uses
the reference's 100.04 s endpoint; trial B deliberately uses its different
100.08 s endpoint at that same declared time. A, B and repeated A each begin
from the identical restored primary/BDF/face-flux state and imposed start
trace. This is a rollback/cache experiment, not a physical B trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
from time import perf_counter

import h5py
import numpy as np

from openonda import fvm
from openonda.tutorial_runner import load_case_module
from source.solvers.fvm.core.solver import FVMSolver
from source.solvers.fvm.fields.gradients import _resolve_gradient_fn
from source.solvers.fvm.io.backup import (
    capture_restart_state,
    restore_restart_state,
    validate_restart_state,
)
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from tests.support.cylinder.run_boundary_condition_study import (
    CASE,
    CELL_FIELDS,
    FLUX_FIELDS,
    check_numerical_sources,
    digest,
    endpoint_comparison,
    fixed_flux_increments,
    force_values,
    impose,
    patch_geometry,
    quiet_setup,
    reference_setup,
    restart_state_comparison,
    validate_saved_settings,
)

TRACE_FIELDS = ("velocity", "normal_velocity", "tangential_gradient")


def interpolated_trial_trace(start, endpoint, substep, substeps=5):
    """Use the same declared interval with an explicitly selected trial endpoint."""
    if (isinstance(substep, bool) or not isinstance(substep, int)
            or not 0 <= substep <= substeps or substeps != 5):
        raise ValueError("The rollback control requires five native substeps")
    fraction = substep / substeps
    result = {}
    for name in TRACE_FIELDS:
        old, new = np.asarray(start[name]), np.asarray(endpoint[name])
        if old.shape != new.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError("Finite matching old and trial endpoint traces are required")
        result[name] = (1 - fraction) * old + fraction * new
    return result


def trace_difference(first, second):
    result = {}
    for name in TRACE_FIELDS:
        delta = np.asarray(first[name]) - np.asarray(second[name])
        result[name] = {"maximum_absolute_component": float(np.max(np.abs(delta), initial=0)),
                        "rms_component": float(np.sqrt(np.mean(delta**2)))}
    return result


def gradient_snapshot(solver):
    """Capture native accepted velocity and pressure gradients independently."""
    return {"velocity_gradient": np.asarray(solver._velocity_gradient()).copy(),
            "pressure_gradient": _resolve_gradient_fn(solver.geo_data)(
                solver.kinematic_pressure, solver.mesh_data, solver.geo_data).copy()}


def gradient_difference(first, second):
    return {name: float(np.max(np.abs(second[name] - values), initial=0))
            for name, values in first.items()}


def run(capture_directory, directory):
    capture_directory, directory = Path(capture_directory).resolve(), Path(directory).resolve()
    if not directory.is_relative_to(Path("/tmp")):
        raise ValueError("Rollback verification requires a private /tmp directory")
    information = json.loads((capture_directory / "inputs/inputs.json").read_text())
    check_numerical_sources(information)
    for name, receipt in information["files"].items():
        if digest(capture_directory / "inputs" / name) != receipt["sha256"]:
            raise ValueError("Frozen study input changed: " + name)
    reference_receipt = json.loads((capture_directory / "reference/report.json").read_text())
    trace_path, initial_path = capture_directory / "reference_traces.h5", capture_directory / "initial_state.npz"
    trace_hash = reference_receipt["trace_sha256"]
    initial_hash = reference_receipt["initial_restriction"]["initial_state_sha256"]
    if digest(trace_path) != trace_hash or digest(initial_path) != initial_hash:
        raise ValueError("Immutable traces or mapped temporal state changed")
    if not np.isclose(information["step_size"], .008, rtol=0, atol=1e-12):
        raise ValueError("Control requires the production .008 s native FVM step")
    module = load_case_module(CASE)
    setup, _particles, _coupling, _mesh = module.build_case(
        end_time=information["end_time"], overrides={"cores": 1})
    if not np.isclose(module.VPM_TIME_STEP_SIZE, .04, rtol=0, atol=1e-12):
        raise ValueError("Control requires the production .04 s exchange")
    setup = quiet_setup(replace(setup, execution=reference_setup(information["end_time"]).execution))
    force = next(sampler for sampler in setup.samplers if isinstance(sampler, fvm.ForceSampler))
    setup = replace(setup, samplers=(), execution=replace(setup.execution, operator_backend="numba"))
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    trials, restorations = {}, []
    with FVMSolver(setup, directory, mesh_data=load_native_mesh(capture_directory / "inputs/small_mesh.npz")) as solver:
        solver.auto_write = False
        metadata = json.loads((directory / "solution/fvm_metadata.json").read_text())
        frozen_metadata = json.loads((capture_directory / "inputs/small_metadata.json").read_text())
        settings = validate_saved_settings(metadata, frozen_metadata)
        with np.load(initial_path, allow_pickle=False) as stored:
            fields = {name: stored[name].copy() for name in stored.files}
        restore_restart_state(solver, validate_restart_state(solver, fields))
        volumes = solver.geo_data["cell_volume"][:solver.mesh_data["n_cells"]]
        with h5py.File(trace_path, "r") as trace:
            if not trace.attrs.get("complete"):
                raise ValueError("Captured reference trace is incomplete")
            _indices, points, normals, areas = patch_geometry(solver.mesh_data, solver.geo_data)
            for name, value in (("face_centre", points), ("face_normal", normals), ("face_area", areas)):
                if not np.array_equal(trace[name][:], value):
                    raise ValueError("Native outer face geometry differs from immutable traces")
            endpoints = {label: {name: trace[name][index] for name in TRACE_FIELDS}
                         for label, index in (("start", 0), ("A", 5), ("B", 10))}
            imposed_difference = trace_difference(endpoints["A"], endpoints["B"])
            if max(values["maximum_absolute_component"] for values in imposed_difference.values()) < 1e-5:
                raise ValueError("The unequal trial traces are too similar for this causal control")
            impose(solver, endpoints["start"], "mixed_subcycled")
            initial = capture_restart_state(solver)
            initial_increments = fixed_flux_increments(solver)
            solver.save_state(directory / "initial_backup.npz")
            for label, endpoint_label in (("first_A", "A"), ("unequal_B", "B"), ("repeated_A", "A")):
                restore_restart_state(solver, initial)
                impose(solver, endpoints["start"], "mixed_subcycled")
                restored = restart_state_comparison(initial, capture_restart_state(solver), volumes)
                restored_increments = {
                    name: float(np.max(np.abs(values - initial_increments[name]), initial=0))
                    for name, values in fixed_flux_increments(solver).items()}
                if not restored["exact_state_equal"] or any(value != 0 for value in restored_increments.values()):
                    raise RuntimeError("Native rollback did not restore the identical accepted starting state")
                restorations.append({"trial": label, "state": restored,
                                     "fixed_flux_pressure_increment_max_abs_difference": restored_increments})
                substep_forces, gradients = [], []
                trial_started = perf_counter()
                for substep in range(1, 6):
                    prescribed = interpolated_trial_trace(endpoints["start"], endpoints[endpoint_label], substep)
                    impose(solver, prescribed, "mixed_subcycled")
                    solver.solve_pimple()
                    solver.advance_time(defer_output=True)
                    expected_time = initial.time + substep * information["step_size"]
                    if abs(solver.time - expected_time) > 1e-9 or solver.step != initial.step + substep:
                        raise RuntimeError("Trial did not use the same declared native accepted clock")
                    substep_forces.append(force_values(force, solver))
                    gradients.append(gradient_snapshot(solver))
                endpoint = (capture_restart_state(solver), fixed_flux_increments(solver), substep_forces)
                trials[label] = {"endpoint": endpoint, "gradients": gradients,
                                 "wall_seconds": perf_counter() - trial_started}
                solver.save_state(directory / (label + "_backup.npz"))
                print(f"{label} completed: {solver.time:.6f} s", flush=True)
            comparisons = {}
            for name in ("unequal_B", "repeated_A"):
                comparison = endpoint_comparison(trials["first_A"]["endpoint"], trials[name]["endpoint"], volumes)
                comparison["gradients_max_abs_difference"] = {
                    field: max(gradient_difference(first, second)[field]
                               for first, second in zip(trials["first_A"]["gradients"], trials[name]["gradients"], strict=True))
                    for field in trials["first_A"]["gradients"][0]}
                comparisons[name] = comparison
            final = comparisons["repeated_A"]
            repeated_difference = max(
                *final["fields_max_abs_difference"].values(),
                *final["fixed_flux_pressure_increment_max_abs_difference"].values(),
                *final["gradients_max_abs_difference"].values(),
                *(value for fields in final["force_max_abs_difference"].values() for value in fields.values()),
                final["eddy_viscosity_max_abs_difference"])
            b_difference = max(comparisons["unequal_B"]["fields_max_abs_difference"]["velocity"],
                               *(values["drag_coefficient"] for values in comparisons["unequal_B"]["force_max_abs_difference"].values()),
                               *(values["lift_coefficient"] for values in comparisons["unequal_B"]["force_max_abs_difference"].values()))
            if b_difference < 1e-6:
                raise RuntimeError("Unequal native trial did not measurably change fields or forces")
            finite = all(np.isfinite(getattr(solver, name)).all() for name in (*CELL_FIELDS, *FLUX_FIELDS))
    check_numerical_sources(information)
    if digest(trace_path) != trace_hash or digest(initial_path) != initial_hash:
        raise ValueError("Original immutable inputs changed during the rollback experiment")
    result = {
        "schema": "openonda-cylinder-asymmetric-rollback/1",
        "status": "complete" if repeated_difference <= 1e-10 and final["clock_and_history_controls_equal"] else "differences_detected",
        "operator_backend": "numba", "declared_trial_interval": [initial.time, initial.time + .04],
        "boundary_endpoint_source_times": {"A": float(information["start_time"] + .04), "B": float(information["start_time"] + .08)},
        "native_substep_size": .008, "native_substeps_per_trial": 5,
        "sequence": ["A", "restore initial", "B", "restore initial", "A"],
        "imposed_unequal_trace_difference": imposed_difference,
        "unequal_trial_field_or_coefficient_difference": b_difference,
        "restorations": restorations, "comparison_with_first_A": comparisons,
        "maximum_repeated_A_difference": repeated_difference,
        "trial_wall_seconds": {name: values["wall_seconds"] for name, values in trials.items()},
        "trial_forces": {name: values["endpoint"][2] for name, values in trials.items()},
        "finite_primary_and_temporal_fields": finite, "frozen_settings_comparison": settings,
        "wall_seconds": perf_counter() - started,
        "peak_process_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "trace_sha256": trace_hash, "initial_state_sha256": initial_hash,
        "source_sha256": information["source_sha256"], "verification_helper_sha256": digest(__file__),
        "scope": "FVM rollback/cache dependence under unequal mixed traces; no VPM evolution, transfer, feedback or mature force amplitude claim. Sampler and output dispatch deferred; native surface forces evaluated directly.",
    }
    (directory / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    options = parser.parse_args(argv)
    result = run(options.capture_directory, options.directory)
    print(json.dumps({name: result[name] for name in ("status", "maximum_repeated_A_difference", "unequal_trial_field_or_coefficient_difference", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
