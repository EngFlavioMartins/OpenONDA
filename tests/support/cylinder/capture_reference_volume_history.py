"""Stream native mature cylinder reference volumes at coupling endpoints.

The immutable reference checkpoint and numerical sources from a completed
boundary study are reused without a restart waiver. Only 0.04 s endpoints are
stored; the 0.008 s reference solver still advances normally. Outer traces,
forces and final primary/BDF fields must reproduce the original continuation.
Private output belongs in /tmp and never replaces the reference or its samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
from time import perf_counter

import h5py
import numpy as np

from source.solvers.fvm.core.solver import FVMSolver
from source.solvers.fvm.io.backup import config_hash
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from tests.support.cylinder.run_boundary_condition_study import (
    CASE,
    CELL_FIELDS,
    FLUX_FIELDS,
    REPOSITORY,
    Reconstruction,
    balanced_trace,
    check_numerical_sources,
    digest,
    patch_geometry,
    quiet_setup,
    read_native_state,
    reference_setup,
)
from tests.support.cylinder.run_coupled_checkpoint_control import read_force_history

SCHEMA = "openonda-cylinder-reference-volume-history/1"
GRADIENT_CONVENTION = "G[i,j]=d(velocity_j)/d(x_i); native FVM ordering"
TRACE_FIELDS = (
    "raw_velocity", "velocity", "jacobian", "normal_velocity",
    "tangential_gradient", "flux_correction",
)


def current_authored_input_hashes():
    """Collect current authored bytes without implying a historical capture."""
    return {
        str(path.relative_to(REPOSITORY)): digest(path)
        for path in (CASE / "setup.py", CASE / "assets/initial_conditions.py",
                     CASE / "reference_flow/setup.py", CASE / "reference_flow/assets/initial_conditions.py")
    }


def authored_input_provenance(information, newly_collected):
    """Separate hashes in the original receipt from newly collected evidence."""
    historically_recorded = "authored_inputs_sha256" in information
    return {
        "historical_hashes_recorded": historically_recorded,
        "historical_sha256": information.get("authored_inputs_sha256", {}),
        "newly_collected_sha256": newly_collected,
        "scope": ("The original study contains authored-input hashes; current hashes are collected separately."
                  if historically_recorded else
                  "The original study did not record authored-input hashes. These current hashes are newly collected; they do not establish historical byte equivalence."),
        "equation_equivalence_basis": "Strict original native configuration and numerical-source hashes, plus deterministic original outer traces, forces and final primary/BDF comparison.",
    }


def endpoint_steps(information, exchange_step_size):
    """Admit only a complete, uniform native FVM/coupling clock schedule."""
    dt = float(information["step_size"])
    exchange_dt = float(exchange_step_size)
    if not np.isfinite((dt, exchange_dt)).all() or min(dt, exchange_dt) <= 0:
        raise ValueError("Finite positive FVM and coupling time steps are required")
    substeps = round(exchange_dt / dt)
    steps = information["steps"]
    if (substeps < 1 or not np.isclose(substeps * dt, exchange_dt, rtol=0, atol=1e-12)
            or isinstance(steps, bool) or not isinstance(steps, int) or steps < 1
            or steps % substeps):
        raise ValueError("Reference horizon must contain complete native coupling intervals")
    if not np.isclose(information["start_time"] + steps * dt,
                      information["end_time"], rtol=0, atol=1e-9):
        raise ValueError("Reference horizon and accepted step count disagree")
    return np.arange(0, steps + 1, substeps, dtype=np.int64)


def bounded_difference(actual, expected, name, *, relative_bound=1e-10):
    """Reject a changed deterministic control, reporting the physical difference."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{name}: finite matching fields are required")
    maximum = float(np.max(np.abs(actual - expected), initial=0))
    scale = max(1., float(np.max(np.abs(expected), initial=0)))
    if maximum > relative_bound * scale:
        raise ValueError(f"{name}: reference determinism bound exceeded ({maximum:g})")
    return maximum


def validate_volume_history(stored, information, *, inspect_fields=False):
    """Validate the completed stream before an image consumer uses its fields."""
    if stored.attrs.get("schema") != SCHEMA or not stored.attrs.get("complete", False):
        raise ValueError("Reference volume history is incomplete or has an unsupported schema")
    if stored.attrs.get("gradient_convention") != GRADIENT_CONVENTION:
        raise ValueError("Reference gradient convention is not the native FVM ordering")
    endpoints = endpoint_steps(information, float(stored.attrs["exchange_step_size"]))
    frames, cells = len(endpoints), int(stored.attrs["cell_count"])
    if cells < 1:
        raise ValueError("Reference volume history has no physical donor cells")
    if stored.attrs.get("published_frame_count") != frames:
        raise ValueError("Reference volume history has incomplete accepted frame publication")
    for name, shape in {
        "time": (frames,), "step": (frames,), "accepted_time_step_size": (frames,),
        "velocity": (frames, cells, 3), "velocity_gradient": (frames, cells, 3, 3),
        "cell_centre": (cells, 3),
    }.items():
        if name not in stored or stored[name].shape != shape:
            raise ValueError(f"Reference volume history has invalid {name} coverage")
    if (stored["velocity"].dtype != np.dtype("float64")
            or stored["velocity_gradient"].dtype != np.dtype("float64")
            or stored["step"].dtype != np.dtype("int64")):
        raise ValueError("Reference volume history changed native field or clock precision")
    expected_time = information["start_time"] + endpoints * information["step_size"]
    bounded_difference(stored["time"][:], expected_time, "accepted endpoint clocks", relative_bound=1e-11)
    if not np.array_equal(stored["step"][:], information["start_step"] + endpoints):
        raise ValueError("Reference volume history has skipped or repeated accepted steps")
    if not np.allclose(stored["accepted_time_step_size"][:], information["step_size"], rtol=0, atol=1e-12):
        raise ValueError("Reference volume history changed the accepted native time step")
    if stored.attrs.get("configuration_hash") != information["reference_config_hash"]:
        raise ValueError("Reference volume history has a different native configuration")
    recorded_sources = json.loads(stored.attrs["source_sha256"])
    if recorded_sources != information["source_sha256"]:
        raise ValueError("Reference volume history has a different frozen numerical source tree")
    provenance = json.loads(stored.attrs["authored_input_provenance"])
    if (provenance["historical_hashes_recorded"] != ("authored_inputs_sha256" in information)
            or provenance["historical_sha256"] != information.get("authored_inputs_sha256", {})):
        raise ValueError("Reference volume history misstates historical physical-input evidence")
    if not provenance["newly_collected_sha256"]:
        raise ValueError("Reference volume history lacks newly collected authored-input hashes")
    for attribute, name in (("reference_mesh_archive_sha256", "reference_mesh.npz"),
                            ("native_reference_checkpoint_sha256", "reference_backup.npz")):
        if stored.attrs.get(attribute) != information["files"][name]["sha256"]:
            raise ValueError("Reference volume history has a different frozen mesh or checkpoint")
    if not stored.attrs.get("native_mesh_hash"):
        raise ValueError("Reference volume history lacks native ordered mesh identity")
    if not np.isfinite(stored["cell_centre"][:]).all():
        raise ValueError("Reference donor geometry is nonfinite")
    if inspect_fields:
        for index in range(frames):
            for name in ("velocity", "velocity_gradient"):
                if not np.isfinite(stored[name][index]).all():
                    raise ValueError(f"Reference volume history has nonfinite {name} at frame {index}")
    return {"frame_count": frames, "physical_cell_count": cells,
            "initial_time": float(stored["time"][0]), "final_time": float(stored["time"][-1]),
            "gradient_convention": GRADIENT_CONVENTION}


def compare_forces(candidate_path, original_path):
    candidate, original = read_force_history(candidate_path), read_force_history(original_path)
    if candidate.dtype.names != original.dtype.names or len(candidate) != len(original):
        raise ValueError("Reference force history coverage or columns changed during recapture")
    maximum = {}
    for name in candidate.dtype.names:
        if candidate.dtype[name].kind in "US":
            if not np.array_equal(candidate[name], original[name]):
                raise ValueError(f"Reference force history {name} labels changed")
        else:
            maximum[name] = bounded_difference(candidate[name], original[name], f"force history {name}")
    return {"record_count": len(candidate), "maximum_absolute_field_difference": maximum,
            "candidate_sha256": digest(candidate_path), "original_sha256": digest(original_path)}


def capture(capture_directory, directory, exchange_step_size=.04):
    capture_directory, directory = Path(capture_directory).resolve(), Path(directory).resolve()
    if not directory.is_relative_to(Path("/tmp")):
        raise ValueError("Reference volume recapture must use a private /tmp directory")
    information_path = capture_directory / "inputs/inputs.json"
    information = json.loads(information_path.read_text())
    endpoints = endpoint_steps(information, exchange_step_size)
    check_numerical_sources(information)
    current_input_hashes = current_authored_input_hashes()
    input_provenance = authored_input_provenance(information, current_input_hashes)
    frozen = capture_directory / "inputs"
    for name, receipt in information["files"].items():
        if digest(frozen / name) != receipt["sha256"]:
            raise ValueError(f"Frozen reference study input changed: {name}")
    original_report_path = capture_directory / "reference/report.json"
    original_report = json.loads(original_report_path.read_text())
    trace_path = capture_directory / "reference_traces.h5"
    initial_path = capture_directory / "initial_state.npz"
    if digest(trace_path) != original_report["trace_sha256"]:
        raise ValueError("Original immutable reference traces changed")
    initial_hash = original_report["initial_restriction"]["initial_state_sha256"]
    if digest(initial_path) != initial_hash:
        raise ValueError("Original mapped small-domain temporal state changed")
    metadata, _state = read_native_state(frozen / "reference_backup.npz")
    setup = quiet_setup(reference_setup(information["end_time"]))
    if config_hash(setup) != information["reference_config_hash"]:
        raise ValueError("Native reference settings changed; no restart waiver is permitted")
    mesh = load_native_mesh(frozen / "reference_mesh.npz")
    small_mesh = load_native_mesh(frozen / "small_mesh.npz")
    small_geometry = compute_mesh_geometry(small_mesh, compute_lsq=False, logger=None)
    _indices, points, normals, areas = patch_geometry(small_mesh, small_geometry)
    directory.mkdir(parents=True, exist_ok=False)
    output_path = directory / "reference_volume_history.h5"
    started = perf_counter()
    maximum = dict.fromkeys(TRACE_FIELDS, 0.)
    with FVMSolver(setup, directory / "reference", mesh_data=mesh) as solver:
        solver.auto_write = False
        solver.load_state(frozen / "reference_backup.npz")
        cells = mesh["n_cells"]
        probe = Reconstruction(solver.geo_data["cell_centre"], points, 12)
        with h5py.File(output_path, "x") as stored, h5py.File(trace_path, "r") as original:
            if not original.attrs.get("complete"):
                raise ValueError("Original reference trace is incomplete")
            for name, values in (("face_centre", points), ("face_normal", normals), ("face_area", areas)):
                if not np.array_equal(original[name][:], values):
                    raise ValueError("Recaptured reference uses different small-domain outer faces")
                stored.create_dataset(name, data=values)
            stored.attrs.update(
                schema=SCHEMA, complete=False, cell_count=cells,
                gradient_convention=GRADIENT_CONVENTION,
                exchange_step_size=exchange_step_size,
                fvm_step_size=information["step_size"],
                configuration_hash=metadata["config_hash"], native_mesh_hash=metadata["mesh_hash"],
                reference_mesh_archive_sha256=digest(frozen / "reference_mesh.npz"),
                native_reference_checkpoint_sha256=digest(frozen / "reference_backup.npz"),
                mapped_initial_state_sha256=initial_hash,
                source_sha256=json.dumps(information["source_sha256"], sort_keys=True),
                authored_input_provenance=json.dumps(input_provenance, sort_keys=True),
            )
            stored.create_dataset("cell_centre", data=solver.geo_data["cell_centre"])
            datasets = {}
            for name, shape in {
                "time": (), "step": (), "accepted_time_step_size": (),
                "velocity": (cells, 3), "velocity_gradient": (cells, 3, 3),
                "raw_velocity": (len(points), 3), "outer_velocity": (len(points), 3),
                "jacobian": (len(points), 3, 3), "normal_velocity": (len(points),),
                "tangential_gradient": (len(points), 3), "flux_correction": (),
            }.items():
                datasets[name] = stored.create_dataset(
                    name, shape=(len(endpoints), *shape), dtype=np.int64 if name == "step" else np.float64,
                    chunks=(1, *shape) if shape else None,
                )
            frame = 0
            for index in range(information["steps"] + 1):
                if index:
                    solver.advance()
                if index != endpoints[frame]:
                    continue
                expected_time = information["start_time"] + index * information["step_size"]
                if solver.step != information["start_step"] + index or abs(solver.time - expected_time) > 1e-9:
                    raise ValueError("Native reference accepted endpoint clock changed")
                velocity = np.asarray(solver.velocity[:cells])
                gradient = np.asarray(solver._velocity_gradient()[:cells])
                if not np.isfinite(velocity).all() or not np.isfinite(gradient).all():
                    raise ValueError("Native reference endpoint has nonfinite fields")
                raw_velocity = probe.apply(velocity)
                jacobian = probe.apply(gradient).swapaxes(1, 2)
                outer_velocity, un, tangent, balance = balanced_trace(raw_velocity, jacobian, normals, areas)
                values = {
                    "raw_velocity": raw_velocity, "velocity": outer_velocity,
                    "jacobian": jacobian, "normal_velocity": un,
                    "tangential_gradient": tangent, "flux_correction": balance["normal_velocity_correction"],
                }
                for name, value in values.items():
                    maximum[name] = max(maximum[name], bounded_difference(value, original[name][index], name))
                    datasets["outer_velocity" if name == "velocity" else name][frame] = value
                for name, value in {
                    "velocity": velocity, "velocity_gradient": gradient,
                    "time": solver.time, "step": solver.step,
                    "accepted_time_step_size": solver._accepted_time_step_size,
                }.items():
                    datasets[name][frame] = value
                frame += 1
                stored.attrs["published_frame_count"] = frame
                if frame % 25 == 0 or frame == len(endpoints):
                    stored.flush()
                    print(f"Reference volume endpoint accepted: {solver.time:.6f} s ({frame}/{len(endpoints)})", flush=True)
            if frame != len(endpoints):
                raise ValueError("Reference volume history lacks accepted endpoint coverage")
            final_checkpoint = Path(solver.save_state(directory / "reference/final_backup.npz"))
        final_time = solver.time
    forces = compare_forces(directory / "reference/samples/forces_history.csv",
                            capture_directory / "reference/samples/forces_history.csv")
    _original_metadata, original_state = read_native_state(capture_directory / "reference/final_backup.npz")
    _final_metadata, final_state = read_native_state(final_checkpoint)
    final_fields = {name: bounded_difference(final_state[name], original_state[name], f"final {name}")
                    for name in (*CELL_FIELDS, *FLUX_FIELDS)}
    check_numerical_sources(information)
    if current_authored_input_hashes() != current_input_hashes:
        raise ValueError("Newly hashed authored case inputs changed during recapture")
    if digest(initial_path) != initial_hash or digest(trace_path) != original_report["trace_sha256"]:
        raise ValueError("Original reference evidence changed during recapture")
    with h5py.File(output_path, "r+") as stored:
        stored.attrs["complete"] = True
        validation = validate_volume_history(stored, information, inspect_fields=True)
    report = {
        "schema": SCHEMA, "status": "complete", "history": str(output_path),
        "validation": validation, "accepted_time": final_time,
        "accepted_native_steps": information["steps"], "endpoint_count": len(endpoints),
        "wall_seconds": perf_counter() - started,
        "peak_process_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "history_sha256": digest(output_path), "history_bytes": output_path.stat().st_size,
        "final_checkpoint_sha256": digest(final_checkpoint),
        "reference_mesh_sha256": digest(frozen / "reference_mesh.npz"),
        "native_reference_config_hash": metadata["config_hash"], "native_mesh_hash": metadata["mesh_hash"],
        "maximum_outer_trace_difference": maximum, "force_determinism": forces,
        "final_primary_and_BDF_difference": final_fields,
        "source_sha256": information["source_sha256"],
        "authored_input_provenance": input_provenance,
        "equation_equivalence_verified": True,
        "original_input_schema_keys": sorted(information),
        "verification_helper_sha256": digest(__file__),
        "original_capture_inputs_sha256": digest(information_path),
        "mapped_initial_state_sha256": initial_hash,
        "scope": "Strict native reference continuation; streamed accepted coupling endpoints. No VPM evolution, numerical-source edits, or changed original fields/histories.",
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--exchange-step-size", type=float, default=.04)
    args = parser.parse_args(argv)
    result = capture(args.capture_directory, args.directory, args.exchange_step_size)
    print(json.dumps({key: result[key] for key in ("accepted_time", "endpoint_count", "wall_seconds", "history_bytes")}), flush=True)


if __name__ == "__main__":
    main()
