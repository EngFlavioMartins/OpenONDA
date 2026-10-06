"""Isolated strict native coupled continuation from the frozen 46 s cylinder state.

The default ``native`` mode runs the ordinary coupled algorithm from the
selected repository. Pass the original physical case directory explicitly
when executing this helper in an isolated candidate worktree.

The optional old-target mode repeats native FVM-to-VPM renewal after the
unchanged VPM advance, using the still-accepted FVM field at t_n. The ordinary boundary evaluation and Picard
iteration then run unchanged. This tests sensitivity to the advanced predictor
representation, but also introduces an extra inverse renewal and a lagged FVM
source. It is neither a pure absorption experiment nor a proposed production fix.

All numerical controls, solid/support/amplification guards and native checkpoint
validation remain unchanged. Only scientific output is reduced to FVM forces.
The instance wrapper exists exclusively in this verification process.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np

from openonda import coupler, fvm
from openonda.tutorial_runner import load_case_module
from source.coupler.backup import (
    BACKUP_FORMAT_VERSION,
    _resolve_checkpoint_path,
    checkpoint_path_hash,
    config_mapping_digest,
)
from source.coupler.parallel import collective_phase
from source.coupler.stable_renewal import vortex_invariants
from source.solvers.fvm.io.backup import FORMAT_VERSION, config_hash
from source.solvers.vpm.config.configuration_values import numerical_configuration
from source.solvers.vpm.config.restart_changes import validate_configuration_changes

from .analyze_force_cycles import FIELDS, complete_cycles

REPOSITORY = Path(__file__).resolve().parents[3]


OLD_TARGET_SCOPE = (
    "Old-target predictor-refresh ablation: unchanged native VPM advance, followed by one "
    "additional guarded native renewal from the accepted FVM t_n field before ordinary "
    "boundary evaluation and Picard. Extra inverse renewal and lagged source are confounded; "
    "this is not a production fix, pure absorption isolation or proof of the paper model."
)
NATIVE_SCOPE = (
    "Ordinary native coupled continuation from the immutable 46 s checkpoint using the "
    "selected repository's numerical sources. No predictor, transfer or boundary wrapper "
    "is installed. Authored physical/numerical settings and all native guards remain strict; "
    "only scientific output is reduced to FVM forces."
)
HISTORY_FIELDS = (
    "_velocity_boundary_condition_old",
    "_normal_velocity_boundary_condition_old",
    "_tangential_gradient_boundary_condition_old",
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def copy_unchanged(source, destination):
    expected = digest(source)
    shutil.copyfile(source, destination)
    if digest(source) != expected or digest(destination) != expected:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Input changed during checkpoint copy: " + str(source))
    return expected


def quiet_setup(setup):
    """Keep physical settings and forces while reducing private scientific output."""
    forces = tuple(sampler for sampler in setup.samplers if isinstance(sampler, fvm.ForceSampler))
    if len(forces) != 1:
        raise ValueError("The cylinder control requires exactly one authored native force sampler")
    return replace(
        setup,
        samplers=forces,
        logging=replace(setup.logging, console=False),
        time=replace(setup.time, output_schedule=fvm.RunSchedule(final_only=True)),
    )


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {name: json_value(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_report(path, report):
    path.write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n")


def read_force_history(path):
    """Read native forces with the patch name retained as text."""
    content = Path(path).read_bytes()
    if not content.endswith(b"\n"):
        content = content[: content.rfind(b"\n") + 1]
    values = np.atleast_1d(
        np.genfromtxt(io.BytesIO(content), delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    if not len(values) or "patch" not in (values.dtype.names or ()):
        raise ValueError("A nonempty native patch-force history is required")
    for name in values.dtype.names:
        field = values[name]
        if name == "patch":
            if field.dtype.kind not in "US" or np.any(field == ""):
                raise ValueError("Force-history patch names must be nonempty text")
        elif not np.issubdtype(field.dtype, np.number) or not np.isfinite(field).all():
            raise ValueError("Non-finite or nonnumeric native force field: " + name)
    if np.any(np.diff(values["time"]) <= 0):
        raise ValueError("Native cylinder force times must increase")
    return values


def source_hashes(case_directory):
    paths = [
        path
        for package in (REPOSITORY / "source", REPOSITORY / "openonda")
        for path in package.rglob("*")
        if path.is_file() and path.suffix in (".py", ".c", ".cpp", ".h", ".so", ".dylib")
    ]
    paths.extend((case_directory / "setup.py", case_directory / "assets/initial_conditions.py"))
    return {str(path.resolve()): digest(path) for path in paths}


def unchanged_sources(expected, case_directory):
    actual = source_hashes(case_directory)
    changed = sorted(
        name for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name)
    )
    if changed:
        raise RuntimeError("Numerical sources changed during the ablation: " + ", ".join(changed))


def prepare_inputs(snapshot, directory, flow, particles, coupling, case_directory):
    """Copy a complete hash-verified committed bundle without modifying its source."""
    target = directory / "inputs"
    target.mkdir(parents=True, exist_ok=False)
    checkpoint = snapshot / "checkpoint"
    metadata_path = checkpoint / "checkpoint_info.json"
    metadata_sha256 = digest(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("format_version") != BACKUP_FORMAT_VERSION
        or metadata.get("kind") != "openonda.coupled_backup"
        or metadata.get("time") != 46.0
        or metadata.get("coupling_step") != 1150
        or metadata.get("config_sha256") != config_mapping_digest(metadata["config"])
    ):
        raise ValueError("A complete current-format native 46 s coupled checkpoint is required")
    if json.loads(json.dumps(coupling.to_dict()["coupler"])) != metadata["config"]["coupler"]:
        raise ValueError("Current authored coupling settings differ from the frozen checkpoint")
    validate_configuration_changes(
        numerical_configuration(particles.numerics), metadata["config"]["vpm"]
    )
    if particles.numerics.compute_device != "AUTO" or particles.numerics.max_n_particles != 200000:
        raise ValueError("The diagnostic retains the original AUTO device and 200000 capacity")
    copied = target / "checkpoint"
    copied.mkdir()
    files, hashes = metadata["checkpoint_files"], metadata["file_sha256"]
    if (
        files.keys() != hashes.keys()
        or not {
            "fvm",
            "vpm",
            "vpm_vtu",
            "vpm_boundary_condition",
        }
        <= files.keys()
    ):
        raise ValueError("Frozen checkpoint does not declare a complete native bundle")
    inputs = {}
    for name, relative in files.items():
        original = _resolve_checkpoint_path(checkpoint, relative)
        destination = _resolve_checkpoint_path(copied, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path_hash(original) != hashes[name]:
            raise ValueError("Frozen native checkpoint file hash differs: " + name)
        if original.is_dir():
            shutil.copytree(original, destination)
        else:
            copy_unchanged(original, destination)
        if checkpoint_path_hash(destination) != hashes[name]:
            raise ValueError("Copied native checkpoint file hash differs: " + name)
        inputs[name] = {"source": str(original), "sha256": hashes[name]}
    copy_unchanged(metadata_path, copied / "checkpoint_info.json")
    inputs["metadata"] = {"source": str(metadata_path), "sha256": metadata_sha256}
    mesh = snapshot / "coupled_mesh.npz"
    inputs["mesh"] = {
        "source": str(mesh),
        "sha256": copy_unchanged(mesh, target / "mesh.npz"),
    }
    with np.load(_resolve_checkpoint_path(copied, files["fvm"]), allow_pickle=False) as state:
        fvm_metadata = json.loads(str(state["metadata"]))
        if fvm_metadata["format_version"] != FORMAT_VERSION or fvm_metadata[
            "config_hash"
        ] != config_hash(flow):
            raise ValueError("Current FVM equation settings differ from frozen native state")
    if digest(metadata_path) != metadata_sha256:
        raise RuntimeError("Frozen checkpoint changed during input copy")
    verify_original_inputs(inputs)
    information = {
        "source_snapshot": str(snapshot),
        "copied_checkpoint": str(copied),
        "inputs": inputs,
        "source_repository": str(REPOSITORY),
        "physical_case_directory": str(case_directory),
        "source_sha256": source_hashes(case_directory),
        "native_coupled_configuration_sha256": metadata["config_sha256"],
        "native_fvm_configuration_sha256": fvm_metadata["config_hash"],
        "initial_time": metadata["time"],
        "initial_coupling_step": metadata["coupling_step"],
    }
    write_report(target / "inputs.json", information)
    return information


def verify_original_inputs(inputs):
    for name, record in inputs.items():
        if checkpoint_path_hash(Path(record["source"])) != record["sha256"]:
            raise RuntimeError("Frozen original input changed: " + name)


def particle_statistics(vpm, bounds):
    position = np.asarray(vpm.particle_position, dtype=float)
    strength = np.asarray(vpm.particle_vortex_strength, dtype=float)
    if not np.isfinite(position).all() or not np.isfinite(strength).all():
        raise ValueError("Ablation phase contains non-finite native particle fields")
    inside = np.all((position >= bounds[::2]) & (position <= bounds[1::2]), axis=1)
    near_wall = np.linalg.norm(position[:, :2], axis=1) < 0.75
    result = {"time": float(vpm.time), "step": int(vpm.step), "regions": {}}
    for name, selected in (
        ("whole_particle_cloud", np.ones(len(position), dtype=bool)),
        ("inside_fvm_domain", inside),
        ("near_cylinder_r_below_0_75", near_wall),
    ):
        values = vortex_invariants(position[selected], strength[selected])
        result["regions"][name] = {
            "particle_count": int(np.count_nonzero(selected)),
            "absolute_axial_strength": float(np.sum(np.abs(strength[selected, 2]))),
            **json_value(values),
        }
    return result


def raw_boundary_values(vpm, points, normals, spacing):
    velocity, gradient = vpm.compute_velocity_and_tangential_normal_gradient_at_points(
        points, normals, particle_spacing=spacing
    )
    return {
        "velocity": np.asarray(velocity),
        "normal_velocity": np.einsum("ij,ij->i", velocity, normals),
        "tangential_gradient": np.asarray(gradient),
    }


@contextmanager
def old_target_predictor_refresh(owner, geometry, report, report_path, trace=None):
    """Install one process-local instance wrapper, restoring it after the test."""
    original = owner._advance_vpm
    points, normals, areas = geometry

    def measure(label, step, clock):
        result = owner.apply_vpm(particle_statistics, np.asarray(owner.fvm_box))
        if trace is not None and owner._is_master:
            values = raw_boundary_values(
                owner.vpm_solver, points, normals, owner.vpm_particle_spacing
            )
            group = trace.create_group(f"step_{step:06d}/{label}")
            group.attrs["time"] = clock
            for name, field in values.items():
                group.create_dataset(name, data=field)
            trace.flush()
        return result

    def advance(step, time_end):
        started = time.perf_counter()
        old_history = (
            {name: getattr(owner, name).copy() for name in HISTORY_FIELDS}
            if owner._is_master
            else None
        )
        before = measure("accepted_t_n", step, owner.fvm_solver.time)
        native_seconds = original(step, time_end)
        advanced = measure("native_advanced_t_n_plus_1", step, time_end)
        transfer_step = owner.vorticity_transfer.step
        refresh_started = time.perf_counter()
        try:
            transfer_result, transfer_seconds = owner._transfer_vorticity_to_vpm(*geometry)
        finally:
            owner.vorticity_transfer.step = transfer_step
        refreshed = measure("renewed_from_fvm_t_n", step, time_end)
        with collective_phase(owner._comm, "old-target ablation accepted history check"):
            if owner._is_master and any(
                not np.array_equal(getattr(owner, name), values)
                for name, values in old_history.items()
            ):
                raise RuntimeError("Old-target refresh changed the accepted old boundary trace")
        if owner._is_master:
            report["predictor_phases"].append(
                {
                    "coupling_step": step,
                    "predictor_time": time_end,
                    "unchanged_fvm_target_time": owner.fvm_solver.time,
                    "native_advance": before,
                    "after_native_advance": advanced,
                    "after_old_target_refresh": refreshed,
                    "native_advance_seconds": native_seconds,
                    "extra_transfer_seconds": transfer_seconds,
                    "extra_transfer_and_measurement_seconds": time.perf_counter() - refresh_started,
                    "native_extra_transfer_result": json_value(transfer_result),
                    "transfer_step_preserved": owner.vorticity_transfer.step == transfer_step,
                    "accepted_old_boundary_history_preserved": True,
                }
            )
            write_report(report_path, report)
        return time.perf_counter() - started

    owner._advance_vpm = advance
    try:
        yield
    finally:
        owner._advance_vpm = original


def accepted_summary(owner):
    return [
        {
            "step": record["step"],
            "time": record["time"],
            "timing_seconds": record["timing_seconds"],
            "backup_phase": record["backup_phase"],
            "interface_iteration": record["interface_iteration"],
        }
        for record in owner.coupling_diagnostics
    ]


def run(
    snapshot,
    directory,
    case_directory,
    steps,
    *,
    mode="native",
    boundary_traces=False,
    prepare_only=False,
):
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 300:
        raise ValueError("The ablation must be bounded to 1–300 accepted exchanges")
    if not directory.is_relative_to(Path("/tmp")):
        raise ValueError("Ablation solver outputs require a private /tmp directory")
    directory.mkdir(parents=True, exist_ok=False)
    scope = NATIVE_SCOPE if mode == "native" else OLD_TARGET_SCOPE
    if boundary_traces and mode == "native":
        raise ValueError("Extra raw boundary queries are exclusive to the old-target ablation")
    module = load_case_module(case_directory)
    flow, particles, coupling, _mesh = module.build_case()
    flow = quiet_setup(flow)
    particles = replace(
        particles, directory=directory, samplers=replace(particles.samplers, samples=())
    )
    information = prepare_inputs(snapshot, directory, flow, particles, coupling, case_directory)
    if prepare_only:
        print(
            json.dumps(
                {"status": "prepared", "directory": str(directory), "mode": mode, "scope": scope}
            )
        )
        return information
    report_path = directory / "continuation_report.json"
    report = {
        "status": "initializing",
        "mode": mode,
        "scope": scope,
        "restart": information,
        "steps_requested": steps,
        "predictor_phases": [],
        "accepted_exchanges": [],
        "boundary_trace_capture_enabled": boundary_traces,
        "boundary_trace_capture_is_extra_diagnostic_work": boundary_traces,
    }
    write_report(report_path, report)
    started = time.perf_counter()
    trace = None
    with coupler.create_coupler(
        flow, particles, coupling, mesh=directory / "inputs/mesh.npz", case_dir=directory
    ) as owner:
        try:
            owner.initialize()
            owner.fvm_solver.auto_write = False
            start = owner.load_backup(directory / "inputs/checkpoint")
            if not np.isclose(owner.fvm_solver.time, 46.0, rtol=0, atol=1e-9) or start != 1150:
                raise RuntimeError("Native restart did not restore the immutable 46 s state")
            geometry, _remaining_steps = owner._prepare_run()
            if boundary_traces and owner._is_master:
                trace = h5py.File(directory / "predictor_boundary_traces.h5", "x")
                for name, value in zip(
                    ("face_centre", "face_normal", "face_area"), geometry, strict=True
                ):
                    trace.create_dataset(name, data=value)
            report["status"] = "running"
            report["native_strict_restart_passed"] = True
            report["compute_device"] = owner.apply_vpm(lambda vpm: vpm.compute_device)
            if owner._is_master:
                write_report(report_path, report)
            if mode == "native":
                final = owner.solve(start_step=start, max_coupling_steps=steps, backup_at_stop=True)
            else:
                with old_target_predictor_refresh(owner, geometry, report, report_path, trace):
                    final = owner.solve(
                        start_step=start, max_coupling_steps=steps, backup_at_stop=True
                    )
            unchanged_sources(information["source_sha256"], case_directory)
            verify_original_inputs(information["inputs"])
            if owner._is_master:
                force_path = directory / "samples/forces_history.csv"
                force = read_force_history(force_path)
                factor = 2.0 / (
                    flow.transport.density
                    * flow.samplers[0].reference_velocity ** 2
                    * flow.samplers[0].reference_area
                )
                checkpoint_path = directory / "solution/backups/checkpoint_info.json"
                checkpoint = json.loads(checkpoint_path.read_text())
                for name, relative in checkpoint["checkpoint_files"].items():
                    if (
                        checkpoint_path_hash(checkpoint_path.parent / relative)
                        != checkpoint["file_sha256"][name]
                    ):
                        raise RuntimeError("Final native checkpoint hash differs: " + name)
                report.update(
                    status="complete",
                    accepted_time=owner.fvm_solver.time,
                    final_coupling_step=final,
                    accepted_new_exchanges=final - start,
                    accepted_exchanges=accepted_summary(owner),
                    forces=[
                        {name: json_value(row[name]) for name in force.dtype.names} for row in force
                    ],
                    complete_cycles={name: complete_cycles(force, name, factor) for name in FIELDS},
                    final_checkpoint_time=checkpoint["time"],
                    final_checkpoint_sha256=digest(checkpoint_path),
                    force_history_sha256=digest(force_path),
                    numerical_sources_and_original_inputs_unchanged=True,
                    wall_seconds_including_initialization=time.perf_counter() - started,
                )
                write_report(report_path, report)
        except BaseException as error:
            if owner._is_master:
                report.update(status="failed", error=repr(error))
                write_report(report_path, report)
            raise
        finally:
            if trace is not None:
                trace.close()
    print(
        json.dumps(
            {
                "status": report["status"],
                "accepted_time": report["accepted_time"],
                "report": str(report_path),
            }
        )
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--case-directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "old_target_refresh"), default="native")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--boundary-traces", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    options = parser.parse_args(argv)
    run(
        options.snapshot.resolve(),
        options.directory.resolve(),
        options.case_directory.resolve(),
        options.steps,
        mode=options.mode,
        boundary_traces=options.boundary_traces,
        prepare_only=options.prepare_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
