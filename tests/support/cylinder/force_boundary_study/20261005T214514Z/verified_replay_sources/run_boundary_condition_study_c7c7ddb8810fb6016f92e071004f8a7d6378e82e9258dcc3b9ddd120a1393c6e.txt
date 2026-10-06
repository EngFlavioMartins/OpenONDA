"""Isolated reference-driven tests of the cylinder's outer boundary condition.

The reference restart is admitted without a configuration waiver. Its accepted
velocity and native velocity gradient are recorded on the small FVM domain at
every FVM time step. Two independent FVM runs replay those same data: normal
velocity/tangential gradient, and full velocity. Both retain fixedFluxPressure.
No VPM evolution, transfer, or production output is involved in these controls.

Run sequentially under an external memory limit, for example::

    python -m tests.support.cylinder.run_boundary_condition_study \
        --directory /tmp/cylinder-boundary-study --duration 12 --mode all

``prepare`` only freezes inputs and checks the native reference configuration.
Initial fields and BDF/face-flux histories are restricted explicitly; they are
not misrepresented as an exact restart on a different mesh. Reconstruction
comparisons and matched-face counts quantify that restriction's limitations.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

from openonda import fvm
from openonda.tutorial_runner import load_case_module
from source.coupler.boundary import tangential_normal_velocity_gradient
from source.solvers.fvm.core.solver import FVMSolver
from source.solvers.fvm.io.backup import (
    FORMAT_VERSION,
    capture_restart_state,
    config_hash,
    decode_state,
    restore_restart_state,
    validate_restart_state,
)
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.sampling.fields import _PointProbe

from .analyze_force_cycles import FIELDS, complete_cycles, window_statistics
from .compare_drag_cycles import read_forces, summarize

REPOSITORY = Path(__file__).resolve().parents[3]
CASE = REPOSITORY / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
REPORTS = Path(__file__).resolve().parent / "force_boundary_study"
PATCH = "numericalBoundary"
CELL_FIELDS = ("velocity", "velocity_old", "velocity_older", "kinematic_pressure")
FLUX_FIELDS = (
    "volumetric_face_flux",
    "volumetric_face_flux_old",
    "volumetric_face_flux_older",
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def copy_unchanged(source, destination):
    """Copy immutable bytes and reject a changing source before publication."""
    before = digest(source)
    shutil.copyfile(source, destination)
    if digest(source) != before or digest(destination) != before:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Input changed while copying: {source}")
    return before


def read_native_state(path):
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError("Reference restart does not use the current native FVM format")
        state = decode_state({key: stored[key].copy() for key in stored.files})
    state.pop("metadata")
    return metadata, state


def check_numerical_sources(information):
    """Keep the captured reference and both controls on identical equations."""
    differences = [
        name
        for name, expected in information["source_sha256"].items()
        if not (REPOSITORY / name).is_file() or digest(REPOSITORY / name) != expected
    ]
    differences.extend(
        name
        for name, expected in information.get("authored_inputs_sha256", {}).items()
        if not (REPOSITORY / name).is_file() or digest(REPOSITORY / name) != expected
    )
    if differences:
        raise RuntimeError(
            "Numerical sources changed during the controlled study: " + ", ".join(differences)
        )


def reference_setup(end_time):
    module = load_case_module(CASE / "reference_flow")
    setup, _ = module.build_case(module.DEFAULT_NAME, module.DEFAULT_H, end_time=end_time, cores=1)
    return setup


def quiet_setup(setup):
    """Change output controls only; equation-affecting settings are retained."""
    force = next(sampler for sampler in setup.samplers if isinstance(sampler, fvm.ForceSampler))
    return replace(
        setup,
        samplers=(force,),
        logging=replace(setup.logging, console=False),
        time=replace(setup.time, output_schedule=fvm.RunSchedule(final_only=True)),
        backup=fvm.BackupConfig(schedule=None, write_at_end=False),
    )


def prepare(directory, duration):
    frozen = directory / "inputs"
    record = frozen / "inputs.json"
    if record.exists():
        data = json.loads(record.read_text())
        if data["duration"] != duration:
            raise ValueError("Existing frozen study has a different duration")
        for name, information in data["files"].items():
            if digest(frozen / name) != information["sha256"]:
                raise ValueError(f"Frozen study input changed: {name}")
        return data
    frozen.mkdir(parents=True, exist_ok=False)
    sources = {
        "reference_backup.npz": CASE / "reference_flow/solution/backup",
        "reference_mesh.npz": CASE / "reference_flow/solution/fvm/mesh.npz",
        "reference_metadata.json": CASE / "reference_flow/solution/fvm_metadata.json",
        "small_mesh.npz": CASE / "solution/fvm/mesh.npz",
        "small_metadata.json": CASE / "solution/fvm_metadata.json",
    }
    files = {
        name: {"source": str(source), "sha256": copy_unchanged(source, frozen / name)}
        for name, source in sources.items()
    }
    metadata, state = read_native_state(frozen / "reference_backup.npz")
    start = float(state["time"])
    step_size = float(state["time_step_size"])
    steps = round(duration / step_size)
    if steps < 1 or not np.isclose(steps * step_size, duration, rtol=0, atol=1e-11):
        raise ValueError("Duration must describe a positive integer number of native FVM steps")
    setup = reference_setup(start + duration)
    active_hash = config_hash(setup)
    if active_hash != metadata["config_hash"]:
        raise ValueError(
            "Reference end extension is incompatible with its immutable native settings"
        )
    source_hashes = {
        str(path.relative_to(REPOSITORY)): digest(path)
        for path in sorted((REPOSITORY / "source").rglob("*.py"))
    }
    data = {
        "schema": "openonda-cylinder-boundary-study/1",
        "duration": duration,
        "start_time": start,
        "end_time": start + duration,
        "start_step": int(state["step"]),
        "step_size": step_size,
        "steps": steps,
        "reference_config_hash": active_hash,
        "reference_native_format": metadata["format_version"],
        "source_sha256": source_hashes,
        "authored_inputs_sha256": {
            str(path.relative_to(REPOSITORY)): digest(path)
            for path in (
                CASE / "setup.py",
                CASE / "assets/initial_conditions.py",
                CASE / "reference_flow/setup.py",
                CASE / "reference_flow/assets/initial_conditions.py",
            )
        },
        "files": files,
        "scope": "Reference-forced single-process FVM controls; no VPM or production changes",
    }
    _json(record, data)
    return data


class Reconstruction:
    """The native affine sampler stencil, extended to arbitrary trailing axes."""

    def __init__(self, source_points, target_points, neighbours=12):
        self.source = np.asarray(source_points, dtype=np.float64)
        self.target = np.asarray(target_points, dtype=np.float64)
        probe = _PointProbe(self.target, k=neighbours, reconstruction="affine")
        self.indices, self.weights = probe._interpolation_stencil(self.source)
        self.distance, self.nearest = cKDTree(self.source).query(self.target)

    def apply(self, values):
        values = np.asarray(values)
        return np.einsum("nk,nk...->n...", self.weights, values[self.indices])


def error_statistics(candidate, comparison):
    difference = np.asarray(candidate) - np.asarray(comparison)
    flat = difference.reshape(len(difference), -1)
    return {
        "maximum_absolute_component_difference": float(np.max(np.abs(difference))),
        "rms_vector_difference": float(np.sqrt(np.mean(np.sum(flat**2, axis=1)))),
    }


def balanced_trace(velocity, jacobian, normals, areas, *, maximum_correction=1e-3):
    """Apply only the closed-surface constant normal-velocity correction."""
    velocity = np.asarray(velocity, dtype=np.float64).copy()
    raw = float(np.dot(np.einsum("ni,ni->n", velocity, normals), areas))
    correction = raw / float(np.sum(areas))
    if abs(correction) > maximum_correction:
        raise ValueError(f"Reference trace requires excessive flux correction: {correction:g} m/s")
    velocity -= correction * normals
    normal_velocity = np.einsum("ni,ni->n", velocity, normals)
    gradient = tangential_normal_velocity_gradient(jacobian, normals)
    return (
        velocity,
        normal_velocity,
        gradient,
        {
            "raw_flux_imbalance": raw,
            "normal_velocity_correction": correction,
            "corrected_flux_imbalance": float(np.dot(normal_velocity, areas)),
        },
    )


def patch_geometry(mesh, geometry):
    patch = next(item for item in mesh["boundary"] if item["name"] == PATCH)
    indices = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
    areas = geometry["face_area"][indices]
    normals = geometry["face_area_vector"][indices] / areas[:, None]
    return indices, geometry["face_centre"][indices], normals, areas


def restrict_initial_state(reference, small_mesh, small_geometry, output):
    """Save mapped primary/history fields, preserving matching native face fluxes."""
    count = reference.mesh_data["n_cells"]
    small_count = small_mesh["n_cells"]
    targets = np.concatenate(
        (
            small_geometry["cell_centre"],
            small_geometry["face_centre"][small_mesh["n_interior_faces"] :],
        )
    )
    primary = Reconstruction(reference.geo_data["cell_centre"], targets, 12)
    comparison = Reconstruction(reference.geo_data["cell_centre"], targets, 24)
    mapped = {}
    errors = {}
    near_body = np.linalg.norm(targets[:small_count, :2], axis=1) < 1.0
    for name in CELL_FIELDS:
        field = getattr(reference, name)[:count]
        mapped[name] = primary.apply(field)
        other = comparison.apply(field)
        errors[name] = {
            "all_fluid_cells": error_statistics(mapped[name][:small_count], other[:small_count]),
            "near_body_cells_r_lt_1": error_statistics(
                mapped[name][:small_count][near_body], other[:small_count][near_body]
            ),
        }
    # Wall ghosts must retain no slip; one periodic layer maps to its owner.
    for boundary in small_mesh["boundary"]:
        start, number = boundary["start_face"], boundary["n_faces"]
        rows = (
            small_mesh["n_cells"]
            + np.arange(start, start + number)
            - small_mesh["n_interior_faces"]
        )
        owners = small_mesh["owners"][start : start + number]
        if boundary["name"] == "cylinder":
            for name in CELL_FIELDS[:3]:
                mapped[name][rows] = 0.0
        elif boundary["name"] in ("zmin", "zmax"):
            for name in CELL_FIELDS:
                mapped[name][rows] = mapped[name][owners]
    face_reconstruction = Reconstruction(
        reference.geo_data["cell_centre"], small_geometry["face_centre"]
    )
    distance, closest = cKDTree(reference.geo_data["face_centre"]).query(
        small_geometry["face_centre"]
    )
    target_sf = small_geometry["face_area_vector"]
    reference_sf = reference.geo_data["face_area_vector"][closest]
    target_area = small_geometry["face_area"]
    reference_area = reference.geo_data["face_area"][closest]
    cosine = np.einsum("ni,ni->n", target_sf, reference_sf) / (target_area * reference_area)
    matching = (distance < 1e-5) & (np.abs(cosine) > 0.999999)
    velocity_names = ("velocity", "velocity_old", "velocity_older")
    for name, velocity_name in zip(FLUX_FIELDS, velocity_names, strict=True):
        source_velocity = getattr(reference, velocity_name)[:count]
        values = np.einsum("ni,ni->n", face_reconstruction.apply(source_velocity), target_sf)
        values[matching] = (
            getattr(reference, name)[closest[matching]]
            * target_area[matching]
            / reference_area[matching]
            * np.sign(cosine[matching])
        )
        for boundary in small_mesh["boundary"]:
            if boundary["name"] in ("cylinder", "zmin", "zmax"):
                values[boundary["start_face"] : boundary["start_face"] + boundary["n_faces"]] = 0.0
        mapped[name] = values
    mapped.update(
        time=np.asarray(reference.time),
        step=np.asarray(reference.step, dtype=np.int64),
        n_committed_time_steps=np.asarray(reference._n_committed_time_steps, dtype=np.int64),
        time_step_size=np.asarray(reference.time_step_size),
        accepted_time_step_size=np.asarray(reference._accepted_time_step_size),
        previous_time_step_size=np.asarray(reference._previous_time_step_size),
        max_courant_number=np.asarray(reference.max_courant_number),
        eddy_viscosity=np.empty(0, dtype=np.float64),
        n_consecutive_accepted_steps=np.asarray(
            [
                reference._n_consecutive_accepted_steps[name]
                for name in sorted(reference._n_consecutive_accepted_steps)
            ],
            dtype=np.int64,
        ),
    )
    np.savez(output, **mapped)
    return {
        "method": "Native affine reconstruction of each primary and BDF field; matching oriented face fluxes retained, other fluxes reconstructed from their matching velocity history",
        "comparison": "12-neighbour versus 24-neighbour reconstruction differences; estimates, not rigorous error bounds",
        "reconstruction_differences": errors,
        "fluid_cells": small_count,
        "maximum_nearest_source_cell_distance": float(primary.distance[:small_count].max()),
        "maximum_near_body_source_cell_distance": float(
            primary.distance[:small_count][near_body].max()
        ),
        "matching_oriented_native_faces": int(matching.sum()),
        "total_faces": int(small_mesh["n_faces"]),
        "matching_face_maximum_centre_distance": float(distance[matching].max()),
        "history_clock": {"step": reference.step, "time": reference.time},
        "initial_state_sha256": digest(output),
    }


def capture(directory, information):
    destination = directory / "reference"
    destination.mkdir(exist_ok=False)
    frozen = directory / "inputs"
    small_mesh = load_native_mesh(frozen / "small_mesh.npz")
    small_geometry = compute_mesh_geometry(small_mesh, compute_lsq=False, logger=None)
    _indices, points, normals, areas = patch_geometry(small_mesh, small_geometry)
    setup = quiet_setup(reference_setup(information["end_time"]))
    started = time.perf_counter()
    maximum = {
        "normal_velocity_correction": 0.0,
        "velocity_reconstruction_difference": 0.0,
        "gradient_reconstruction_difference": 0.0,
    }
    with FVMSolver(
        setup, destination, mesh_data=load_native_mesh(frozen / "reference_mesh.npz")
    ) as solver:
        solver.auto_write = False
        solver.load_state(frozen / "reference_backup.npz")
        initial_report = restrict_initial_state(
            solver, small_mesh, small_geometry, directory / "initial_state.npz"
        )
        probe = Reconstruction(solver.geo_data["cell_centre"], points, 12)
        comparison = Reconstruction(solver.geo_data["cell_centre"], points, 24)
        trace_path = directory / "reference_traces.h5"
        with h5py.File(trace_path, "x") as stored:
            stored.attrs["gradient_convention"] = (
                "J[i,j]=d(velocity_i)/d(x_j); native FVM gradient transposed"
            )
            stored.attrs["initial_state_sha256"] = digest(directory / "initial_state.npz")
            stored.create_dataset("face_centre", data=points)
            stored.create_dataset("face_normal", data=normals)
            stored.create_dataset("face_area", data=areas)
            datasets = {}
            for name, shape in {
                "time": (),
                "raw_velocity": (len(points), 3),
                "velocity": (len(points), 3),
                "jacobian": (len(points), 3, 3),
                "kinematic_pressure": (len(points),),
                "normal_velocity": (len(points),),
                "tangential_gradient": (len(points), 3),
                "flux_correction": (),
            }.items():
                datasets[name] = stored.create_dataset(
                    name, shape=(information["steps"] + 1, *shape), dtype=np.float64
                )
            for index in range(information["steps"] + 1):
                if index:
                    solver.advance()
                count = solver.mesh_data["n_cells"]
                velocity = probe.apply(solver.velocity[:count])
                raw_velocity = velocity.copy()
                jacobian = probe.apply(solver._velocity_gradient()[:count]).swapaxes(1, 2)
                velocity, normal_velocity, tangent, balance = balanced_trace(
                    velocity, jacobian, normals, areas
                )
                pressure = probe.apply(solver.kinematic_pressure[:count])
                datasets["time"][index] = solver.time
                for name, value in {
                    "raw_velocity": raw_velocity,
                    "velocity": velocity,
                    "jacobian": jacobian,
                    "kinematic_pressure": pressure,
                    "normal_velocity": normal_velocity,
                    "tangential_gradient": tangent,
                    "flux_correction": balance["normal_velocity_correction"],
                }.items():
                    datasets[name][index] = value
                maximum["normal_velocity_correction"] = max(
                    maximum["normal_velocity_correction"],
                    abs(balance["normal_velocity_correction"]),
                )
                maximum["velocity_reconstruction_difference"] = max(
                    maximum["velocity_reconstruction_difference"],
                    error_statistics(
                        probe.apply(solver.velocity[:count]),
                        comparison.apply(solver.velocity[:count]),
                    )["maximum_absolute_component_difference"],
                )
                maximum["gradient_reconstruction_difference"] = max(
                    maximum["gradient_reconstruction_difference"],
                    error_statistics(
                        jacobian,
                        comparison.apply(solver._velocity_gradient()[:count]).swapaxes(1, 2),
                    )["maximum_absolute_component_difference"],
                )
                if index % 125 == 0:
                    stored.flush()
                    print(f"Reference trace accepted: {solver.time:.6f} s", flush=True)
            stored.attrs["complete"] = True
        saved = Path(solver.save_state(destination / "final_backup.npz"))
        final_time = solver.time
    report = {
        "status": "complete",
        "accepted_time": final_time,
        "accepted_new_steps": information["steps"],
        "face_count": len(points),
        "step_size": information["step_size"],
        "wall_seconds": time.perf_counter() - started,
        "strict_native_reference_restart": True,
        "initial_restriction": initial_report,
        "maximum_trace_reconstruction_differences": maximum,
        "trace_sha256": digest(trace_path),
        "final_checkpoint_sha256": digest(saved),
    }
    _json(destination / "report.json", report)
    return report


def trace_at_step(trace, index, *, coupling_substeps=1):
    """Return direct traces or the native coupler's linear endpoint subcycling."""
    fields = ("velocity", "normal_velocity", "tangential_gradient")
    if (
        not isinstance(coupling_substeps, int)
        or isinstance(coupling_substeps, bool)
        or coupling_substeps < 1
    ):
        raise ValueError("Coupling substeps must be a positive integer")
    if index < 0 or index >= len(trace["time"]):
        raise ValueError("Requested trace step is outside captured data")
    if coupling_substeps == 1 or index == 0:
        return {name: np.asarray(trace[name][index]) for name in fields}
    lower = ((index - 1) // coupling_substeps) * coupling_substeps
    upper = lower + coupling_substeps
    if upper >= len(trace["time"]):
        raise ValueError("Coupling interval lacks its accepted endpoint trace")
    fraction = (index - lower) / coupling_substeps
    return {
        name: (1.0 - fraction) * np.asarray(trace[name][lower])
        + fraction * np.asarray(trace[name][upper])
        for name in fields
    }


def impose(solver, values, mode):
    if mode in ("mixed", "mixed_subcycled"):
        solver.set_normal_velocity_tangential_gradient_boundary_condition(
            values["normal_velocity"], values["tangential_gradient"], PATCH
        )
    else:
        solver.set_dirichlet_velocity_boundary_condition_vec(values["velocity"], PATCH)
    solver.set_flux_consistent_pressure_boundary_condition(PATCH)


def numerical_metadata(configuration):
    """Canonical equation settings from native saved metadata, without output controls."""

    def strip_types(value):
        if isinstance(value, dict):
            return {key: strip_types(item) for key, item in value.items() if key != "type"}
        if isinstance(value, list):
            return [strip_types(item) for item in value]
        return value

    value = strip_types(configuration)
    for name in ("output", "logging", "backup", "samplers"):
        value.pop(name, None)
    for name in ("end_time", "output_schedule"):
        value["time"].pop(name, None)
    # This is the intentionally varied arithmetic implementation. All other
    # execution controls and all discretization settings must still match.
    value["execution"].pop("operator_backend", None)
    return value


def validate_saved_settings(actual_metadata, frozen_metadata):
    actual = numerical_metadata(actual_metadata["configuration"])
    frozen = numerical_metadata(frozen_metadata["configuration"])
    if actual != frozen:
        changed = sorted(
            key for key in actual.keys() | frozen.keys() if actual.get(key) != frozen.get(key)
        )
        raise ValueError(
            "Replay equation settings differ from the frozen case: " + ", ".join(changed)
        )
    return {
        "matched": True,
        "canonical_settings_sha256": hashlib.sha256(
            json.dumps(actual, sort_keys=True).encode()
        ).hexdigest(),
        "operator_backend": actual_metadata["configuration"]["execution"]["operator_backend"],
        "comparison": "All equation settings match frozen native case metadata; output controls/horizon excluded and recorded operator backend deliberately varied.",
    }


def replay(directory, information, mode, operator_backend="reference"):
    reference_report = json.loads((directory / "reference/report.json").read_text())
    if digest(directory / "reference_traces.h5") != reference_report["trace_sha256"]:
        raise ValueError("Reference trace changed after capture")
    if (
        digest(directory / "initial_state.npz")
        != reference_report["initial_restriction"]["initial_state_sha256"]
    ):
        raise ValueError("Restricted initial fields changed after capture")
    label = mode + "_numba" if operator_backend == "numba" else mode
    destination = directory / label
    destination.mkdir(exist_ok=False)
    module = load_case_module(CASE)
    setup, _particles, _exchange, _builder = module.build_case(
        end_time=information["end_time"], overrides={"cores": 1}
    )
    # Match the reference's operator implementation; retain its pressure model.
    setup = quiet_setup(
        replace(setup, execution=reference_setup(information["end_time"]).execution)
    )
    if operator_backend != "reference":
        setup = replace(
            setup, execution=replace(setup.execution, operator_backend=operator_backend)
        )
    exchange_dt = float(module.VPM_TIME_STEP_SIZE)
    coupling_substeps = round(exchange_dt / information["step_size"])
    if coupling_substeps < 1 or not np.isclose(
        coupling_substeps * information["step_size"], exchange_dt, rtol=0, atol=1e-12
    ):
        raise ValueError("Authored coupling interval is incompatible with the captured FVM clock")
    if mode == "mixed_subcycled" and information["steps"] % coupling_substeps:
        raise ValueError("Subcycled replay requires complete native coupling intervals")
    if mode != "mixed_subcycled":
        coupling_substeps = 1
    started = time.perf_counter()
    with FVMSolver(
        setup, destination, mesh_data=load_native_mesh(directory / "inputs/small_mesh.npz")
    ) as solver:
        solver.auto_write = False
        saved_metadata = json.loads((destination / "solution/fvm_metadata.json").read_text())
        frozen_metadata = json.loads((directory / "inputs/small_metadata.json").read_text())
        settings_comparison = validate_saved_settings(saved_metadata, frozen_metadata)
        with np.load(directory / "initial_state.npz", allow_pickle=False) as stored:
            state = {name: stored[name].copy() for name in stored.files}
        accepted = validate_restart_state(solver, state)
        restore_restart_state(solver, accepted)
        with h5py.File(directory / "reference_traces.h5", "r") as trace:
            if not trace.attrs.get("complete"):
                raise ValueError("Reference trace is incomplete")
            _, points, normals, areas = patch_geometry(solver.mesh_data, solver.geo_data)
            if (
                not np.array_equal(points, trace["face_centre"][:])
                or not np.array_equal(normals, trace["face_normal"][:])
                or not np.array_equal(areas, trace["face_area"][:])
            ):
                raise ValueError("Replay outer faces differ from captured reference trace")
            impose(solver, trace_at_step(trace, 0, coupling_substeps=coupling_substeps), mode)
            solver.save_state(destination / "initial_backup.npz")
            maximum_normal_mismatch = 0.0
            maximum_temporal_normal_difference = 0.0
            maximum_temporal_gradient_difference = 0.0
            for index in range(1, information["steps"] + 1):
                prescribed = trace_at_step(trace, index, coupling_substeps=coupling_substeps)
                impose(solver, prescribed, mode)
                maximum_temporal_normal_difference = max(
                    maximum_temporal_normal_difference,
                    float(
                        np.max(
                            np.abs(prescribed["normal_velocity"] - trace["normal_velocity"][index])
                        )
                    ),
                )
                maximum_temporal_gradient_difference = max(
                    maximum_temporal_gradient_difference,
                    float(
                        np.max(
                            np.abs(
                                prescribed["tangential_gradient"]
                                - trace["tangential_gradient"][index]
                            )
                        )
                    ),
                )
                solver.advance()
                if abs(solver.time - float(trace["time"][index])) > 1e-9:
                    raise RuntimeError("Replay accepted clock differs from reference trace")
                patch = next(item for item in solver.boundaries if item["name"] == PATCH)
                start = (
                    patch["start_face"]
                    - solver.mesh_data["n_interior_faces"]
                    + solver.mesh_data["n_cells"]
                )
                ghost = solver.velocity[start : start + patch["n_faces"]]
                maximum_normal_mismatch = max(
                    maximum_normal_mismatch,
                    float(
                        np.max(
                            np.abs(
                                np.einsum("ni,ni->n", ghost, normals)
                                - prescribed["normal_velocity"]
                            )
                        )
                    ),
                )
                if index % 125 == 0:
                    print(f"{label} replay accepted: {solver.time:.6f} s", flush=True)
        saved = Path(solver.save_state(destination / "final_backup.npz"))
        final_time = solver.time
        final_finite = all(
            np.isfinite(getattr(solver, name)).all() for name in (*CELL_FIELDS, *FLUX_FIELDS)
        )
    report = {
        "status": "complete",
        "accepted_time": final_time,
        "accepted_new_steps": information["steps"],
        "mode": mode,
        "label": label,
        "pressure_condition": "fixedFluxPressure",
        "operator_backend": setup.execution.operator_backend,
        "frozen_settings_comparison": settings_comparison,
        "wall_seconds": time.perf_counter() - started,
        "maximum_normal_velocity_mismatch": maximum_normal_mismatch,
        "trace_schedule": {
            "method": "linear interpolation of accepted coupling endpoint traces"
            if mode == "mixed_subcycled"
            else "reference trace at every accepted FVM endpoint",
            "fvm_step_size": information["step_size"],
            "coupling_step_size": exchange_dt if mode == "mixed_subcycled" else None,
            "fvm_substeps_per_coupling_interval": coupling_substeps,
            "maximum_normal_velocity_difference_from_stepwise_reference": maximum_temporal_normal_difference,
            "maximum_tangential_gradient_component_difference_from_stepwise_reference": maximum_temporal_gradient_difference,
        },
        "finite_primary_and_temporal_fields": final_finite,
        "trace_sha256": reference_report["trace_sha256"],
        "initial_state_sha256": reference_report["initial_restriction"]["initial_state_sha256"],
        "final_checkpoint_sha256": digest(saved),
    }
    _json(destination / "report.json", report)
    return report


def restart_state_comparison(reference, actual, volumes):
    """Compare complete accepted states, separating a constant pressure gauge."""
    fields = {
        name: float(np.max(np.abs(actual.fields[name] - values), initial=0))
        for name, values in reference.fields.items()
    }
    pressure_difference = (
        actual.fields["kinematic_pressure"] - reference.fields["kinematic_pressure"]
    )
    pressure_gauge = float(np.average(pressure_difference[: len(volumes)], weights=volumes))
    scalar_names = (
        "time",
        "step",
        "n_committed_time_steps",
        "time_step_size",
        "accepted_time_step_size",
        "previous_time_step_size",
        "kinematic_viscosity",
        "max_courant_number",
    )
    scalar_difference = {
        name: abs(float(getattr(actual, name)) - float(getattr(reference, name)))
        for name in scalar_names
    }
    controls_equal = (
        all(scalar_difference[name] == 0 for name in scalar_names if name != "max_courant_number")
        and actual.n_consecutive_accepted_steps == reference.n_consecutive_accepted_steps
    )
    return {
        "fields_max_abs_difference": fields,
        "pressure_gauge_difference": pressure_gauge,
        "gauge_aligned_pressure_max_abs_difference": float(
            np.max(np.abs(pressure_difference - pressure_gauge), initial=0)
        ),
        "eddy_viscosity_max_abs_difference": float(
            np.max(np.abs(actual.eddy_viscosity - reference.eddy_viscosity), initial=0)
        ),
        "scalar_abs_difference": scalar_difference,
        "clock_and_history_controls_equal": controls_equal,
        "exact_state_equal": bool(
            controls_equal
            and scalar_difference["max_courant_number"] == 0
            and all(value == 0 for value in fields.values())
            and np.array_equal(actual.eddy_viscosity, reference.eddy_viscosity)
        ),
    }


def fixed_flux_increments(solver):
    """Copy accepted pressure increments from every fixed-flux patch."""
    return {
        patch["name"]: np.asarray(patch["fixed_flux_pressure_delta"]).copy()
        for patch in solver.boundaries
        if patch.get("pressure_type") == "fixedFluxPressure"
    }


def force_values(sampler, solver):
    """Evaluate native surface loads directly, without appending sample histories."""
    return {
        patch: {
            **{name: float(value) for name, value in loads["coeffs"].items()},
            **{
                f"{name}_{axis}": float(value)
                for name in ("pressure_force", "viscous_force", "total_force")
                for axis, value in zip("xyz", loads[name], strict=True)
            },
        }
        for patch, loads in sampler.sample(solver).items()
    }


def endpoint_comparison(reference, actual, volumes):
    """Compare native state, pressure increments and every substep's surface loads."""
    reference_state, reference_increments, reference_forces = reference
    actual_state, actual_increments, actual_forces = actual
    comparison = restart_state_comparison(reference_state, actual_state, volumes)
    comparison["fixed_flux_pressure_increment_max_abs_difference"] = {
        name: float(np.max(np.abs(values - reference_increments[name]), initial=0))
        for name, values in actual_increments.items()
    }
    comparison["force_max_abs_difference"] = {
        patch: {
            name: max(
                abs(values[patch][name] - reference[patch][name])
                for values, reference in zip(actual_forces, reference_forces, strict=True)
            )
            for name in actual_forces[0][patch]
        }
        for patch in actual_forces[0]
    }
    return comparison


def rollback_replay(directory, information, exchanges=2, trials=4, operator_backend="numba"):
    """Replay identical native coupling intervals repeatedly from accepted state.

    This isolates the FVM side of ordinary Picard rollback. Every trial uses
    the same reference endpoint traces and the native momentum/pressure
    solvers, retaining their production preconditioner reuse. Only the final
    trial becomes the next accepted interval's starting state. A separate
    one-trial trajectory begins from the same initial state for comparison.
    No VPM transfer or predictor trial is performed, and all sampler/output
    dispatch is deferred.
    """
    if exchanges not in (2, 5) or trials != 4:
        raise ValueError("Rollback coverage requires two or five exchanges and four trials")
    reference_report = json.loads((directory / "reference/report.json").read_text())
    if digest(directory / "reference_traces.h5") != reference_report["trace_sha256"]:
        raise ValueError("Reference trace changed after capture")
    initial_sha256 = reference_report["initial_restriction"]["initial_state_sha256"]
    if digest(directory / "initial_state.npz") != initial_sha256:
        raise ValueError("Restricted initial fields changed after capture")
    label = f"rollback_{operator_backend}_{exchanges}_exchanges_{trials}_trials"
    destination = directory / label
    destination.mkdir(exist_ok=False)
    module = load_case_module(CASE)
    setup, _particles, _exchange, _builder = module.build_case(
        end_time=information["end_time"], overrides={"cores": 1}
    )
    setup = quiet_setup(
        replace(setup, execution=reference_setup(information["end_time"]).execution)
    )
    force = next(sampler for sampler in setup.samplers if isinstance(sampler, fvm.ForceSampler))
    setup = replace(
        setup,
        samplers=(),
        execution=replace(setup.execution, operator_backend=operator_backend),
    )
    coupling_substeps = round(module.VPM_TIME_STEP_SIZE / information["step_size"])
    if coupling_substeps != 5 or not np.isclose(
        coupling_substeps * information["step_size"], module.VPM_TIME_STEP_SIZE, atol=1e-12
    ):
        raise ValueError("Rollback control requires the native five-substep cylinder interval")
    if exchanges * coupling_substeps > information["steps"]:
        raise ValueError("Captured reference trace is too short for the rollback control")
    interval_records = []
    started = time.perf_counter()
    with FVMSolver(
        setup, destination, mesh_data=load_native_mesh(directory / "inputs/small_mesh.npz")
    ) as solver:
        solver.auto_write = False
        metadata = json.loads((destination / "solution/fvm_metadata.json").read_text())
        frozen_metadata = json.loads((directory / "inputs/small_metadata.json").read_text())
        settings_comparison = validate_saved_settings(metadata, frozen_metadata)
        with np.load(directory / "initial_state.npz", allow_pickle=False) as stored:
            state = {name: stored[name].copy() for name in stored.files}
        restore_restart_state(solver, validate_restart_state(solver, state))
        volumes = solver.geo_data["cell_volume"][: solver.mesh_data["n_cells"]]
        with h5py.File(directory / "reference_traces.h5", "r") as trace:
            if not trace.attrs.get("complete"):
                raise ValueError("Reference trace is incomplete")
            _, points, normals, areas = patch_geometry(solver.mesh_data, solver.geo_data)
            if any(
                not np.array_equal(values, trace[name][:])
                for name, values in (
                    ("face_centre", points),
                    ("face_normal", normals),
                    ("face_area", areas),
                )
            ):
                raise ValueError("Rollback outer faces differ from the frozen reference trace")
            impose(solver, trace_at_step(trace, 0, coupling_substeps=5), "mixed_subcycled")
            solver.save_state(destination / "initial_backup.npz")

            def advance_interval(exchange):
                substep_forces = []
                for substep in range(1, coupling_substeps + 1):
                    index = exchange * coupling_substeps + substep
                    impose(
                        solver,
                        trace_at_step(trace, index, coupling_substeps=5),
                        "mixed_subcycled",
                    )
                    solver.solve_pimple()
                    solver.advance_time(defer_output=True)
                    if abs(solver.time - float(trace["time"][index])) > 1e-9:
                        raise RuntimeError("Rollback replay clock differs from frozen trace")
                    substep_forces.append(force_values(force, solver))
                return (
                    capture_restart_state(solver),
                    fixed_flux_increments(solver),
                    substep_forces,
                )

            initial = capture_restart_state(solver)
            single_trial_trajectory = [advance_interval(exchange) for exchange in range(exchanges)]
            restore_restart_state(solver, initial)
            if not restart_state_comparison(initial, capture_restart_state(solver), volumes)[
                "exact_state_equal"
            ]:
                raise RuntimeError("Native rollback changed the control's initial state")
            for exchange in range(exchanges):
                interval_start = capture_restart_state(solver)
                start_increments = fixed_flux_increments(solver)
                first_endpoint = None
                records = []
                for trial in range(trials):
                    if trial:
                        restore_restart_state(solver, interval_start)
                        restored = restart_state_comparison(
                            interval_start, capture_restart_state(solver), volumes
                        )
                        increments = fixed_flux_increments(solver)
                        restored_increments = all(
                            np.allclose(values, increments[name], rtol=0, atol=1e-14)
                            for name, values in start_increments.items()
                        )
                        if not restored["exact_state_equal"] or not restored_increments:
                            raise RuntimeError("Native Picard rollback changed accepted state")
                    endpoint = advance_interval(exchange)
                    if first_endpoint is None:
                        first_endpoint = endpoint
                    comparison = endpoint_comparison(first_endpoint, endpoint, volumes)
                    records.append(
                        {
                            "trial": trial + 1,
                            "accepted_endpoint_time": endpoint[0].time,
                            "restored_exact_accepted_state": trial == 0
                            or restored["exact_state_equal"],
                            "restored_fixed_flux_increments": trial == 0 or restored_increments,
                            "comparison_with_first_trial": comparison,
                            "substep_forces": endpoint[2],
                        }
                    )
                interval_records.append(
                    {
                        "exchange": exchange + 1,
                        "trials": records,
                        "comparison_with_single_trial_trajectory": endpoint_comparison(
                            single_trial_trajectory[exchange], endpoint, volumes
                        ),
                    }
                )
                print(f"rollback interval accepted: {solver.time:.6f} s", flush=True)
            saved = Path(solver.save_state(destination / "final_backup.npz"))
            final_time = solver.time
            finite = all(
                np.isfinite(getattr(solver, name)).all() for name in (*CELL_FIELDS, *FLUX_FIELDS)
            )
    report = {
        "status": "complete",
        "operator_backend": operator_backend,
        "accepted_time": final_time,
        "accepted_new_exchanges": exchanges,
        "fvm_substeps_per_exchange": coupling_substeps,
        "identical_trials_per_exchange": trials,
        "independent_single_trial_trajectory": True,
        "finite_primary_and_temporal_fields": finite,
        "sampler_and_output_dispatch_deferred": True,
        "frozen_settings_comparison": settings_comparison,
        "trace_schedule": "linear interpolation of immutable 0.04 s endpoint traces",
        "trial_comparison": "Each trial compared with the first; fourth trial supplies the next interval's state. Pressure is reported raw and after removing its volume-weighted constant gauge.",
        "intervals": interval_records,
        "wall_seconds": time.perf_counter() - started,
        "trace_sha256": reference_report["trace_sha256"],
        "initial_state_sha256": initial_sha256,
        "final_checkpoint_sha256": digest(saved),
    }
    _json(destination / "report.json", report)
    return report


def compare(directory, information, settling, report_directory):
    start = information["start_time"] + settling
    end = information["end_time"]
    histories = {}
    reports = {}
    cycle_statistics = {}
    modes = ["reference", "mixed", "full_velocity"]
    modes.extend(
        path.parent.name
        for path in sorted(directory.glob("*/report.json"))
        if path.parent.name not in modes and path.parent.name.startswith(("mixed", "full_velocity"))
    )
    for mode in modes:
        path = directory / mode / "samples/forces_history.csv"
        values, force_digest = read_forces(path)
        histories[mode] = summarize(values, start, end)
        histories[mode].update(force_history_sha256=force_digest, source=str(path))
        reports[mode] = json.loads((directory / mode / "report.json").read_text())
        metadata = json.loads((directory / mode / "solution/fvm_metadata.json").read_text())
        frozen_name = "reference_metadata.json" if mode == "reference" else "small_metadata.json"
        reports[mode]["frozen_settings_comparison"] = validate_saved_settings(
            metadata, json.loads((directory / "inputs" / frozen_name).read_text())
        )
        configuration = metadata["configuration"]
        force_sampler = next(
            sampler for sampler in configuration["samplers"] if sampler["type"] == "ForceSampler"
        )
        factor = 2.0 / (
            configuration["transport"]["density"]
            * force_sampler["reference_velocity"] ** 2
            * force_sampler["reference_area"]
        )
        for field, axis in zip(FIELDS, ("x", "y"), strict=True):
            if np.max(np.abs(values[field] - factor * values[f"total_force_{axis}"])) > 1e-10:
                raise ValueError(f"Inconsistent dimensional force normalization in {mode}")
        if histories[mode]["requested_interval_covered"]:
            cycles = {field: complete_cycles(values, field, factor) for field in FIELDS}
            cycle_statistics[mode] = {
                "covered_window": True,
                "statistics": window_statistics(values, cycles, start, end, factor),
            }
        else:
            cycle_statistics[mode] = {
                "covered_window": False,
                "statistics": None,
                "limitation": "Requested window is only partially covered; no complete-window amplitude comparison is admitted.",
            }
    result = {
        "schema": "openonda-cylinder-boundary-study/1",
        "directory": str(directory),
        "statistics_window": [start, end],
        "initial_settling_seconds_excluded": settling,
        "scope": "Identical reference traces imposed on two isolated small-domain FVM runs. Reconstruction and initial-history remapping remain measured limitations; saturation is not inferred.",
        "series": histories,
        "complete_cycle_statistics": cycle_statistics,
        "cycle_definition": "A peak bracketed by troughs entirely within the requested window; Cl and Cd are measured separately. Whole-window ranges remain separately labelled.",
        "runs": reports,
        "frozen_inputs": information,
    }
    for mode in modes[1:]:
        ref, candidate = histories["reference"]["window"], histories[mode]["window"]
        if ref is not None and candidate is not None:
            result[mode + "_relative_to_reference"] = {
                metric: candidate[metric] / ref[metric]
                if candidate[metric] is not None and ref[metric] not in (None, 0)
                else None
                for metric in (
                    "mean_drag_coefficient",
                    "mean_drag_peak_to_peak_detrended",
                    "lift_peak_to_peak",
                )
            }
        reference_cycles = cycle_statistics["reference"]["statistics"]
        candidate_cycles = cycle_statistics[mode]["statistics"]
        if reference_cycles is not None and candidate_cycles is not None:
            result[mode + "_complete_cycle_relative_to_reference"] = {
                field: {
                    metric: candidate_cycles[field][metric] / reference_cycles[field][metric]
                    if candidate_cycles[field][metric] is not None
                    and reference_cycles[field][metric] not in (None, 0)
                    else None
                    for metric in (
                        "median_peak_to_peak_raw",
                        "median_peak_to_peak_drift_corrected",
                    )
                }
                for field in FIELDS
            }
    _json(directory / "comparison.json", result)
    report_directory.mkdir(parents=True, exist_ok=True)
    _json(report_directory / (directory.name + ".json"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--settling", type=float, default=2.0)
    parser.add_argument(
        "--mode",
        choices=(
            "prepare",
            "capture",
            "mixed",
            "mixed_subcycled",
            "rollback",
            "full_velocity",
            "compare",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--report-directory", type=Path, default=REPORTS)
    parser.add_argument(
        "--operator-backend", choices=("reference", "numpy", "numba"), default="reference"
    )
    parser.add_argument("--rollback-exchanges", type=int, choices=(2, 5), default=2)
    parser.add_argument("--rollback-trials", type=int, choices=(4,), default=4)
    options = parser.parse_args(argv)
    if (
        not np.isfinite(options.duration)
        or options.duration <= 0
        or not 0 <= options.settling < options.duration
    ):
        raise ValueError("Duration must be positive and settling must lie inside it")
    directory = options.directory.resolve()
    if not directory.is_relative_to(Path("/tmp")):
        raise ValueError("Diagnostic solver outputs must use a private /tmp directory")
    directory.mkdir(parents=True, exist_ok=True)
    if options.mode == "all" and options.operator_backend == "numba":
        raise ValueError(
            "Run the original reference controls first, then an explicit Numba replay mode"
        )
    information = prepare(directory, options.duration)
    print(
        f"Frozen reference: {information['start_time']:.6f} s; {information['steps']} native FVM steps",
        flush=True,
    )
    phases = (
        ("capture", "mixed", "full_velocity", "compare")
        if options.mode == "all"
        else (options.mode,)
    )
    for phase in phases:
        check_numerical_sources(information)
        if phase == "capture":
            capture(directory, information)
        elif phase in ("mixed", "mixed_subcycled", "full_velocity"):
            replay(directory, information, phase, operator_backend=options.operator_backend)
        elif phase == "rollback":
            backend = (
                "numba" if options.operator_backend == "reference" else options.operator_backend
            )
            rollback_replay(
                directory,
                information,
                exchanges=options.rollback_exchanges,
                trials=options.rollback_trials,
                operator_backend=backend,
            )
        elif phase == "compare":
            result = compare(
                directory, information, options.settling, options.report_directory.resolve()
            )
            print(
                json.dumps(
                    {key: value for key, value in result.items() if "relative_to_reference" in key}
                ),
                flush=True,
            )
        check_numerical_sources(information)
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
