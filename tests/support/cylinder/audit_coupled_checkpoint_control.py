"""Read-only audit of a strict single-rank native cylinder continuation.

Require current coupled format 13 and FVM format 10. Decode every primary and
BDF history field without constructing a solver. Scientific outputs and the
source/input hashes recorded by run_coupled_checkpoint_control remain strict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from source.coupler.backup import checkpoint_path_hash, config_mapping_digest
from source.coupler.geometry import SolidBoundary, TriangulatedWall
from source.solvers.fvm.io.backup import decode_state, mesh_hash
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

from .run_coupled_checkpoint_control import json_value, read_force_history

FVM_FIELDS = (
    "velocity", "velocity_old", "velocity_older", "kinematic_pressure",
    "volumetric_face_flux", "volumetric_face_flux_old", "volumetric_face_flux_older",
    "eddy_viscosity",
)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_state(path):
    with np.load(path, allow_pickle=False) as stored:
        return decode_state({name: stored[name].copy() for name in stored.files})


def declared_path(directory, relative):
    relative = Path(relative)
    result = (directory / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not result.is_relative_to(directory):
        raise ValueError("Native checkpoint path escapes its directory")
    return result


def native_wall_geometry(mesh):
    """Use the same stationary wall triangles as native coupled initialization."""
    triangles = []
    vertices = mesh["vertex_position"]
    for patch in mesh["boundary"]:
        if patch["type"] != "wall":
            continue
        for face in range(patch["start_face"], patch["start_face"] + patch["n_faces"]):
            indices = np.asarray(mesh["faces"][face], dtype=int)
            polygon = vertices[indices[indices >= 0]]
            centre = polygon.mean(axis=0)
            for edge in range(len(polygon)):
                triangle = np.array([centre, polygon[edge], polygon[(edge + 1) % len(polygon)]])
                if np.linalg.norm(np.cross(triangle[1] - centre, triangle[2] - centre)) > 0:
                    triangles.append(triangle)
    if not triangles:
        return None, None
    geometry = compute_mesh_geometry(mesh, compute_lsq=False, logger=None)
    faces = np.concatenate([
        np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
        for patch in mesh["boundary"] if patch["type"] != "wall"
    ])
    centres = geometry["face_centre"][faces]
    bounds = np.column_stack((centres.min(axis=0), centres.max(axis=0))).ravel()
    wall = TriangulatedWall(np.asarray(triangles), bounds)
    return SolidBoundary((wall,)), wall


def audit(directory, *, expected_time, expected_new_exchanges):
    """Validate a complete unchanged generation and its accepted observations."""
    directory = directory.resolve()
    checkpoint_directory = directory / "solution/backups"
    metadata_path = checkpoint_directory / "checkpoint_info.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    inputs = json.loads((directory / "inputs/inputs.json").read_text())
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(name)

    check("native_coupled_format_13", metadata["kind"] == "openonda.coupled_backup"
          and metadata["format_version"] == 13)
    check("coupled_configuration_sha256", config_mapping_digest(metadata["config"])
          == metadata["config_sha256"] == inputs["native_coupled_configuration_sha256"])
    check("complete_checkpoint_file_hash_keys", metadata["checkpoint_files"].keys()
          == metadata["file_sha256"].keys()
          and {"fvm", "vpm", "vpm_vtu", "vpm_boundary_condition"}
          <= metadata["checkpoint_files"].keys())
    paths = {name: declared_path(checkpoint_directory, relative)
             for name, relative in metadata["checkpoint_files"].items()}
    for name, path in paths.items():
        check("native_" + name + "_sha256", checkpoint_path_hash(path)
              == metadata["file_sha256"][name])
    state = read_state(paths["fvm"])
    fvm_metadata = json.loads(str(state["metadata"]))
    check("native_single_rank_fvm_format_10", fvm_metadata["format_version"] == 10)
    check("fvm_configuration_sha256", fvm_metadata["config_hash"]
          == inputs["native_fvm_configuration_sha256"])
    check("expected_accepted_time", np.isfinite(expected_time)
          and abs(metadata["time"] - expected_time) <= 1e-10)
    check("expected_new_exchanges", isinstance(expected_new_exchanges, int)
          and not isinstance(expected_new_exchanges, bool) and expected_new_exchanges > 0
          and metadata["coupling_step"] - inputs["initial_coupling_step"]
          == expected_new_exchanges)
    check("native_clock_step_ratios", metadata["fvm_step"]
          == metadata["coupling_step"] * metadata["n_fvm_substeps"]
          and metadata["vpm_step"] == metadata["coupling_step"]
          and metadata["n_fvm_substeps"] == 5)
    dt = metadata["config"]["vpm"]["time_step_size"]
    check("accepted_exchange_time_interval", abs(metadata["time"] - inputs["initial_time"]
          - expected_new_exchanges * dt) <= 1e-9)
    check("fvm_accepted_clock", abs(float(state["time"]) - metadata["time"]) <= 1e-10
          and int(state["step"]) == metadata["fvm_step"]
          == int(state["n_committed_time_steps"]))
    check("fvm_time_step_and_bdf_history", all(abs(float(state[name]) - dt / 5) < 1e-14
          for name in ("time_step_size", "accepted_time_step_size", "previous_time_step_size")))
    for name in FVM_FIELDS:
        check("finite_fvm_" + name, np.isfinite(state[name]).all())
    mesh = load_native_mesh(directory / "inputs/mesh.npz")
    fvm_run_metadata = json.loads((directory / "solution/fvm_metadata.json").read_text())
    # The native archive is saved before configured cyclic/wall mesh types
    # are applied. Reproduce only those declared type assignments for its hash.
    boundary_settings = {item["name"]: item
                         for item in fvm_run_metadata["configuration"]["boundaries"]}
    check("complete_declared_mesh_boundary_settings", set(boundary_settings)
          == {patch["name"] for patch in mesh["boundary"]})
    for patch in mesh["boundary"]:
        mesh_type = boundary_settings[patch["name"]]["mesh_type"]
        if mesh_type is not None:
            patch["type"] = mesh_type
    check("native_fvm_mesh_hash", fvm_metadata["mesh_hash"] == mesh_hash(mesh))
    cells, faces = mesh["n_cells"], mesh["n_faces"]
    boundary_count = faces - mesh["n_interior_faces"]
    cell_shape = (cells + boundary_count,)
    check("complete_global_mesh_coverage", len(mesh["faces"]) == faces
          and all(state[name].shape == (*cell_shape, 3)
                  for name in ("velocity", "velocity_old", "velocity_older"))
          and state["kinematic_pressure"].shape == cell_shape
          and all(state[name].shape == (faces,) for name in FVM_FIELDS
                  if name.startswith("volumetric_face_flux"))
          and state["eddy_viscosity"].shape in ((0,), (cells,)))
    with h5py.File(paths["vpm"], "r") as stored:
        attributes = dict(stored["solver"].attrs)
        particles = {name: value[:] for name, value in stored["particles"].items()}
    check("native_vpm_backup_format_10_3", str(attributes["backup_format_version"]) == "10.3")
    count = int(attributes["n_particles_total"])
    check("vpm_accepted_clock", abs(float(attributes["time"]) - metadata["time"]) <= 1e-10
          and int(attributes["step"]) == metadata["vpm_step"])
    vpm_config = json.loads(attributes["numerical_configuration"])
    check("vpm_configuration_match", vpm_config == metadata["config"]["vpm"]
          and config_mapping_digest(vpm_config) == attributes["numerical_configuration_sha256"])
    check("native_particle_capacity_and_spacing", 0 < count <= vpm_config["max_n_particles"]
          == 200000 and vpm_config["viscous"]["particle_spacing"] == 0.04)
    required = {"position", "vortex_strength", "core_radius", "particle_volume", "velocity",
                "vorticity", "kinematic_viscosity", "effective_viscosity", "eddy_viscosity"}
    check("complete_native_particle_fields", required <= particles.keys())
    for name, values in particles.items():
        check("finite_complete_vpm_" + name, len(values) == count and np.isfinite(values).all())
    induction = vpm_config["induction"]
    check("native_planar_particle_state", induction["method"] == "PLANAR"
          and particles["position"].shape == particles["vortex_strength"].shape == (count, 3)
          and np.max(np.abs(particles["position"][:, 2] - induction["plane_z"])) < 1e-6
          and np.max(np.abs(particles["vortex_strength"][:, :2])) == 0
          and np.all(particles["core_radius"] > 0) and np.all(particles["particle_volume"] > 0))
    expected_dtype = np.dtype({"f32": "float32", "f64": "float64"}[vpm_config["precision"]])
    check("declared_particle_storage_precision", all(particles[name].dtype == expected_dtype
          for name in ("position", "vortex_strength", "core_radius")))
    check("constant_final_freestream", np.array_equal(attributes["freestream_velocity"], [1, 0, 0]))
    solid, wall = native_wall_geometry(mesh)
    if solid is not None:
        check("native_wall_geometry_revision", wall.revision
              in metadata["config"]["solid_geometry"]["wall_revisions"])
        inside = solid.contains(particles["position"].astype(np.float64), include_boundary=False)
        check("stored_positions_outside_strict_solid", not inside.any())
    else:
        inside = np.zeros(count, dtype=bool)
    boundary = read_state(paths["vpm_boundary_condition"])
    patch = next(p for p in mesh["boundary"]
                 if p["name"] == metadata["config"]["coupler"]["coupling_patch"])
    patch_count = patch["n_faces"]
    check("native_boundary_history_schema_4", int(boundary["boundary_schema_version"]) == 4)
    check("native_boundary_history_present", all(bool(boundary[name]) for name in
          ("has_velocity", "has_normal_velocity", "has_tangential_gradient")))
    check("finite_complete_native_boundary_history", boundary["velocity"].shape
          == boundary["tangential_gradient"].shape == (patch_count, 3)
          and boundary["normal_velocity"].shape == (patch_count,)
          and all(np.isfinite(boundary[name]).all()
                  for name in ("velocity", "normal_velocity", "tangential_gradient")))
    forces = read_force_history(directory / "samples/forces_history.csv")
    latest_observed_force_time = float(forces["time"][-1])
    forces = forces[forces["time"] <= metadata["time"] + 1e-9]
    expected_times = inputs["initial_time"] + dt * np.arange(1, expected_new_exchanges + 1)
    check("accepted_force_records", len(forces) == expected_new_exchanges
          and np.allclose(forces["time"], expected_times, rtol=0, atol=1e-9)
          and np.all(forces["patch"] == "cylinder"))
    diagnostic_bytes = (directory / "solution/coupler_diagnostics.jsonl").read_bytes()
    if not diagnostic_bytes.endswith(b"\n"):
        diagnostic_bytes = diagnostic_bytes[:diagnostic_bytes.rfind(b"\n") + 1]
    diagnostics = [json.loads(line) for line in diagnostic_bytes.splitlines() if line]
    diagnostics = [record for record in diagnostics if record["time"] <= metadata["time"] + 1e-9]
    check("accepted_diagnostic_records", len(diagnostics) == expected_new_exchanges
          and [r["step"] for r in diagnostics]
          == list(range(inputs["initial_coupling_step"] + 1, metadata["coupling_step"] + 1))
          and np.allclose([r["time"] for r in diagnostics], expected_times, rtol=0, atol=1e-9))
    config = metadata["config"]["coupler"]
    check("unchanged_interface_tolerances_and_convergence", config["interface_iterations"] == 6
          and config["interface_normal_tolerance"] == config["interface_gradient_tolerance"] == 1e-5
          and all(r["interface_iteration"]["converged"]
                  and r["interface_iteration"]["sweeps"] <= 6 for r in diagnostics))
    check("finite_converged_interface_residuals", all(
        np.isfinite(r["interface_iteration"]["residuals"][-1][name])
        and 0 <= r["interface_iteration"]["residuals"][-1][name] <= 1e-5
        for r in diagnostics for name in ("normal_residual_rms", "gradient_residual_rms")))
    check("finite_nonnegative_exchange_timings", all(
        {"total", "vpm", "fvm", "transfer"} <= r["timing_seconds"].keys()
        and all(np.isfinite(value) and value >= 0 for value in r["timing_seconds"].values())
        for r in diagnostics))
    check("native_one_second_backup_cadence", config["backup_interval_steps"] == 25
        and all(r["backup_phase"] == {"scheduled": True, "status": "complete", "error": None}
                for r in diagnostics if r["step"] % 25 == 0))
    check("numerical_sources_unchanged", bool(inputs["source_sha256"])
          and all(Path(path).is_file() and digest(path) == value
                  for path, value in inputs["source_sha256"].items()))
    check("frozen_original_inputs_unchanged", bool(inputs["inputs"])
          and all(checkpoint_path_hash(Path(record["source"])) == record["sha256"]
                  for record in inputs["inputs"].values()))
    check("committed_checkpoint_unchanged_during_audit", metadata_path.read_bytes() == metadata_bytes
          and all(checkpoint_path_hash(paths[name]) == metadata["file_sha256"][name]
                  for name in paths))
    return {
        "status": "passed", "scope": "Read-only current native single-rank checkpoint audit; no solver is initialized.",
        "directory": str(directory), "accepted_time": metadata["time"],
        "coupling_step": metadata["coupling_step"], "accepted_new_exchanges": expected_new_exchanges,
        "latest_observed_force_time": latest_observed_force_time,
        "native_fvm_format": fvm_metadata["format_version"], "n_cells": cells,
        "n_faces": faces, "particle_count": count, "check_count": len(checks), "checks": checks,
        "checkpoint_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "checkpoint_file_sha256": metadata["file_sha256"],
        "stored_particle_dtypes": {name: str(particles[name].dtype)
                                   for name in ("position", "vortex_strength", "core_radius")},
        "stored_particle_strict_solid_node_count": int(inside.sum()),
        "accepted_exchanges": diagnostics,
        "forces": [{name: json_value(row[name]) for name in forces.dtype.names} for row in forces],
        "amplitude_recovery_demonstrated": False,
        "amplitude_limitation": "Native validity and exchange timing do not establish force amplitude recovery; compare complete cycles separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected-time", type=float, required=True)
    parser.add_argument("--expected-new-exchanges", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    report = audit(options.directory, expected_time=options.expected_time,
                   expected_new_exchanges=options.expected_new_exchanges)
    with options.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "accepted_time": report["accepted_time"],
                      "checks": report["check_count"], "report": str(options.output)}))


if __name__ == "__main__":
    main()
