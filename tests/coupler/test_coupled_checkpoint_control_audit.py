"""Current single-rank native audit detects corruption beyond the file digest."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from source.coupler.backup import checkpoint_path_hash, config_mapping_digest
from source.solvers.fvm.io.backup import encode_state, mesh_hash
from source.solvers.fvm.io.mesh_storage import load_native_mesh, save_native_mesh
from source.solvers.fvm.sampling.forces import ForceSampler
from tests.support.cylinder.audit_coupled_checkpoint_control import audit, digest


def native_checkpoint_fixture(directory):
    """Write a tiny current-format bundle without initializing any solver."""
    inputs, backups = directory / "inputs", directory / "solution/backups"
    inputs.mkdir(parents=True)
    backups.mkdir(parents=True)
    samples = directory / "samples"
    samples.mkdir()
    mesh = {
        "vertex_position": np.array([[x, y, z] for x in (-1., 1.)
                                     for y in (-1., 1.) for z in (-.5, .5)]),
        "faces": np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                           [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]),
        "owners": np.zeros(6, dtype=int), "neighbours": np.empty(0, dtype=int),
        "n_cells": 1, "n_faces": 6, "n_interior_faces": 0,
        "boundary": [{"name": "numericalBoundary", "start_face": 0,
                      "n_faces": 6, "type": "patch"}],
    }
    save_native_mesh(mesh, inputs / "mesh.npz")
    mesh = load_native_mesh(inputs / "mesh.npz")
    (directory / "solution/fvm_metadata.json").write_text(json.dumps({
        "configuration": {"boundaries": [{"name": "numericalBoundary", "mesh_type": None}]}
    }))
    config = {
        "coupler": {"coupling_patch": "numericalBoundary", "interface_iterations": 6,
                    "interface_normal_tolerance": 1e-5, "interface_gradient_tolerance": 1e-5,
                    "backup_interval_steps": 25},
        "vpm": {"time_step_size": .04, "max_n_particles": 200000, "precision": "f32",
                "viscous": {"particle_spacing": .04},
                "induction": {"method": "PLANAR", "plane_z": 0.0}},
    }
    state = {
        "metadata": np.asarray(json.dumps({"format_version": 10, "config_hash": "equations",
                                           "mesh_hash": mesh_hash(mesh)})),
        **{name: np.ones((7, 3)) for name in ("velocity", "velocity_old", "velocity_older")},
        "kinematic_pressure": np.zeros(7),
        **{name: np.zeros(6) for name in ("volumetric_face_flux", "volumetric_face_flux_old",
                                        "volumetric_face_flux_older")},
        "eddy_viscosity": np.empty(0), "time": np.asarray(46.08), "step": np.asarray(5760),
        "n_committed_time_steps": np.asarray(5760),
        **{name: np.asarray(.008) for name in ("time_step_size", "accepted_time_step_size",
                                              "previous_time_step_size")},
    }
    np.savez(backups / "fvm.npz", **encode_state(state))
    with h5py.File(backups / "vpm.h5", "w") as stored:
        solver = stored.create_group("solver")
        solver.attrs.update({"backup_format_version": "10.3", "n_particles_total": 2,
                             "time": 46.08, "step": 1152, "freestream_velocity": [1., 0., 0.],
                             "numerical_configuration": json.dumps(config["vpm"]),
                             "numerical_configuration_sha256": config_mapping_digest(config["vpm"])})
        particles = stored.create_group("particles")
        for name in ("position", "vortex_strength", "velocity", "vorticity"):
            values = np.array([[.6, 0, 0], [-.6, 0, 0]], dtype=np.float32)
            if name == "vortex_strength":
                values = np.array([[0, 0, .01], [0, 0, -.01]], dtype=np.float32)
            particles.create_dataset(name, data=values)
        for name in ("core_radius", "particle_volume", "kinematic_viscosity", "effective_viscosity",
                     "eddy_viscosity"):
            particles.create_dataset(name, data=np.full(2, .04, dtype=np.float32))
    np.savez(backups / "boundary.npz", **encode_state({
        "boundary_schema_version": np.asarray(4), "has_velocity": np.asarray(True),
        "has_normal_velocity": np.asarray(True), "has_tangential_gradient": np.asarray(True),
        "velocity": np.zeros((6, 3)), "normal_velocity": np.zeros(6),
        "tangential_gradient": np.zeros((6, 3)),
    }))
    (backups / "vpm.vtu").write_text("native-file-hash-fixture\n")
    files = {"fvm": "fvm.npz", "vpm": "vpm.h5", "vpm_vtu": "vpm.vtu",
             "vpm_boundary_condition": "boundary.npz"}
    metadata = {"kind": "openonda.coupled_backup", "format_version": 13,
                "config": config, "config_sha256": config_mapping_digest(config),
                "time": 46.08, "coupling_step": 1152, "fvm_step": 5760, "vpm_step": 1152,
                "n_fvm_substeps": 5, "checkpoint_files": files,
                "file_sha256": {name: checkpoint_path_hash(backups / path)
                                for name, path in files.items()}}
    (backups / "checkpoint_info.json").write_text(json.dumps(metadata))
    source = directory / "frozen_source.py"
    source.write_text("unchanged_numerical_source = True\n")
    (inputs / "inputs.json").write_text(json.dumps({
        "native_coupled_configuration_sha256": metadata["config_sha256"],
        "native_fvm_configuration_sha256": "equations", "initial_time": 46.,
        "initial_coupling_step": 1150, "source_sha256": {str(source): digest(source)},
        "inputs": {"source": {"source": str(source), "sha256": digest(source)}},
    }))
    sampler = ForceSampler(file_name="forces_history")
    diagnostics = []
    for index in (1, 2):
        time = 46. + index * .04
        context = SimpleNamespace(time=time, step=5750 + index * 5, _accepted_time_step_size=.008)
        sampler.write_csv(context, str(samples), {"cylinder": {
            "coeffs": dict.fromkeys(("drag_coefficient", "lift_coefficient",
                                     "side_force_coefficient", "pitching_moment_coefficient"), .2)}})
        diagnostics.append({"time": time, "step": 1150 + index,
                            "timing_seconds": {"total": 1., "vpm": .1, "fvm": .8, "transfer": .1},
                            "interface_iteration": {"converged": True, "sweeps": 4,
                                "residuals": [{"normal_residual_rms": 1e-6,
                                               "gradient_residual_rms": 1e-6}]},
                            "backup_phase": {"scheduled": False, "status": "not_scheduled",
                                             "error": None}})
    (directory / "solution/coupler_diagnostics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in diagnostics))
    return metadata, state


def test_current_native_audit_decodes_histories_and_rejects_rehashed_nonfinite_state(tmp_path):
    metadata, state = native_checkpoint_fixture(tmp_path)
    report = audit(tmp_path, expected_time=46.08, expected_new_exchanges=2)
    assert report["status"] == "passed"
    assert report["stored_particle_dtypes"]["position"] == "float32"
    assert report["accepted_new_exchanges"] == 2
    assert report["forces"][0]["patch"] == "cylinder"
    with pytest.raises(ValueError, match="expected_accepted_time"):
        audit(tmp_path, expected_time=47., expected_new_exchanges=2)
    state["velocity_older"][0, 0] = np.nan
    backups = tmp_path / "solution/backups"
    np.savez(backups / "fvm.npz", **encode_state(state))
    metadata["file_sha256"]["fvm"] = checkpoint_path_hash(backups / "fvm.npz")
    (backups / "checkpoint_info.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="finite_fvm_velocity_older"):
        audit(tmp_path, expected_time=46.08, expected_new_exchanges=2)
