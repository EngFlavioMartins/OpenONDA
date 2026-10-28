#!/usr/bin/env python3
"""Convert a coupled run written before 28 October 2026 to the current names and formats.

Converts solution/coupler_diagnostics.jsonl, solution/coupler_metadata.json and the
coupled backups solution/coupled/coupled_* of a coupled run in place (atomic rewrite) to
the names of docs/terminology.md and the current backup format, so that the run restarts
with the current coupler. Files already converted are left unchanged.

    python convert_coupler_diagnostics.py <solution directory> [...]
"""

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from openonda.coupler.backup import (
    BACKUP_FORMAT_VERSION,
    BOUNDARY_SCHEMA_VERSION,
    configuration_digest,
    file_digest,
)
from openonda.fvm.io.backup import decode_state, encode_state

TOP = {
    "fvm_boundary_trace": "fvm_boundary_data",
    "boundary_normal_velocity": "transfer_surface_normal_velocity",
    "vortex_line_closure": "transfer_surface_normal_vorticity",
    "gbd_moment_recovery": "gbd_moment_correction",
    "interface_iteration": "coupling_iteration",
}
GBD = {"pruned_node_count": "removed_node_count"}
ITERATION = {"sweeps": "iterations", "initial_guess": "first_estimate"}
FIRST_ESTIMATE = {"vpm_trace": "vpm_predictor"}
ROW = {
    "sweep": "iteration",
    "normal_residual_rms": "normal_velocity_residual",
    "gradient_residual_rms": "normal_gradient_residual",
}
TIMING = {"last_sweep_donor_gather": "last_transfer_fvm_gather"}
TRANSFER_EXACT = {
    "maximum_transfer_amplification": "strength_ratio",
    "eta_blending_enabled": "blending_enabled",
    "representation_residual_before_prune": "representation_residual_before_removal",
    "representation_residual_after_prune": "representation_residual_after_removal",
}
TRANSFER_DROPPED = {"transfer_method"}
CONFIGURATION = {
    "coupling_patch": "outer_boundary_patch",
    "eta_blend_width": "blend_end_depth",
    "vpm_only_width": "blend_start_depth",
    "interface_iterations": "max_coupling_iterations",
    "interface_normal_tolerance": "normal_velocity_tolerance",
    "interface_gradient_tolerance": "normal_gradient_tolerance",
    "transfer_amplification_cap": "max_strength_ratio",
    "transfer_discretization_error_limit": "max_moment_correction_fraction",
    "transfer_vorticity_cutoff": "transfer_removal_threshold",
}
COUPLING = {"fvm_box": "fvm_domain_bounds"}
BOUNDARY = {
    "has_tangential_gradient": "has_normal_gradient",
    "tangential_gradient": "normal_gradient",
}


def transfer_key(name):
    """Mechanical rules of docs/terminology.md, in this order."""
    if name in TRANSFER_EXACT:
        return TRANSFER_EXACT[name]
    for old, new in (
        ("population_renewal_", "particle_set_transfer_"),
        ("population_", "particle_set_"),
        ("renewal_", "transfer_"),
        ("renewed_", "transfer_"),
        ("pruned_", "removed_"),
    ):
        name = name.replace(old, new)
    return name


def rename(mapping, table):
    return {table.get(key, key): value for key, value in mapping.items()}


def convert_record(record):
    record = rename(record, TOP)
    if isinstance(record.get("gbd_moment_correction"), dict):
        record["gbd_moment_correction"] = rename(record["gbd_moment_correction"], GBD)
    iteration = record.get("coupling_iteration")
    if isinstance(iteration, dict):
        iteration = rename(iteration, ITERATION)
        if "first_estimate" in iteration:
            iteration["first_estimate"] = FIRST_ESTIMATE.get(
                iteration["first_estimate"], iteration["first_estimate"]
            )
        iteration["residuals"] = [rename(row, ROW) for row in iteration.get("residuals", [])]
        record["coupling_iteration"] = iteration
    if isinstance(record.get("timing_seconds"), dict):
        record["timing_seconds"] = rename(record["timing_seconds"], TIMING)
    if isinstance(record.get("transfer"), dict):
        record["transfer"] = {
            transfer_key(key): value
            for key, value in record["transfer"].items()
            if key not in TRANSFER_DROPPED
        }
    return record


def write_atomic(path, text):
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(temporary, path)


def convert_jsonl(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        record = convert_record(json.loads(line))
        out.append(json.dumps(record, sort_keys=True, allow_nan=False))
    text = "\n".join(out) + "\n"
    if text != path.read_text(encoding="utf-8"):
        write_atomic(path, text)
        return True
    return False


def convert_metadata(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document.get("configuration"), dict):
        document["configuration"] = rename(document["configuration"], CONFIGURATION)
    if isinstance(document.get("coupling"), dict):
        document["coupling"] = rename(document["coupling"], COUPLING)
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if text != path.read_text(encoding="utf-8"):
        write_atomic(path, text)
        return True
    return False


def convert_backup(path):
    """Rename the boundary data and the configuration of one coupled backup; renew its digests.

    Format 16 leaves the interval of the full transfer diagnostics out of the configuration.
    """
    info = json.loads((path / "checkpoint_info.json").read_text(encoding="utf-8"))
    if info["format_version"] == BACKUP_FORMAT_VERSION:
        return False
    info["configuration"]["coupler"].pop("transfer_diagnostic_interval_steps", None)
    boundary = path / info["files"]["boundary_condition"]
    with np.load(boundary, allow_pickle=False) as stored:
        state = rename(decode_state({name: stored[name] for name in stored.files}), BOUNDARY)
    state["boundary_schema_version"] = np.asarray(BOUNDARY_SCHEMA_VERSION)
    handle, temporary = tempfile.mkstemp(dir=path, prefix=f".{boundary.name}.")
    with os.fdopen(handle, "wb") as stream:
        np.savez_compressed(stream, **encode_state(state))
    os.replace(temporary, boundary)
    info["configuration"]["coupler"] = rename(info["configuration"]["coupler"], CONFIGURATION)
    info["configuration_sha256"] = configuration_digest(info["configuration"])
    info["file_sha256"]["boundary_condition"] = file_digest(boundary)
    info["format_version"] = BACKUP_FORMAT_VERSION
    write_atomic(path / "checkpoint_info.json", json.dumps(info, indent=2) + "\n")
    return True


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        directory = Path(argument)
        for name, convert in (
            ("coupler_diagnostics.jsonl", convert_jsonl),
            ("coupler_metadata.json", convert_metadata),
        ):
            path = directory / name
            if path.is_file():
                print(f"{path}: {'converted' if convert(path) else 'unchanged'}")
        for path in sorted((directory / "coupled").glob("coupled_*")):
            print(f"{path}: {'converted' if convert_backup(path) else 'unchanged'}")
