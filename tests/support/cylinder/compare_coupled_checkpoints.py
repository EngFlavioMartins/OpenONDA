"""Read-only matched native checkpoint and observation comparison.

No solver import, interpolation, coordinate tolerance, time shift, or inferred
accuracy pass is used. Coupled checkpoint_file digests, numerical configurations and
committed clocks are validated before any numerical differences are reported.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

_HISTORY_REFERENCE = {
    "velocity_old": "velocity",
    "velocity_older": "velocity",
    "volumetric_face_flux_old": "volumetric_face_flux",
    "volumetric_face_flux_older": "volumetric_face_flux",
}
_FVM_FIELDS = (
    "velocity",
    "kinematic_pressure",
    "volumetric_face_flux",
    "volumetric_face_flux_old",
    "volumetric_face_flux_older",
    "velocity_old",
    "velocity_older",
    "eddy_viscosity",
)
_CLOCK_ATOL = 1e-10


def mapping_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def checkpoint_path_hash(path):
    """The native coupled checkpoint_info comparison_settings, including names inside directories."""
    digest = hashlib.sha256()
    children = (
        sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    )
    for child in children:
        if path.is_dir():
            digest.update(child.relative_to(path).as_posix().encode())
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def contained_path(root, name):
    relative = Path(name)
    result = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not result.is_relative_to(root.resolve())
    ):
        raise ValueError(f"Checkpoint file path escapes checkpoint: {name}")
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def decode_npz(path):
    """Invert the native v8 lossless byte shuffle/history XOR without a solver."""
    with np.load(path, allow_pickle=False) as stored:
        layout = json.loads(str(stored["storage_layout"]))
        values = {key: stored[key].copy() for key in stored.files if key != "storage_layout"}
    for key, (dtype, shape) in layout.items():
        values[key] = np.ascontiguousarray(values[key].T).view(np.dtype(dtype)).reshape(shape)
    for key, reference in _HISTORY_REFERENCE.items():
        if key in values:
            values[key] = (
                values[key].view(np.uint64)
                ^ np.ascontiguousarray(values[reference]).view(np.uint64)
            ).view(values[reference].dtype)
    return values


def _clock_equal(actual, expected):
    return bool(np.isfinite(actual) and abs(float(actual) - float(expected)) <= _CLOCK_ATOL)


def load_checkpoint(directory):
    directory = directory.resolve()
    metadata_path = directory / "checkpoint_info.json"
    metadata_bytes = metadata_path.read_bytes()
    checkpoint_info = json.loads(metadata_bytes)
    if (
        checkpoint_info.get("kind") != "openonda.coupled_backup"
        or checkpoint_info.get("format_version") != 13
    ):
        raise ValueError("Unsupported coupled checkpoint schema; require native v13")
    if mapping_digest(checkpoint_info["config"]) != checkpoint_info["config_sha256"]:
        raise ValueError("Coupled numerical configuration digest mismatch")
    if set(checkpoint_info["checkpoint_files"]) != set(checkpoint_info["file_sha256"]):
        raise ValueError("Coupled checkpoint file/digest keys disagree")
    checkpoint_files = {
        name: contained_path(directory, value)
        for name, value in checkpoint_info["checkpoint_files"].items()
    }
    for name, path in checkpoint_files.items():
        if checkpoint_path_hash(path) != checkpoint_info["file_sha256"][name]:
            raise ValueError(f"Coupled checkpoint_file SHA256 mismatch: {name}")
    with h5py.File(checkpoint_files["vpm"], "r") as saved:
        solver = {
            name: value.tolist()
            if isinstance(value, np.ndarray)
            else value.item()
            if isinstance(value, np.generic)
            else value
            for name, value in saved["solver"].attrs.items()
        }
        particle = {name: value[:] for name, value in saved["particles"].items()}
    vpm_config = json.loads(solver["numerical_configuration"])
    if mapping_digest(vpm_config) != solver["numerical_configuration_sha256"]:
        raise ValueError("VPM numerical configuration digest mismatch")
    if vpm_config != checkpoint_info["config"]["vpm"]:
        raise ValueError("VPM numerical configuration disagrees with coupled checkpoint_info")
    if int(solver["step"]) != checkpoint_info["vpm_step"] or not _clock_equal(
        solver["time"], checkpoint_info["time"]
    ):
        raise ValueError("VPM committed clock disagrees with coupled checkpoint_info")
    particle_count = int(solver["n_particles_total"])
    if any(
        len(value) != particle_count or not np.isfinite(value).all() for value in particle.values()
    ):
        raise ValueError("Invalid or nonfinite VPM particle field")
    if particle["position"].shape != (particle_count, 3):
        raise ValueError("Invalid VPM coordinate shape")
    fvm_root = checkpoint_files["fvm"]
    fvm_checkpoint_info = json.loads((fvm_root / "checkpoint_info.json").read_text())
    if (
        fvm_checkpoint_info.get("format_version") != 9
        or len(fvm_checkpoint_info["files"]) != fvm_checkpoint_info["n_ranks"]
    ):
        raise ValueError("Unsupported or incomplete partitioned FVM checkpoint")
    ranks = [decode_npz(contained_path(fvm_root, name)) for name in fvm_checkpoint_info["files"]]
    for state in ranks:
        if (
            int(state["step"]) != checkpoint_info["fvm_step"]
            or int(state["n_committed_time_steps"]) != checkpoint_info["fvm_step"]
        ):
            raise ValueError("FVM committed step disagrees with coupled checkpoint_info")
        if not _clock_equal(state["time"], checkpoint_info["time"]):
            raise ValueError("FVM committed time disagrees with coupled checkpoint_info")
        if not all(np.isfinite(state[name]).all() for name in _FVM_FIELDS):
            raise ValueError("Nonfinite FVM primary field")
    if (
        checkpoint_info["fvm_step"]
        != checkpoint_info["coupling_step"] * checkpoint_info["n_fvm_substeps"]
        or checkpoint_info["vpm_step"] != checkpoint_info["coupling_step"]
    ):
        raise ValueError("Coupled/FVM/VPM step ratios disagree")
    boundary = decode_npz(checkpoint_files["vpm_boundary_condition"])
    for name, path in checkpoint_files.items():
        if checkpoint_path_hash(path) != checkpoint_info["file_sha256"][name]:
            raise ValueError(f"Checkpoint changed while being read: {name}")
    if metadata_path.read_bytes() != metadata_bytes:
        raise ValueError("Checkpoint checkpoint_info changed while being read")
    return {
        "directory": directory,
        "checkpoint_info": checkpoint_info,
        "fvm_checkpoint_info": fvm_checkpoint_info,
        "ranks": ranks,
        "particle": particle,
        "solver": solver,
        "boundary": boundary,
        "checkpoint_info_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }


def _json_comparison_value(value):
    """Exact JSON configuration; in particular True is not interchangeable with 1."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_matching_checkpoints(control, candidate):
    left, right = control["checkpoint_info"], candidate["checkpoint_info"]
    for checkpoint_info in (left, right):
        if mapping_digest(checkpoint_info["config"]) != checkpoint_info["config_sha256"]:
            raise ValueError("Comparison configuration digest mismatch")
    for key in ("coupling_step", "fvm_step", "vpm_step", "n_fvm_substeps"):
        if left[key] != right[key]:
            raise ValueError(f"Unmatched coupled checkpoints: {key}")
    if not _clock_equal(left["time"], right["time"]):
        raise ValueError("Unmatched coupled checkpoint times")
    for key in ("config", "config_sha256"):
        if _json_comparison_value(left[key]) != _json_comparison_value(right[key]):
            raise ValueError(f"Unmatched coupled checkpoints: {key}")


def field_difference(reference, candidate):
    left, right = np.asarray(reference), np.asarray(candidate)
    result = {
        "reference_shape": list(left.shape),
        "candidate_shape": list(right.shape),
        "reference_dtype": str(left.dtype),
        "candidate_dtype": str(right.dtype),
    }
    if left.shape != right.shape:
        return {
            **result,
            "comparable": False,
            "reason": "different shapes; no interpolation performed",
        }
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite comparison field")
    result.update(
        comparable=True, exactly_equal=bool(np.array_equal(left, right)), entries=left.size
    )
    if left.size == 0:
        return {
            **result,
            "max_absolute_difference": 0.0,
            "l2_difference": 0.0,
            "relative_l2_difference": None,
            "peak": None,
        }
    delta = right.astype(np.float64) - left.astype(np.float64)
    maximum = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
    denominator = float(np.linalg.norm(left.ravel().astype(np.float64)))
    return {
        **result,
        "max_absolute_difference": float(np.max(np.abs(delta))),
        "rms_absolute_difference": float(np.sqrt(np.mean(delta**2))),
        "l2_difference": float(np.linalg.norm(delta.ravel())),
        "reference_l2": denominator,
        "relative_l2_difference": float(np.linalg.norm(delta.ravel()) / denominator)
        if denominator
        else None,
        "peak": {
            "index": [int(value) for value in maximum],
            "reference": float(left[maximum]),
            "candidate": float(right[maximum]),
            "difference": float(delta[maximum]),
        },
    }


def coordinate_alignment(left, right):
    """Match exactly equal numeric coordinates only when unique on both sides."""
    keys = []
    for value in (left, right):
        coordinates = np.array(value, dtype=np.float64, order="C", copy=True)
        coordinates[coordinates == 0.0] = 0.0  # Numeric +0 and -0 are the same point.
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
            raise ValueError("Coordinates must be finite (N,3) arrays")
        structured = coordinates.view(
            np.dtype([(axis, np.float64) for axis in ("x", "y", "z")])
        ).reshape(-1)
        unique, first, count = np.unique(structured, return_index=True, return_counts=True)
        keys.append((unique[count == 1], first[count == 1], int(np.sum(count[count > 1]))))
    _, a, b = np.intersect1d(keys[0][0], keys[1][0], return_indices=True)
    left_indices, right_indices = keys[0][1][a], keys[1][1][b]
    return (
        left_indices,
        right_indices,
        {
            "matched_unique_coordinates": len(a),
            "reference_unmatched_particles": len(left) - len(a),
            "candidate_unmatched_particles": len(right) - len(a),
            "reference_particles_at_duplicate_coordinates": keys[0][2],
            "candidate_particles_at_duplicate_coordinates": keys[1][2],
            "matching": "exact numeric coordinates, unique on BOTH sides; no rounding, nearest neighbours or interpolation",
        },
    )


def compare_particles(left, right):
    a, b, alignment = coordinate_alignment(left["position"], right["position"])
    result = {
        "alignment": alignment,
        "reference_fields": sorted(left),
        "candidate_fields": sorted(right),
        "missing_fields": sorted(set(left) ^ set(right)),
        "coordinate_matched_fields": {},
    }
    ordered = np.array_equal(left["position"], right["position"])
    result["ordered_positions_identical"] = bool(ordered)
    for name in sorted(set(left) & set(right)):
        summary = field_difference(left[name][a], right[name][b])
        if summary.get("peak") is not None:
            row = summary["peak"]["index"][0]
            summary["peak"].update(
                reference_particle=int(a[row]),
                candidate_particle=int(b[row]),
                position=left["position"][a[row]].tolist(),
            )
        result["coordinate_matched_fields"][name] = summary
    if ordered:
        result["ordered_storage_fields"] = {
            name: field_difference(left[name], right[name])
            for name in sorted(set(left) & set(right))
        }
        result["ordered_storage_note"] = (
            "Storage-order comparison; duplicate coordinates alone do not establish particle correspondence"
        )
    for role, fields, matched in (("reference", left, a), ("candidate", right, b)):
        unmatched = np.setdiff1d(np.arange(len(fields["position"])), matched, assume_unique=True)
        gamma = fields["vortex_strength"][unmatched].astype(np.float64)
        largest = np.argsort(-np.linalg.norm(gamma, axis=1), kind="stable")[:8]
        result[role + "_unmatched"] = {
            "count": len(unmatched),
            "vortex_strength_l1": float(np.linalg.norm(gamma, axis=1).sum()),
            "net_vortex_strength": gamma.sum(axis=0).tolist(),
            "largest": [
                {
                    "particle": int(unmatched[index]),
                    "position": fields["position"][unmatched[index]].tolist(),
                    "vortex_strength": gamma[index].tolist(),
                }
                for index in largest
            ],
        }
    return result


def compare_fvm(control, candidate):
    left_checkpoint_info, right_checkpoint_info = (
        control["fvm_checkpoint_info"],
        candidate["fvm_checkpoint_info"],
    )
    for name in ("config_hash", "mesh_hash", "n_global_cells", "n_ranks", "kinematic_viscosity"):
        if left_checkpoint_info[name] != right_checkpoint_info[name]:
            raise ValueError(f"FVM configuration mismatch: {name}")
    result = []
    for rank, (left, right) in enumerate(zip(control["ranks"], candidate["ranks"], strict=True)):
        for name in ("global_cell_id", "global_face_id"):
            if not np.array_equal(left[name], right[name]):
                raise ValueError(f"FVM rank {rank} {name} ordering differs")
        record = {
            "rank": rank,
            "scope": "all stored local entries, including halos and boundary ghosts",
            "fields": {name: field_difference(left[name], right[name]) for name in _FVM_FIELDS},
            "clock_and_acceptance_state": {
                name: field_difference(left[name], right[name])
                for name in (
                    "time",
                    "step",
                    "n_committed_time_steps",
                    "time_step_size",
                    "accepted_time_step_size",
                    "previous_time_step_size",
                    "max_courant_number",
                    "n_consecutive_accepted_steps",
                )
            },
        }
        for name, summary in record["fields"].items():
            if summary.get("peak") is None:
                continue
            row = summary["peak"]["index"][0]
            ids = (
                left["global_face_id"]
                if name.startswith("volumetric_face_flux")
                else left["global_cell_id"]
            )
            summary["peak"]["global_entity_id"] = int(ids[row]) if row < len(ids) else None
            summary["peak"]["boundary_ghost_without_cell_id"] = bool(row >= len(ids))
        result.append(record)
    return result


def matching_diagnostic(path, step, clock):
    if not path.is_file():
        return {"available": False, "path": str(path), "reason": "diagnostic history absent"}
    matches = [
        row
        for row in map(json.loads, path.read_text().splitlines())
        if row.get("step") == step and _clock_equal(row.get("time", np.nan), clock)
    ]
    if len(matches) != 1:
        return {
            "available": False,
            "path": str(path),
            "matching_rows": len(matches),
            "reason": "require one unambiguous accepted row at the checkpoint clock",
        }
    return {
        "available": True,
        "path": str(path),
        "sha256": checkpoint_path_hash(path),
        "record": matches[0],
    }


def numerical_checks(checkpoint, diagnostics_path):
    checkpoint_info = checkpoint["checkpoint_info"]
    limits = checkpoint_info["config"]["coupler"]
    diagnostic = matching_diagnostic(
        diagnostics_path, checkpoint_info["coupling_step"], checkpoint_info["time"]
    )
    result = {
        "vpm_lagrangian_cfl": checkpoint["solver"].get("lagrangian_cfl"),
        "vpm_lagrangian_cfl_limit": checkpoint_info["config"]["vpm"]["state_limits"][
            "lagrangian_cfl"
        ]["maximum"],
        "fvm_max_courant_per_rank": [
            float(rank["max_courant_number"]) for rank in checkpoint["ranks"]
        ],
        "fvm_residual_check_note": "Residual convergence is not stored in native FVM backup; no residual pass inferred from a committed clock",
        "accepted_diagnostic": diagnostic,
    }
    fvm_diagnostic = matching_diagnostic(
        diagnostics_path.parent / "diagnostics.jsonl",
        checkpoint_info["fvm_step"],
        checkpoint_info["time"],
    )
    result["fvm_accepted_diagnostic"] = fvm_diagnostic
    metadata_path = diagnostics_path.parent / "fvm_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        result["fvm_acceptance_configuration"] = metadata.get("configuration", {}).get("acceptance")
        result["fvm_metadata_sha256"] = checkpoint_path_hash(metadata_path)
    if fvm_diagnostic["available"]:
        fvm_row = fvm_diagnostic["record"]
        solves = fvm_row.get("linear_solves", [])
        result["fvm_all_recorded_linear_solves_converged"] = (
            all(solve.get("converged") is True for solve in solves) if solves else None
        )
        result["fvm_nonfinite_value_count"] = fvm_row.get("n_nonfinite_values")
        result["fvm_final_residuals"] = fvm_row.get("residuals")
        result["fvm_residual_check_note"] = (
            "Convergence flags and residuals are from the unique accepted FVM diagnostic row; "
            "no new residual threshold or pass criterion introduced"
        )
    value, maximum = result["vpm_lagrangian_cfl"], result["vpm_lagrangian_cfl_limit"]
    result["vpm_lagrangian_cfl_within_limit"] = (
        bool(value <= maximum) if value is not None and maximum is not None else None
    )
    if diagnostic["available"]:
        iteration = diagnostic["record"].get("interface_iteration", {})
        selected = [
            row
            for row in iteration.get("residuals", [])
            if row.get("sweep") == iteration.get("accepted_sweep")
        ]
        if len(selected) == 1:
            row = selected[0]
            result["interface"] = {
                "normal_residual_rms": row["normal_residual_rms"],
                "gradient_residual_rms": row["gradient_residual_rms"],
                "normal_limit": limits["interface_normal_tolerance"],
                "gradient_limit": limits["interface_gradient_tolerance"],
                "recorded_converged": iteration.get("converged"),
                "checks_passed": bool(
                    row["normal_residual_rms"] <= limits["interface_normal_tolerance"]
                    and row["gradient_residual_rms"] <= limits["interface_gradient_tolerance"]
                ),
            }
    return result


def _sample_rows(path, clock):
    if not path.is_file():
        return None, []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        if header is None or "time" not in header:
            raise ValueError(f"Sample history has no time column: {path}")
        rows = [row for row in reader if _clock_equal(float(row["time"]), clock)]
    return header, rows


def _sample_key(row):
    coordinates = ("position_x", "position_y", "position_z")
    if all(name in row for name in coordinates):
        return tuple(float(row[name]) for name in coordinates)
    return (row.get("patch", "single observation"),)


def compare_samples(control, candidate, clock, fvm_step, vpm_step):
    if not control.is_dir() or not candidate.is_dir():
        raise ValueError("Both sample directories must exist")
    records = {}
    names = sorted({path.name for folder in (control, candidate) for path in folder.glob("*.csv")})
    if not names:
        raise ValueError("No CSV observations in either sample directory")
    for name in names:
        paths = (control / name, candidate / name)
        data = [_sample_rows(path, clock) for path in paths]
        record = {
            "clock": clock,
            "clock_absolute_tolerance": _CLOCK_ATOL,
            "reference_rows": len(data[0][1]),
            "candidate_rows": len(data[1][1]),
            "hashes": {
                role: checkpoint_path_hash(path) if path.is_file() else None
                for role, path in zip(("reference", "candidate"), paths, strict=True)
            },
        }
        if data[0][0] is None or data[1][0] is None:
            record["status"] = "missing history"
        elif data[0][0] != data[1][0]:
            record["status"] = "different columns"
            record["columns"] = [item[0] for item in data]
        elif not data[0][1] and not data[1][1]:
            record["status"] = "neither history samples this clock; no time interpolation"
        else:
            keys = [[_sample_key(row) for row in item[1]] for item in data]
            if any(len(set(key)) != len(key) for key in keys):
                record["status"] = "ambiguous duplicate observation keys at this clock"
            else:
                lookup = [
                    dict(zip(key, item[1], strict=True))
                    for key, item in zip(keys, data, strict=True)
                ]
                common = sorted(set(lookup[0]) & set(lookup[1]))
                record.update(
                    status="compared exact observation keys",
                    matched_rows=len(common),
                    reference_unmatched_rows=len(keys[0]) - len(common),
                    candidate_unmatched_rows=len(keys[1]) - len(common),
                )
                expected_step = vpm_step if name.startswith("vpm_") else fvm_step
                for item in data:
                    if any(int(row["step"]) != expected_step for row in item[1]):
                        raise ValueError(f"Observation clock/step mismatch: {name}")
                record["fields"] = {}
                for column in data[0][0]:
                    if column in {
                        "patch",
                        "time",
                        "step",
                        "position_x",
                        "position_y",
                        "position_z",
                    }:
                        continue
                    values = [
                        np.asarray([float(table[key][column]) for key in common])
                        for table in lookup
                    ]
                    summary = field_difference(*values)
                    if summary["peak"] is not None:
                        summary["peak"]["observation_key"] = list(
                            common[summary["peak"]["index"][0]]
                        )
                    record["fields"][column] = summary
        records[name] = record
    return records


def compare_checkpoints(
    control,
    candidate,
    control_samples,
    candidate_samples,
    control_diagnostics,
    candidate_diagnostics,
):
    left, right = control["checkpoint_info"], candidate["checkpoint_info"]
    validate_matching_checkpoints(control, candidate)
    boundary = {
        key: field_difference(control["boundary"][key], candidate["boundary"][key])
        for key in sorted(set(control["boundary"]) & set(candidate["boundary"]))
    }
    solver_state = {}
    for name in sorted(set(control["solver"]) | set(candidate["solver"])):
        if name not in control["solver"] or name not in candidate["solver"]:
            solver_state[name] = {"comparable": False, "reason": "attribute absent on one side"}
            continue
        a, b = control["solver"][name], candidate["solver"][name]
        if np.asarray(a).dtype.kind in "biufc" and np.asarray(b).dtype.kind in "biufc":
            solver_state[name] = field_difference(a, b)
        else:
            solver_state[name] = {"exactly_equal": a == b}
            if name != "numerical_configuration":
                solver_state[name].update(reference=a, candidate=b)
    return {
        "status": "comparison_complete_no_accuracy_pass_inferred",
        "time": left["time"],
        "coupling_step": left["coupling_step"],
        "config_sha256": left["config_sha256"],
        "control": {
            "directory": str(control["directory"]),
            "checkpoint_info_sha256": control["checkpoint_info_sha256"],
            "file_sha256": left["file_sha256"],
            "config_sha256": left["config_sha256"],
        },
        "candidate": {
            "directory": str(candidate["directory"]),
            "checkpoint_info_sha256": candidate["checkpoint_info_sha256"],
            "file_sha256": right["file_sha256"],
            "config_sha256": right["config_sha256"],
        },
        "scope": "One matched accepted checkpoint, not long-time trajectory, periodicity, phase or performance validation",
        "fvm": compare_fvm(control, candidate),
        "vpm": compare_particles(control["particle"], candidate["particle"]),
        "vpm_solver_state": solver_state,
        "boundary_history": boundary,
        "boundary_missing_fields": sorted(set(control["boundary"]) ^ set(candidate["boundary"])),
        "numerical_checks": {
            "control": numerical_checks(control, control_diagnostics),
            "candidate": numerical_checks(candidate, candidate_diagnostics),
        },
        "samples": compare_samples(
            control_samples, candidate_samples, left["time"], left["fvm_step"], left["vpm_step"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "control-backup",
        "candidate-backup",
        "control-samples",
        "candidate-samples",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--control-diagnostics", type=Path)
    parser.add_argument("--candidate-diagnostics", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        output.parent
        != (
            Path(__file__).resolve().parents[3]
            / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
        )
        / "solution"
        or output.suffix != ".json"
    ):
        raise ValueError("Comparison JSON must stay in this tutorial's ordinary solution directory")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {output}")
    control = load_checkpoint(args.control_backup)
    candidate = load_checkpoint(args.candidate_backup)
    report = compare_checkpoints(
        control,
        candidate,
        args.control_samples,
        args.candidate_samples,
        args.control_diagnostics or args.control_backup.parent / "coupler_diagnostics.jsonl",
        args.candidate_diagnostics or args.candidate_backup.parent / "coupler_diagnostics.jsonl",
    )
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "time": report["time"],
                "particle_alignment": report["vpm"]["alignment"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
