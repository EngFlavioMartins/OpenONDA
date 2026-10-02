"""Read-only matched native checkpoint and observation comparison.

No solver import, interpolation, coordinate tolerance, time shift, or inferred
accuracy pass is used. Coupled artifact digests, numerical configurations and
committed clocks are admitted before any numerical differences are reported.
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
_MESH_POLICY_PATH = "vpm.induction.gaussian_mesh_policy"


def mapping_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact_digest(path):
    """The native coupled manifest contract, including names inside directories."""
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
        raise ValueError(f"Artifact path escapes checkpoint: {name}")
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
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("kind") != "openonda.coupled_backup" or manifest.get("format_version") != 11:
        raise ValueError("Unsupported coupled checkpoint schema; require native v11")
    if mapping_digest(manifest["config"]) != manifest["config_sha256"]:
        raise ValueError("Coupled numerical configuration digest mismatch")
    if set(manifest["artifacts"]) != set(manifest["artifact_sha256"]):
        raise ValueError("Coupled manifest artifact/digest keys disagree")
    artifacts = {
        name: contained_path(directory, value) for name, value in manifest["artifacts"].items()
    }
    for name, path in artifacts.items():
        if artifact_digest(path) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Coupled artifact SHA256 mismatch: {name}")
    with h5py.File(artifacts["vpm"], "r") as saved:
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
    if vpm_config != manifest["config"]["vpm"]:
        raise ValueError("VPM numerical configuration disagrees with coupled manifest")
    if int(solver["step"]) != manifest["vpm_step"] or not _clock_equal(
        solver["time"], manifest["time"]
    ):
        raise ValueError("VPM committed clock disagrees with coupled manifest")
    particle_count = int(solver["n_particles_total"])
    if any(
        len(value) != particle_count or not np.isfinite(value).all() for value in particle.values()
    ):
        raise ValueError("Invalid or nonfinite VPM particle field")
    if particle["position"].shape != (particle_count, 3):
        raise ValueError("Invalid VPM coordinate shape")
    fvm_root = artifacts["fvm"]
    fvm_manifest = json.loads((fvm_root / "manifest.json").read_text())
    if (
        fvm_manifest.get("format_version") != 8
        or len(fvm_manifest["files"]) != fvm_manifest["n_ranks"]
    ):
        raise ValueError("Unsupported or incomplete partitioned FVM checkpoint")
    ranks = [decode_npz(contained_path(fvm_root, name)) for name in fvm_manifest["files"]]
    for state in ranks:
        if (
            int(state["step"]) != manifest["fvm_step"]
            or int(state["n_committed_time_steps"]) != manifest["fvm_step"]
        ):
            raise ValueError("FVM committed step disagrees with coupled manifest")
        if not _clock_equal(state["time"], manifest["time"]):
            raise ValueError("FVM committed time disagrees with coupled manifest")
        if not all(np.isfinite(state[name]).all() for name in _FVM_FIELDS):
            raise ValueError("Nonfinite FVM primary field")
    if (
        manifest["fvm_step"] != manifest["coupling_step"] * manifest["n_fvm_substeps"]
        or manifest["vpm_step"] != manifest["coupling_step"]
    ):
        raise ValueError("Coupled/FVM/VPM step ratios disagree")
    boundary = decode_npz(artifacts["vpm_boundary_condition"])
    for name, path in artifacts.items():
        if artifact_digest(path) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Checkpoint changed while being read: {name}")
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Checkpoint manifest changed while being read")
    return {
        "directory": directory,
        "manifest": manifest,
        "fvm_manifest": fvm_manifest,
        "ranks": ranks,
        "particle": particle,
        "solver": solver,
        "boundary": boundary,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def _json_identity(value):
    """Exact JSON identity; in particular True is not interchangeable with 1."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _configuration_changes(left, right, path=""):
    """Describe differences without deleting/replacing any configuration keys."""
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                result.append({"path": child,
                               "stored": {"present": True, "value": left[key]}
                               if key in left else {"present": False},
                               "current": {"present": True, "value": right[key]}
                               if key in right else {"present": False}})
            else:
                result.extend(_configuration_changes(left[key], right[key], child))
        return result
    if _json_identity(left) == _json_identity(right):
        return []
    return [{"path": path, "stored": {"present": True, "value": left},
             "current": {"present": True, "value": right}}]


def _sha256(value):
    if (type(value) is not str or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("Explicit lowercase SHA-256 digest required")
    return value


def admit_mesh_transition(control, candidate, report_path, expected_report_sha256):
    """Authenticate one recorded missing-to-policy transition, never a waiver.

    The independently pinned completed benchmark report authenticates its
    original source manifest/artifacts and immutable implementation inventory.
    Both comparison checkpoints retain their own native config/digest/clock
    admission. This is evidence of the recorded transition, not an accuracy
    pass or a cryptographic claim about which process wrote the candidate.
    """
    report_path = Path(report_path).resolve(strict=True)
    raw = report_path.read_bytes()
    report_sha = hashlib.sha256(raw).hexdigest()
    if report_sha != _sha256(expected_report_sha256):
        raise ValueError("Transition benchmark report SHA-256 mismatch")
    report = json.loads(raw)
    evidence = report.get("restart_admission")
    if (report.get("status") != "complete" or report.get("source_checkpoint_unchanged") is not True
            or report.get("qualification_controls", {}).get("gaussian_mesh_explicit_opt_in") is not True
            or report.get("loaded_source_files_changed_during_run") != []
            or not isinstance(evidence, dict)):
        raise ValueError("Require completed, explicit, immutable mesh-transition benchmark evidence")
    before, after = (report.get(name) for name in
                     ("source_hashes_before_initialization", "source_inventory_at_completion"))
    loaded = report.get("source_hashes_at_completion")
    if (not isinstance(before, dict) or not before or not isinstance(after, dict)
            or _json_identity(before) != _json_identity(after) or not isinstance(loaded, dict)
            or not loaded or any(before.get(path) != digest for path, digest in loaded.items())):
        raise ValueError("Transition implementation inventories disagree or are incomplete")
    for digest in before.values():
        _sha256(digest)

    left, right = control["manifest"], candidate["manifest"]
    changes = _configuration_changes(left["config"], right["config"])
    if (len(changes) != 1 or changes[0]["path"] != _MESH_POLICY_PATH
            or changes[0]["stored"] != {"present": False}
            or changes[0]["current"].get("present") is not True
            or not isinstance(changes[0]["current"].get("value"), dict)
            or not changes[0]["current"]["value"]
            or _json_identity(evidence.get("permissions")) != _json_identity(changes)):
        raise ValueError("Only the exact recorded missing-to-mesh-policy delta is comparable")

    source_path = Path(evidence["manifest_path"]).resolve(strict=True)
    if (source_path.name != "manifest.json"
            or Path(report["checkpoint"]).resolve() != source_path.parent):
        raise ValueError("Transition source checkpoint path disagrees")
    source = load_checkpoint(source_path.parent)
    original = source["manifest"]
    if (source["manifest_sha256"] != _sha256(evidence["manifest_sha256"])
            or source["manifest_sha256"] != _sha256(evidence["expected_manifest_sha256"])
            or original.get("backend") != "fvm"
            or _json_identity(original["config"]) != _json_identity(left["config"])
            or evidence["source_configuration_sha256"] != original["config_sha256"]
            or evidence["source_vpm_configuration_sha256"] != mapping_digest(original["config"]["vpm"])
            or evidence["current_vpm_configuration_sha256"] != mapping_digest(right["config"]["vpm"])):
        raise ValueError("Transition source/configuration identity disagrees")
    artifacts = {name: {"path": str(contained_path(source_path.parent, relative)),
                        "sha256": original["artifact_sha256"][name]}
                 for name, relative in original["artifacts"].items()}
    if (not {"fvm", "vpm", "vpm_vtu", "vpm_boundary_condition"} <= set(artifacts)
            or _json_identity(evidence.get("source_artifacts")) != _json_identity(artifacts)):
        raise ValueError("Transition source artifact evidence disagrees")
    exchanges = report.get("exchanges")
    final, start = right["coupling_step"], original["coupling_step"]
    if (type(report.get("final_step")) is not int or report["final_step"] != final
            or type(report.get("max_coupling_steps")) is not int
            or not 0 < final-start <= report["max_coupling_steps"]
            or original["n_fvm_substeps"] != right["n_fvm_substeps"]
            or not isinstance(exchanges, list)
            or [item.get("step") for item in exchanges] != list(range(start+1, final+1))
            or not _clock_equal(exchanges[-1].get("time", np.nan), right["time"])
            or not np.isfinite(original["time"]) or not original["time"] < right["time"]):
        raise ValueError("Transition benchmark accepted clock/step does not match candidate")
    if report_path.read_bytes() != raw:
        raise ValueError("Transition benchmark report changed while being read")
    return {"report": str(report_path), "report_sha256": report_sha,
            "configuration_changes": changes,
            "source_manifest_sha256": source["manifest_sha256"],
            "source_artifacts": artifacts,
            "source_configuration_sha256": original["config_sha256"],
            "candidate_configuration_sha256": right["config_sha256"],
            "implementation_inventory_sha256": mapping_digest(before),
            "implementation_files": len(before), "accepted_final_step": final,
            "scope": "Authenticated recorded numerical transition; no accuracy pass inferred"}


def admit_comparison_identity(control, candidate, *, transition_report=None,
                              expected_transition_report_sha256=None):
    left, right = control["manifest"], candidate["manifest"]
    for manifest in (left, right):
        if mapping_digest(manifest["config"]) != manifest["config_sha256"]:
            raise ValueError("Comparison configuration digest mismatch")
    for key in ("coupling_step", "fvm_step", "vpm_step", "n_fvm_substeps"):
        if left[key] != right[key]:
            raise ValueError(f"Unmatched coupled checkpoints: {key}")
    if not _clock_equal(left["time"], right["time"]):
        raise ValueError("Unmatched coupled checkpoint times")
    if transition_report is None:
        if expected_transition_report_sha256 is not None:
            raise ValueError("Transition report digest requires an explicit report")
        for key in ("config", "config_sha256"):
            if _json_identity(left[key]) != _json_identity(right[key]):
                raise ValueError(f"Unmatched coupled checkpoints: {key}")
        return None
    if expected_transition_report_sha256 is None:
        raise ValueError("Transition comparison requires the explicit report SHA-256")
    return admit_mesh_transition(control, candidate, transition_report,
                                 expected_transition_report_sha256)


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
            "Storage-order comparison; duplicate coordinates alone do not establish physical lineage"
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
    left_manifest, right_manifest = control["fvm_manifest"], candidate["fvm_manifest"]
    for name in ("config_hash", "mesh_hash", "n_global_cells", "n_ranks", "kinematic_viscosity"):
        if left_manifest[name] != right_manifest[name]:
            raise ValueError(f"FVM identity mismatch: {name}")
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
        "sha256": artifact_digest(path),
        "record": matches[0],
    }


def gates(checkpoint, diagnostics_path):
    manifest = checkpoint["manifest"]
    limits = manifest["config"]["coupler"]
    diagnostic = matching_diagnostic(diagnostics_path, manifest["coupling_step"], manifest["time"])
    result = {
        "vpm_lagrangian_cfl": checkpoint["solver"].get("lagrangian_cfl"),
        "vpm_lagrangian_cfl_limit": manifest["config"]["vpm"]["health_limits"]["lagrangian_cfl"][
            "maximum"
        ],
        "fvm_max_courant_per_rank": [
            float(rank["max_courant_number"]) for rank in checkpoint["ranks"]
        ],
        "fvm_residual_gate_note": "Residual convergence is not stored in native FVM backup; no residual pass inferred from a committed clock",
        "accepted_diagnostic": diagnostic,
    }
    fvm_diagnostic = matching_diagnostic(
        diagnostics_path.parent / "diagnostics.jsonl", manifest["fvm_step"], manifest["time"]
    )
    result["fvm_accepted_diagnostic"] = fvm_diagnostic
    metadata_path = diagnostics_path.parent / "fvm_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        result["fvm_acceptance_configuration"] = metadata.get("configuration", {}).get("acceptance")
        result["fvm_metadata_sha256"] = artifact_digest(metadata_path)
    if fvm_diagnostic["available"]:
        fvm_row = fvm_diagnostic["record"]
        solves = fvm_row.get("linear_solves", [])
        result["fvm_all_recorded_linear_solves_converged"] = (
            all(solve.get("converged") is True for solve in solves) if solves else None
        )
        result["fvm_nonfinite_value_count"] = fvm_row.get("n_nonfinite_values")
        result["fvm_final_residuals"] = fvm_row.get("residuals")
        result["fvm_residual_gate_note"] = (
            "Convergence flags and residuals are from the unique accepted FVM diagnostic row; "
            "no new residual threshold or pass criterion introduced"
        )
    value, maximum = result["vpm_lagrangian_cfl"], result["vpm_lagrangian_cfl_limit"]
    result["vpm_lagrangian_cfl_gate_met"] = (
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
                "gates_met": bool(
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
                role: artifact_digest(path) if path.is_file() else None
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
    *,
    transition_report=None,
    expected_transition_report_sha256=None,
):
    left, right = control["manifest"], candidate["manifest"]
    transition = admit_comparison_identity(
        control, candidate, transition_report=transition_report,
        expected_transition_report_sha256=expected_transition_report_sha256,
    )
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
        "configuration_transition": transition,
        "control": {
            "directory": str(control["directory"]),
            "manifest_sha256": control["manifest_sha256"],
            "artifact_sha256": left["artifact_sha256"],
            "config_sha256": left["config_sha256"],
        },
        "candidate": {
            "directory": str(candidate["directory"]),
            "manifest_sha256": candidate["manifest_sha256"],
            "artifact_sha256": right["artifact_sha256"],
            "config_sha256": right["config_sha256"],
        },
        "scope": "One matched accepted checkpoint, not long-time trajectory, periodicity, phase or performance certification",
        "fvm": compare_fvm(control, candidate),
        "vpm": compare_particles(control["particle"], candidate["particle"]),
        "vpm_solver_state": solver_state,
        "boundary_history": boundary,
        "boundary_missing_fields": sorted(set(control["boundary"]) ^ set(candidate["boundary"])),
        "gates": {
            "control": gates(control, control_diagnostics),
            "candidate": gates(candidate, candidate_diagnostics),
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
    parser.add_argument("--transition-report", type=Path,
                        help="Explicit completed mesh-transition benchmark report; default is identical config only")
    parser.add_argument("--expected-transition-report-sha256",
                        help="Required exact SHA-256 of --transition-report")
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        output.parent != (Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow") / "solution"
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
        transition_report=args.transition_report,
        expected_transition_report_sha256=args.expected_transition_report_sha256,
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
                "configuration_transition": report["configuration_transition"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
