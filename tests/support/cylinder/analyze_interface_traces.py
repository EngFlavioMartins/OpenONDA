"""Read-only admission and offline initial-guess analysis of captured traces.

Endpoint prediction error is not a fixed-point residual: this tool cannot
establish convergence, fewer sweeps, stability, or accuracy of a new solver.
It never advances a state or shifts, interpolates, or reorders observations.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

_FIELDS = ("velocity", "normal_velocity", "tangential_gradient")
_GEOMETRY = ("face_centre", "face_normal", "face_area")
_CLOCK_ATOL = 1e-10


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        value.dtype.str.encode() + repr(value.shape).encode() + value.tobytes()
    ).hexdigest()


def _equal(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)


def _clock(value):
    if set(value) != {"fvm", "vpm"}:
        raise ValueError("Trace clock requires both FVM and VPM")
    for part in value.values():
        if (
            set(part) != {"time", "step"}
            or type(part["step"]) is not int
            or part["step"] < 0
            or not np.isfinite(part["time"])
        ):
            raise ValueError("Invalid trace clock")
    return value


def _same_clock(left, right):
    return all(
        left[name]["step"] == right[name]["step"]
        and abs(left[name]["time"] - right[name]["time"]) <= _CLOCK_ATOL
        for name in ("fvm", "vpm")
    )


def load_manifest(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = json.loads(payload)
    if manifest.get("kind") != "openonda.coupled_backup" or manifest.get("format_version") != 12:
        raise ValueError("Trace analysis requires a native v12 coupled manifest")
    config = manifest["config"]
    if mapping_digest(config) != manifest["config_sha256"]:
        raise ValueError("Manifest numerical configuration digest mismatch")
    settings = {
        "normal_tolerance": float(config["coupler"]["interface_normal_tolerance"]),
        "gradient_tolerance": float(config["coupler"]["interface_gradient_tolerance"]),
        "exchange_dt": float(config["vpm"]["time_step_size"]),
        "fvm_substeps": manifest["n_fvm_substeps"],
    }
    if (
        type(settings["fvm_substeps"]) is not int
        or settings["fvm_substeps"] < 1
        or any(
            not np.isfinite(settings[key]) or settings[key] <= 0
            for key in settings
            if key != "fvm_substeps"
        )
    ):
        raise ValueError("Invalid unchanged interface tolerances or exchange clocks")
    if file_digest(path) != digest:
        raise ValueError("Manifest changed while being read")
    return settings, {
        "path": str(path),
        "sha256": digest,
        "config_sha256": manifest["config_sha256"],
        "admission_scope": "Manifest schema/config digest; solver checkpoint artifacts are not reread",
    }


def _trace(arrays, event, count):
    if set(event["arrays"]) != set(_FIELDS):
        raise ValueError(
            "Trace event requires full velocity, normal velocity and tangential gradient"
        )
    values = {field: arrays[event["arrays"][field]] for field in _FIELDS}
    for field, value in values.items():
        shape = (count,) if field == "normal_velocity" else (count, 3)
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"Invalid/nonfinite trace field: {field}")
    return values


def _gate_values(before, after, areas):
    # Preserve the recorded field dtypes/arithmetic when auditing production
    # diagnostics. Analysis vectors are separately promoted to f64 below.
    normal = after["normal_velocity"] - before["normal_velocity"]
    gradient = after["tangential_gradient"] - before["tangential_gradient"]
    return (
        float(np.sqrt(np.average(normal**2, weights=areas))),
        float(np.sqrt(np.average(np.sum(gradient**2, axis=1), weights=areas))),
    )


def load_trace(path, settings):
    path = Path(path).resolve()
    digest = file_digest(path)
    with np.load(path, allow_pickle=False) as stored:
        if len(stored.files) != len(set(stored.files)):
            raise ValueError("Duplicate arrays in trace archive")
        metadata = json.loads(str(stored["metadata_json"]))
        arrays = {key: stored[key].copy() for key in stored.files if key != "metadata_json"}
    if file_digest(path) != digest:
        raise ValueError("Trace changed while being read")
    schema = metadata.get("schema_version")
    if schema != 2 or metadata.get("status") != "complete":
        raise ValueError("Trace is missing, failed, incomplete or unsupported")
    diagnostics = metadata["interface_iteration"]
    if diagnostics.get("converged") is not True:
        raise ValueError("Nonconverged exchange cannot supply a validated accepted endpoint")
    geometry = {name: arrays[name] for name in _GEOMETRY}
    count = len(geometry["face_area"])
    if count < 1:
        raise ValueError("Empty trace geometry")
    for name, value in geometry.items():
        shape = (count,) if name == "face_area" else (count, 3)
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("Invalid/nonfinite trace geometry")
    if np.any(geometry["face_area"] < 0) or np.sum(geometry["face_area"]) <= 0:
        raise ValueError("Invalid face-area weights")
    unit_error = np.max(np.abs(np.linalg.norm(geometry["face_normal"], axis=1) - 1))
    if unit_error > 64 * np.finfo(geometry["face_normal"].dtype).eps:
        raise ValueError("Face normals are not unit vectors")
    entry, exit_clock = _clock(metadata["entry_clocks"]), _clock(metadata["exit_clocks"])
    step = entry["vpm"]["step"]
    end_time = entry["vpm"]["time"]
    expected_exit = {
        "fvm": {"step": step * settings["fvm_substeps"], "time": end_time},
        "vpm": entry["vpm"],
    }
    expected_entry = {
        "fvm": {
            "step": (step - 1) * settings["fvm_substeps"],
            "time": end_time - settings["exchange_dt"],
        },
        "vpm": entry["vpm"],
    }
    if not _same_clock(exit_clock, expected_exit) or not _same_clock(entry, expected_entry):
        raise ValueError("Discontinuous FVM/VPM exchange clocks")
    rows = diagnostics["residuals"]
    sweeps = diagnostics["sweeps"]
    accepted = diagnostics["accepted_sweep"]
    if (
        type(sweeps) is not int
        or sweeps < 1
        or len(rows) != sweeps
        or type(accepted) is not int
        or not 1 <= accepted <= sweeps
        or metadata["accepted_sweep"] != accepted
    ):
        raise ValueError("Missing/inconsistent interface trial diagnostics")
    labels = ["old_physical_endpoint", "raw_predictor"]
    labels += [label for _ in rows for label in ("trial_input", "trial_output")]
    labels += ["accepted_endpoint"]
    events = metadata["events"]
    if [event["label"] for event in events] != labels:
        raise ValueError("Missing or reordered trace events")
    values = []
    claimed_keys = set(_GEOMETRY)
    for sequence, event in enumerate(events):
        if event["sequence"] != sequence:
            raise ValueError("Discontinuous trace event sequence")
        expected_clock = (
            exit_clock if event["label"] in {"trial_output", "accepted_endpoint"} else entry
        )
        if not _same_clock(_clock(event["clocks"]), expected_clock):
            raise ValueError("Trace event clock disagrees with its physical endpoint")
        keys = set(event["arrays"].values())
        if len(keys) != len(_FIELDS) or keys & claimed_keys:
            raise ValueError("Trace events alias/reuse an observation array")
        claimed_keys |= keys
        value = _trace(arrays, event, count)
        reconstructed = np.einsum("ij,ij->i", value["velocity"], geometry["face_normal"])
        epsilon = max(np.finfo(array.dtype).eps for array in value.values())
        scale = max(1.0, float(np.max(np.abs(value["velocity"]))))
        if np.max(np.abs(reconstructed - value["normal_velocity"])) > 64 * epsilon * scale:
            raise ValueError("Normal velocity is inconsistent with the full vector trace")
        values.append(value)
    if set(arrays) != claimed_keys:
        raise ValueError("Unexpected unassociated arrays in trace archive")
    for index, row in enumerate(rows, 1):
        event_in, event_out = events[2 * index], events[2 * index + 1]
        if (
            row["sweep"] != index
            or event_in.get("trial") != index
            or event_out.get("trial") != index
        ):
            raise ValueError("Missing/discontinuous trial numbering")
        before, after = values[2 * index : 2 * index + 2]
        normal, gradient = _gate_values(before, after, geometry["face_area"])
        expected = {
            "normal_residual_rms": normal,
            "gradient_residual_rms": gradient,
            "scaled_residual": max(
                normal / settings["normal_tolerance"], gradient / settings["gradient_tolerance"]
            ),
        }
        epsilon = max(np.finfo(array.dtype).eps for array in before.values())
        if any(
            not math.isclose(row[key], value, rel_tol=64 * epsilon, abs_tol=0.0)
            for key, value in expected.items()
        ):
            raise ValueError("Recorded trial residual disagrees with captured arrays/tolerances")
        converged = (
            normal <= settings["normal_tolerance"] and gradient <= settings["gradient_tolerance"]
        )
        if row["converged"] is not converged:
            raise ValueError("Recorded trial convergence disagrees with unchanged gates")
    selected = rows[accepted - 1]
    if selected.get("accepted") is not True or selected.get("converged") is not True:
        raise ValueError("Selected accepted endpoint did not meet both unchanged gates")
    prediction = diagnostics.get("prediction", {"attempted": False})
    if not isinstance(prediction, dict) or type(prediction.get("attempted")) is not bool:
        raise ValueError("Invalid interface-prediction diagnostics")
    attempted = prediction["attempted"]
    raw_trial = 2
    if attempted:
        if rows[0].get("prediction_probe") is not True:
            raise ValueError("Seed probe lacks explicit supported trace provenance")
        if prediction.get("reason") != "previous_accepted_correction":
            raise ValueError("Unknown interface seed strategy")
        rejected = not rows[0]["converged"]
        if (
            rows[0].get("prediction_rejected") is not rejected
            or rows[0]["accepted"] is rejected
            or prediction.get("accepted") is rejected
            or prediction.get("fallback") is not rejected
            or any(row.get("prediction_probe") is True for row in rows[1:])
            or [row.get("picard_sweep") for row in rows] != list(range(sweeps))
            or (not rejected and (sweeps != 1 or accepted != 1))
            or (rejected and (sweeps < 2 or accepted == 1))
        ):
            raise ValueError("Inconsistent seed acceptance/fallback evidence")
        raw_trial = 4 if rejected else None
    elif any(row.get("prediction_probe") is True for row in rows):
        raise ValueError("Unreported seed probe")
    for field in _FIELDS:
        if raw_trial is not None and not _equal(values[1][field], values[raw_trial][field]):
            raise ValueError("First baseline trial input is not the recorded raw predictor")
        if not _equal(values[-1][field], values[2 * accepted + 1][field]):
            raise ValueError("Accepted endpoint does not match its selected trial output")
    return {
        "path": str(path),
        "sha256": digest,
        "step": step,
        "time": end_time,
        "entry": entry,
        "exit": exit_clock,
        "geometry": geometry,
        "old": values[0],
        "raw": values[1],
        "accepted": values[-1],
        "sweeps": sweeps,
        "accepted_sweep": accepted,
        "accepted_diagnostics": selected,
        "prediction": prediction,
        "first_trial": values[2],
    }


def _difference(left, right):
    return {field: np.asarray(left[field], dtype=np.float64) - right[field] for field in _FIELDS}


def _rms(value, areas):
    squared = value**2 if value.ndim == 1 else np.sum(value**2, axis=1)
    return float(np.sqrt(np.average(squared, weights=areas)))


def _inner(left, right, areas, settings):
    products = (
        np.sum(left["velocity"] * right["velocity"], axis=1) / settings["normal_tolerance"] ** 2
    )
    products += (
        np.sum(left["tangential_gradient"] * right["tangential_gradient"], axis=1)
        / settings["gradient_tolerance"] ** 2
    )
    return float(np.average(products, weights=areas))


def _metrics(vector, areas, settings):
    if any(not np.isfinite(value).all() for value in vector.values()):
        raise ValueError("Nonfinite derived trace difference")
    rms = {field: _rms(value, areas) for field, value in vector.items()}
    inner = _inner(vector, vector, areas, settings)
    if not all(np.isfinite(value) for value in (*rms.values(), inner)):
        raise ValueError("Trace-difference norm overflow")
    return {
        "area_weighted_rms": rms,
        "max_abs_component": {
            field: float(np.max(np.abs(value))) for field, value in vector.items()
        },
        "normal_error_over_unchanged_tolerance": rms["normal_velocity"]
        / settings["normal_tolerance"],
        "gradient_error_over_unchanged_tolerance": rms["tangential_gradient"]
        / settings["gradient_tolerance"],
        "full_velocity_gradient_scaled_norm": math.sqrt(max(0.0, inner)),
    }


def _run_provenance(paths, traces, manifest):
    reports, provenance, linked, observed_sources = [], [], {}, {}
    for path in paths:
        path = Path(path).resolve()
        digest = file_digest(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if file_digest(path) != digest:
            raise ValueError("Benchmark report changed while being read")
        before = report.get("source_hashes_at_construction", {})
        after = report.get("source_hashes_at_completion", {})
        if (
            report.get("status") != "complete"
            or not before
            or not after
            or report.get("loaded_source_files_changed_during_run") != []
            or any(after.get(key) != value for key, value in before.items())
        ):
            raise ValueError("Benchmark source provenance is missing, incomplete or changed")
        for key, value in after.items():
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                or key in observed_sources
                and observed_sources[key] != value
            ):
                raise ValueError("Source digests are invalid or differ between consecutive runs")
            observed_sources[key] = value
        _, checkpoint = load_manifest(Path(report["checkpoint"]) / "manifest.json")
        if checkpoint["config_sha256"] != manifest["config_sha256"]:
            raise ValueError("Benchmark checkpoint configuration differs from analysis manifest")
        for trace in report["interface_traces"]:
            key = str(Path(trace["path"]).resolve())
            if key in linked:
                raise ValueError("A trace is claimed by multiple benchmark records")
            linked[key] = (trace, report)
        reports.append(report)
        provenance.append(
            {
                "path": str(path),
                "sha256": digest,
                "source_root": report["source_root"],
                "source_hashes_at_construction": before,
                "source_hashes_at_completion": after,
                "checkpoint_manifest": checkpoint,
            }
        )
    if not reports or len({report["source_root"] for report in reports}) != 1:
        raise ValueError("A consistent benchmark source root and completed report are required")
    for trace in traces:
        if trace["path"] not in linked:
            raise ValueError("Trace is not associated with a supplied completed benchmark")
        recorded, report = linked[trace["path"]]
        if (
            recorded["status"] != "complete"
            or recorded["step"] != trace["step"]
            or abs(recorded["time"] - trace["time"]) > _CLOCK_ATOL
            or recorded["trials"] != trace["sweeps"]
            or recorded["accepted_sweep"] != trace["accepted_sweep"]
        ):
            raise ValueError("Benchmark trace index disagrees with captured exchange")
        exchange = [row for row in report["exchanges"] if row["step"] == trace["step"]]
        if len(exchange) != 1 or abs(exchange[0]["time"] - trace["time"]) > _CLOCK_ATOL:
            raise ValueError("Benchmark has no unique accepted exchange at the trace clock")
    return provenance


def analyze(paths, manifest_path, run_reports):
    settings, manifest = load_manifest(manifest_path)
    traces = [load_trace(path, settings) for path in paths]
    if not traces or len({trace["path"] for trace in traces}) != len(traces):
        raise ValueError("Provide one or more unique traces in physical time order")
    provenance = _run_provenance(run_reports, traces, manifest)
    first = traces[0]
    areas, normals = first["geometry"]["face_area"], first["geometry"]["face_normal"]
    vectors, rows, corrections = {}, [], []
    for index, trace in enumerate(traces):
        if any(not _equal(trace["geometry"][key], first["geometry"][key]) for key in _GEOMETRY):
            raise ValueError("Geometry or face ordering changed; no remapping is permitted")
        if index:
            previous = traces[index - 1]
            if (
                trace["step"] != previous["step"] + 1
                or abs(trace["time"] - previous["time"] - settings["exchange_dt"]) > _CLOCK_ATOL
                or trace["entry"]["fvm"]["step"] != previous["exit"]["fvm"]["step"]
                or abs(trace["entry"]["fvm"]["time"] - previous["exit"]["fvm"]["time"])
                > _CLOCK_ATOL
            ):
                raise ValueError("Trace records are not consecutive exchanges")
            if any(not _equal(trace["old"][key], previous["accepted"][key]) for key in _FIELDS):
                raise ValueError(
                    "Old physical endpoint does not exactly continue the previous accepted trace"
                )
        correction = _difference(trace["accepted"], trace["raw"])
        if index and trace["prediction"]["attempted"]:
            previous_correction = _difference(previous["accepted"], previous["raw"])
            expected_seed = {
                field: trace["raw"][field] + previous_correction[field] for field in _FIELDS
            }
            expected_seed["normal_velocity"] = np.einsum(
                "ij,ij->i", expected_seed["velocity"], normals
            )
            if any(
                not _equal(expected_seed[field], trace["first_trial"][field]) for field in _FIELDS
            ):
                raise ValueError("Recorded seed is not the previous accepted-minus-raw correction")
        corrections.append(correction)
        row = {
            key: trace[key]
            for key in ("step", "time", "sweeps", "accepted_sweep", "accepted_diagnostics")
        }
        row["correction"] = _metrics(correction, areas, settings)
        row["prediction"] = trace["prediction"]
        row["first_trial_endpoint_error"] = _metrics(
            _difference(trace["accepted"], trace["first_trial"]), areas, settings
        )
        row["seed_history_reconstruction"] = (
            "verified_against_previous_trace"
            if index and trace["prediction"]["attempted"]
            else "not_attempted"
            if not trace["prediction"]["attempted"]
            else "previous_exchange_not_supplied; gate_verified_only"
        )
        row["correction_arrays"] = {}
        for field, value in correction.items():
            key = f"step_{trace['step']:06d}_correction_{field}"
            vectors[key] = value
            row["correction_arrays"][field] = key
        if index:
            before = corrections[index - 1]
            product = _inner(before, correction, areas, settings)
            norm_before = math.sqrt(max(0.0, _inner(before, before, areas, settings)))
            norm_now = row["correction"]["full_velocity_gradient_scaled_norm"]
            field_cosines = {}
            for field in _FIELDS:
                denominator = _rms(before[field], areas) * _rms(correction[field], areas)
                products = before[field] * correction[field]
                if products.ndim > 1:
                    products = np.sum(products, axis=1)
                field_cosines[field] = (
                    None
                    if denominator == 0
                    else float(np.clip(np.average(products, weights=areas) / denominator, -1, 1))
                )
            row["successive_corrections"] = {
                "scaled_full_trace_cosine": None
                if norm_before * norm_now == 0
                else float(np.clip(product / (norm_before * norm_now), -1, 1)),
                "scaled_norm_ratio": None if norm_before == 0 else norm_now / norm_before,
                "area_weighted_field_cosines": field_cosines,
                "drift": _metrics(_difference(correction, before), areas, settings),
            }
        candidates = {"raw_predictor": None}
        if index:
            candidates["previous_correction_reuse"] = corrections[index - 1]
        if index >= 2:
            candidates["linear_correction_extrapolation"] = {
                field: 2 * corrections[index - 1][field] - corrections[index - 2][field]
                for field in _FIELDS
            }
        row["offline_candidate_endpoint_errors"] = {}
        for name, added in candidates.items():
            predicted = {
                field: np.asarray(trace["raw"][field], dtype=np.float64).copy() for field in _FIELDS
            }
            if added is not None:
                for field in ("velocity", "tangential_gradient"):
                    predicted[field] += added[field]
                predicted["normal_velocity"] = np.einsum("ij,ij->i", predicted["velocity"], normals)
            error = _difference(predicted, trace["accepted"])
            metrics = _metrics(error, areas, settings)
            metrics["error_arrays"] = {}
            for field, value in error.items():
                key = f"step_{trace['step']:06d}_{name}_error_{field}"
                vectors[key] = value
                metrics["error_arrays"][field] = key
            row["offline_candidate_endpoint_errors"][name] = metrics
        rows.append(row)
    for trace in traces:
        if file_digest(trace["path"]) != trace["sha256"]:
            raise ValueError("Trace changed during analysis")
    for document in [manifest, *provenance]:
        if file_digest(document["path"]) != document["sha256"]:
            raise ValueError("Manifest or benchmark report changed during analysis")
    report = {
        "schema_version": 1,
        "status": "validated_offline_endpoint_analysis",
        "scope": "Candidate-to-accepted endpoint errors are not fixed-point residuals or convergence/accuracy proofs; no time shifts or solver changes",
        "provenance_limit": "Captured traces have no embedded solver/config hash; completed benchmark trace indexes and unchanged source hashes bind explicit external provenance, audited against captured residuals and clocks",
        "manifest": manifest,
        "settings": settings,
        "benchmark_reports": provenance,
        "inputs": [
            {key: trace[key] for key in ("path", "sha256", "step", "time")} for trace in traces
        ],
        "geometry_sha256": {key: array_digest(value) for key, value in first["geometry"].items()},
        "vector_units": {"velocity": "m/s", "normal_velocity": "m/s", "tangential_gradient": "1/s"},
        "weighting_definition": "Exact captured face areas; full-vector norm scales u by the existing normal tolerance and gradient by its existing tolerance, without double-counting the dependent normal component",
        "correction_definition": "accepted_endpoint minus raw_predictor at the same physical endpoint",
        "linear_extrapolation_definition": "raw_n + 2*correction_(n-1) - correction_(n-2), fixed exchange dt; reconstruct normal component from full velocity",
        "exchanges": rows,
        "analyzer_sha256": file_digest(__file__),
    }
    return report, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--run-reports", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New prefix directly under this case's solution/"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    case = (
        Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"
    )
    if output.parent != case / "solution":
        raise ValueError(
            "Analysis output must stay directly in the ordinary case solution directory"
        )
    json_path, npz_path = output.with_suffix(".json"), output.with_suffix(".npz")
    if json_path.exists() or npz_path.exists():
        raise FileExistsError(
            "Analysis outputs must be new; existing evidence is never overwritten"
        )
    report, vectors = analyze(args.traces, args.manifest, args.run_reports)
    with npz_path.open("xb") as stream:
        np.savez(stream, **vectors)
    report["vectors"] = {"path": str(npz_path), "sha256": file_digest(npz_path)}
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "exchanges": len(report["exchanges"]),
                "report": str(json_path),
            }
        )
    )


if __name__ == "__main__":
    main()
