"""Offline trace estimates require complete, ordered, converged evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def rig(tmp_path):
    path = (
        Path(__file__).resolve().parents[2]
        / "tests/support/cylinder/analyze_interface_traces.py"
    )
    spec = importlib.util.spec_from_file_location("trace_analysis_asset", path)
    asset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asset)
    config = {
        "coupler": {"interface_normal_tolerance": 1.0, "interface_gradient_tolerance": 2.0},
        "vpm": {"time_step_size": 0.04},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "openonda.coupled_backup",
                "format_version": 11,
                "config": config,
                "config_sha256": asset.mapping_digest(config),
                "n_fvm_substeps": 5,
            }
        )
    )
    geometry = {
        "face_centre": np.array([[0.0, 0, 0], [1.0, 0, 0]]),
        "face_normal": np.array([[1.0, 0, 0], [0.0, 1, 0]]),
        "face_area": np.array([1.0, 3.0]),
    }
    old = {
        "velocity": np.zeros((2, 3)),
        "normal_velocity": np.zeros(2),
        "tangential_gradient": np.zeros((2, 3)),
    }
    paths, index, exchanges = [], [], []
    for step in range(1, 4):
        raw = {field: value + 0.001 for field, value in old.items()}
        raw["normal_velocity"] = np.einsum("ij,ij->i", raw["velocity"], geometry["face_normal"])
        correction = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]) * step
        accepted = {
            "velocity": raw["velocity"] + correction,
            "tangential_gradient": raw["tangential_gradient"] + correction * 2,
        }
        accepted["normal_velocity"] = np.einsum(
            "ij,ij->i", accepted["velocity"], geometry["face_normal"]
        )
        entry = {
            "vpm": {"step": step, "time": step * 0.04},
            "fvm": {"step": (step - 1) * 5, "time": (step - 1) * 0.04},
        }
        exit_clock = {"vpm": entry["vpm"], "fvm": {"step": step * 5, "time": step * 0.04}}
        arrays = {name: value.copy() for name, value in geometry.items()}
        events = []
        for label, trace in (
            ("old_physical_endpoint", old),
            ("raw_predictor", raw),
            ("trial_input", raw),
            ("trial_output", accepted),
            ("accepted_endpoint", accepted),
        ):
            sequence = len(events)
            event = {
                "sequence": sequence,
                "label": label,
                "clocks": exit_clock if label in {"trial_output", "accepted_endpoint"} else entry,
                "arrays": {},
            }
            if label.startswith("trial_"):
                event["trial"] = 1
            for field, value in trace.items():
                key = f"event_{sequence:03d}_{field}"
                event["arrays"][field] = key
                arrays[key] = value.copy()
            events.append(event)
        normal, gradient = asset._gate_values(raw, accepted, geometry["face_area"])
        row = {
            "sweep": 1,
            "normal_residual_rms": normal,
            "gradient_residual_rms": gradient,
            "scaled_residual": max(normal, gradient / 2),
            "accepted": True,
            "converged": True,
        }
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "entry_clocks": entry,
            "exit_clocks": exit_clock,
            "events": events,
            "accepted_sweep": 1,
            "interface_iteration": {
                "sweeps": 1,
                "accepted_sweep": 1,
                "converged": True,
                "residuals": [row],
            },
        }
        trace_path = tmp_path / f"trace-{step}.npz"
        np.savez(trace_path, metadata_json=np.asarray(json.dumps(metadata)), **arrays)
        paths.append(trace_path)
        index.append(
            {
                "path": str(trace_path),
                "status": "complete",
                "step": step,
                "time": step * 0.04,
                "trials": 1,
                "accepted_sweep": 1,
            }
        )
        exchanges.append({"step": step, "time": step * 0.04})
        old = accepted
    run_report = tmp_path / "benchmark.json"
    run_report.write_text(
        json.dumps(
            {
                "status": "complete",
                "checkpoint": str(tmp_path),
                "source_root": "/qualified/source",
                "source_hashes_at_construction": {"/qualified/source/operator.py": "0" * 64},
                "source_hashes_at_completion": {"/qualified/source/operator.py": "0" * 64},
                "loaded_source_files_changed_during_run": [],
                "interface_traces": index,
                "exchanges": exchanges,
            }
        )
    )
    return asset, paths, manifest, run_report


def _mutate_trace(path, transform):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        arrays = {key: saved[key].copy() for key in saved.files if key != "metadata_json"}
    transform(metadata, arrays)
    np.savez(path, metadata_json=np.asarray(json.dumps(metadata)), **arrays)


def _seed_record(asset, path, seed, *, reject=False):
    def transform(metadata, arrays):
        metadata["schema_version"] = 2
        original = metadata["events"]

        def values(event):
            return {field: arrays[key].copy() for field, key in event["arrays"].items()}

        old, raw, _, accepted, _ = [values(event) for event in original]
        geometry = {key: arrays[key] for key in asset._GEOMETRY}
        arrays.clear()
        arrays.update(geometry)
        sequence = [
            ("old_physical_endpoint", old),
            ("raw_predictor", raw),
            ("trial_input", seed),
            ("trial_output", accepted),
        ]
        if reject:
            sequence.extend((("trial_input", raw), ("trial_output", accepted)))
        sequence.append(("accepted_endpoint", accepted))
        events = []
        for index, (label, trace) in enumerate(sequence):
            event = {
                "sequence": index,
                "label": label,
                "arrays": {},
                "clocks": metadata["exit_clocks"]
                if label in ("trial_output", "accepted_endpoint")
                else metadata["entry_clocks"],
            }
            if label.startswith("trial_"):
                event["trial"] = index // 2
            for field, value in trace.items():
                key = f"event_{index:03d}_{field}"
                arrays[key] = value.copy()
                event["arrays"][field] = key
            events.append(event)
        rows = []
        for index in range(1 + int(reject)):
            normal, gradient = asset._gate_values(
                seed if index == 0 else raw, accepted, geometry["face_area"]
            )
            rows.append(
                {
                    "sweep": index + 1,
                    "normal_residual_rms": normal,
                    "gradient_residual_rms": gradient,
                    "scaled_residual": max(normal, gradient / 2),
                    "accepted": index == 1 or not reject,
                    "converged": normal <= 1 and gradient <= 2,
                    "prediction_probe": index == 0,
                    "prediction_rejected": index == 0 and reject,
                    "picard_sweep": index,
                }
            )
        metadata["events"] = events
        metadata["accepted_sweep"] = len(rows)
        metadata["interface_iteration"] = {
            "sweeps": len(rows),
            "accepted_sweep": len(rows),
            "converged": True,
            "residuals": rows,
            "prediction": {
                "attempted": True,
                "reason": "previous_accepted_correction",
                "accepted": not reject,
                "fallback": reject,
            },
        }

    _mutate_trace(path, transform)


def test_seeded_initial_trace_is_verified_against_previous_raw_correction(rig):
    asset, paths, manifest, run_report = rig
    settings, _ = asset.load_manifest(manifest)
    previous = asset.load_trace(paths[0], settings)
    current = asset.load_trace(paths[1], settings)
    # Preserve the exact production operation order (raw + (accepted - raw)).
    seed = {
        field: current["raw"][field] + (previous["accepted"][field] - previous["raw"][field])
        for field in asset._FIELDS
    }
    seed["normal_velocity"] = np.einsum(
        "ij,ij->i", seed["velocity"], current["geometry"]["face_normal"]
    )
    _seed_record(asset, paths[1], seed)
    report, _ = asset.analyze(paths, manifest, [run_report])
    assert (
        report["exchanges"][1]["seed_history_reconstruction"] == "verified_against_previous_trace"
    )
    seed["velocity"][0, 2] += 1e-4
    _seed_record(asset, paths[1], seed)
    with pytest.raises(ValueError, match="previous accepted-minus-raw"):
        asset.analyze(paths, manifest, [run_report])


def test_rejected_seed_requires_a_fresh_raw_baseline_trace(rig):
    asset, paths, manifest, _ = rig
    settings, _ = asset.load_manifest(manifest)
    current = asset.load_trace(paths[0], settings)
    seed = {field: value + 10 for field, value in current["raw"].items()}
    seed["normal_velocity"] = np.einsum(
        "ij,ij->i", seed["velocity"], current["geometry"]["face_normal"]
    )
    _seed_record(asset, paths[0], seed, reject=True)
    checked = asset.load_trace(paths[0], settings)
    assert checked["accepted_sweep"] == 2 and checked["prediction"]["fallback"]

    def corrupt(metadata, arrays):
        metadata["interface_iteration"]["prediction"]["fallback"] = False

    _mutate_trace(paths[0], corrupt)
    with pytest.raises(ValueError, match="seed acceptance/fallback"):
        asset.load_trace(paths[0], settings)


def test_legacy_schema_does_not_silently_admit_a_seed(rig):
    asset, paths, manifest, _ = rig
    settings, _ = asset.load_manifest(manifest)
    current = asset.load_trace(paths[0], settings)
    _seed_record(asset, paths[0], current["accepted"])

    def legacy(metadata, arrays):
        metadata["schema_version"] = 1

    _mutate_trace(paths[0], legacy)
    with pytest.raises(ValueError, match="supported trace provenance"):
        asset.load_trace(paths[0], settings)


def test_linear_correction_reconstruction_and_area_weighted_errors(rig):
    asset, paths, manifest, run_report = rig
    report, vectors = asset.analyze(paths, manifest, [run_report])
    assert len(report["exchanges"]) == 3
    first, second, third = report["exchanges"]
    assert set(first["offline_candidate_endpoint_errors"]) == {"raw_predictor"}
    assert third["successive_corrections"]["scaled_full_trace_cosine"] == pytest.approx(1)
    assert third["successive_corrections"]["scaled_norm_ratio"] == pytest.approx(1.5)
    metric = third["offline_candidate_endpoint_errors"]["linear_correction_extrapolation"]
    assert metric["full_velocity_gradient_scaled_norm"] < 1e-15
    expected_normal = np.sqrt((0.01**2 + 3 * 0.05**2) / 4)
    assert first["correction"]["area_weighted_rms"]["normal_velocity"] == pytest.approx(
        expected_normal
    )
    assert second["offline_candidate_endpoint_errors"]["previous_correction_reuse"][
        "area_weighted_rms"
    ]["normal_velocity"] == pytest.approx(expected_normal)
    np.testing.assert_allclose(
        vectors[first["correction_arrays"]["velocity"]], [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]
    )
    assert "not fixed-point residuals" in report["scope"]


@pytest.mark.parametrize(
    "mutation",
    [
        "failed",
        "nonconverged",
        "missing",
        "residual",
        "nonfinite",
        "clock",
        "accepted",
        "geometry",
        "old_endpoint",
    ],
)
def test_incomplete_or_discontinuous_evidence_is_rejected(rig, mutation):
    asset, paths, manifest, run_report = rig

    def corrupt(metadata, arrays):
        if mutation == "failed":
            metadata["status"] = "failed"
        elif mutation == "nonconverged":
            metadata["interface_iteration"]["converged"] = False
        elif mutation == "missing":
            metadata["events"].pop(2)
        elif mutation == "residual":
            metadata["interface_iteration"]["residuals"][0]["normal_residual_rms"] *= 2
        elif mutation == "nonfinite":
            arrays["event_001_tangential_gradient"][0, 0] = np.nan
        elif mutation == "clock":
            metadata["exit_clocks"]["fvm"]["time"] += 0.001
        elif mutation == "accepted":
            arrays["event_004_tangential_gradient"][0, 0] += 0.001
        elif mutation == "geometry":
            arrays["face_centre"] = arrays["face_centre"][::-1]
        elif mutation == "old_endpoint":
            arrays["event_000_tangential_gradient"][0, 0] += 0.001

    _mutate_trace(paths[1], corrupt)
    with pytest.raises(ValueError):
        asset.analyze(paths, manifest, [run_report])


@pytest.mark.parametrize("order", [[1, 0, 2], [0, 2], [0, 0, 1]])
def test_reordered_skipped_or_duplicate_exchanges_are_rejected(rig, order):
    asset, paths, manifest, run_report = rig
    with pytest.raises(ValueError):
        asset.analyze([paths[index] for index in order], manifest, [run_report])


def test_changed_benchmark_source_is_rejected(rig):
    asset, paths, manifest, run_report = rig
    report = json.loads(run_report.read_text())
    report["source_hashes_at_completion"]["/qualified/source/operator.py"] = "1" * 64
    run_report.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source provenance"):
        asset.analyze(paths, manifest, [run_report])


def test_manifest_configuration_tampering_is_rejected(rig):
    asset, paths, manifest, run_report = rig
    record = json.loads(manifest.read_text())
    record["config"]["coupler"]["interface_normal_tolerance"] *= 2
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="digest mismatch"):
        asset.analyze(paths, manifest, [run_report])


def test_missing_benchmark_trace_association_is_rejected(rig):
    asset, paths, manifest, run_report = rig
    report = json.loads(run_report.read_text())
    report["interface_traces"].pop()
    run_report.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not associated"):
        asset.analyze(paths, manifest, [run_report])
