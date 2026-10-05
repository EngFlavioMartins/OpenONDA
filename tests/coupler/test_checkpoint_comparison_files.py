"""Read-only checkpoint comparison admits identities and exposes unmatched data."""

import csv
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

_PATH = (
    Path(__file__).resolve().parents[2] / "tests/support/cylinder/compare_coupled_checkpoints.py"
)
_SPEC = importlib.util.spec_from_file_location("checkpoint_comparison", _PATH)
comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparison)


def _encode(arrays):
    values, layout = {}, {}
    for name, original in arrays.items():
        value = np.asarray(original)
        if name in comparison._HISTORY_REFERENCE:
            value = value.view(np.uint64) ^ arrays[comparison._HISTORY_REFERENCE[name]].view(
                np.uint64
            )
        if value.ndim and value.size:
            layout[name] = [str(value.dtype), list(value.shape)]
            value = (
                np.ascontiguousarray(value)
                .view(np.uint8)
                .reshape(-1, value.dtype.itemsize)
                .T.copy()
            )
        values[name] = value
    return {**values, "storage_layout": np.asarray(json.dumps(layout))}


def _checkpoint(path):
    path.mkdir()
    config = {
        "vpm": {"state_limits": {"lagrangian_cfl": {"maximum": 1.0}}},
        "coupler": {"interface_normal_tolerance": 1e-5, "interface_gradient_tolerance": 1e-5},
    }
    with h5py.File(path / "vpm.h5", "w") as saved:
        solver = saved.create_group("solver")
        solver.attrs.update(
            time=11.04,
            step=276,
            n_particles_total=3,
            lagrangian_cfl=0.3,
            numerical_configuration=json.dumps(config["vpm"]),
            numerical_configuration_sha256=comparison.mapping_digest(config["vpm"]),
        )
        saved["particles/position"] = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        saved["particles/vortex_strength"] = np.ones((3, 3), dtype=np.float32)
        saved["particles/core_radius"] = np.ones(3, dtype=np.float32)
    fvm = path / "fvm"
    fvm.mkdir()
    fields = {
        "global_cell_id": np.arange(3),
        "global_face_id": np.arange(4),
        "time": np.asarray(11.04),
        "step": np.asarray(1380),
        "n_committed_time_steps": np.asarray(1380),
        "time_step_size": np.asarray(0.008),
        "accepted_time_step_size": np.asarray(0.008),
        "previous_time_step_size": np.asarray(0.008),
        "max_courant_number": np.asarray(0.55),
        "n_consecutive_accepted_steps": np.ones(4, np.int64),
    }
    for name in comparison._FVM_FIELDS:
        shape = (
            (4,)
            if name.startswith("volumetric")
            else (3, 3)
            if name.startswith("velocity")
            else (3,)
        )
        fields[name] = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    np.savez(fvm / "rank.npz", **_encode(fields))
    (fvm / "checkpoint_info.json").write_text(
        json.dumps(
            {
                "format_version": 9,
                "files": ["rank.npz"],
                "config_hash": "fvm-config",
                "mesh_hash": "mesh",
                "n_global_cells": 3,
                "n_ranks": 1,
                "kinematic_viscosity": 0.01,
            }
        )
    )
    np.savez(
        path / "boundary.npz",
        **_encode({"velocity": np.ones((2, 3)), "has_velocity": np.asarray(True)}),
    )
    checkpoint_info = {
        "format_version": 13,
        "kind": "openonda.coupled_backup",
        "config": config,
        "config_sha256": comparison.mapping_digest(config),
        "time": 11.04,
        "coupling_step": 276,
        "vpm_step": 276,
        "fvm_step": 1380,
        "n_fvm_substeps": 5,
        "checkpoint_files": {
            "vpm": "vpm.h5",
            "fvm": "fvm",
            "vpm_boundary_condition": "boundary.npz",
        },
    }
    checkpoint_info["file_sha256"] = {
        key: comparison.checkpoint_path_hash(path / name)
        for key, name in checkpoint_info["checkpoint_files"].items()
    }
    (path / "checkpoint_info.json").write_text(json.dumps(checkpoint_info))
    return fields, checkpoint_info


def test_native_archive_codec_and_digest_validation(tmp_path):
    fields, _ = _checkpoint(tmp_path / "backup")
    loaded = comparison.load_checkpoint(tmp_path / "backup")
    for name, value in fields.items():
        np.testing.assert_array_equal(loaded["ranks"][0][name], value)
    assert loaded["solver"]["lagrangian_cfl"] == 0.3
    with (tmp_path / "backup" / "boundary.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        comparison.load_checkpoint(tmp_path / "backup")


@pytest.mark.parametrize("change", ["config", "clock", "path"])
def test_checkpoint_rejects_bad_configuration_clocks_and_path_escape(tmp_path, change):
    path = tmp_path / "backup"
    _, checkpoint_info = _checkpoint(path)
    if change == "config":
        checkpoint_info["config"]["vpm"]["changed"] = True
    elif change == "clock":
        checkpoint_info["time"] = 11.08
    else:
        checkpoint_info["checkpoint_files"]["vpm"] = "../vpm.h5"
    (path / "checkpoint_info.json").write_text(json.dumps(checkpoint_info))
    with pytest.raises(ValueError):
        comparison.load_checkpoint(path)


def test_coordinate_matching_never_hides_duplicates_or_displaced_particles():
    left = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    right = np.array([[2, 0, 0], [0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=np.float32)
    a, b, detail = comparison.coordinate_alignment(left, right)
    np.testing.assert_array_equal(left[a], right[b])
    assert detail["matched_unique_coordinates"] == 2
    assert detail["reference_particles_at_duplicate_coordinates"] == 2
    assert detail["reference_unmatched_particles"] == detail["candidate_unmatched_particles"] == 2
    result = comparison.compare_particles(
        {"position": left, "vortex_strength": np.ones((4, 3))},
        {"position": right, "vortex_strength": np.full((4, 3), 2.0)},
    )
    assert result["reference_unmatched"]["count"] == 2
    assert (
        result["candidate_unmatched"]["vortex_strength_l1"]
        > result["reference_unmatched"]["vortex_strength_l1"]
    )
    assert not result["ordered_positions_identical"]
    json.dumps(result, allow_nan=False)


def _csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_samples_exact_clock_and_duplicate_handling(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    row = {
        "time": 11.04,
        "step": 1380,
        "position_x": 0.6,
        "position_y": 0.1,
        "position_z": 0.0,
        "velocity_x": 1.0,
    }
    _csv(a / "probe.csv", [row])
    _csv(b / "probe.csv", [{**row, "velocity_x": 1.1}])
    result = comparison.compare_samples(a, b, 11.04, 1380, 276)["probe.csv"]
    assert result["matched_rows"] == 1
    assert result["fields"]["velocity_x"]["max_absolute_difference"] == pytest.approx(0.1)
    _csv(b / "probe.csv", [row, row])
    assert (
        "ambiguous duplicate"
        in comparison.compare_samples(a, b, 11.04, 1380, 276)["probe.csv"]["status"]
    )
    _csv(b / "probe.csv", [{**row, "time": 11.08, "step": 1385}])
    result = comparison.compare_samples(a, b, 11.04, 1380, 276)["probe.csv"]
    assert result["matched_rows"] == 0 and result["reference_unmatched_rows"] == 1


def test_matched_checkpoint_reports_metrics_and_original_checks_without_pass(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _checkpoint(a)
    _checkpoint(b)
    samples = tmp_path / "samples"
    samples.mkdir()
    _csv(
        samples / "forces_history.csv",
        [{"time": 11.04, "step": 1380, "patch": "body", "drag_coefficient": 1.2}],
    )
    diagnostics = tmp_path / "coupler_diagnostics.jsonl"
    diagnostics.write_text(
        json.dumps(
            {
                "step": 276,
                "time": 11.04,
                "interface_iteration": {
                    "accepted_sweep": 3,
                    "converged": True,
                    "residuals": [
                        {"sweep": 3, "normal_residual_rms": 5e-6, "gradient_residual_rms": 8e-6}
                    ],
                },
            }
        )
        + "\n"
    )
    (tmp_path / "diagnostics.jsonl").write_text(
        json.dumps(
            {
                "step": 1380,
                "time": 11.04,
                "linear_solves": [{"converged": True}],
                "residuals": {"velocity": 1e-8},
                "n_nonfinite_values": 0,
            }
        )
        + "\n"
    )
    (tmp_path / "fvm_metadata.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "acceptance": {"max_courant_number_abort": None},
                }
            }
        )
    )
    result = comparison.compare_checkpoints(
        comparison.load_checkpoint(a),
        comparison.load_checkpoint(b),
        samples,
        samples,
        diagnostics,
        diagnostics,
    )
    assert result["status"] == "comparison_complete_no_accuracy_pass_inferred"
    assert result["numerical_checks"]["control"]["vpm_lagrangian_cfl_within_limit"]
    assert result["numerical_checks"]["candidate"]["interface"]["checks_passed"]
    assert result["numerical_checks"]["candidate"]["fvm_all_recorded_linear_solves_converged"]
    assert result["numerical_checks"]["candidate"]["fvm_final_residuals"] == {"velocity": 1e-8}
    assert result["fvm"][0]["fields"]["velocity"]["exactly_equal"]
    assert result["vpm"]["alignment"]["matched_unique_coordinates"] == 3
    json.dumps(result, allow_nan=False)
