"""Portable cube comparisons retain a complete, immutable input closure."""

import hashlib
import json

import pytest

from tests._tutorial_helpers import load_tutorial_module


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    util = load_tutorial_module("coupled_fvm_vpm/cube_flow", "assets.postprocess")
    monkeypatch.setattr(util, "CASE_DIR", tmp_path)
    monkeypatch.setattr(util, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(util, "COMPARISON", tmp_path / "samples/comparison")
    util.prepared_inputs.cache_clear()
    paths = [
        "samples/comparison/manifest.json",
        "samples/comparison/fields_t0.250000000.npz",
        "samples/vpm_slice_z0.pvd",
        "samples/vpm_slice_z0_000005.vts",
        "samples/vpm_centreline.csv",
        "samples/vpm_offaxis_y075.csv",
        "samples/forces_history.csv",
        "solution/run_metadata.json",
        "solution/fvm_metadata.json",
        "solution/coupler_diagnostics.jsonl",
        "solution/fvm/mesh.npz",
        "reference_flow/solution/grid_h0045/fvm_metadata.json",
        "reference_flow/solution/grid_h0045/diagnostics.jsonl",
        "reference_flow/solution/grid_h0045/fvm/mesh.npz",
        "reference_flow/samples/grid_h0045/forces_history.csv",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("native accepted input")
    manifest = {
        "method": util.PREPARATION_METHOD,
        "frames": [{"time": 0.25, "file": "fields_t0.250000000.npz"}],
    }
    (util.COMPARISON / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "samples/vpm_slice_z0.pvd").write_text(
        '<VTKFile><Collection><DataSet timestep="0.25" file="vpm_slice_z0_000005.vts"/></Collection></VTKFile>'
    )
    marker = {
        "schema_version": 1,
        "method": util.PREPARATION_METHOD,
        "comparison": manifest,
        "reference": {
            "name": "grid_h0045",
            "solution": "reference_flow/solution/grid_h0045",
            "samples": "reference_flow/samples/grid_h0045",
            "target_spacing": 0.045,
        },
        "meshes": {
            "coupled": "solution/fvm/mesh.npz",
            "reference": "reference_flow/solution/grid_h0045/fvm/mesh.npz",
        },
        "files": [
            {"path": name, "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
            for name in paths
        ],
    }
    (util.COMPARISON / "prepared_inputs.json").write_text(json.dumps(marker))
    yield util, marker
    util.prepared_inputs.cache_clear()


def test_portable_fields_reuse_without_raw_volume_files(prepared):
    util, _ = prepared
    assert util.prepare_comparison_fields() == 0
    reference = util.reference_run()
    assert reference.solution == util.CASE_DIR / "reference_flow/solution/grid_h0045"
    assert reference.target_spacing == 0.045
    assert not (reference.solution / "fvm.pvd").exists()


def test_changed_sample_rejected(prepared):
    util, _ = prepared
    (util.SAMPLES / "vpm_centreline.csv").write_text("different scientific samples")
    with pytest.raises(ValueError, match="input changed"):
        util.prepare_comparison_fields()


def test_missing_provenance_rejected(prepared):
    util, marker = prepared
    marker["files"] = [
        item for item in marker["files"] if item["path"] != "solution/fvm_metadata.json"
    ]
    (util.COMPARISON / "prepared_inputs.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="omit required scientific provenance"):
        util.prepare_comparison_fields()


def test_absolute_input_path_rejected(prepared):
    util, marker = prepared
    marker["files"].append({"path": "/outside/file", "sha256": "invalid"})
    (util.COMPARISON / "prepared_inputs.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="case-relative"):
        util.prepare_comparison_fields()


def test_export_preserves_all_sample_bytes_and_original_manifest(prepared, monkeypatch, tmp_path):
    util, marker = prepared
    original = (util.COMPARISON / "manifest.json").read_bytes()
    (util.COMPARISON / "prepared_inputs.json").unlink()
    monkeypatch.setattr(util, "SOLUTION", util.CASE_DIR / "solution")
    monkeypatch.setattr(util, "prepare_comparison_fields", lambda: 0)
    monkeypatch.setattr(util, "validate_plot_inputs", lambda: {})
    monkeypatch.setattr(
        util,
        "_fvm_artifacts",
        lambda solution: (solution / "fvm.pvd", solution / "fvm/mesh.npz"),
    )
    reference = util.ReferenceRun(
        marker["reference"]["name"],
        util.CASE_DIR / marker["reference"]["solution"],
        util.CASE_DIR / marker["reference"]["samples"],
        0.045,
    )
    monkeypatch.setattr(util, "reference_run", lambda: reference)
    extra = reference.samples.parent / "other_grid" / "all_samples.bin"
    extra.parent.mkdir()
    extra.write_bytes(b"\x00\xffnative sample bytes")
    exporter = load_tutorial_module("coupled_fvm_vpm/cube_flow", "assets.export_plot_inputs")
    destination = tmp_path / "export"
    paths = exporter.export_inputs(destination)
    relative = extra.relative_to(util.CASE_DIR)
    assert relative.as_posix() in paths
    assert (destination / relative).read_bytes() == extra.read_bytes()
    assert (destination / "samples/comparison/manifest.json").read_bytes() == original
    assert (destination / "samples/vpm_slice_z0.pvd").read_bytes() == (
        util.SAMPLES / "vpm_slice_z0.pvd"
    ).read_bytes()
