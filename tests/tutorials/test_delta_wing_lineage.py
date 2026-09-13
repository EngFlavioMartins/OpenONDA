"""Unit checks for the explicit Delta-wing source lineage."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tests._tutorial_helpers import load_tutorial_module

_plots = load_tutorial_module("vpm/delta_wing", "assets._delta_wing_plots")
_gif = load_tutorial_module("vpm/delta_wing", "assets.render_delta_wing_gif")
_read_lineage_csv = _plots._read_lineage_csv
_validate_boundary = _plots._validate_boundary
_validate_duplicate_csv_rows = _plots._validate_duplicate_csv_rows
_wake_frames = _plots._wake_frames
load_accepted_lineage = _plots.load_accepted_lineage
load_animation_lineage = _plots.load_animation_lineage
coupled_frames = _gif.coupled_frames


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_backup(path: Path, step: int, time: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as archive:
        solver = archive.create_group("solver")
        solver.attrs["step"] = step
        solver.attrs["time"] = time
        vlm = solver.create_group("vlm")
        vlm.create_dataset("panel_corner_position", data=np.zeros((1, 4, 3)))
        vlm.create_dataset("circulation", data=np.zeros(1))
        vlm.create_dataset("panel_force", data=np.zeros((1, 3)))


def _write_manifest_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    case = tmp_path / "case"
    assets = case / "assets"
    assets.mkdir(parents=True)
    for name in ("original", "continuation", "active"):
        (case / "solution" / name).mkdir(parents=True)
        (case / "samples" / name).mkdir(parents=True)

    source_steps = {
        "original": [(0, 0.0), (1, 0.1), (2, 0.2)],
        "continuation": [(3, 0.3), (4, 0.4)],
        "active": [(5, 0.5), (6, 0.6)],
    }
    for name, clocks in source_steps.items():
        for step, time in clocks:
            _write_backup(case / "solution" / name / f"vpm_{step:06d}.h5", step, time)
        force_rows = [
            {"time": time, "step": step, "surface": "front_wing", "force_z": float(step)}
            for step, time in clocks
        ]
        # Include one raw tail row in each of the first two namespaces. The
        # manifest interval, not a directory-name heuristic, must exclude it.
        if name == "original":
            force_rows.append({"time": 0.3, "step": 3, "surface": "front_wing", "force_z": 3.0})
        if name == "continuation":
            force_rows.append({"time": 0.5, "step": 5, "surface": "front_wing", "force_z": 5.0})
        pd.DataFrame(force_rows).to_csv(
            case / "samples" / name / "vlm_surface_forces.csv", index=False
        )
        flow_rows = [
            {"time": time, "step": step, "vortex_strength_magnitude_sum": float(step)}
            for step, time in clocks
        ]
        if name == "original":
            flow_rows.append({"time": 0.3, "step": 3, "vortex_strength_magnitude_sum": 3.0})
        if name == "continuation":
            flow_rows.append({"time": 0.5, "step": 5, "vortex_strength_magnitude_sum": 5.0})
        pd.DataFrame(flow_rows).to_csv(case / "samples" / name / "flow_integrals.csv", index=False)
        wake = case / "samples" / name / "wake_1span.pvd"
        pvd_rows = []
        for step, time in clocks:
            frame = wake.parent / f"wake_1span_{step:06d}.vts"
            frame.write_text("tiny plane fixture\n", encoding="utf-8")
            pvd_rows.append(f'    <DataSet timestep="{time}" file="{frame.name}"/>')
        if name == "original":
            pvd_rows.append('    <DataSet timestep="0.3" file="wake_1span_000003.vts"/>')
        if name == "continuation":
            pvd_rows.append('    <DataSet timestep="0.5" file="wake_1span_000005.vts"/>')
        wake.write_text(
            '<?xml version="1.0"?>\n<VTKFile type="Collection"><Collection>\n'
            + "\n".join(pvd_rows)
            + "\n</Collection></VTKFile>\n",
            encoding="utf-8",
        )

    payload = {
        "schema_version": 1,
        "case_root": "..",
        "default_statuses": ["accepted"],
        "animation_source": {
            "status": "active",
            "segment_ids": ["active"],
            "selection": "dense_continuation_only",
        },
        "segments": [
            {
                "id": "original",
                "status": "accepted",
                "solution": "solution/original",
                "samples": "samples/original",
                "accepted_interval": {
                    "first_step": 0,
                    "last_step": 2,
                    "first_time": 0.0,
                    "last_time": 0.2,
                },
                "boundary_checkpoints": [
                    {
                        "role": "end",
                        "path": "solution/original/vpm_000002.h5",
                        "sha256": _sha256(case / "solution/original/vpm_000002.h5"),
                        "step": 2,
                        "time": 0.2,
                    }
                ],
            },
            {
                "id": "continuation",
                "status": "accepted",
                "solution": "solution/continuation",
                "samples": "samples/continuation",
                "accepted_interval": {
                    "first_step": 3,
                    "last_step": 4,
                    "first_time": 0.3,
                    "last_time": 0.4,
                },
                "boundary_checkpoints": [
                    {
                        "role": "end",
                        "path": "solution/continuation/vpm_000004.h5",
                        "sha256": _sha256(case / "solution/continuation/vpm_000004.h5"),
                        "step": 4,
                        "time": 0.4,
                    }
                ],
            },
            {
                "id": "active",
                "status": "active",
                "solution": "solution/active",
                "samples": "samples/active",
                "accepted_interval": {
                    "first_step": 5,
                    "last_step": None,
                    "first_time": 0.5,
                    "last_time": None,
                },
                "boundary_checkpoints": [
                    {
                        "role": "start",
                        "path": "solution/continuation/vpm_000004.h5",
                        "sha256": _sha256(case / "solution/continuation/vpm_000004.h5"),
                        "step": 4,
                        "time": 0.4,
                    }
                ],
            },
        ],
    }
    manifest = assets / "lineage.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return case, manifest, payload


def _write_fresh_origin_fixture(tmp_path: Path) -> tuple[Path, Path]:
    case = tmp_path / "fresh-case"
    (case / "assets").mkdir(parents=True)
    solution = case / "solution"
    samples = case / "samples/delta_wing"
    solution.mkdir(parents=True)
    samples.mkdir(parents=True)
    _write_backup(solution / "vpm_000010.h5", 10, 0.025)
    pd.DataFrame([{"step": 10, "time": 0.025, "surface": "front_wing"}]).to_csv(
        samples / "vlm_surface_forces.csv", index=False
    )
    payload = {
        "schema_version": 1,
        "case_root": "..",
        "default_statuses": ["accepted"],
        "animation_source": {
            "status": "active",
            "segment_ids": ["fresh_run"],
            "selection": "single_dense_run",
        },
        "segments": [
            {
                "id": "fresh_run",
                "status": "active",
                "origin": "fresh_initial_value",
                "solution": "solution",
                "samples": "samples/delta_wing",
                "accepted_interval": {
                    "first_step": 0,
                    "last_step": None,
                    "first_time": 0.0,
                    "last_time": None,
                },
                "boundary_checkpoints": [],
            }
        ],
    }
    manifest = case / "assets/lineage.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return case, manifest


def _finalize_fixture(case: Path, manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    active = payload["segments"][-1]
    active["status"] = "accepted"
    active["accepted_interval"]["last_step"] = 6
    active["accepted_interval"]["last_time"] = 0.6
    active["boundary_checkpoints"].append(
        {
            "role": "end",
            "path": "solution/active/vpm_000006.h5",
            "sha256": _sha256(case / "solution/active/vpm_000006.h5"),
            "step": 6,
            "time": 0.6,
        }
    )
    payload["animation_source"]["status"] = "accepted"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_default_lineage_excludes_active_and_counterfactual_tails(tmp_path):
    _, manifest, _ = _write_manifest_fixture(tmp_path)
    segments = load_accepted_lineage(manifest)
    assert [segment["id"] for segment in segments] == ["original", "continuation"]
    force = _read_lineage_csv("vlm_surface_forces.csv", manifest)
    flow = _read_lineage_csv("flow_integrals.csv", manifest)
    assert int(force.step.max()) == 4
    assert int(flow.step.max()) == 4
    assert force.source_segment.tolist() == [
        "original",
        "original",
        "original",
        "continuation",
        "continuation",
    ]
    assert flow.time.is_monotonic_increasing


def test_selected_csv_clocks_are_finite_integer_and_ordered(tmp_path):
    case, manifest, _ = _write_manifest_fixture(tmp_path)
    force_path = case / "samples/original/vlm_surface_forces.csv"
    force = pd.read_csv(force_path)
    force["step"] = force["step"].astype(float)
    force.loc[1, "step"] = 1.5
    force.to_csv(force_path, index=False)
    with pytest.raises(ValueError, match="finite integer"):
        _read_lineage_csv("vlm_surface_forces.csv", manifest)

    force.loc[1, "step"] = 1
    force.loc[1, "time"] = np.nan
    force.to_csv(force_path, index=False)
    with pytest.raises(ValueError, match="time must be finite"):
        _read_lineage_csv("vlm_surface_forces.csv", manifest)

    force.loc[1, "time"] = 0.1
    force.loc[2, "time"] = 0.05
    force.to_csv(force_path, index=False)
    with pytest.raises(ValueError, match="timestamps must increase"):
        _read_lineage_csv("vlm_surface_forces.csv", manifest)


def test_selected_csv_step_and_surface_clock_consistency(tmp_path):
    case, manifest, _ = _write_manifest_fixture(tmp_path / "nonmonotonic")
    force_path = case / "samples/original/vlm_surface_forces.csv"
    force = pd.read_csv(force_path)
    force.loc[:2, "step"] = [0, 2, 1]
    force.loc[:2, "time"] = [0.0, 0.1, 0.2]
    force.to_csv(force_path, index=False)
    with pytest.raises(ValueError, match="step clocks must be nondecreasing"):
        _read_lineage_csv("vlm_surface_forces.csv", manifest)

    case, manifest, _ = _write_manifest_fixture(tmp_path / "surface-conflict")
    force_path = case / "samples/original/vlm_surface_forces.csv"
    force = pd.read_csv(force_path)
    rear = pd.DataFrame([{"time": 0.15, "step": 1, "surface": "rear_wing", "force_z": 1.0}])
    force = pd.concat([force.iloc[:2], rear, force.iloc[2:]], ignore_index=True)
    force.to_csv(force_path, index=False)
    with pytest.raises(ValueError, match="multiple timestamps for step 1"):
        _read_lineage_csv("vlm_surface_forces.csv", manifest)

    case, manifest, _ = _write_manifest_fixture(tmp_path / "surface-valid")
    force_path = case / "samples/original/vlm_surface_forces.csv"
    force = pd.read_csv(force_path)
    rear = pd.DataFrame([{"time": 0.1, "step": 1, "surface": "rear_wing", "force_z": 1.0}])
    force = pd.concat([force.iloc[:2], rear, force.iloc[2:]], ignore_index=True)
    force.to_csv(force_path, index=False)
    selected = _read_lineage_csv("vlm_surface_forces.csv", manifest)
    assert len(selected[selected.step == 1]) == 2


def test_manifest_interval_and_boundary_times_are_reconciled(tmp_path):
    _, manifest, payload = _write_manifest_fixture(tmp_path)
    altered = deepcopy(payload)
    altered["segments"][0]["accepted_interval"]["last_time"] = 99.0
    manifest.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="end boundary time"):
        load_accepted_lineage(manifest)

    altered = deepcopy(payload)
    altered["segments"][1]["accepted_interval"]["first_time"] = 0.1
    manifest.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted clocks"):
        load_accepted_lineage(manifest)

    altered = deepcopy(payload)
    altered["segments"][1]["accepted_interval"]["first_step"] = 2
    manifest.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap or are unordered"):
        load_accepted_lineage(manifest)


def test_fresh_initial_value_origin_has_no_invented_start_checkpoint(tmp_path):
    _, manifest = _write_fresh_origin_fixture(tmp_path)
    segments = load_accepted_lineage(manifest, include_active=True)
    assert [segment["id"] for segment in segments] == ["fresh_run"]
    assert segments[0]["accepted_interval"]["first_step"] == 0
    assert segments[0]["accepted_interval"]["first_time"] == 0.0

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["segments"][0]["origin"] = "restart_checkpoint"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="start boundary checkpoint"):
        load_accepted_lineage(manifest, include_active=True)


def test_boundary_hash_state_and_role_validation(tmp_path):
    case, manifest, _ = _write_manifest_fixture(tmp_path)
    segment = load_accepted_lineage(manifest)[0]
    boundary = dict(segment["boundary_checkpoints"][0])
    boundary["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_boundary(boundary, case, segment["accepted_interval"], segment["id"])

    boundary = dict(segment["boundary_checkpoints"][0])
    boundary["step"] = 1
    with pytest.raises(ValueError, match="clock mismatch"):
        _validate_boundary(boundary, case, segment["accepted_interval"], segment["id"])

    boundary = dict(segment["boundary_checkpoints"][0])
    boundary["role"] = "middle"
    with pytest.raises(ValueError, match="unsupported boundary role"):
        _validate_boundary(boundary, case, segment["accepted_interval"], segment["id"])


def test_native_wake_and_dense_animation_selection_share_lineage(tmp_path):
    case, manifest, _ = _write_manifest_fixture(tmp_path)
    with pytest.raises(ValueError, match="not finalized"):
        load_animation_lineage(manifest)
    _finalize_fixture(case, manifest)
    animation = load_animation_lineage(manifest)
    assert [segment["id"] for segment in animation] == ["active"]
    records = coupled_frames(None, manifest)
    assert [int(path.stem.rsplit("_", 1)[1]) for _, path, _ in records] == [5, 6]
    assert all(time > 0.4 for time, _, _ in records)

    segments = load_accepted_lineage(manifest)
    samples = [segment["samples_path"] for segment in segments]
    wake = _wake_frames(samples, "wake_1span.pvd", segments)
    assert [int(path.stem.rsplit("_", 1)[1]) for _, path, _ in wake] == [0, 1, 2, 3, 4, 5, 6]
    assert wake[-1][0] == pytest.approx(0.6)


def test_native_and_wake_selected_clocks_reject_out_of_interval(tmp_path):
    case, manifest, _ = _write_manifest_fixture(tmp_path)
    segments = load_accepted_lineage(manifest)
    interval = segments[1]["accepted_interval"]
    _validate_selected_clock = _plots._validate_selected_clock

    with pytest.raises(ValueError, match="outside its lineage interval"):
        _validate_selected_clock(4, 0.9, interval, "native fixture")

    pvd = case / "samples/continuation/wake_1span.pvd"
    pvd.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection"><Collection>\n'
        '    <DataSet timestep="0.9" file="wake_1span_000004.vts"/>\n'
        "</Collection></VTKFile>\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside its lineage interval"):
        _wake_frames([segments[1]["samples_path"]], "wake_1span.pvd", segments)


def test_authoritative_conflicting_duplicate_is_rejected():
    left = pd.DataFrame(
        {
            "step": [4],
            "surface": ["front_wing"],
            "time": [0.4],
            "force_z": [1.0],
            "source_segment": ["left"],
            "source_samples_directory": ["left"],
        }
    )
    right = left.copy()
    right["force_z"] = 2.0
    right["source_segment"] = "right"
    with pytest.raises(ValueError, match="conflicts in force_z"):
        _validate_duplicate_csv_rows("vlm_surface_forces.csv", [left, right], ["step", "surface"])
