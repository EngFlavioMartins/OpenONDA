"""Fixture checks for a fresh prefix followed by a checkpoint restart."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import pyvista as pv

from tutorials.vpm.delta_wing.assets._delta_wing_plots import (
    _read_lineage_csv,
    load_accepted_lineage,
)
from tutorials.vpm.delta_wing.assets.finalize_delta_wing_lineage import (
    _check_csv,
    _clock,
    _require_exact_steps,
    _require_force_surface_steps,
    finalize_lineage,
)

FIXTURE_NUMERICS = {
    "axisymmetric_no_swirl_axis": None,
    "compute_device": "CPU",
    "cutoff_radius_factor": 100,
    "domain_bounds": None,
    "health_limits": {
        "divergence": {"maximum": None},
        "finite_state": {"enabled": True},
        "growth": {"maximum": None},
        "lagrangian_cfl": {"maximum": 1.0},
        "maximum_particle_strength": {"maximum": None},
        "misalignment": {"maximum_degrees": None},
    },
    "induction": {
        "kernel": "GAUSSIAN",
        "method": "FMM",
        "stretching_scheme": "TRANSPOSED",
    },
    "integrator": {"a": [[0, 0, 0]], "b": [1.0], "c": [0.0], "name": "FIXTURE", "order": 1},
    "max_n_particles": 1,
    "particle_kernel": "GAUSSIAN",
    "precision": "f32",
    "random_seed": 42,
    "stabilization": {},
    "time_step_size": 0.0025,
    "turbulence": {"flow_model": "DNS"},
    "viscous": {"scheme": "CS"},
}
FIXTURE_SURFACES = [
    {
        "name": surface,
        "geometry": {
            "wings": [
                {
                    "symmetry": 0,
                    "uid": "main_wing",
                    "segments": [
                        {
                            "uid": "segment_0",
                            "n_chordwise_panels": 3,
                            "n_spanwise_panels": 2,
                        }
                    ],
                }
            ]
        },
    }
    for surface in ("front_wing", "rear_wing")
]
FIXTURE_NATIVE_CONFIG = json.dumps(FIXTURE_NUMERICS, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_backup(path: Path, step: int, time: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as archive:
        solver = archive.create_group("solver")
        numerical_configuration = FIXTURE_NATIVE_CONFIG
        solver.attrs["backup_format_version"] = "10.0"
        solver.attrs["numerical_configuration"] = numerical_configuration
        solver.attrs["numerical_configuration_sha256"] = hashlib.sha256(
            numerical_configuration.encode("utf-8")
        ).hexdigest()
        solver.attrs["write_precision"] = "f32"
        solver.attrs["time_step_size"] = 0.0025
        solver.attrs["step"] = step
        solver.attrs["time"] = time
        vlm = solver.create_group("vlm")
        vlm.attrs["version"] = 7
        vlm.attrs["identity"] = "fixture-restart"
        vlm.attrs["physics_identity"] = "fixture-physics"
        vlm.attrs["time"] = time
        vlm.create_dataset("panel_corner_position", data=np.zeros((12, 4, 3)))
        vlm.create_dataset("circulation", data=np.zeros(12))
        vlm.create_dataset("panel_force", data=np.zeros((12, 3)))


def _write_surface_samples(path: Path, start_step: int, end_step: int) -> None:
    rows = [
        {
            "step": step,
            "time": step * 0.0025,
            "surface": surface,
            "force_x": 1.0,
            "force_y": 1.0,
            "force_z": float(step),
            "moment_x": 1.0,
            "moment_y": 1.0,
            "moment_z": 1.0,
            "power": 1.0,
            "translation_velocity_z": 1.0,
            "centroid_z": 1.0,
        }
        for step in range(start_step, end_step + 1)
        for surface in ("front_wing", "rear_wing")
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_owner_companions(solution: Path, steps: list[int]) -> None:
    for step in steps:
        time = step * 0.0025
        _write_backup(solution / f"vpm_{step:06d}.h5", step, time)
        (solution / f"vpm_{step:06d}.xdmf").write_text(
            f'<Xdmf><Domain><Grid><Time Value="{time}"/></Grid></Domain></Xdmf>',
            encoding="utf-8",
        )
        surface = pv.PolyData(np.zeros((4, 3)))
        surface.field_data["time"] = np.array([time])
        surface.field_data["TimeValue"] = np.array([time])
        surface.save(solution / f"vlm_{step:06d}.vtp")
    rows = "".join(
        f'<DataSet timestep="{step * 0.0025}" file="vlm_{step:06d}.vtp"/>' for step in steps
    )
    (solution / "vlm.pvd").write_text(
        f"<VTKFile><Collection>{rows}</Collection></VTKFile>", encoding="utf-8"
    )


def _write_metadata(
    solution: Path,
    sample_directory: str,
    *,
    initial_step: int,
    initial_time: float,
    step: int,
    time: float,
    requested_steps: int,
    status: str,
    backup_directory: str,
) -> None:
    numerics = deepcopy(FIXTURE_NUMERICS)
    numerics.update(
        {
            "write_precision": "f32",
            "vlm": {
                "physics_identity": "fixture-physics",
                "restart_identity": "fixture-restart",
                "surfaces": FIXTURE_SURFACES,
            },
        }
    )
    metadata = {
        "schema_version": 1,
        "solver": "VPM",
        "configuration": {
            "numerics": numerics,
            "run": {
                "steps": requested_steps,
                "initial_samples": True,
                "final_backup": True,
                "health_limit_action": "RAISE",
                "wall_time_limit_seconds": None,
                "resource_limits": None,
                "runtime_compute_device": None,
            },
            "backup": {
                "directory": backup_directory,
                "log_directory": backup_directory,
                "interval_steps": 10,
            },
            "samplers": {"directory": sample_directory, "items": [{"type": "fixture"}]},
            "initial_conditions": [],
            "initial_weak_particle_percent": 0.0,
        },
        "state": {
            "initial_step": initial_step,
            "initial_time": initial_time,
            "step": step,
            "time": time,
        },
        "lifecycle": {"status": status},
    }
    (solution / "vpm_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _write_segment_samples(
    samples: Path, *, force_start: int, force_end: int, owner_steps: list[int]
) -> None:
    samples.mkdir(parents=True, exist_ok=True)
    force_steps = range(force_start, force_end + 1)
    pd.DataFrame(
        [
            {
                "step": step,
                "time": step * 0.0025,
                "force_x": 1.0,
                "force_y": 1.0,
                "force_z": 1.0,
                "lift": 1.0,
                "drag": 1.0,
                "power": 1.0,
            }
            for step in force_steps
        ]
    ).to_csv(samples / "vlm_forces.csv", index=False)
    pd.DataFrame(
        [
            {
                "step": step,
                "time": step * 0.0025,
                "surface": surface,
                "force_x": 1.0,
                "force_y": 1.0,
                "force_z": 1.0,
                "moment_x": 1.0,
                "moment_y": 1.0,
                "moment_z": 1.0,
                "power": 1.0,
                "translation_velocity_z": 1.0,
                "centroid_z": 1.0,
            }
            for step in force_steps
            for surface in ("front_wing", "rear_wing")
        ]
    ).to_csv(samples / "vlm_surface_forces.csv", index=False)
    for name, surface in (
        ("vlm_chordwise_front_wing.csv", "front_wing"),
        ("vlm_chordwise_rear_wing.csv", "rear_wing"),
        ("vlm_spanwise_front_wing.csv", "front_wing"),
        ("vlm_spanwise_rear_wing.csv", "rear_wing"),
    ):
        station_count = 6 if "chordwise" in name else 2
        pd.DataFrame(
            [
                {
                    "step": step,
                    "time": step * 0.0025,
                    "surface": surface,
                    "wing_uid": f"{surface}_main_wing",
                    "segment_uid": "segment_0",
                    "station_id": f"{surface}_main_wing:segment_0:orig:{station}",
                    "half": "orig",
                    "span_index": station,
                    **({"chord_index": chord} if "chordwise" in name else {}),
                    "panel_force_x" if "chordwise" in name else "section_force_x": 1.0,
                    "panel_force_y" if "chordwise" in name else "section_force_y": 1.0,
                    "panel_force_z" if "chordwise" in name else "section_force_z": 1.0,
                    "panel_circulation" if "chordwise" in name else "circulation": 1.0,
                }
                for step in force_steps
                for station in range(2 if "chordwise" in name else station_count)
                for chord in (range(3) if "chordwise" in name else [None])
            ]
        ).to_csv(samples / name, index=False)
    pd.DataFrame(
        [
            {
                "step": step,
                "time": step * 0.0025,
                "total_kinetic_energy": 1.0,
                "total_enstrophy": 1.0,
                "vortex_strength_magnitude_sum": 1.0,
                "n_particles_total": 1,
                "lagrangian_cfl": 0.1,
            }
            for step in owner_steps
        ]
    ).to_csv(samples / "flow_integrals.csv", index=False)


def _write_resume_manifest(
    case: Path, *, continuation_status: str, continuation_end: int | None
) -> Path:
    root_checkpoint = case / "solution/vpm_000310.h5"
    continuation_checkpoint = case / "solution/continuation_from_000310/vpm_004000.h5"
    continuation = {
        "id": "continuation_from_000310",
        "status": continuation_status,
        "origin": "restart_checkpoint",
        "solution": "solution/continuation_from_000310",
        "samples": "samples/delta_wing_continuation_from_000310",
        "accepted_interval": {
            "first_step": 311,
            "last_step": continuation_end,
            "first_time": 311 * 0.0025,
            "last_time": None if continuation_end is None else continuation_end * 0.0025,
        },
        "boundary_checkpoints": [
            {
                "role": "start",
                "path": "solution/vpm_000310.h5",
                "sha256": _sha256(root_checkpoint),
                "step": 310,
                "time": 310 * 0.0025,
            }
        ],
    }
    if continuation_end is not None:
        continuation["boundary_checkpoints"].append(
            {
                "role": "end",
                "path": "solution/continuation_from_000310/vpm_004000.h5",
                "sha256": _sha256(continuation_checkpoint),
                "step": continuation_end,
                "time": continuation_end * 0.0025,
            }
        )
    payload = {
        "schema_version": 1,
        "case_root": "..",
        "default_statuses": ["accepted"],
        "animation_source": {
            "status": "accepted" if continuation_status == "accepted" else "active",
            "segment_ids": ["fresh_prefix", "continuation_from_000310"],
            "selection": "fresh_prefix_plus_checkpoint_continuation",
        },
        "segments": [
            {
                "id": "fresh_prefix",
                "status": "accepted",
                "origin": "fresh_initial_value",
                "solution": "solution",
                "samples": "samples/delta_wing",
                "accepted_interval": {
                    "first_step": 0,
                    "last_step": 310,
                    "first_time": 0.0,
                    "last_time": 310 * 0.0025,
                },
                "boundary_checkpoints": [
                    {
                        "role": "end",
                        "path": "solution/vpm_000310.h5",
                        "sha256": _sha256(root_checkpoint),
                        "step": 310,
                        "time": 310 * 0.0025,
                    }
                ],
            },
            continuation,
        ],
    }
    manifest = case / "assets/delta_wing_accepted_lineage.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_resume_fixture(tmp_path: Path) -> tuple[Path, Path]:
    case = tmp_path / "resume-case"
    (case / "assets").mkdir(parents=True)
    (case / "solution").mkdir()
    (case / "samples/delta_wing").mkdir(parents=True)
    (case / "solution/continuation_from_000310").mkdir(parents=True)
    (case / "samples/delta_wing_continuation_from_000310").mkdir(parents=True)

    _write_backup(case / "solution/vpm_000310.h5", 310, 310 * 0.0025)
    _write_backup(
        case / "solution/continuation_from_000310/vpm_004000.h5",
        4000,
        4000 * 0.0025,
    )
    # The canonical prefix directory deliberately retains rows 311..315 from
    # the interrupted in-memory tail. The interval cap must exclude them.
    _write_surface_samples(case / "samples/delta_wing/vlm_surface_forces.csv", 1, 315)
    _write_surface_samples(
        case / "samples/delta_wing_continuation_from_000310/vlm_surface_forces.csv",
        311,
        4000,
    )
    manifest = _write_resume_manifest(case, continuation_status="active", continuation_end=None)
    return case, manifest


def _write_complete_resume_fixture(tmp_path: Path) -> tuple[Path, Path]:
    case = tmp_path / "complete-resume-case"
    (case / "assets").mkdir(parents=True)
    (case / "solution").mkdir()
    (case / "samples/delta_wing").mkdir(parents=True)
    continuation_solution = case / "solution/continuation_from_000310"
    continuation_samples = case / "samples/delta_wing_continuation_from_000310"
    continuation_solution.mkdir(parents=True)
    continuation_samples.mkdir(parents=True)

    prefix_owners = list(range(10, 311, 10))
    continuation_owners = list(range(320, 4001, 10))
    _write_owner_companions(case / "solution", prefix_owners)
    _write_owner_companions(continuation_solution, continuation_owners)
    _write_metadata(
        case / "solution",
        "delta_wing",
        initial_step=0,
        initial_time=0.0,
        step=310,
        time=0.775,
        requested_steps=4000,
        status="running",
        backup_directory="solution",
    )
    _write_metadata(
        continuation_solution,
        "delta_wing_continuation_from_000310",
        initial_step=310,
        initial_time=0.775,
        step=4000,
        time=10.0,
        requested_steps=3690,
        status="completed",
        backup_directory="solution/continuation_from_000310",
    )
    # Preserve a raw post-checkpoint tail in the prefix. The finalizer must
    # validate it, then exclude it by the declared interval before checking
    # the non-overlapping union with the continuation samples.
    _write_segment_samples(
        case / "samples/delta_wing",
        force_start=1,
        force_end=315,
        owner_steps=prefix_owners,
    )
    _write_segment_samples(
        continuation_samples,
        force_start=311,
        force_end=4000,
        owner_steps=continuation_owners,
    )
    manifest = _write_resume_manifest(case, continuation_status="active", continuation_end=None)
    return case, manifest


def test_fresh_prefix_and_checkpoint_continuation_lineage_fixture(tmp_path):
    case, manifest = _write_resume_fixture(tmp_path)

    root_only = load_accepted_lineage(manifest)
    assert [segment["id"] for segment in root_only] == ["fresh_prefix"]
    with_active = load_accepted_lineage(manifest, include_active=True)
    assert [segment["id"] for segment in with_active] == [
        "fresh_prefix",
        "continuation_from_000310",
    ]
    assert _clock(case / "solution/vpm_000310.h5") == (310, 0.775)
    assert _clock(case / "solution/continuation_from_000310/vpm_004000.h5") == (
        4000,
        10.0,
    )

    prefix = _read_lineage_csv("vlm_surface_forces.csv", manifest)
    assert int(prefix.step.max()) == 310
    assert prefix.source_segment.unique().tolist() == ["fresh_prefix"]
    assert set(prefix.step) == set(range(1, 311))

    finalized_manifest = _write_resume_manifest(
        case, continuation_status="accepted", continuation_end=4000
    )
    combined = _read_lineage_csv("vlm_surface_forces.csv", finalized_manifest)
    assert int(combined.step.min()) == 1
    assert int(combined.step.max()) == 4000
    assert set(combined.loc[combined.step.between(311, 315), "source_segment"]) == {
        "continuation_from_000310"
    }
    assert not combined.duplicated(["step", "surface"]).any()

    continuation = _check_csv(
        case / "samples/delta_wing_continuation_from_000310/vlm_surface_forces.csv",
        end_step=4000,
        initial_time=0.0,
        dt=0.0025,
    )
    expected_steps = list(range(311, 4001))
    _require_exact_steps(continuation, expected_steps, "checkpoint continuation")
    _require_force_surface_steps(
        continuation, expected_steps, "checkpoint continuation surface forces"
    )


def test_public_finalizer_promotes_declared_resume_lineage(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)

    finalized = finalize_lineage(case)
    payload = json.loads(finalized.read_text(encoding="utf-8"))
    assert [segment["status"] for segment in payload["segments"]] == ["accepted", "accepted"]
    assert payload["segments"][1]["accepted_interval"]["last_step"] == 4000
    assert payload["segments"][1]["boundary_checkpoints"][-1]["role"] == "end"
    assert payload["animation_source"]["status"] == "accepted"
    combined = _read_lineage_csv("vlm_surface_forces.csv", manifest)
    assert int(combined.step.max()) == 4000
    assert not combined.duplicated(["step", "surface"]).any()


def test_public_finalizer_rejects_missing_continuation_coverage_atomically(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    path = case / "samples/delta_wing_continuation_from_000310/vlm_forces.csv"
    frame = pd.read_csv(path)
    frame = frame[frame.step != 350]
    frame.to_csv(path, index=False)
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="do not cover every required step"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before
    assert json.loads(manifest.read_text(encoding="utf-8"))["segments"][1]["status"] == "active"


def test_public_finalizer_rejects_missing_loading_station_atomically(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    path = case / "samples/delta_wing_continuation_from_000310/vlm_chordwise_front_wing.csv"
    frame = pd.read_csv(path)
    frame = frame[~((frame.step == 350) & (frame.chord_index == 1))]
    frame.to_csv(path, index=False)
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="station membership"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_public_finalizer_rejects_consistent_missing_loading_station_atomically(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    for path in (
        case / "samples/delta_wing/vlm_chordwise_front_wing.csv",
        case / "samples/delta_wing_continuation_from_000310/vlm_chordwise_front_wing.csv",
    ):
        frame = pd.read_csv(path)
        frame = frame[frame.chord_index != 1]
        frame.to_csv(path, index=False)
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="station membership"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_public_finalizer_rejects_consistent_invalid_loading_identity_atomically(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    valid_station = "front_wing_main_wing:segment_0:orig:1"
    invalid_station = "front_wing_main_wing:segment_0:orig:999"
    for path in (
        case / "samples/delta_wing/vlm_chordwise_front_wing.csv",
        case / "samples/delta_wing_continuation_from_000310/vlm_chordwise_front_wing.csv",
    ):
        frame = pd.read_csv(path)
        frame.loc[frame.station_id == valid_station, "station_id"] = invalid_station
        frame.to_csv(path, index=False)
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="station membership"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_public_finalizer_rejects_metadata_native_contradiction_atomically(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    for path in (
        case / "solution/vpm_metadata.json",
        case / "solution/continuation_from_000310/vpm_metadata.json",
    ):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["configuration"]["numerics"]["cutoff_radius_factor"] = 999
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="metadata numerical configuration"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_public_finalizer_rejects_invalid_predecessor_boundary_atomically(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["segments"][1]["boundary_checkpoints"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="declared lineage is invalid"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_public_finalizer_rejects_native_identity_mismatch_atomically(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    with h5py.File(case / "solution/continuation_from_000310/vpm_004000.h5", "r+") as archive:
        archive["solver/vlm"].attrs["physics_identity"] = "wrong-physics"
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="native identity mismatch"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_public_finalizer_rejects_short_continuation_endpoint_atomically(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    metadata_path = case / "solution/continuation_from_000310/vpm_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["configuration"]["run"]["steps"] = 10
    metadata["state"]["step"] = 320
    metadata["state"]["time"] = 0.8
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="does not reach the declared step-4000"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_public_finalizer_rejects_missing_required_physical_column_atomically(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    path = case / "samples/delta_wing_continuation_from_000310/vlm_forces.csv"
    frame = pd.read_csv(path).drop(columns=["force_z"])
    frame.to_csv(path, index=False)
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="required scientific columns"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_public_finalizer_rejects_non_numeric_physical_column_atomically(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    path = case / "samples/delta_wing_continuation_from_000310/vlm_forces.csv"
    frame = pd.read_csv(path)
    frame["force_z"] = frame["force_z"].astype(object)
    frame.loc[0, "force_z"] = "not-a-force"
    frame.to_csv(path, index=False)
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="required numeric column force_z"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_public_finalizer_ties_boundaries_to_selected_owner_namespace(tmp_path):
    case, manifest = _write_complete_resume_fixture(tmp_path)
    alternate = case / "solution/alternate"
    _write_backup(alternate / "vpm_000310.h5", 310, 0.775)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    alternate_boundary = {
        "path": "solution/alternate/vpm_000310.h5",
        "sha256": _sha256(alternate / "vpm_000310.h5"),
        "step": 310,
        "time": 0.775,
    }
    payload["segments"][0]["boundary_checkpoints"][0].update(alternate_boundary)
    payload["segments"][1]["boundary_checkpoints"][0].update(alternate_boundary)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    before = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="not its selected owner H5"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before
