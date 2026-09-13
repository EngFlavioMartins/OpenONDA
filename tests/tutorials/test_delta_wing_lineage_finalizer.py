"""Small completion-gate checks for the canonical Delta native output."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import pyvista as pv

from tests._tutorial_helpers import load_tutorial_module

load_animation_lineage = load_tutorial_module(
    "vpm/delta_wing", "assets._delta_wing_plots"
).load_animation_lineage
finalize_lineage = load_tutorial_module(
    "vpm/delta_wing", "assets.finalize_delta_wing_lineage"
).finalize_lineage

FRESH_NUMERICS = {
    "compute_device": "CPU",
    "induction": {"kernel": "GAUSSIAN"},
    "particle_kernel": "GAUSSIAN",
    "precision": "f32",
    "time_step_size": 0.0025,
}
FRESH_SURFACES = [
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
                            "n_chordwise_panels": 1,
                            "n_spanwise_panels": 1,
                        }
                    ],
                }
            ]
        },
    }
    for surface in ("front_wing", "rear_wing")
]
FRESH_NATIVE_CONFIG = json.dumps(FRESH_NUMERICS, sort_keys=True, separators=(",", ":"))


def _write_complete_case(tmp_path: Path) -> tuple[Path, Path]:
    case = tmp_path / "case"
    solution = case / "solution"
    samples = case / "samples/delta_wing"
    assets = case / "assets"
    solution.mkdir(parents=True)
    samples.mkdir(parents=True)
    assets.mkdir(parents=True)
    owner_steps = (10, 20)

    for step in owner_steps:
        time = step * 0.0025
        with h5py.File(solution / f"vpm_{step:06d}.h5", "w") as archive:
            solver = archive.create_group("solver")
            numerical_configuration = FRESH_NATIVE_CONFIG
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
            vlm.create_dataset("panel_corner_position", data=np.zeros((2, 4, 3)))
            vlm.create_dataset("circulation", data=np.zeros(2))
            vlm.create_dataset("panel_force", data=np.zeros((2, 3)))
        (solution / f"vpm_{step:06d}.xdmf").write_text(
            f'<Xdmf><Domain><Grid><Time Value="{time}"/></Grid></Domain></Xdmf>',
            encoding="utf-8",
        )
        surface = pv.PolyData(np.zeros((4, 3)))
        surface.field_data["time"] = np.array([time])
        surface.field_data["TimeValue"] = np.array([time])
        surface.save(solution / f"vlm_{step:06d}.vtp")

    pvd_rows = "".join(
        f'<DataSet timestep="{step * 0.0025}" file="vlm_{step:06d}.vtp"/>' for step in owner_steps
    )
    (solution / "vlm.pvd").write_text(
        f"<VTKFile><Collection>{pvd_rows}</Collection></VTKFile>", encoding="utf-8"
    )

    accepted_steps = range(1, 21)
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
            for step in accepted_steps
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
            for step in accepted_steps
            for surface in ("front_wing", "rear_wing")
        ]
    ).to_csv(samples / "vlm_surface_forces.csv", index=False)
    for name, surface in (
        ("vlm_chordwise_front_wing.csv", "front_wing"),
        ("vlm_chordwise_rear_wing.csv", "rear_wing"),
        ("vlm_spanwise_front_wing.csv", "front_wing"),
        ("vlm_spanwise_rear_wing.csv", "rear_wing"),
    ):
        station_column = "chord_index" if "chordwise" in name else "span_index"
        pd.DataFrame(
            [
                {
                    "step": step,
                    "time": step * 0.0025,
                    "surface": surface,
                    "wing_uid": f"{surface}_main_wing",
                    "segment_uid": "segment_0",
                    "station_id": f"{surface}_main_wing:segment_0:orig:0",
                    "half": "orig",
                    "span_index": 0,
                    **({station_column: 0} if "chordwise" in name else {}),
                    "panel_force_x" if "chordwise" in name else "section_force_x": 1.0,
                    "panel_force_y" if "chordwise" in name else "section_force_y": 1.0,
                    "panel_force_z" if "chordwise" in name else "section_force_z": 1.0,
                    "panel_circulation" if "chordwise" in name else "circulation": 1.0,
                }
                for step in accepted_steps
            ]
        ).to_csv(samples / name, index=False)

    metadata = {
        "configuration": {
            "run": {"steps": 20},
            "numerics": {
                **deepcopy(FRESH_NUMERICS),
                "write_precision": "f32",
                "vlm": {
                    "physics_identity": "fixture-physics",
                    "restart_identity": "fixture-restart",
                    "surfaces": FRESH_SURFACES,
                },
            },
            "backup": {"interval_steps": 10},
        },
        "state": {"initial_step": 0, "initial_time": 0.0, "step": 20, "time": 0.05},
        "lifecycle": {"status": "completed"},
    }
    (solution / "vpm_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = assets / "delta_wing_accepted_lineage.json"
    manifest.write_text('{"status":"active"}\n', encoding="utf-8")
    return case, manifest


def test_finalizer_promotes_only_a_complete_native_case(tmp_path):
    case, manifest = _write_complete_case(tmp_path)
    finalized = finalize_lineage(case)
    payload = json.loads(finalized.read_text(encoding="utf-8"))
    assert payload["animation_source"]["status"] == "accepted"
    assert payload["segments"][0]["accepted_interval"]["last_step"] == 20
    assert load_animation_lineage(manifest)[0]["id"] == "clean_dense_run"


def test_finalizer_rejects_nonnumeric_required_time_without_promotion(tmp_path):
    case, manifest = _write_complete_case(tmp_path)
    force_path = case / "samples/delta_wing/vlm_forces.csv"
    force = pd.read_csv(force_path)
    force["time"] = force["time"].astype(object)
    force.loc[3, "time"] = "not-a-time"
    force.to_csv(force_path, index=False)
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="required step/time columns must be finite numeric"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_finalizer_rejects_stale_h5_vlm_clock_without_promotion(tmp_path):
    case, manifest = _write_complete_case(tmp_path)
    with h5py.File(case / "solution/vpm_000020.h5", "r+") as archive:
        archive["solver/vlm"].attrs["time"] = 0.049
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="saved VLM time"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize(
    ("artifact", "pattern"),
    [("xdmf", "native XDMF time"), ("vtp", "native VTP time")],
)
def test_finalizer_rejects_stale_companion_clock_without_promotion(tmp_path, artifact, pattern):
    case, manifest = _write_complete_case(tmp_path)
    if artifact == "xdmf":
        (case / "solution/vpm_000020.xdmf").write_text(
            '<Xdmf><Domain><Grid><Time Value="0.049"/></Grid></Domain></Xdmf>',
            encoding="utf-8",
        )
    else:
        surface = pv.PolyData(np.zeros((4, 3)))
        surface.field_data["time"] = np.array([0.049])
        surface.field_data["TimeValue"] = np.array([0.049])
        surface.save(case / "solution/vlm_000020.vtp")
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match=pattern):
        finalize_lineage(case)
    assert manifest.read_bytes() == before


def test_finalizer_rejects_incomplete_loading_steps_without_promotion(tmp_path):
    case, manifest = _write_complete_case(tmp_path)
    path = case / "samples/delta_wing/vlm_spanwise_front_wing.csv"
    frame = pd.read_csv(path)
    frame = frame[frame.step != 7]
    frame.to_csv(path, index=False)
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="do not cover every required step"):
        finalize_lineage(case)
    assert manifest.read_bytes() == before
