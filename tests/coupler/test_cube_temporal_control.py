"""Same-mesh temporal comparison must not qualify missing or confounded evidence."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/postprocess_temporal_control.py"
)


def module():
    spec = importlib.util.spec_from_file_location("cube_temporal_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def case(root, name, dt, *, bias=0, drift=0, accepted_dt=True, frequency=0.5):
    samples = root / "samples" / name
    solution = root / "solution" / name
    samples.mkdir(parents=True)
    (solution / "fvm").mkdir(parents=True)
    (samples / "grid_run.json").write_text(
        json.dumps({"case": name, "cell_size": 0.045, "cell_count": 2, "end_time": 120})
    )
    config = {
        "transport": {"density": 1, "kinematic_viscosity": 0.001},
        "turbulence": {},
        "boundaries": [],
        "schemes": {"time_scheme": "backward"},
        "linear": {},
        "pimple": {},
        "initial_velocity": [1, 0, 0],
        "initial_kinematic_pressure": 0,
        "time": {
            "time_step_size": dt,
            "adjustment": {"maximum": 0.5, "maximum_time_step_size": dt},
        },
        "samplers": [
            {
                "type": "ForceSampler",
                "file_name": "forces_history",
                "reference_length": 1,
                "reference_velocity": 1,
                "reference_area": 1,
                "patch_names": ["cube"],
            }
        ],
    }
    (solution / "fvm_metadata.json").write_text(
        json.dumps(
            {"lifecycle": {"status": "complete"}, "state": {"time": 120}, "configuration": config}
        )
    )
    generation = {
        "resolved_surface_patch_sizes": {"cube": 0.045},
        "method": "cartesian",
        "cartesian_report": {"surface_hashes": {"cube": "geometry-a"}},
        "requested_sizes": [{"name": "cube", "requested": 0.045}],
    }
    np.savez(
        solution / "fvm/mesh.npz",
        vertex_position=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        owners=np.array([0, 1]),
        cell_sizes=np.array([0.045, 0.045]),
        metadata=json.dumps({"mesh_generation": generation}),
    )
    with (samples / "forces_history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        names = ["time", "drag_coefficient", "lift_coefficient", "side_force_coefficient"]
        if accepted_dt:
            names.append("accepted_time_step_size")
        writer.writerow(names)
        for t in np.linspace(0, 120, 1201):
            row = [
                t,
                1 + bias + 0.01 * np.sin(4 * np.pi * frequency * t) + drift * t,
                0.1 * np.sin(2 * np.pi * frequency * t),
                0.1 * np.cos(2 * np.pi * frequency * t),
            ]
            writer.writerow(row + [dt] if accepted_dt else row)
    for profile, y in (("centreline", 0), ("offaxis_y075", 0.75)):
        with (samples / f"{profile}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "time",
                    "position_x",
                    "position_y",
                    "position_z",
                    "velocity_x",
                    "velocity_y",
                    "velocity_z",
                ]
            )
            for t in np.linspace(0, 120, 25):
                for x in (0.6, 1.0, 2.0, 4.0, 8.0):
                    writer.writerow([t, x, y, 0, 0.8 + 0.02 * x + bias, 0, 0])
    return samples, solution


def campaign(tmp_path, **kwargs):
    case(tmp_path, "grid_h0045", 0.005, bias=0.001, **kwargs)
    case(tmp_path / "temporal", "time_h0045_dt_half", 0.0025, **kwargs)
    return module().analyse_temporal_control(tmp_path, tmp_path / "report")


def test_completed_matching_control_qualifies_without_spatial_claim(tmp_path):
    report = campaign(tmp_path)
    assert report["status"] == "passes_engineering_temporal_screen"
    assert report["mean_drag_and_profile_screen"]["status"] == "passes_engineering_screen"
    assert report["identity"]["matching"]
    assert report["sampled_dt_refinement_ratios"]["p90"] == 0.5
    assert "not established" in report["scope"]
    assert (tmp_path / "report/temporal_control.md").exists()


def test_missing_control_still_writes_unqualified_report(tmp_path):
    report = module().analyse_temporal_control(tmp_path, tmp_path / "report")
    assert report["status"] == "unqualified"
    assert len(report["reasons"]) == 2
    assert (tmp_path / "report/temporal_control.json").exists()


def test_stationary_small_differences_cannot_hide_mesh_identity_change(tmp_path):
    campaign(tmp_path)
    path = tmp_path / "temporal/solution/time_h0045_dt_half/fvm/mesh.npz"
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["owners"] = np.array([1, 0])
    np.savez(path, **arrays)
    report = module().analyse_temporal_control(tmp_path, tmp_path / "report")
    assert report["status"] == "unqualified"
    assert "realized_mesh_array_sha256" in report["identity"]["differences"]


def test_drift_and_missing_actual_dt_withhold_qualification(tmp_path):
    report = campaign(tmp_path, drift=0.001, accepted_dt=False)
    assert report["status"] == "unqualified"
    assert any("Actual accepted" in reason for reason in report["reasons"])
    assert any("Cd drift" in reason for reason in report["reasons"])


def test_frequency_resolution_is_separate_from_mean_cd_and_profiles(tmp_path):
    report = campaign(tmp_path, frequency=0.2)
    assert report["status"] == "unqualified"
    assert report["mean_drag_and_profile_screen"]["status"] == "passes_engineering_screen"
    assert not report["metrics"]["strouhal_lift"]["passes"]
    assert report["metrics"]["strouhal_lift"]["spectral_resolution"] == [1 / 60, 1 / 60]


def test_short_horizon_cannot_be_extrapolated_to_full_window(tmp_path):
    campaign(tmp_path)
    forces = tmp_path / "temporal/samples/time_h0045_dt_half/forces_history.csv"
    lines = forces.read_text().splitlines()
    forces.write_text("\n".join(lines[:500]) + "\n")
    report = module().analyse_temporal_control(tmp_path, tmp_path / "report")
    assert report["status"] == "unqualified"
    assert any("outside a sampled history" in reason for reason in report["reasons"])


def test_cfl_limited_runs_with_same_actual_dt_are_not_a_temporal_control(tmp_path):
    campaign(tmp_path)
    paths = [
        tmp_path / "samples/grid_h0045/forces_history.csv",
        tmp_path / "temporal/samples/time_h0045_dt_half/forces_history.csv",
    ]
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            names, rows = reader.fieldnames, list(reader)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            for row in rows:
                row["accepted_time_step_size"] = 0.001
                writer.writerow(row)
    report = module().analyse_temporal_control(tmp_path, tmp_path / "report")
    assert report["status"] == "unqualified"
    assert any(
        "insufficient observed timestep separation" in reason for reason in report["reasons"]
    )


def test_different_physical_profile_line_is_not_compared_as_same_line(tmp_path):
    campaign(tmp_path)
    path = tmp_path / "temporal/samples/time_h0045_dt_half/offaxis_y075.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names, rows = reader.fieldnames, list(reader)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            row["position_y"] = 1.0
            writer.writerow(row)
    report = module().analyse_temporal_control(tmp_path, tmp_path / "report")
    assert report["status"] == "unqualified"
    assert not report["profiles"]["offaxis_y075"]["available"]
