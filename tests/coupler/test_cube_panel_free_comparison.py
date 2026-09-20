"""Qualification gates must measure known offsets and reject incomplete histories."""

import json

import numpy as np
import pandas as pd
import pytest

from studies.panel_removal import compare_cube_run as comparison


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    trial = tmp_path / "trial"
    monkeypatch.setattr(comparison, "BASELINE", baseline)
    times = np.arange(1, 81) * 0.05
    for directory, offset, prefix in (
        (trial / "samples", 0.01, "fvm_"),
        (baseline / "samples", 0.0, "fvm_"),
        (baseline / "reference_flow/samples/fine", 0.0, ""),
    ):
        directory.mkdir(parents=True)
        pd.DataFrame(
            {
                "time": times,
                "drag_coefficient": np.full(80, 1 + offset),
                "lift_coefficient": np.zeros(80),
            }
        ).to_csv(directory / "forces_history.csv", index=False)
        for name in ("centreline", "offaxis_y075"):
            x = np.linspace(-1.5, 1.5, 31)
            frames = [
                pd.DataFrame(
                    {
                        "time": np.full(len(x), time),
                        "position_x": x,
                        "velocity_x": 1 + 0.1 * x + offset,
                        "velocity_y": np.zeros(len(x)),
                        "velocity_z": np.zeros(len(x)),
                    }
                )
                for time in np.arange(0.25, 4.01, 0.25)
            ]
            pd.concat(frames).to_csv(directory / f"{prefix}{name}.csv", index=False)
    for directory, panel, sweeps in ((baseline, "PanelSolver", 3), (trial, None, 6)):
        solution = directory / "solution"
        (solution / "fvm").mkdir(parents=True)
        (solution / "fvm/mesh.npz").write_bytes(b"identical saved mesh")
        (solution / "fvm_metadata.json").write_text(
            json.dumps(
                {
                    "configuration": {
                        "time": {"output_schedule": None, "time_step_size": 0.01},
                        "samplers": [],
                    }
                }
            )
        )
        (solution / "vpm_metadata.json").write_text(
            json.dumps(
                {
                    "configuration": {
                        "numerics": {
                            "panel_solver": panel,
                            "bodies": [{}] if panel else [],
                            "viscosity": 0.001,
                        }
                    }
                }
            )
        )
        (solution / "run_metadata.json").write_text(
            json.dumps(
                {
                    "coupler": {
                        "interface_iterations": sweeps,
                        "interface_normal_tolerance": 1e-6,
                        "interface_gradient_tolerance": 1e-6,
                    }
                }
            )
        )
        for name in ("centreline", "offaxis_y075"):
            (directory / f"samples/vpm_{name}.csv").write_bytes(
                (directory / f"samples/fvm_{name}.csv").read_bytes()
            )
    records = [
        json.dumps(
            {
                "time": float(t),
                "step": index + 1,
                "interface_iteration": {"converged": True, "sweeps": 3},
            }
        )
        for index, t in enumerate(times)
    ]
    (trial / "solution/coupler_diagnostics.jsonl").write_text("\n".join(records))
    (baseline / "solution/coupler_diagnostics.jsonl").write_text("\n".join(records))
    (trial / "experiment.json").write_text(json.dumps({"panel": False}))
    return trial


def test_known_drag_and_velocity_offsets_are_not_fitted_away(experiment):
    report = comparison.report(experiment)
    assert report["gate_passed"]
    assert report["forces"]["cd_mean_change_relative"] == pytest.approx(0.01)
    assert report["forces"]["cd_rms_change_relative"] == pytest.approx(0.01)
    for profile in report["profiles"].values():
        assert profile["change_rms_Uinf"] == pytest.approx(0.01)
        assert profile["trial_reference_rms_Uinf"] == pytest.approx(0.01)


def test_missing_final_profile_cannot_pass_four_second_gate(experiment):
    path = experiment / "samples/fvm_centreline.csv"
    data = pd.read_csv(path)
    data.loc[data.time < 4].to_csv(path, index=False)
    report = comparison.report(experiment)
    assert not report["gate_passed"]
    assert not report["criteria"]["centreline_complete_window"]


def test_panel_run_cannot_admit_cylinder_experiment(experiment):
    (experiment / "experiment.json").write_text(json.dumps({"panel": True}))
    report = comparison.report(experiment)
    assert not report["gate_passed"]
    assert not report["criteria"]["verified_panel_free_experiment"]


def test_short_profile_excursion_cannot_hide_in_three_second_average(experiment):
    path = experiment / "samples/fvm_offaxis_y075.csv"
    data = pd.read_csv(path)
    data.loc[np.isclose(data.time, 1.25), "velocity_x"] += 0.04
    data.to_csv(path, index=False)
    report = comparison.report(experiment)
    sustained = report["sustained_window"]["profiles"]["offaxis_y075"]
    assert sustained["change_rms_Uinf"] < 0.02
    assert sustained["change_max_snapshot_rms_Uinf"] == pytest.approx(0.05)
    assert not report["gate_passed"]
    assert not report["criteria"]["sustained_offaxis_y075_every_snapshot_below_2pct_Uinf"]
    assert report["profiles"]["offaxis_y075"]["change_rms_Uinf"] == pytest.approx(0.01)


@pytest.mark.parametrize("missing", ["force", "profile", "interface"])
def test_internal_sampling_gaps_cannot_pass_by_preserving_endpoints(experiment, missing):
    if missing == "interface":
        path = experiment / "solution/coupler_diagnostics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text(
            "\n".join(json.dumps(row) for row in rows if not np.isclose(row["time"], 2.5))
        )
    else:
        file = "forces_history.csv" if missing == "force" else "fvm_centreline.csv"
        path = experiment / "samples" / file
        data = pd.read_csv(path)
        data.loc[~np.isclose(data.time, 2.5)].to_csv(path, index=False)
    report = comparison.report(experiment)
    assert not report["gate_passed"]
    key = {
        "force": "complete_force_window",
        "profile": "centreline_complete_window",
        "interface": "complete_interface_window",
    }[missing]
    assert not report["criteria"][key]


def test_identity_rejects_viscosity_change_and_mesh_change(experiment):
    path = experiment / "solution/vpm_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["configuration"]["numerics"]["viscosity"] = 0.002
    path.write_text(json.dumps(metadata))
    report = comparison.report(experiment)
    assert not report["gate_passed"]
    assert (
        report["identity_audit"]["unexpected_differences"][0]["path"] == ".vpm.numerics.viscosity"
    )
    (experiment / "solution/fvm/mesh.npz").write_bytes(b"another mesh")
    assert not comparison.identity_audit(experiment)["matching_mesh"]


def test_startup_is_reported_and_persistent_vpm_error_rejects_admission(experiment):
    path = experiment / "samples/forces_history.csv"
    data = pd.read_csv(path)
    data.loc[np.isclose(data.time, 0.05), "drag_coefficient"] += 1
    data.to_csv(path, index=False)
    path = experiment / "samples/vpm_offaxis_y075.csv"
    data = pd.read_csv(path)
    data.loc[data.position_x > 0.5, "velocity_x"] += 0.1
    data.to_csv(path, index=False)
    report = comparison.report(experiment)
    assert not report["gate_passed"]
    assert not report["criteria"]["sustained_vpm_offaxis_y075_wake_only_change_below_2pct_Uinf"]
    assert report["full_transient_forces"]["cd_max_change"] == pytest.approx(1.01)
    wake = report["vpm_full_transient_profiles"]["offaxis_y075"]["wake_only"]
    assert wake["change_rms_Uinf"] == pytest.approx(0.11)
    assert len(report["full_transient_profiles"]["centreline"]["snapshots"]) == 16


def test_missing_vpm_wake_output_cannot_admit_cylinder(experiment):
    (experiment / "samples/vpm_centreline.csv").unlink()
    report = comparison.report(experiment)
    assert not report["gate_passed"]
    assert not report["criteria"]["sustained_vpm_centreline_wake_only_complete_window"]


def test_profile_excursion_between_reference_samples_is_still_checked(experiment):
    for directory, offset in ((comparison.BASELINE, 0.0), (experiment, 0.04)):
        path = directory / "samples/fvm_centreline.csv"
        data = pd.read_csv(path)
        extra = data.loc[np.isclose(data.time, 1.25)].copy()
        extra["time"] = 1.10  # No reference output exists at this time.
        extra["velocity_x"] += offset
        pd.concat([data, extra]).sort_values(["time", "position_x"]).to_csv(path, index=False)
    report = comparison.report(experiment)
    profile = report["sustained_window"]["profiles"]["centreline"]
    assert profile["samples"] == profile["reference_samples"] + 1
    assert profile["change_max_snapshot_rms_Uinf"] == pytest.approx(0.05)
    assert not report["criteria"]["sustained_centreline_every_snapshot_below_2pct_Uinf"]
    assert not report["gate_passed"]
