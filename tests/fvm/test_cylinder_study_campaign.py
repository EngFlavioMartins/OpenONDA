"""Numerical fixtures and lifecycle tests; no synthetic data is production evidence."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import study
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow.assets import (
    study_analysis as analysis,
)


def config():
    return study.read_config(study.CASE_DIR / "study_config.json")


def history(*, end=250.0, dt=0.01, frequency=0.2, drag=1.3, amplitude=0.4, drift=0.0):
    t = np.arange(0.0, end + dt / 2, dt)
    return {
        "time": t,
        "drag_coefficient": drag + 0.03 * np.cos(4 * np.pi * frequency * t) + drift * t,
        "lift_coefficient": amplitude * np.sin(2 * np.pi * frequency * t),
    }


def write_history(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["time", "patch", "drag_coefficient", "lift_coefficient"]
        )
        writer.writeheader()
        for t, cd, cl in zip(
            data["time"], data["drag_coefficient"], data["lift_coefficient"], strict=True
        ):
            writer.writerow(
                {"time": t, "patch": "cylinder", "drag_coefficient": cd, "lift_coefficient": cl}
            )


def test_cycle_statistics_resolve_coefficients_and_stationarity():
    result = analysis.cycle_statistics(history(), config())
    assert result["qualified"], result["reasons"]
    assert result["cycles"] == 20
    assert result["mean_cd"] == pytest.approx(1.3, abs=2e-7)
    assert result["mean_cl"] == pytest.approx(0.0, abs=2e-7)
    assert result["cd_rms"] == pytest.approx(0.03 / np.sqrt(2), rel=2e-4)
    assert result["cl_rms"] == pytest.approx(0.4 / np.sqrt(2), rel=2e-4)
    assert result["cl_amplitude"] == pytest.approx(0.4, rel=2e-4)
    assert result["strouhal"] == pytest.approx(0.2, abs=1e-7)


@pytest.mark.parametrize(
    "data, message",
    [
        (history(end=150.0), "end time"),
        (history(amplitude=0.0), "No resolved shedding"),
        (history(frequency=0.05), "complete settled cycles"),
        (history(dt=0.1), "sampling has gaps"),
    ],
)
def test_insufficient_histories_fail(data, message):
    with pytest.raises(ValueError, match=message):
        analysis.cycle_statistics(data, config())


def test_drifting_statistics_are_inconclusive():
    result = analysis.cycle_statistics(history(drift=0.0005), config())
    assert not result["qualified"]
    assert any("drift" in message for message in result["reasons"])


def test_duplicate_restart_times_are_rejected(tmp_path):
    data = history(end=2.0)
    data["time"][-1] = data["time"][-2]
    path = tmp_path / "forces.csv"
    write_history(path, data)
    with pytest.raises(ValueError, match="restart seam"):
        analysis.read_history(path)


def synthetic_records():
    values = {
        "mean_cd": 1.3,
        "cd_rms": 0.03,
        "cd_peak_to_peak": 0.06,
        "cl_rms": 0.28,
        "cl_amplitude": 0.4,
        "strouhal": 0.2,
    }
    records = {}
    for spec in study.run_matrix(config()):
        # Known second-order spatial error and first-order temporal error.
        spatial = 0.016 * (spec["dx"] / study.GRIDS["coarse"]) ** 2
        temporal = 0.0001 * spec["dt"] / config()["dt"]
        iterative = 0 if spec["tight"] else 0.00001
        row = {key: value * (1 + spatial + temporal + iterative) for key, value in values.items()}
        row.update(
            {
                "case": spec["name"],
                "dx": spec["dx"],
                "effective_h": 8 * spec["dx"],
                "mean_cl": 0.0,
                "qualified": True,
                "sampling_uncertainty": dict.fromkeys((*analysis.METRICS, "mean_cl"), 0.0),
            }
        )
        records[spec["name"]] = row
    return records


def test_budget_selects_fine_not_medium():
    _, candidates, selected = analysis.convergence_verdict(synthetic_records(), config())
    assert selected == "fine"
    assert not candidates["medium"]["passed"]
    assert candidates["fine"]["passed"]


def test_budget_selects_medium_when_it_is_sufficient():
    records = synthetic_records()
    for row in records.values():
        for key in analysis.METRICS:
            row[key] -= row[key] * 0.006 * (row["dx"] / study.GRIDS["coarse"]) ** 2
    _, _, selected = analysis.convergence_verdict(records, config())
    assert selected == "medium"


@pytest.mark.parametrize(
    "mode",
    ["equal", "noise", "temporal", "iteration", "unqualified", "oscillatory", "domain", "span"],
)
def test_no_false_grid_independence(mode):
    records = synthetic_records()
    if mode == "equal":
        for name in ("coarse", "medium", "fine"):
            records[name]["mean_cd"] = 1.3
    elif mode == "noise":
        for row in records.values():
            row["sampling_uncertainty"]["mean_cd"] = 0.02
    elif mode == "temporal":
        records["fine_dt4"]["mean_cd"] -= 0.1
    elif mode == "iteration":
        records["fine_tight"]["mean_cd"] -= 0.1
    elif mode == "unqualified":
        records["coarse"]["qualified"] = False
    elif mode in ("domain", "span"):
        records["fine_" + mode]["mean_cd"] += 0.1
    else:
        records["medium"]["mean_cd"] = 1.2
    _, _, selected = analysis.convergence_verdict(records, config())
    assert selected is None


def test_run_matrix_separates_time_and_iteration():
    runs = study.run_matrix(config())
    assert len(runs) == 11
    assert {row["dt"] for row in runs if row["name"] in study.GRIDS} == {config()["dt"]}
    for name in ("medium", "fine"):
        assert [
            row["dt"] for row in runs if row["name"] in (name, name + "_dt2", name + "_dt4")
        ] == [0.001, 0.0005, 0.00025]
    controls = study.flow_setup(runs[0], config())
    assert controls.time.adjustment is None
    assert controls.mesh.max_lsq_condition == 9  # No hidden quality waiver.
    assert controls.cores == 1


def test_resource_gate_prevents_accidental_fine_launch():
    assert study.resource_check(50000, config()) < 2.5
    with pytest.raises(RuntimeError, match="Resource gate"):
        study.resource_check(33506176, config())


def test_laptop_estimates_fit_budget_and_cover_measured_domain_probe():
    estimates = {name: study.estimated_cells(name) for name in study.MESH_CASES}
    # Recorded 2026-09-06 build, not a claim about solver peak memory.
    assert estimates["fine_domain"] >= 266429
    assert max(estimates.values()) == estimates["fine_domain"]
    for cells in estimates.values():
        assert study.resource_check(cells, config()) <= config()["memory_budget_gib"]


def test_laptop_physics_and_span_force_normalization():
    assert study.DOMAIN == (-8.0, 24.0, -10.0, 10.0, -0.5, 0.5)
    runs = {row["name"]: row for row in study.run_matrix(config())}
    base = study.flow_setup(runs["fine"], config())
    span = study.flow_setup(runs["fine_span"], config())
    assert base.transport.kinematic_viscosity == pytest.approx(1 / 150)
    assert base.samplers[0].reference_area == pytest.approx(1.0)
    assert span.samplers[0].reference_area == pytest.approx(0.75)
    assert {bc.name: bc.velocity_type for bc in base.boundaries} == {
        "inlet": "fixedValue",
        "outlet": "inletOutlet",
        "cylinder": "fixedValue",
        "ymin": "slip",
        "ymax": "slip",
        "zmin": "slip",
        "zmax": "slip",
    }


def test_checkpoint_retention_preserves_latest_and_unrelated_files(tmp_path):
    for step in (0, 2, 4):
        (tmp_path / f"checkpoint-{step:010d}.npz").write_bytes(b"generated fixture")
    (tmp_path / "checkpoint-user.npz").write_bytes(b"keep")
    study.prune_checkpoints(tmp_path, 2)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "checkpoint-0000000002.npz",
        "checkpoint-0000000004.npz",
        "checkpoint-user.npz",
    ]


def test_invalid_config_rejected_before_work(tmp_path):
    invalid = config()
    invalid["force_interval"] = 0.0101
    path = tmp_path / "config.json"
    study.write_json(path, invalid)
    with pytest.raises(ValueError, match="integer multiple"):
        study.read_config(path)


def test_unrelated_output_is_not_adopted(tmp_path):
    (tmp_path / "important.txt").write_text("keep")
    with pytest.raises(ValueError, match="nonempty directory"):
        study.campaign(tmp_path, config())
    assert (tmp_path / "important.txt").read_text() == "keep"


def test_missing_campaign_runs_still_produce_report_and_plots(tmp_path):
    from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow.assets.plot_grid_study import (
        plot_campaign,
    )

    study.write_json(
        tmp_path / "campaign.json", {"config": config(), "runs": study.run_matrix(config())}
    )
    report = analysis.report_campaign(tmp_path, failure="Synthetic resource-gate fixture")
    assert not report["grid_independent"] and report["selected_mesh"] is None
    assert any("fine" in reason for reason in report["reasons"])
    plot_campaign(tmp_path, report)
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "figures/grid_study.png").stat().st_size > 1000


def test_full_report_and_plots_from_synthetic_campaign(tmp_path, monkeypatch):
    """End-to-end report only; fake mesh metadata is explicitly injected."""
    from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow.assets.plot_grid_study import (
        plot_campaign,
    )

    cfg = config()
    study.write_json(tmp_path / "campaign.json", {"config": cfg, "runs": study.run_matrix(cfg)})
    monkeypatch.setattr(study, "verify_mesh", lambda *_: None)
    for grid in study.MESH_CASES:
        dx = study.spacing_for(grid)
        directory = tmp_path / "meshes" / grid
        directory.mkdir(parents=True)
        (directory / "mesh.npz").write_bytes(f"SYNTHETIC TEST ONLY {grid}".encode())
        (directory / "checkMesh.log").write_text("SYNTHETIC TEST ONLY")
        study.write_json(
            directory / "independent_check.json",
            {
                "passed": True,
                "mesh_sha256": study.sha256(directory / "mesh.npz"),
                "log_sha256": study.sha256(directory / "checkMesh.log"),
            },
        )
        study.write_json(
            directory / "mesh_report.json",
            {
                "mesh_identity": grid,
                "counts": {"cells": int(1000 * (0.025 / dx) ** 3)},
                "mesh_generation": {
                    "resolved_background_cell_size": 8 * dx,
                    "resolved_surface_patch_sizes": {"cylinder": dx},
                },
            },
        )
    for spec in study.run_matrix(cfg):
        directory = tmp_path / "runs" / spec["name"]
        # Keep temporal changes above sampled-extrema interpolation noise.
        error = 0.016 * (spec["dx"] / study.GRIDS["coarse"]) ** 2 + 0.0002 * spec["dt"] / cfg["dt"]
        data = history(
            frequency=0.2 * (1 + error), drag=1.3 * (1 + error), amplitude=0.4 * (1 + error)
        )
        data["drag_coefficient"] += (
            0.03 * error * np.cos(4 * np.pi * 0.2 * (1 + error) * data["time"])
        )
        path = tmp_path / "samples" / spec["name"] / "forces_history.csv"
        write_history(path, data)
        study.write_json(
            directory / "progress.json",
            {
                "completed": True,
                "time": cfg["end_time"],
                "spec": spec,
                "mesh_sha256": study.sha256(tmp_path / "meshes" / spec["grid"] / "mesh.npz"),
                "force_sha256": study.force_digest(path),
                "health": {
                    "all_finite": True,
                    "all_linear_converged": True,
                    "max_courant": 0.2,
                    "max_continuity": 1e-7,
                    "max_residual": 1e-7,
                },
            },
        )
    report = analysis.report_campaign(tmp_path)
    assert len(report["cases"]) == 11
    assert "medium" in report["candidate_meshes"]
    assert report["grid_independent"] and report["selected_mesh"] == "fine"
    report["evidence_label"] = "SYNTHETIC TEST ONLY — "
    plot_campaign(tmp_path, report)
    for name in ("grid_study", "force_histories", "error_budgets"):
        assert (tmp_path / f"figures/{name}.png").stat().st_size > 1000
    # Missing fine may never select medium on incomplete evidence.
    (tmp_path / "runs/fine/progress.json").unlink()
    incomplete = analysis.report_campaign(tmp_path)
    assert incomplete["selected_mesh"] is None
    previous_plot = (tmp_path / "figures/error_budgets.png").read_bytes()
    plot_campaign(tmp_path, incomplete)
    assert (tmp_path / "figures/error_budgets.png").read_bytes() != previous_plot


def test_real_fvm_worker_and_checkpoint_resume(tmp_path, monkeypatch):
    """Eight-cell actual FVM integration exercises the campaign lifecycle."""
    import openonda.fvm as fvm
    from source.solvers.fvm.io.mesh_storage import save_native_mesh
    from source.solvers.fvm.mesh.cartesian import structured_box

    cfg = {
        **config(),
        "dt": 0.001,
        "end_time": 0.004,
        "force_interval": 0.001,
        "checkpoint_interval": 0.002,
    }
    spec = study.run_matrix(cfg)[0]
    study.write_json(tmp_path / "campaign.json", {"config": cfg, "runs": [spec]})
    directory = tmp_path / "meshes/coarse"
    directory.mkdir(parents=True)
    mesh = structured_box(2, 2, 2)
    save_native_mesh(mesh, directory / "mesh.npz")
    monkeypatch.setattr(study, "verify_mesh", lambda *_: None)

    def small_setup(spec, cfg):
        return fvm.FVMSetup(
            case_name=spec["name"],
            cores=1,
            time=fvm.TimeConfig(time_step_size=spec["dt"], end_time=cfg["end_time"]),
            linear=fvm.LinearSolverConfig(linear_solver="spsolve", pressure_solver="spsolve"),
            backup=fvm.BackupConfig(schedule=None, write_at_end=False),
            transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.02),
            boundaries=[
                fvm.BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
                fvm.BoundaryConfig.outlet("xmax"),
                *[fvm.BoundaryConfig.wall(name) for name in ("ymin", "ymax", "zmin", "zmax")],
            ],
            samplers=(
                fvm.ForceSampler(
                    patch_names=["ymin"],
                    reference_velocity=1.0,
                    reference_area=1.0,
                    reference_length=1.0,
                    moment_centre=[0.0, 0.0, 0.0],
                    file_name="forces_history",
                    schedule=fvm.RunSchedule(every_n_steps=1),
                ),
            ),
        )

    monkeypatch.setattr(study, "flow_setup", small_setup)
    study.run_flow(tmp_path, "coarse")
    progress_path = tmp_path / "runs/coarse/progress.json"
    result = json.loads(progress_path.read_text())
    assert result["completed"] and result["health"]["steps"] == 4
    archive = tmp_path / "runs/coarse" / result["checkpoint"]
    assert archive.exists()
    # Exact completed-run reuse does not append samples.
    assert (tmp_path / "solution/coarse/mesh.vtu").is_file()
    before = (tmp_path / "samples/coarse/forces_history.csv").read_bytes()
    study.run_flow(tmp_path, "coarse")
    assert (tmp_path / "samples/coarse/forces_history.csv").read_bytes() == before
    # Resume t=.002 even with later samples present; the solver rewinds them.
    result.update(completed=False, time=0.002, checkpoint="checkpoint-0000000002.npz")
    result["checkpoint_sha256"] = study.sha256(archive.with_name(result["checkpoint"]))
    result["health"]["steps"] = 2
    result["force_sha256"] = study.force_digest(
        tmp_path / "samples/coarse/forces_history.csv", 0.002
    )
    study.write_json(progress_path, result)
    study.run_flow(tmp_path, "coarse")
    result = json.loads(progress_path.read_text())
    assert result["completed"] and result["health"]["steps"] == 4
    with (tmp_path / "samples/coarse/forces_history.csv").open() as stream:
        times = [float(row["time"]) for row in csv.DictReader(stream)]
    np.testing.assert_allclose(times, [0.001, 0.002, 0.003, 0.004])
    force_path = tmp_path / "samples/coarse/forces_history.csv"
    force_path.write_text(force_path.read_text().replace("0.001", "0.00101", 1))
    with pytest.raises(ValueError, match="checksum changed"):
        study.run_flow(tmp_path, "coarse")


@pytest.mark.parametrize(
    "verdict, exit_code, accepted",
    [("Mesh OK.\n", 0, True), ("Failed 1 mesh checks.\n", 0, False), ("Mesh OK.\n", 1, False)],
)
def test_independent_checker_requires_explicit_pass(
    tmp_path, monkeypatch, verdict, exit_code, accepted
):
    from types import SimpleNamespace

    from source.solvers.fvm.io.mesh_storage import save_native_mesh
    from source.solvers.fvm.mesh.cartesian import structured_box

    save_native_mesh(structured_box(2, 2, 2), tmp_path / "mesh.npz")
    executable = tmp_path / "checkMesh"
    executable.write_text("synthetic checker identity")
    monkeypatch.setattr(study.native, "CHECKMESH", executable)
    monkeypatch.setattr(study.native, "LAUNCHER", tmp_path / "absent-launcher")

    def check(command, **kwargs):
        assert "-allGeometry" in command and "-allTopology" in command
        kwargs["stdout"].write(verdict)
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(study.subprocess, "run", check)
    if accepted:
        assert study.independent_check(tmp_path)["passed"]
    else:
        with pytest.raises(RuntimeError, match="qualification failed"):
            study.independent_check(tmp_path)
    assert json.loads((tmp_path / "independent_check.json").read_text())["passed"] is accepted


def test_concurrent_campaign_is_rejected(tmp_path):
    import fcntl

    output = tmp_path / "study"
    with (tmp_path / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already owns"):
            study.campaign(output, config())
    assert not output.exists()
