"""Regression test for the six-case vortexInteractions launcher."""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openonda.vpm import RingDiagnosticsSampler


def test_launcher_runs_six_cases_then_their_physics_gate():
    tutorial = Path(__file__).parents[2] / "tutorials/VPM/vortexInteractions"
    launcher = (tutorial / "allrun.sh").read_text(encoding="utf-8")

    assert 'python -u rings_setup.py "$case_name"' in launcher
    assert "python assets/check_run.py" in launcher
    assert "rm " not in launcher
    assert "METHODS" not in launcher
    assert "RUN_FAMILIES" not in launcher
    for name in (
        "leapfrog_dns",
        "leapfrog_les",
        "leapfrog_les_stabilized",
        "collide_dns",
        "collide_les",
        "collide_les_stabilized",
    ):
        assert name in launcher


def test_stabilized_cases_cannot_use_the_baseline_resolution_stop():
    tutorial = Path(__file__).parents[2] / "tutorials/VPM/vortexInteractions"
    setup = (tutorial / "rings_setup.py").read_text(encoding="utf-8")
    gate = (tutorial / "assets/check_run.py").read_text(encoding="utf-8")

    assert 'if variant == "les_stabilized" or solver.time_step % DIAGNOSTIC_FREQUENCY:' in setup
    assert 'status="resolution_lost" if resolution_lost else "completed"' in setup
    assert 'if variant == "les_stabilized":' in gate
    assert "stabilized simulation did not reach its requested end time" in gate


def test_tutorial_uses_dense_diagnostics_and_animation_snapshots():
    tutorial = Path(__file__).parents[2] / "tutorials/VPM/vortexInteractions"
    setup = (tutorial / "rings_setup.py").read_text(encoding="utf-8")

    assert "PARTICLE_SPACING = 0.04" in setup
    assert 'LES_COEFFICIENT = {"leapfrog": 0.16, "collide": 0.32}' in setup
    assert "DIAGNOSTIC_FREQUENCY = 5" in setup
    assert "SNAPSHOT_FREQUENCY = 10" in setup
    assert "samplers=(RingDiagnosticsSampler(),)" in setup
    assert 'solver.backup_solution(str(output_dir / f"vpm_{case_name}"))' in setup


def test_plots_use_solver_managed_diagnostics_only():
    assets = Path(__file__).parents[2] / "tutorials/VPM/vortexInteractions/assets"
    common = (assets / "_common.py").read_text(encoding="utf-8")
    trajectory = (assets / "plot_rings_trajectory.py").read_text(encoding="utf-8")

    assert "flow_integrals.csv" in common
    assert "ring_diagnostics.csv" in common
    assert "read_ring_diagnostics" in trajectory
    assert "read_log_diagnostics" not in common
    assert "stability_metrics.csv" not in common
    assert "h5py" not in common
    assert "h5py" not in trajectory


def test_builtin_ring_sampler_writes_each_particle_group(tmp_path):
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    positions = []
    circulations = []
    groups = []
    for group_id, center in enumerate((-0.5, 0.5)):
        position = np.column_stack(
            (
                np.full_like(theta, center),
                np.cos(theta),
                np.sin(theta),
            )
        )
        circulation = np.column_stack(
            (
                np.zeros_like(theta),
                -np.sin(theta),
                np.cos(theta),
            )
        )
        positions.append(position)
        circulations.append(circulation)
        groups.append(np.full(len(theta), group_id))

    solver = SimpleNamespace(
        particles_positions=np.vstack(positions),
        particles_circulation=np.vstack(circulations),
        particles_group_ids=np.concatenate(groups),
    )
    output = tmp_path / "ring_diagnostics.csv"
    RingDiagnosticsSampler().save_csv(solver, output, time=0.25, step=5)

    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["group_id"]) for row in rows] == [0, 1]
    assert [int(row["time_step"]) for row in rows] == [5, 5]
    np.testing.assert_allclose([float(row["major_radius"]) for row in rows], 1.0)
