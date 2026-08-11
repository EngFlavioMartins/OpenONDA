"""Regression test for the six-case vortexInteractions launcher."""

from pathlib import Path


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

    assert 'if variant == "les_stabilized" or solver.time_step % OUTPUT_FREQUENCY:' in setup
    assert 'status="resolution_lost" if resolution_lost else "completed"' in setup
    assert 'if variant == "les_stabilized":' in gate
    assert "stabilized simulation did not reach its requested end time" in gate
