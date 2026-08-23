"""Guard the runtime logs against stale narrative labels."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "forbidden"),
    [
        ("source/solvers/vpm/core/evolution.py", "Grid diffusion audit"),
        ("source/solvers/vpm/core/evolution.py", "Performing GBD"),
        ("source/solvers/vpm/core/evolution.py", "Applying global GBD"),
        ("source/solvers/vpm/physics/diffusion/grid.py", "Moment recovery retained"),
        ("source/coupler/boundary.py", "VPM boundary-condition deficit"),
        ("source/coupler/boundary.py", "[Resync]"),
        ("source/coupler/boundary.py", "[Sub-cycle]"),
        ("source/coupler/boundary.py", "u_x/U∞"),
        ("source/coupler/vorticity_transfer.py", "compatible velocity-defect curl ready"),
        ("source/coupler/vorticity_transfer.py", "[Transfer step"),
        ("source/coupler/reporting.py", "[Timing step"),
        ("source/coupler/solver.py", "FVM-VPM COUPLED SOLVER"),
        ("source/solvers/vpm/io/checkpoint.py", "Checkpoint saved:"),
        ("source/solvers/vpm/io/solver_io.py", "[INFO]"),
        ("source/solvers/vpm/io/sampler.py", "(Warning)"),
        ("source/solvers/vpm/particles/distribution.py", "(warning)"),
        ("source/solvers/vpm/particles/container.py", "(numpy arrays)"),
        ("source/solvers/vpm/stabilization/manager.py", "|dGamma|/|Gamma|"),
        ("source/solvers/vpm/io/logging.py", "Time for this step:"),
        ("source/solvers/vpm/io/logging.py", "PARTICLE CLEANUP:"),
        ("source/solvers/vpm/io/logging.py", "reduce time-step"),
        ("source/solvers/fvm/io/logging.py", "Time for this step:"),
        ("source/solvers/fvm/io/logging.py", "Total simulation time:"),
    ],
)
def test_runtime_source_excludes_stale_log_phrases(relative_path, forbidden):
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    assert forbidden not in text


def test_runtime_logs_use_subsystem_operation_prefixes():
    expected = {
        "source/solvers/vpm/core/evolution.py": "[VPM][GBD]",
        "source/solvers/fvm/io/logging.py": "[FVM][Warning]",
        "source/coupler/boundary.py": "[Coupler][BoundaryFlux]",
        "source/coupler/vorticity_transfer.py": "[Coupler][Transfer]",
    }
    for relative_path, prefix in expected.items():
        assert prefix in (ROOT / relative_path).read_text(encoding="utf-8")
