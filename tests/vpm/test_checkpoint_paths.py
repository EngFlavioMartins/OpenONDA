"""Checkpoint output must be rooted in ``case_dir``, never the caller's cwd.

Regression for a bug where ``SolverIO`` read the *raw*, possibly-relative
``setup.checkpoint_directory`` instead of the solver's already
``case_dir``-resolved ``checkpoint_directory``. A relative
``checkpoint_directory`` (e.g. the tutorial default ``"solution"``) combined
with a caller whose cwd differs from ``case_dir`` silently wrote checkpoints
under cwd instead of the case directory.
"""

from __future__ import annotations

import os

import numpy as np

from source.solvers.VPM import VPMSetup, VPMSolver
from source.solvers.VPM.config.types import AdvectionConfig, StretchingConfig, ViscousConfig

_SIGMA = 0.2


def _make_solver(case_dir) -> VPMSolver:
    config = VPMSetup(
        time_step_size=0.05,
        compute_device="CPU",
        advection=AdvectionConfig(scheme="RK2"),
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig.inviscid(),
        checkpoint_interval_steps=1,
        logging_interval_steps=0,
        # Relative, as tutorials commonly configure it -- the whole point of
        # the regression is that this must resolve against case_dir, not cwd.
        checkpoint_directory="solution",
    )
    solver = VPMSolver(setup=config, case_dir=case_dir)
    volume = (4.0 / 3.0) * np.pi * _SIGMA**3
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        vortex_strength=np.array([[0.0, 0.0, 1.0]]),
        core_radius=np.array([_SIGMA]),
        volume=np.array([volume]),
        kinematic_viscosity=np.array([1e-5]),
    )
    return solver


def test_checkpoint_is_rooted_in_case_dir_not_cwd(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    elsewhere = tmp_path / "elsewhere"
    case_dir.mkdir()
    elsewhere.mkdir()

    monkeypatch.chdir(elsewhere)

    solver = _make_solver(case_dir)
    solver.advance()

    assert solver.io.export_dir == str((case_dir / "solution").resolve())
    assert (case_dir / "solution").is_dir()
    assert any((case_dir / "solution").glob("vpm_*.h5"))

    # No stray output was written under cwd (elsewhere) or its "solution".
    assert not (elsewhere / "solution").exists()
    assert list(elsewhere.iterdir()) == []


def test_checkpoint_directory_matches_solver_resolved_path(tmp_path, monkeypatch):
    case_dir = tmp_path / "case2"
    case_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    solver = _make_solver(case_dir)

    assert os.path.isabs(solver.checkpoint_directory)
    assert solver.io.export_dir == solver.checkpoint_directory
