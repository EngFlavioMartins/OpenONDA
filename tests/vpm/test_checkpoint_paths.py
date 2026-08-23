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
from types import SimpleNamespace

import numpy as np

from source.solvers.vpm import VPMSetup, VPMSolver
from source.solvers.vpm.config.types import AdvectionConfig, StretchingConfig, ViscousConfig
from source.solvers.vpm.io.solver_io import SolverIO

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
    particle_volume = (4.0 / 3.0) * np.pi * _SIGMA**3
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        vortex_strength=np.array([[0.0, 0.0, 1.0]]),
        core_radius=np.array([_SIGMA]),
        particle_volume=np.array([particle_volume]),
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


def test_panel_force_csv_is_rooted_in_case_samples(tmp_path):
    case_dir = tmp_path / "case"
    checkpoint_directory = case_dir / "solution"

    class PanelSolver:
        lattice = SimpleNamespace(n_panels=1)
        density = 1.0
        freestream_velocity = np.array([1.0, 0.0, 0.0])

        @staticmethod
        def compute_forces_coefficients(**_kwargs):
            return {
                "lift_coefficient": 0.2,
                "drag_coefficient": 0.1,
                "force_x": 0.1,
                "force_y": 0.0,
                "force_z": 0.2,
            }

    solver = SimpleNamespace(
        case_dir=case_dir,
        checkpoint_directory=str(checkpoint_directory),
        checkpoint_interval_steps=0,
        checkpoint_name="",
        setup=SimpleNamespace(sample_subdirectory=None),
        panel_solver=PanelSolver(),
    )
    SolverIO(solver)._export_panel_loads(0.1)

    assert (case_dir / "samples" / "vpm_forces.csv").is_file()
    assert not (checkpoint_directory / "samples").exists()
