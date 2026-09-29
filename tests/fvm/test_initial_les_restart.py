"""LES backups distinguish an unevaluated initial model from missing saved state."""

from dataclasses import replace

import numpy as np
import pytest

from source.solvers.fvm import FVMSolver, TurbulenceConfig
from source.solvers.fvm.io.backup import decode_state, encode_state
from source.solvers.fvm.mesh.cartesian import structured_box
from tests.fvm.test_restart_and_diagnostics import _setup


def _solver(directory):
    setup = replace(_setup(), turbulence=TurbulenceConfig.equilibrium_smagorinsky())
    return FVMSolver(setup, str(directory), mesh_data=structured_box(2, 2, 2))


def test_initial_les_backup_replays_the_first_step(tmp_path):
    with _solver(tmp_path / "reference") as reference, _solver(tmp_path / "resumed") as resumed:
        assert reference.eddy_viscosity is None
        checkpoint = tmp_path / "initial.npz"
        reference.save_state(checkpoint)
        resumed.load_state(checkpoint)
        assert resumed.eddy_viscosity is None
        reference.advance()
        resumed.advance()
        for name in ("velocity", "kinematic_pressure", "eddy_viscosity"):
            np.testing.assert_array_equal(getattr(resumed, name), getattr(reference, name))


def test_accepted_les_backup_cannot_omit_eddy_viscosity(tmp_path):
    with _solver(tmp_path / "accepted") as solver:
        solver.advance()
        checkpoint = tmp_path / "accepted.npz"
        solver.save_state(checkpoint)
        with np.load(checkpoint, allow_pickle=False) as archive:
            state = decode_state({name: archive[name] for name in archive.files})
        state["eddy_viscosity"] = np.empty(0)
        np.savez(checkpoint, **encode_state(state))
        with pytest.raises(RuntimeError, match="eddy-viscosity shape"):
            solver.load_state(checkpoint)
