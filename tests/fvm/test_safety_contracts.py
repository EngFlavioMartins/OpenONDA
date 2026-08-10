"""Regression tests for fail-fast native FVM configuration."""

import numpy as np
import pytest

from source.solvers.FVM import DynamicMeshConfig, FVMSetup, Solver
from source.solvers.FVM.fields.diagnostics import _should_compute_yplus


def test_dynamic_mesh_is_explicitly_unsupported(hand_built_3d_mesh, tmp_path):
    config = FVMSetup(
        case_name="moving",
        dynamic_mesh=DynamicMeshConfig.rigid(velocity=[1.0, 0.0, 0.0]),
    )
    with pytest.raises(NotImplementedError, match="ALE mesh-flux"):
        Solver(config, case_dir=str(tmp_path), mesh_data=hand_built_3d_mesh)


def test_turbulence_failure_does_not_switch_to_laminar():
    class BrokenModel:
        def compute_nut(self, *args):
            raise RuntimeError("LES failed")

    solver = Solver.__new__(Solver)
    solver.turbulence = BrokenModel()
    solver.U = np.zeros((1, 3))
    solver.mesh_data = {}
    solver.geo_data = {}
    solver.config = FVMSetup(case_name="les")
    with pytest.raises(RuntimeError, match="LES failed"):
        solver.compute_effective_viscosity()


def test_fixed_value_inlet_is_not_auto_selected_as_wall():
    inlet = {"name": "inlet", "type": "patch", "bc_type_U": "fixedValue"}
    wall = {"name": "body", "type": "wall", "bc_type_U": "fixedValue"}
    assert not _should_compute_yplus(inlet, None)
    assert _should_compute_yplus(wall, None)
